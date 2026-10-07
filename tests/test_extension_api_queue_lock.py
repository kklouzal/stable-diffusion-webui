"""Extension endpoints that use state a generation uses run between generations under queue_lock, in the threadpool
rather than on the event loop: /controlnet/detect (the preprocessor models and their cache),
/controlnet/model_list?update=true (the ControlNet model registry) and /sdapi/v1/refresh-loras (the LoRA registries)."""

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
from pydantic import BaseModel

SOURCE = Path("extensions/sd-webui-controlnet/scripts/api.py")
LORA_SOURCE = Path("extensions-builtin/Lora/scripts/lora_script.py")


class RecordingLock:
    def __init__(self, events):
        self.events = events
        self.held = False

    def __enter__(self):
        self.events.append("lock")
        self.held = True

    def __exit__(self, *exc):
        self.held = False
        self.events.append("unlock")


class FakeApp:
    def __init__(self):
        self.routes = {}

    def _route(self, path):
        def register(func):
            self.routes[path] = func
            return func

        return register

    get = post = _route


def load_function(path, name, namespace):
    tree = ast.parse(path.read_text())
    func = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[func], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace.setdefault("gr", SimpleNamespace(Blocks=object))  # annotations of the route-registering function
    namespace.setdefault("FastAPI", object)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def load_routes(namespace):
    namespace.update(
        Body=lambda default, **_kwargs: default,
        BaseModel=BaseModel,
        List=List,
        Optional=Optional,
        np=np,
        HTTPException=RuntimeError,
        logger=SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, debug=lambda *a, **k: None),
    )
    app = FakeApp()
    load_function(SOURCE, "controlnet_api", namespace)(None, app)
    return app.routes


def test_detect_runs_the_preprocessor_under_queue_lock_off_the_event_loop():
    events = []
    lock = RecordingLock(events)

    class FakePreprocessor:
        label = "depth"
        requires_mask = accepts_mask = False
        returns_image = True

        def cached_call(self, img, **_kwargs):
            events.append(("call", lock.held))
            return SimpleNamespace(display_images=["detected"])

        def unload(self):
            events.append(("unload", lock.held))

    def decode(_b64):
        events.append(("decode", lock.held))
        return np.zeros((2, 2, 3), dtype=np.uint8)

    routes = load_routes({
        "call_queue": SimpleNamespace(queue_lock=lock),
        "Preprocessor": SimpleNamespace(get_preprocessor=lambda _name: FakePreprocessor()),
        "external_code": SimpleNamespace(to_base64_nparray=decode),
        "ControlNetUnit": lambda **kwargs: SimpleNamespace(**kwargs),
        "encode_to_base64": lambda image: image,
    })
    detect = routes["/controlnet/detect"]

    assert not inspect.iscoroutinefunction(detect)  # FastAPI runs a plain def in the threadpool
    result = detect(controlnet_module="depth", controlnet_input_images=["a", "b"], controlnet_processor_res=512,
                    controlnet_threshold_a=-1, controlnet_threshold_b=-1, controlnet_masks=[], low_vram=False)

    assert result == {"info": "Success", "images": ["detected", "detected"]}
    # inputs decode before the lock; every preprocessor call and the unload hold it
    assert events == [("decode", False), ("decode", False), "lock", ("call", True), ("call", True), ("unload", True), "unlock"]


def test_model_list_update_rebuilds_the_registry_under_queue_lock():
    events = []
    lock = RecordingLock(events)

    def get_models(update):
        events.append(("get_models", update, lock.held))
        return ["None"]

    routes = load_routes({"call_queue": SimpleNamespace(queue_lock=lock), "external_code": SimpleNamespace(get_models=get_models)})
    model_list = routes["/controlnet/model_list"]

    assert not inspect.iscoroutinefunction(model_list)
    assert model_list() == {"model_list": ["None"]}
    assert model_list(update=False) == {"model_list": ["None"]}
    assert events == ["lock", ("get_models", True, True), "unlock", ("get_models", False, False)]


def test_refresh_loras_rebuilds_the_registries_under_queue_lock():
    events = []
    lock = RecordingLock(events)

    def list_available_networks():
        events.append(("rescan", lock.held))

    app = FakeApp()
    namespace = {"call_queue": SimpleNamespace(queue_lock=lock), "networks": SimpleNamespace(list_available_networks=list_available_networks)}
    load_function(LORA_SOURCE, "api_networks", namespace)(None, app)
    refresh = app.routes["/sdapi/v1/refresh-loras"]

    assert not inspect.iscoroutinefunction(refresh)
    refresh()
    assert events == ["lock", ("rescan", True), "unlock"]
