import ast
from pathlib import Path


class _TaskStarted(Exception):
    pass


def _img2imgapi(events):
    """Api.img2imgapi on its own, with a decoder that rejects "bad" inputs and a recording _run_generation_task."""
    import time
    from types import SimpleNamespace

    from fastapi.exceptions import HTTPException

    source = Path("modules/api/api.py").read_text(encoding="utf-8")
    api_class = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "Api")
    method = next(node for node in api_class.body if isinstance(node, ast.FunctionDef) and node.name == "img2imgapi")
    method.decorator_list = []  # decode_inline_images_once only memoizes decoding within the call
    module = ast.Module(body=[ast.ClassDef(name="Api", bases=[], keywords=[], body=[method], decorator_list=[])], type_ignores=[])

    def decode(text):
        events.append(("decode", text))
        if text.startswith("bad"):
            raise HTTPException(status_code=400, detail="Invalid encoded image")
        return f"image:{text}"

    namespace = {
        "HTTPException": HTTPException,
        "create_task_id": lambda kind: f"task({kind})",
        "decode_base64_to_image": decode,
        "scripts": SimpleNamespace(scripts_img2img="img2img-runner"),
        "time": time,
        "models": SimpleNamespace(StableDiffusionImg2ImgProcessingAPI=object),
    }
    exec(compile(ast.fix_missing_locations(module), "modules/api/api.py", "exec"), namespace)
    api = namespace["Api"]()
    api.default_script_arg_img2img = []

    def prepare(request, tabname, script_runner, default_script_args, update=None, extra_pop_fields=()):
        events.append(("prepare", update["mask"]))
        return {}, True, None, [], {}

    def run_generation_task(task_id, tabname, args, *rest, configure=None):
        events.append(("task", task_id))
        p = SimpleNamespace()
        configure(p)
        events.append(("init_images", p.init_images))
        raise _TaskStarted

    api._prepare_generation_api_request = prepare
    api._run_generation_task = run_generation_task
    return api, HTTPException


def _request(init_images, mask=None):
    from types import SimpleNamespace

    return SimpleNamespace(force_task_id=None, init_images=init_images, mask=mask, save_images=False, include_init_images=False)


def test_img2img_decodes_every_input_image_before_starting_the_task():
    # Invalid init images or masks must be rejected before the request waits for queue_lock and starts its task.
    import pytest

    for init_images, mask in ((["good"], "bad-mask"), (["good", "bad-init"], None)):
        events = []
        api, HTTPException = _img2imgapi(events)
        with pytest.raises(HTTPException):
            api.img2imgapi(_request(init_images, mask))
        assert not [event for event in events if event[0] == "task"]

    events = []
    api, _ = _img2imgapi(events)
    with pytest.raises(_TaskStarted):
        api.img2imgapi(_request(["a", "b"], "m"))
    assert events == [
        ("decode", "m"),
        ("prepare", "image:m"),
        ("decode", "a"),
        ("decode", "b"),
        ("task", "task(img2img)"),
        ("init_images", ["image:a", "image:b"]),
    ]
