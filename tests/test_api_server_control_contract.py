import ast
import os
from pathlib import Path
from types import SimpleNamespace


def load_api_control_class():
    source = Path("modules/api/api.py").read_text(encoding="utf8")
    module = ast.parse(source)
    api_class = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Api")
    wanted = {
        "__init__",
        "_call_with_queue_lock",
        "add_api_route",
        "_env_flag",
        "apply_openclaw_runtime_defaults",
        "get_cuda_graphs",
        "set_cuda_graphs",
        "get_openclaw_generation_diagnostics",
        "get_memory",
        "refresh_embeddings",
        "refresh_checkpoints",
        "refresh_vae",
        "unloadapi",
        "reloadapi",
        "kill_webui",
        "restart_webui",
        "stop_webui",
    }
    methods = [node for node in api_class.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    subset = ast.Module(body=[ast.ClassDef(name="Api", bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(subset)

    namespace = {"Any": object, "FastAPI": object, "Lock": object, "APIRouter": lambda: object(), "os": os}
    exec(compile(subset, "api-server-control", "exec"), namespace)
    return namespace["Api"]


class DummyLock:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        self.events.append("lock-enter")

    def __exit__(self, exc_type, exc, tb):
        self.events.append("lock-exit")


class DummyApp:
    def __init__(self):
        self.routes = []

    def add_api_route(self, path, endpoint, **kwargs):
        self.routes.append((path, endpoint.__name__, kwargs))


def _patch_minimal_init_globals(monkeypatch, api_class, *, api_server_stop=False):
    cmd_opts = SimpleNamespace(
        api_auth=None,
        api_server_stop=api_server_stop,
        timeout_keep_alive=30,
        tls_keyfile=None,
        tls_certfile=None,
    )
    shared_stub = SimpleNamespace(cmd_opts=cmd_opts)
    scripts_stub = SimpleNamespace(
        scripts_txt2img=SimpleNamespace(scripts=[object()]),
        scripts_img2img=SimpleNamespace(scripts=[object()]),
    )
    models_stub = SimpleNamespace(
        TextToImageResponse=object,
        ImageToImageResponse=object,
        ExtrasSingleImageResponse=object,
        ExtrasBatchImagesResponse=object,
        PNGInfoResponse=object,
        ProgressResponse=object,
        OptionsModel=object,
        FlagsModel=object,
        SamplerItem=object,
        SchedulerItem=object,
        UpscalerItem=object,
        LatentUpscalerModeItem=object,
        SDModelItem=object,
        SDVaeItem=object,
        HypernetworkItem=object,
        FaceRestorerItem=object,
        RealesrganItem=object,
        PromptStyleItem=object,
        EmbeddingsResponse=object,
        CreateResponse=object,
        TrainResponse=object,
        MemoryResponse=object,
        ScriptsList=object,
        ScriptInfo=object,
        ExtensionItem=object,
    )

    monkeypatch.setitem(api_class.__init__.__globals__, "api_middleware", lambda app: None)
    monkeypatch.setitem(api_class.__init__.__globals__, "shared", shared_stub)
    monkeypatch.setitem(api_class.__init__.__globals__, "scripts", scripts_stub)
    monkeypatch.setitem(api_class.__init__.__globals__, "models", models_stub)
    monkeypatch.setitem(api_class.__init__.__globals__, "ui", SimpleNamespace(create_ui=lambda: None))
    monkeypatch.setattr(api_class, "init_default_script_args", lambda self, runner: [], raising=False)
    monkeypatch.setattr(api_class, "apply_openclaw_runtime_defaults", lambda self: None, raising=False)

    for method_name in [
        "text2imgapi",
        "img2imgapi",
        "extras_single_image_api",
        "extras_batch_images_api",
        "pnginfoapi",
        "progressapi",
        "interrogateapi",
        "interruptapi",
        "skip",
        "get_config",
        "set_config",
        "get_sdpa_backend",
        "set_sdpa_backend",
        "get_precision_map",
        "get_cmd_flags",
        "get_samplers",
        "get_schedulers",
        "get_upscalers",
        "get_latent_upscale_modes",
        "get_sd_models",
        "get_sd_vaes",
        "get_hypernetworks",
        "get_face_restorers",
        "get_realesrgan_models",
        "get_prompt_styles",
        "get_embeddings",
        "create_embedding",
        "create_hypernetwork",
        "train_embedding",
        "train_hypernetwork",
        "get_memory",
        "get_scripts_list",
        "get_script_info",
        "get_extensions_list",
    ]:
        monkeypatch.setattr(api_class, method_name, lambda self: None, raising=False)
    return shared_stub


def test_refresh_endpoints_keep_queue_lock_and_side_effect_targets(monkeypatch):
    api_class = load_api_control_class()
    events = []
    api = api_class.__new__(api_class)
    api.queue_lock = DummyLock(events)

    embedding_db = SimpleNamespace(load_textual_inversion_embeddings=lambda **kwargs: events.append(("embeddings", kwargs)))
    sd_hijack_stub = SimpleNamespace(model_hijack=SimpleNamespace(embedding_db=embedding_db))
    shared_stub = SimpleNamespace(refresh_checkpoints=lambda: events.append("checkpoints"))
    shared_items_stub = SimpleNamespace(refresh_vae_list=lambda: events.append("vae"))

    monkeypatch.setitem(api_class.refresh_embeddings.__globals__, "sd_hijack", sd_hijack_stub)
    monkeypatch.setitem(api_class.refresh_checkpoints.__globals__, "shared", shared_stub)
    monkeypatch.setitem(api_class.refresh_vae.__globals__, "shared_items", shared_items_stub)

    api.refresh_embeddings()
    api.refresh_checkpoints()
    api.refresh_vae()

    assert events == [
        "lock-enter",
        ("embeddings", {"force_reload": True}),
        "lock-exit",
        "lock-enter",
        "checkpoints",
        "lock-exit",
        "lock-enter",
        "vae",
        "lock-exit",
    ]


def test_checkpoint_reload_endpoints_preserve_public_empty_response(monkeypatch):
    api_class = load_api_control_class()
    calls = []
    sd_model = object()
    sd_models_stub = SimpleNamespace(
        unload_model_weights=lambda: calls.append("unload"),
        send_model_to_device=lambda model: calls.append(("send", model)),
    )
    monkeypatch.setitem(api_class.unloadapi.__globals__, "sd_models", sd_models_stub)
    monkeypatch.setitem(api_class.reloadapi.__globals__, "sd_models", sd_models_stub)
    monkeypatch.setitem(api_class.reloadapi.__globals__, "shared", SimpleNamespace(sd_model=sd_model))

    api = api_class.__new__(api_class)
    assert api.unloadapi() == {}
    assert api.reloadapi() == {}
    assert calls == ["unload", ("send", sd_model)]


def test_server_control_routes_are_gated_by_api_server_stop(monkeypatch):
    api_class = load_api_control_class()

    disabled_app = DummyApp()
    _patch_minimal_init_globals(monkeypatch, api_class, api_server_stop=False)
    api_class(disabled_app, DummyLock([]))
    disabled_paths = {path for path, _, _ in disabled_app.routes}
    assert "/sdapi/v1/server-kill" not in disabled_paths
    assert "/sdapi/v1/server-restart" not in disabled_paths
    assert "/sdapi/v1/server-stop" not in disabled_paths

    enabled_app = DummyApp()
    _patch_minimal_init_globals(monkeypatch, api_class, api_server_stop=True)
    api_class(enabled_app, DummyLock([]))
    enabled_paths = {path for path, _, _ in enabled_app.routes}
    assert {"/sdapi/v1/server-kill", "/sdapi/v1/server-restart", "/sdapi/v1/server-stop"} <= enabled_paths


def test_server_control_side_effects_and_response_status(monkeypatch):
    api_class = load_api_control_class()
    calls = []

    class Response:
        def __init__(self, content=None, status_code=200):
            self.content = content
            self.status_code = status_code

    restart_stub = SimpleNamespace(
        stop_program=lambda: calls.append("stop-program"),
        is_restartable=lambda: False,
        restart_program=lambda: calls.append("restart-program"),
    )
    shared_stub = SimpleNamespace(state=SimpleNamespace(server_command=None))
    monkeypatch.setitem(api_class.kill_webui.__globals__, "restart", restart_stub)
    monkeypatch.setitem(api_class.restart_webui.__globals__, "restart", restart_stub)
    monkeypatch.setitem(api_class.restart_webui.__globals__, "Response", Response)
    monkeypatch.setitem(api_class.stop_webui.__globals__, "shared", shared_stub)
    monkeypatch.setitem(api_class.stop_webui.__globals__, "Response", Response)

    api = api_class.__new__(api_class)
    assert api.kill_webui() is None
    assert calls == ["stop-program"]

    restart_response = api.restart_webui()
    assert restart_response.status_code == 501
    assert calls == ["stop-program"]

    restart_stub.is_restartable = lambda: True
    assert api.restart_webui().status_code == 501
    assert calls == ["stop-program", "restart-program"]

    stop_response = api.stop_webui()
    assert shared_stub.state.server_command == "stop"
    assert stop_response.content == "Stopping."
    assert stop_response.status_code == 200


def test_runtime_metadata_endpoints_preserve_delegation_and_public_fallbacks(monkeypatch):
    api_class = load_api_control_class()
    calls = []

    class CudaGraphs:
        @staticmethod
        def status():
            calls.append("status")
            return {"enabled": True, "cache_size": 2}

        @staticmethod
        def set_enabled(enabled, clear=False):
            calls.append(("set", enabled, clear))
            return {"enabled": enabled, "cleared": clear}

    class Diagnostics:
        current = None

        @classmethod
        def last_generation_diagnostics(cls):
            calls.append("diagnostics")
            return cls.current

    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "modules" and "openclaw_cuda_graphs" in fromlist:
            return SimpleNamespace(openclaw_cuda_graphs=CudaGraphs)
        if name == "modules" and "openclaw_generation_diagnostics" in fromlist:
            return SimpleNamespace(openclaw_generation_diagnostics=Diagnostics)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setitem(api_class.get_cuda_graphs.__globals__["__builtins__"], "__import__", fake_import)

    api = api_class.__new__(api_class)
    assert api.get_cuda_graphs() == {"enabled": True, "cache_size": 2}
    assert api.set_cuda_graphs({"enabled": 1, "clear": "yes"}) == {"enabled": True, "cleared": True}
    assert api.set_cuda_graphs(None) == {"enabled": False, "cleared": False}
    assert api.get_openclaw_generation_diagnostics() == {}

    Diagnostics.current = {"cuda_graphs_delta": {"captures": 1}}
    assert api.get_openclaw_generation_diagnostics() == {"cuda_graphs_delta": {"captures": 1}}
    assert calls == [
        "status",
        ("set", True, True),
        ("set", False, False),
        "diagnostics",
        "diagnostics",
    ]


def test_openclaw_runtime_defaults_apply_env_values_without_raising(monkeypatch):
    api_class = load_api_control_class()
    calls = []

    monkeypatch.setenv("OPENCLAW_SDPA_BACKEND", "math")
    monkeypatch.setenv("OPENCLAW_CUDA_GRAPHS", "true")

    sdpa_stub = SimpleNamespace(set_sdpa_backend=lambda value: calls.append(("sdpa", value)))
    errors_stub = SimpleNamespace(report=lambda *args, **kwargs: calls.append(("error", args, kwargs)))

    class CudaGraphs:
        @staticmethod
        def set_enabled(enabled, clear=False):
            calls.append(("graphs", enabled, clear))

    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "modules" and "openclaw_cuda_graphs" in fromlist:
            return SimpleNamespace(openclaw_cuda_graphs=CudaGraphs)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setitem(api_class.apply_openclaw_runtime_defaults.__globals__, "sd_hijack_optimizations", sdpa_stub)
    monkeypatch.setitem(api_class.apply_openclaw_runtime_defaults.__globals__, "errors", errors_stub)
    monkeypatch.setitem(api_class.apply_openclaw_runtime_defaults.__globals__["__builtins__"], "__import__", fake_import)

    api = api_class.__new__(api_class)
    api.apply_openclaw_runtime_defaults()

    assert calls == [("sdpa", "math"), ("graphs", True, True)]
    assert api_class._env_flag("OPENCLAW_CUDA_GRAPHS") is True
    monkeypatch.setenv("OPENCLAW_CUDA_GRAPHS", "off")
    assert api_class._env_flag("OPENCLAW_CUDA_GRAPHS") is False
    monkeypatch.delenv("OPENCLAW_CUDA_GRAPHS")
    assert api_class._env_flag("OPENCLAW_CUDA_GRAPHS") is None


def test_get_memory_preserves_ram_and_cuda_response_shape(monkeypatch):
    api_class = load_api_control_class()

    class MemoryResponse:
        def __init__(self, ram, cuda):
            self.ram = ram
            self.cuda = cuda

    class Process:
        def __init__(self, pid):
            self.pid = pid

        @staticmethod
        def memory_info():
            return SimpleNamespace(rss=25)

        @staticmethod
        def memory_percent():
            return 25

    cuda_stats = {
        "allocated_bytes.all.current": 1,
        "allocated_bytes.all.peak": 2,
        "reserved_bytes.all.current": 3,
        "reserved_bytes.all.peak": 4,
        "active_bytes.all.current": 5,
        "active_bytes.all.peak": 6,
        "inactive_split_bytes.all.current": 7,
        "inactive_split_bytes.all.peak": 8,
        "num_alloc_retries": 9,
        "num_ooms": 10,
    }

    psutil_stub = SimpleNamespace(Process=Process)
    torch_stub = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            mem_get_info=lambda: (40, 100),
            memory_stats=lambda device: cuda_stats,
        )
    )
    shared_stub = SimpleNamespace(device="cuda:0")
    models_stub = SimpleNamespace(MemoryResponse=MemoryResponse)
    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "psutil":
            return psutil_stub
        if name == "torch":
            return torch_stub
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setitem(api_class.get_memory.__globals__, "shared", shared_stub)
    monkeypatch.setitem(api_class.get_memory.__globals__, "models", models_stub)
    monkeypatch.setitem(api_class.get_memory.__globals__["__builtins__"], "__import__", fake_import)

    response = api_class.__new__(api_class).get_memory()

    assert response.ram == {"free": 75, "used": 25, "total": 100}
    assert response.cuda == {
        "system": {"free": 40, "used": 60, "total": 100},
        "active": {"current": 5, "peak": 6},
        "allocated": {"current": 1, "peak": 2},
        "reserved": {"current": 3, "peak": 4},
        "inactive": {"current": 7, "peak": 8},
        "events": {"retries": 9, "oom": 10},
    }
