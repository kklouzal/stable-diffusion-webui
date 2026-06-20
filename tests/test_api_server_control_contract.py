import ast
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

    namespace = {"FastAPI": object, "Lock": object, "APIRouter": lambda: object()}
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
        "get_cuda_graphs",
        "set_cuda_graphs",
        "get_openclaw_generation_diagnostics",
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
