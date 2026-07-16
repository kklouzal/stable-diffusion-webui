from __future__ import annotations

from modules import timer
from modules import initialize_util
from modules import initialize

startup_timer = timer.startup_timer
startup_timer.record("launcher")

initialize_util.fix_pytorch_lightning()

initialize.imports()

initialize.check_versions()


def create_api(app):
    from modules.api.api import Api
    from modules.call_queue import queue_lock

    api = Api(app, queue_lock)
    return api


def api_only():
    from fastapi import FastAPI
    from modules.shared_cmd_options import cmd_opts

    initialize.initialize()

    # API-only startup does not construct the full Gradio settings UI. In this
    # headless fork, Hypertile's on_ui_settings callback is not reliably present
    # in the API-only callback registry, so register its persistent options
    # directly when the built-in extension is available. This restores
    # /sdapi/v1/options visibility without enabling the full web UI.
    try:
        from modules import paths
        import importlib.util
        hypertile_script = paths.script_path + "/extensions-builtin/hypertile/scripts/hypertile_script.py"
        spec = importlib.util.spec_from_file_location("openclaw_api_hypertile_settings", hypertile_script)
        hypertile_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hypertile_module)
        if hasattr(hypertile_module, "on_ui_settings"):
            hypertile_module.on_ui_settings()
            startup_timer.record("register Hypertile settings")
    except Exception:
        from modules import errors
        errors.report("Failed to register Hypertile settings in API-only startup", exc_info=True)

    app = FastAPI()
    initialize_util.setup_middleware(app)
    api = create_api(app)

    from modules import script_callbacks
    script_callbacks.before_ui_callback()

    # API-only startup does not construct the full Gradio settings UI. In this
    # headless fork, Hypertile's on_ui_settings callback is not reliably present
    # in the API-only callback registry, so register its persistent options
    # directly when the built-in extension is available. This restores
    # /sdapi/v1/options visibility without enabling the full web UI.
    try:
        from modules import paths
        import importlib.util
        hypertile_script = paths.script_path + "/extensions-builtin/hypertile/scripts/hypertile_script.py"
        spec = importlib.util.spec_from_file_location("openclaw_api_hypertile_settings", hypertile_script)
        hypertile_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hypertile_module)
        if hasattr(hypertile_module, "on_ui_settings"):
            hypertile_module.on_ui_settings()
            startup_timer.record("register Hypertile settings")
    except Exception:
        from modules import errors
        errors.report("Failed to register Hypertile settings in API-only startup", exc_info=True)

    script_callbacks.app_started_callback(None, app)

    print(f"Startup time: {startup_timer.summary()}.")
    api.launch(
        server_name=initialize_util.server_name(),
        port=cmd_opts.port if cmd_opts.port else 7861,
        root_path=f"/{cmd_opts.subpath}" if cmd_opts.subpath else ""
    )


def webui():
    raise SystemExit(
        "The browser UI has been removed from this GB10 fork. "
        "Run with --nowebui --api to start the API/headless server."
    )


if __name__ == "__main__":
    from modules.shared_cmd_options import cmd_opts

    if cmd_opts.nowebui:
        api_only()
    else:
        webui()
