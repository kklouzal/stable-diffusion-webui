from pathlib import Path


def _run_api_only(monkeypatch, *, skip_load_model_at_start):
    """Run webui.api_only (compiled from source: importing webui runs initialize.imports()) against recording stubs and
    return the startup steps in the order they ran."""
    import ast
    import sys
    import types
    from types import SimpleNamespace

    events = []

    def record(name, result=None):
        return lambda *args, **kwargs: events.append(name) or result

    tree = ast.parse(Path('webui.py').read_text(encoding='utf8'))
    api_only = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'api_only')
    namespace = {
        'initialize': SimpleNamespace(initialize=record('initialize')),
        'startup_timer': SimpleNamespace(record=lambda label: events.append(f'timer: {label}'), summary=lambda: ''),
        'initialize_util': SimpleNamespace(setup_middleware=record('setup_middleware'), server_name=lambda: '127.0.0.1'),
        'create_api': record('create_api', SimpleNamespace(launch=record('launch'))),
    }

    fake_modules = types.ModuleType('modules')
    fake_modules.shared = SimpleNamespace(cmd_opts=SimpleNamespace(skip_load_model_at_start=skip_load_model_at_start))
    fake_modules.sd_models = SimpleNamespace(model_data=SimpleNamespace(get_sd_model=record('get_sd_model')))
    fake_modules.script_callbacks = SimpleNamespace(before_ui_callback=record('before_ui_callback'), app_started_callback=record('app_started_callback'))
    fake_modules.openclaw_warmup = SimpleNamespace(install=record('openclaw_warmup.install'))
    cmd_options = types.ModuleType('modules.shared_cmd_options')
    cmd_options.cmd_opts = SimpleNamespace(port=None, subpath=None)
    fastapi = types.ModuleType('fastapi')
    fastapi.FastAPI = record('FastAPI', object())
    monkeypatch.setitem(sys.modules, 'modules', fake_modules)
    monkeypatch.setitem(sys.modules, 'modules.shared_cmd_options', cmd_options)
    monkeypatch.setitem(sys.modules, 'fastapi', fastapi)

    exec(compile(ast.fix_missing_locations(ast.Module(body=[api_only], type_ignores=[])), 'webui.py', 'exec'), namespace)
    namespace['api_only']()
    return events


def test_api_only_waits_for_startup_model_before_serving(monkeypatch):
    # The warm-up installs its route and app_started callback once the Api exists, before the callbacks run.
    serving = ['FastAPI', 'setup_middleware', 'before_ui_callback', 'create_api', 'openclaw_warmup.install', 'app_started_callback', 'launch']

    assert _run_api_only(monkeypatch, skip_load_model_at_start=False) == ['initialize', 'get_sd_model', 'timer: load startup SD model', *serving]
    assert _run_api_only(monkeypatch, skip_load_model_at_start=True) == ['initialize', *serving]


def test_api_only_runs_before_ui_callbacks_before_the_api_snapshots_xyz_axes(monkeypatch):
    """create_api() runs the headless script setup, where each X/Y/Z runner copies the module axis list
    (xyz_grid Script.ui -> current_axis_options) that run() indexes. Axes appended by before_ui callbacks
    (Hypertile, Incantations PAG/SEG/DynThres) are selectable only if those callbacks ran first."""
    events = _run_api_only(monkeypatch, skip_load_model_at_start=True)

    assert events.count('before_ui_callback') == 1
    assert events.index('before_ui_callback') < events.index('create_api') < events.index('app_started_callback')
