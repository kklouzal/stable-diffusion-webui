from pathlib import Path


def test_api_only_waits_for_startup_model_before_serving():
    source = Path('webui.py').read_text(encoding='utf8')

    initialize_pos = source.index('    initialize.initialize()')
    load_pos = source.index('sd_models.model_data.get_sd_model()')
    app_pos = source.index('    app = FastAPI()')
    launch_pos = source.index('    api.launch(')

    assert initialize_pos < load_pos < app_pos < launch_pos
    assert 'if not shared.cmd_opts.skip_load_model_at_start:' in source
    assert 'startup_timer.record("load startup SD model")' in source


def test_api_only_runs_before_ui_callbacks_before_the_api_snapshots_xyz_axes():
    """create_api() runs the headless script setup, where each X/Y/Z runner copies the module axis list
    (xyz_grid Script.ui -> current_axis_options) that run() indexes. Axes appended by before_ui callbacks
    (Hypertile, Incantations PAG/SEG/DynThres) are selectable only if those callbacks ran first."""
    import ast

    tree = ast.parse(Path('webui.py').read_text(encoding='utf8'))
    api_only = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'api_only')
    calls = [ast.unparse(node.func) for node in ast.walk(api_only) if isinstance(node, ast.Call)]
    lines = {ast.unparse(node.func): node.lineno for node in ast.walk(api_only) if isinstance(node, ast.Call)}

    assert calls.count('script_callbacks.before_ui_callback') == 1
    assert lines['script_callbacks.before_ui_callback'] < lines['create_api'] < lines['script_callbacks.app_started_callback']
