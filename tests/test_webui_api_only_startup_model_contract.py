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
