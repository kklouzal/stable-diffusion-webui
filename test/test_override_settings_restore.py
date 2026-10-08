import types


def test_override_of_an_option_absent_from_saved_settings_is_restored(initialize):
    from modules import processing, shared

    key = "profiling_filename"
    saved = shared.opts.data.pop(key, None)
    try:
        default = shared.opts.get_default(key)
        p = types.SimpleNamespace(override_settings={key: "/tmp/override-trace.json"})

        stored = processing.store_processing_override_settings(p)
        shared.opts.set(key, "/tmp/override-trace.json", is_api=True, run_callbacks=False)
        processing.restore_processing_override_settings(stored)

        assert stored == {key: default}
        assert getattr(shared.opts, key) == default
    finally:
        if saved is None:
            shared.opts.data.pop(key, None)
        else:
            shared.opts.data[key] = saved


def test_unknown_override_keys_are_not_stored(initialize):
    from modules import processing

    p = types.SimpleNamespace(override_settings={"no_such_option_anywhere": 1})
    assert processing.store_processing_override_settings(p) == {}
