import types

import pytest


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


def _override(monkeypatch, **settings):
    from modules import shared

    for key in settings:  # the test's option changes are undone after it
        if key in shared.opts.data:
            monkeypatch.setitem(shared.opts.data, key, shared.opts.data[key])
        else:
            monkeypatch.setitem(shared.opts.data, key, shared.opts.get_default(key))
            monkeypatch.delitem(shared.opts.data, key)
    return types.SimpleNamespace(override_settings=dict(settings), override_settings_restore_afterwards=False)


def _fail(*args, **kwargs):
    raise RuntimeError("reload failed")


def test_a_failing_checkpoint_reload_restores_the_request_options(initialize, monkeypatch):
    from modules import processing, shared, sd_models

    monkeypatch.setattr(sd_models, "checkpoint_aliases", {"bad.safetensors": object()})
    monkeypatch.setattr(sd_models, "reload_model_weights", _fail)
    p = _override(monkeypatch, sd_model_checkpoint="bad.safetensors", CLIP_stop_at_last_layers=7)
    before = (shared.opts.sd_model_checkpoint, shared.opts.CLIP_stop_at_last_layers)

    with pytest.raises(RuntimeError, match="reload failed"):
        processing.apply_processing_override_settings(p)

    # restore_afterwards=False does not keep a checkpoint every later request would retry and fail on
    assert (shared.opts.sd_model_checkpoint, shared.opts.CLIP_stop_at_last_layers) == before


def test_a_failing_vae_reload_restores_the_vae_option(initialize, monkeypatch):
    from modules import processing, shared, sd_models, sd_vae

    vae_reloads = []

    def reload_vae(*args, **kwargs):
        vae_reloads.append(shared.opts.sd_vae)
        if shared.opts.sd_vae == "missing.safetensors":
            _fail()

    monkeypatch.setattr(sd_models, "reload_model_weights", lambda *a, **k: None)
    monkeypatch.setattr(sd_vae, "reload_vae_weights", reload_vae)
    p = _override(monkeypatch, sd_vae="missing.safetensors")
    before = shared.opts.sd_vae

    with pytest.raises(RuntimeError, match="reload failed"):
        processing.apply_processing_override_settings(p)

    assert shared.opts.sd_vae == before
    assert vae_reloads == ["missing.safetensors", before]  # the restore reloads the previous VAE


def test_a_failing_restore_reload_still_restores_later_options(initialize, monkeypatch):
    from modules import processing, shared, sd_vae

    monkeypatch.setattr(sd_vae, "reload_vae_weights", _fail)
    p = _override(monkeypatch, sd_vae="other.safetensors", CLIP_stop_at_last_layers=7)
    stored = processing.store_processing_override_settings(p)
    shared.opts.set("sd_vae", "other.safetensors", is_api=True, run_callbacks=False)
    shared.opts.set("CLIP_stop_at_last_layers", 7, is_api=True, run_callbacks=False)

    with pytest.raises(RuntimeError, match="restoring settings") as exc:
        processing.restore_processing_override_settings(stored)

    assert str(exc.value.__cause__) == "reload failed"
    assert {key: getattr(shared.opts, key) for key in stored} == stored
