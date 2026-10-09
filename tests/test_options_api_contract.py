import importlib.util
import json
import sys
import threading
import types

import pytest


@pytest.fixture
def options_module(monkeypatch):
    cmd_opts = types.SimpleNamespace(freeze_settings=False, freeze_settings_in_sections=None, freeze_specific_settings=None, hide_ui_dir_config=False)
    monkeypatch.setitem(sys.modules, "modules.shared_cmd_options", types.SimpleNamespace(cmd_opts=cmd_opts))
    spec = importlib.util.spec_from_file_location("options_under_test", "modules/options.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_opts(module):
    labels = {
        "enable_feature": module.OptionInfo(False, "Feature"),
        "clip_skip": module.OptionInfo(1, "Clip skip"),
        "strength": module.OptionInfo(0.5, "Strength"),
        "vae": module.OptionInfo("Automatic", "VAE"),
        "checkpoint": module.OptionInfo(None, "Checkpoint"),
    }
    return module.Options(labels, restricted_opts=set())


def test_api_set_rejects_a_value_of_another_type_instead_of_storing_it(options_module):
    opts = make_opts(options_module)

    # The JSON string "false" stored in a bool option would read as True.
    with pytest.raises(ValueError) as raised:
        opts.set("enable_feature", "false", is_api=True)
    assert str(raised.value) == "setting 'enable_feature' expects a value of type bool, got str 'false'"
    with pytest.raises(ValueError, match="clip_skip"):
        opts.set("clip_skip", "2", is_api=True)
    with pytest.raises(ValueError, match="vae"):
        opts.set("vae", 5, is_api=True)
    with pytest.raises(ValueError, match="enable_feature"):
        opts.set("enable_feature", 1.0, is_api=True)

    assert (opts.enable_feature, opts.clip_skip, opts.vae) == (False, 1, "Automatic")


def test_api_set_accepts_values_of_the_option_type(options_module):
    opts = make_opts(options_module)

    assert opts.set("enable_feature", True, is_api=True) is True
    assert opts.set("clip_skip", 2, is_api=True) is True
    assert opts.set("strength", 1, is_api=True) is True  # int and float are interchangeable (Options.typemap)
    assert opts.set("vae", "None", is_api=True) is True
    assert opts.set("checkpoint", "model.safetensors", is_api=True) is True  # a None default accepts any value
    assert opts.set("vae", None, is_api=True) is True

    assert (opts.enable_feature, opts.clip_skip, opts.strength, opts.vae, opts.checkpoint) == (True, 2, 1, None, "model.safetensors")


def test_onchange_failure_restores_the_value_and_propagates(options_module):
    opts = make_opts(options_module)

    def reload_vae():
        raise RuntimeError("VAE reload failed")

    opts.data_labels["vae"].onchange = reload_vae

    with pytest.raises(options_module.OptionChangeFailed, match="VAE reload failed") as raised:
        opts.set("vae", "other.safetensors", is_api=True)

    assert isinstance(raised.value, RuntimeError)
    assert str(raised.value.__cause__) == "VAE reload failed"
    assert opts.vae == "Automatic"


def test_save_failure_leaves_the_settings_file_intact(options_module, tmp_path):
    opts = make_opts(options_module)
    settings_file = tmp_path / "config.json"
    settings_file.write_text('{"vae": "Automatic"}', encoding="utf8")

    opts.data["unserializable"] = object()
    with pytest.raises(TypeError):
        opts.save(str(settings_file))

    assert json.loads(settings_file.read_text(encoding="utf8")) == {"vae": "Automatic"}


class _BlockingItemsDict(dict):
    """A settings dict whose JSON serialization waits for `release` after signalling `started`."""

    def __init__(self, *args, started, release, **kwargs):
        super().__init__(*args, **kwargs)
        self.started = started
        self.release = release

    def items(self):
        self.started.set()
        assert self.release.wait(10)
        return super().items()


def test_concurrent_saves_do_not_interleave_writes(options_module, tmp_path):
    settings_file = tmp_path / "config.json"
    started, release = threading.Event(), threading.Event()

    short = make_opts(options_module)
    short.data = _BlockingItemsDict({"a": 1}, started=started, release=release)
    long = make_opts(options_module)
    long.data = {"a_much_longer_settings_key": "x" * 200}
    failures = []

    def save(opts):
        try:
            opts.save(str(settings_file))
        except Exception as e:  # reported by the assertion below
            failures.append(e)

    first = threading.Thread(target=save, args=(short,))
    first.start()
    assert started.wait(10)
    second = threading.Thread(target=save, args=(long,))
    second.start()
    second.join(0.5)  # an unserialized save completes here, before the first one writes
    release.set()
    first.join(10)
    second.join(10)

    assert failures == []
    assert json.loads(settings_file.read_text(encoding="utf8")) in ({"a": 1}, {"a_much_longer_settings_key": "x" * 200})


def test_setting_a_key_loaded_from_the_settings_file_without_a_registered_option(options_module):
    opts = make_opts(options_module)
    opts.data["removed_extension_option"] = 1

    opts.removed_extension_option = 2

    assert opts.data["removed_extension_option"] == 2


def test_cast_value_rejects_unrecognized_boolean_text(options_module):
    opts = make_opts(options_module)

    assert opts.cast_value("enable_feature", "False") is False
    assert opts.cast_value("enable_feature", " true ") is True
    with pytest.raises(ValueError) as raised:
        opts.cast_value("enable_feature", "maybe")
    assert str(raised.value) == "setting 'enable_feature' expects a boolean, got 'maybe'"
