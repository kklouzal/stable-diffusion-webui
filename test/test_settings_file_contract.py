"""modules/settings_file: config.json is read and written in place (gb10/run.sh bind-mounts it as a single file, which
rename cannot replace), a completed write is fsynced, and an unusable file reverts to the defaults with a copy kept
under tmp/ instead of stopping the server."""
import json
import os
import types

import pytest

from modules import settings_file
from test.helpers import load_source, module


@pytest.fixture
def webui_root(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_file, "script_path", str(tmp_path))
    return tmp_path


def _quarantined(root):
    return sorted((root / "tmp").glob("config.json.corrupt-*"))


def test_read_returns_the_stored_settings_and_leaves_the_file_alone(webui_root):
    config = webui_root / "config.json"
    config.write_text('{"sd_model_checkpoint": "a.safetensors", "clip_stop_at_last_layers": 2}', encoding="utf8")

    assert settings_file.read(str(config)) == {"sd_model_checkpoint": "a.safetensors", "clip_stop_at_last_layers": 2}
    assert config.read_text(encoding="utf8") == '{"sd_model_checkpoint": "a.safetensors", "clip_stop_at_last_layers": 2}'
    assert not (webui_root / "tmp").exists()


def test_read_of_a_missing_file_is_empty_settings(webui_root):
    assert settings_file.read(str(webui_root / "config.json")) == {}
    assert not (webui_root / "config.json").exists()


@pytest.mark.parametrize("content", [
    b'{"sd_model_checkpoint": "a.safe',  # truncated mid-write
    b"",  # empty: a save cut short before its first byte
    b'["not", "an", "object"]',
    b'"text"',
    b'{"key": "\xff\xfe"}',  # not UTF-8
], ids=["truncated", "empty", "list", "string", "not-utf8"])
def test_an_unusable_file_is_copied_aside_and_reset_in_place(webui_root, content, capsys):
    config = webui_root / "config.json"
    config.write_bytes(content)
    inode = config.stat().st_ino

    assert settings_file.read(str(config)) == {}

    # Rewritten in place (the same inode: a single-file bind mount cannot be renamed over), now valid JSON.
    assert config.stat().st_ino == inode
    assert json.loads(config.read_text(encoding="utf8")) == {}
    [backup] = _quarantined(webui_root)
    assert backup.read_bytes() == content
    assert f'Its content was copied to "{backup}"' in capsys.readouterr().err


def test_each_unusable_file_keeps_its_own_copy(webui_root):
    config = webui_root / "config.json"
    for content in (b"{broken", b"[1]"):
        config.write_bytes(content)
        assert settings_file.read(str(config)) == {}

    assert sorted(path.read_bytes() for path in _quarantined(webui_root)) == [b"[1]", b"{broken"]


def test_a_failed_copy_leaves_the_damaged_file_untouched(webui_root):
    config = webui_root / "config.json"
    config.write_bytes(b"{broken")
    (webui_root / "tmp").write_text("a file where the tmp directory belongs", encoding="utf8")

    with pytest.raises(OSError):
        settings_file.read(str(config))

    assert config.read_bytes() == b"{broken"


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads files regardless of their mode")
def test_an_unreadable_file_raises_instead_of_reverting_to_defaults(webui_root):
    config = webui_root / "config.json"
    config.write_text('{"a": 1}', encoding="utf8")
    config.chmod(0)
    try:
        with pytest.raises(PermissionError):
            settings_file.read(str(config))
    finally:
        config.chmod(0o600)

    assert config.read_text(encoding="utf8") == '{"a": 1}'
    assert not (webui_root / "tmp").exists()


def test_write_rewrites_in_place_and_fsyncs_before_returning(webui_root, monkeypatch):
    config = webui_root / "config.json"
    config.write_text('{"a_much_longer_previous_key": "' + "x" * 100 + '"}', encoding="utf8")
    inode = config.stat().st_ino
    synced = []
    real_fsync = os.fsync

    def fsync(fd):
        synced.append(os.readlink(f"/proc/self/fd/{fd}"))
        real_fsync(fd)

    monkeypatch.setattr(settings_file.os, "fsync", fsync)

    settings_file.write(str(config), '{"a": "é"}')

    assert synced == [str(config)]
    assert config.stat().st_ino == inode
    assert config.read_bytes() == '{"a": "é"}'.encode("utf8")


@pytest.fixture
def options_module():
    cmd_opts = types.SimpleNamespace(freeze_settings=False, freeze_settings_in_sections=None, freeze_specific_settings=None, hide_ui_dir_config=False)
    return load_source("options_under_test", "modules/options.py", {"modules.shared_cmd_options": module("modules.shared_cmd_options", cmd_opts=cmd_opts)})


def test_options_load_of_a_damaged_file_starts_from_defaults_and_save_restores_the_file(options_module, webui_root):
    config = webui_root / "config.json"
    config.write_bytes(b'{"vae": "ft')
    opts = options_module.Options({"vae": options_module.OptionInfo("Automatic", "VAE")}, restricted_opts=set())

    opts.load(str(config))
    assert opts.vae == "Automatic"
    assert len(_quarantined(webui_root)) == 1

    opts.set("vae", "ft.safetensors")
    opts.save(str(config))
    assert json.loads(config.read_text(encoding="utf8")) == {"vae": "ft.safetensors"}


def test_launch_extension_list_reads_a_damaged_file_as_defaults(webui_root, monkeypatch):
    from modules import launch_utils

    config = webui_root / "config.json"
    config.write_bytes(b'{"disabled_extensions": ["a"')
    extensions_dir = webui_root / "extensions"
    (extensions_dir / "a").mkdir(parents=True)
    monkeypatch.setattr(launch_utils, "extensions_dir", str(extensions_dir))
    monkeypatch.setattr(launch_utils.args, "disable_extra_extensions", False)
    monkeypatch.setattr(launch_utils.args, "disable_all_extensions", False)

    assert launch_utils.list_extensions(str(config)) == ["a"]
    assert json.loads(config.read_text(encoding="utf8")) == {}
    assert len(_quarantined(webui_root)) == 1
