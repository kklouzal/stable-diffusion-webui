import importlib
import sys
import time
from types import SimpleNamespace


def load_ui_extensions():
    sys.modules.setdefault("git", SimpleNamespace(Repo=None))
    sys.modules["modules.headless_ui"] = SimpleNamespace(
        Dropdown=SimpleNamespace(update=lambda **kwargs: kwargs),
        CheckboxGroup=SimpleNamespace(update=lambda **kwargs: kwargs),
    )
    sys.modules["modules.extensions"] = SimpleNamespace(extensions=[], extensions_dir="/tmp/extensions")
    sys.modules["modules.shared"] = SimpleNamespace(
        cmd_opts=SimpleNamespace(disable_extension_access=False, disable_extra_extensions=False, disable_all_extensions=False),
        opts=SimpleNamespace(disable_all_extensions="none", concurrent_git_fetch_limit=1),
        state=SimpleNamespace(job_count=0, textinfo="", nextjob=lambda: None, request_restart=lambda: None),
        config_filename="/tmp/config.json",
    )
    sys.modules["modules.paths"] = SimpleNamespace(data_path="/tmp")
    sys.modules["modules.config_states"] = SimpleNamespace(
        get_config=lambda: {},
        get_webui_config=lambda: {"remote": "", "branch": "", "commit_hash": ""},
        all_config_states={},
        list_config_states=lambda: {},
    )
    sys.modules["modules.errors"] = SimpleNamespace(report=lambda *args, **kwargs: None)
    sys.modules["modules.restart"] = SimpleNamespace(is_restartable=lambda: False, restart_program=lambda: None, stop_program=lambda: None)
    sys.modules["modules.call_queue"] = SimpleNamespace(wrap_ui_gpu_call=lambda fn, extra_outputs=None: fn)
    sys.modules["modules.paths_internal"] = SimpleNamespace(config_states_dir="/tmp")
    sys.modules.pop("modules.ui_extensions", None)
    return importlib.import_module("modules.ui_extensions")


def test_make_commit_link_escapes_text_and_href():
    ui_extensions = load_ui_extensions()

    link = ui_extensions.make_commit_link(
        "abc123\" onclick=\"alert(1)",
        "https://github.com/example/project.git",
        "<b>main</b>",
    )

    assert "&lt;b&gt;main&lt;/b&gt;" in link
    assert "&quot; onclick=&quot;alert(1)" in link
    assert 'abc123" onclick="alert(1)' not in link
    assert "<b>main</b>" not in link


def test_config_state_table_escapes_stored_backup_fields(monkeypatch):
    ui_extensions = load_ui_extensions()
    config = {
        "name": "<script>backup</script>",
        "created_at": time.time(),
        "filepath": "/tmp/<bad>.json",
        "webui": {
            "remote": "",
            "branch": "<img src=x>",
            "commit_hash": "abc123",
            "commit_date": None,
        },
        "extensions": {
            "evil<ext>": {
                "remote": "",
                "branch": "<branch>",
                "enabled": True,
                "commit_hash": "def456",
                "commit_date": None,
            }
        },
    }
    monkeypatch.setattr(ui_extensions.config_states, "get_config", lambda: config)
    monkeypatch.setattr(ui_extensions.extensions, "extensions", [])

    html = ui_extensions.update_config_states_table("Current")

    assert "&lt;script&gt;backup&lt;/script&gt;" in html
    assert "/tmp/&lt;bad&gt;.json" in html
    assert "&lt;img src=x&gt;" in html
    assert "evil&lt;ext&gt;" in html
    assert "&lt;branch&gt;" in html
    assert "<script>backup</script>" not in html
    assert "<img src=x>" not in html
    assert "<branch>" not in html
