import ast
import gc
import sys
import types
import weakref
from pathlib import Path

import pytest

import modules
from modules import initialize


@pytest.fixture
def frozen_before():
    # CPython 3.12 already starts with a few hundred frozen interpreter objects, so compare against this count.
    count = gc.get_freeze_count()
    yield count
    gc.unfreeze()


def _fake_model_data(monkeypatch, loaded_sd_models):
    # freeze_startup_heap only reads model_data.loaded_sd_models; the real sd_models drags in ldm/sgm.
    fake = types.ModuleType("modules.sd_models")
    fake.model_data = types.SimpleNamespace(loaded_sd_models=loaded_sd_models)
    # Some tests in this session leave a stub in sys.modules["modules"]; resolve against the real package.
    monkeypatch.setitem(sys.modules, "modules", modules)
    monkeypatch.setitem(sys.modules, "modules.sd_models", fake)
    monkeypatch.setattr(modules, "sd_models", fake, raising=False)


def _initialize_rest_module_imports():
    tree = ast.parse(Path("modules/initialize.py").read_text(encoding="utf8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "initialize_rest")
    return {alias.name for node in ast.walk(function) if isinstance(node, ast.ImportFrom) and node.module == "modules" for alias in node.names}


class _Recorder:
    """Stands in for a module initialize_rest imports: records every call made through it, by dotted name."""

    def __init__(self, calls, name):
        self._calls, self._name = calls, name

    def __getattr__(self, attr):
        return _Recorder(self._calls, f"{self._name}.{attr}")

    def __call__(self, *args, **kwargs):
        self._calls.append(self._name)
        return _Recorder(self._calls, f"{self._name}()")


def test_heap_is_frozen_after_scripts_and_upscalers_but_before_the_startup_model_load(monkeypatch):
    import contextlib

    calls = []
    monkeypatch.setitem(sys.modules, "modules", modules)
    for name in _initialize_rest_module_imports():
        monkeypatch.setattr(modules, name, _Recorder(calls, name), raising=False)
    monkeypatch.setattr(modules, "shared", types.SimpleNamespace(cmd_opts=types.SimpleNamespace(skip_load_model_at_start=False)), raising=False)
    cmd_options = types.ModuleType("modules.shared_cmd_options")
    cmd_options.cmd_opts = types.SimpleNamespace(ui_debug_mode=False)
    monkeypatch.setitem(sys.modules, "modules.shared_cmd_options", cmd_options)
    monkeypatch.setattr(initialize, "startup_timer", types.SimpleNamespace(record=lambda label: None, subcategory=lambda label: contextlib.nullcontext()))
    monkeypatch.setattr(initialize, "freeze_startup_heap", lambda: calls.append("freeze_startup_heap"))
    monkeypatch.setattr(initialize, "Thread", lambda target: types.SimpleNamespace(start=lambda: calls.append("start startup model load")))

    initialize.initialize_rest()

    assert calls.count("freeze_startup_heap") == 1
    freeze = calls.index("freeze_startup_heap")
    assert calls.index("scripts.load_scripts") < freeze
    assert calls.index("modelloader.load_upscalers") < freeze
    assert freeze < calls.index("start startup model load")


def test_freeze_keeps_cycles_created_afterwards_collectable(monkeypatch, frozen_before):
    _fake_model_data(monkeypatch, [])

    class Node:
        pass

    before = Node()
    before.self = before

    initialize.freeze_startup_heap()

    assert gc.get_freeze_count() > frozen_before
    # Frozen objects live in the permanent generation, which collections (and get_objects) never visit.
    assert id(before) not in {id(obj) for obj in gc.get_objects()}

    after = Node()
    after.self = after
    after_ref = weakref.ref(after)
    del after
    gc.collect()
    assert after_ref() is None


def test_freeze_is_skipped_when_a_checkpoint_is_already_loaded(monkeypatch, capsys, frozen_before):
    _fake_model_data(monkeypatch, [object()])

    initialize.freeze_startup_heap()

    assert gc.get_freeze_count() == frozen_before
    assert "Not freezing the startup heap" in capsys.readouterr().out
