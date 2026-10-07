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
    monkeypatch.setitem(sys.modules, "modules.sd_models", fake)
    monkeypatch.setattr(modules, "sd_models", fake, raising=False)


def _initialize_rest_calls():
    tree = ast.parse(Path("modules/initialize.py").read_text(encoding="utf8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "initialize_rest")
    return [ast.unparse(node.func) for node in ast.walk(function) if isinstance(node, ast.Call)]


def test_heap_is_frozen_after_scripts_and_upscalers_but_before_the_startup_model_load():
    source = Path("modules/initialize.py").read_text(encoding="utf8")
    body = source[source.index("def initialize_rest():"):]

    freeze = body.index("    freeze_startup_heap()")
    assert body.index("        scripts.load_scripts()\n", body.index("startup_timer.subcategory")) < freeze
    assert body.index("    modelloader.load_upscalers()") < freeze
    assert freeze < body.index("Thread(target=load_model).start()")
    assert _initialize_rest_calls().count("freeze_startup_heap") == 1


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
