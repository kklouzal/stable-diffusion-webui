import sys
from pathlib import Path

import pytest

from modules import script_loading


def _write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def test_load_module_registers_during_execution_and_keeps_success(tmp_path):
    path = _write(
        tmp_path / "one" / "contract.py",
        "from dataclasses import dataclass\n"
        "@dataclass\nclass Contract:\n    value: int = 1\n",
    )
    module = script_loading.load_module(path)
    assert module.Contract().value == 1
    assert sys.modules[module.__name__] is module
    assert script_loading.loaded_scripts[str(path)] is module


def test_load_module_uses_collision_resistant_identity(tmp_path):
    first_path = _write(tmp_path / "one" / "same.py", "VALUE = 1\n")
    second_path = _write(tmp_path / "two" / "same.py", "VALUE = 2\n")
    first = script_loading.load_module(first_path)
    second = script_loading.load_module(second_path)
    assert first.__name__ != second.__name__
    assert first.VALUE == 1 and second.VALUE == 2
    assert sys.modules[first.__name__] is first
    assert sys.modules[second.__name__] is second


def test_load_module_name_is_identifier_for_unusual_basename(tmp_path):
    path = _write(tmp_path / "123 odd.name-å.py", "VALUE = 1\n")
    name = script_loading.module_name_for_path(path)
    assert name.isidentifier()
    assert script_loading.load_module(path).VALUE == 1


def test_load_module_failure_removes_registration(tmp_path):
    path = _write(tmp_path / "broken.py", "raise RuntimeError('boom')\n")
    name = script_loading.module_name_for_path(path)
    with pytest.raises(RuntimeError, match="boom"):
        script_loading.load_module(path)
    assert name not in sys.modules
    assert str(path) not in script_loading.loaded_scripts


def test_load_module_failure_restores_previous_registration(tmp_path):
    path = _write(tmp_path / "broken.py", "raise RuntimeError('boom')\n")
    name = script_loading.module_name_for_path(path)
    sentinel = object()
    sys.modules[name] = sentinel
    try:
        with pytest.raises(RuntimeError, match="boom"):
            script_loading.load_module(path)
        assert sys.modules[name] is sentinel
        assert str(path) not in script_loading.loaded_scripts
    finally:
        sys.modules.pop(name, None)


def test_load_module_failure_preserves_replacement_made_by_script(tmp_path):
    path = tmp_path / "replaced.py"
    name = script_loading.module_name_for_path(path)
    path = _write(
        path,
        "import sys\n"
        f"sys.modules[{name!r}] = 'replacement'\n"
        "raise RuntimeError('boom')\n",
    )
    try:
        with pytest.raises(RuntimeError, match="boom"):
            script_loading.load_module(path)
        assert sys.modules[name] == "replacement"
    finally:
        sys.modules.pop(name, None)


def test_load_module_reload_replaces_module_and_clears_stale_names(tmp_path):
    path = _write(tmp_path / "reload.py", "VALUE = 1\nSTALE = True\n")
    first = script_loading.load_module(path)
    path.write_text("VALUE = 2\n")
    second = script_loading.load_module(path)
    assert second is not first
    assert second.VALUE == 2
    assert not hasattr(second, "STALE")
    assert sys.modules[second.__name__] is second


def test_load_module_supports_package_relative_imports(tmp_path):
    package = tmp_path / "fixture_package"
    _write(package / "__init__.py", "VALUE = 40\n")
    script = _write(package / "script.py", "from . import VALUE\nRESULT = VALUE + 2\n")
    sys.path.insert(0, str(tmp_path))
    try:
        module = script_loading.load_module(script)
        assert module.RESULT == 42
        assert module.__package__ == "fixture_package"
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("fixture_package", None)
