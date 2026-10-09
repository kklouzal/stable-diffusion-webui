import importlib.util
import sys
import types

import pytest

import modules.prompt_parser  # noqa: F401  (the real parser is used by parse_generation_parameters)


@pytest.fixture
def infotext_utils(monkeypatch):
    shared = types.SimpleNamespace(opts=types.SimpleNamespace(infotext_skip_pasting=[], infotext_styles="Ignore", use_old_hires_fix_width_height=False))
    infotext_versions = types.SimpleNamespace(parse_version=lambda text: None, v180_hr_styles=None, backcompat=lambda d: None)
    monkeypatch.setitem(sys.modules, "modules", types.ModuleType("modules"))
    monkeypatch.setitem(sys.modules, "modules.shared", shared)
    monkeypatch.setitem(sys.modules, "modules.processing", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "modules.infotext_versions", infotext_versions)
    spec = importlib.util.spec_from_file_location("infotext_utils_under_test", "modules/infotext_utils.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("value", [
    '"masterpiece"',
    '"say \\"hi\\""',
    '"quoted" and more',
    'plain words',
    'a, b',
    'key: value',
    'two\nlines',
    'say "hi"',
])
def test_quoted_infotext_values_read_back_verbatim(infotext_utils, value):
    infotext = f"a cat\nSteps: 20, Sampler: Euler, Hires prompt: {infotext_utils.quote(value)}, Seed: 1"

    params = infotext_utils.parse_generation_parameters(infotext, skip_fields=[])

    assert params["Hires prompt"] == value
    assert params["Seed"] == "1"


def test_values_that_read_back_verbatim_are_written_unchanged(infotext_utils):
    assert infotext_utils.quote("plain words") == "plain words"
    assert infotext_utils.quote('say "hi"') == 'say "hi"'
    assert infotext_utils.quote(7.5) == 7.5
    assert infotext_utils.quote("a, b") == '"a, b"'
