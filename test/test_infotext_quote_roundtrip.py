from types import SimpleNamespace

import pytest

from modules import prompt_parser
from test.helpers import load_source, module


@pytest.fixture
def infotext_utils():
    return load_source("infotext_utils_under_test", "modules/infotext_utils.py", {
        "modules": module("modules", package=True),
        "modules.shared": module("modules.shared", opts=SimpleNamespace(infotext_skip_pasting=[], infotext_styles="Ignore", use_old_hires_fix_width_height=False)),
        "modules.processing": module("modules.processing"),
        "modules.infotext_versions": module("modules.infotext_versions", parse_version=lambda text: None, v180_hr_styles=None, backcompat=lambda d: None),
        "modules.prompt_parser": prompt_parser,  # the real parser is used by parse_generation_parameters
    })


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
