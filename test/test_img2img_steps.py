from fractions import Fraction
from types import SimpleNamespace

import pytest

from modules import shared, shared_init

if getattr(shared, "opts", None) is None:
    shared_init.initialize()

# sd_samplers first: importing sd_samplers_common on its own runs into an import cycle.
from modules import sd_samplers, sd_samplers_common  # noqa: E402,F401

# Oracle: the same formulas in exact rational arithmetic on the decimal strength.
STRENGTHS = [Fraction(n, 1000) for n in range(1, 1000)]


def _p(steps, strength):
    return SimpleNamespace(steps=steps, denoising_strength=float(strength))


def test_fixed_steps_schedule_length_is_the_exact_quotient(monkeypatch):
    monkeypatch.setattr(shared.opts, "img2img_fix_steps", True, raising=False)
    mismatches = []
    for requested in range(1, 151):
        for strength in STRENGTHS:
            steps, t_enc = sd_samplers_common.setup_img2img_steps(_p(requested, strength))
            expected = (requested / min(strength, Fraction(999, 1000))).__floor__()
            if (steps, t_enc) != (expected, requested - 1):
                mismatches.append((requested, float(strength), steps, expected))

    assert not mismatches, mismatches[:10]
    # e.g. 33 / 0.55 evaluates to 59.99999999999999 in binary floating point
    assert sd_samplers_common.setup_img2img_steps(_p(33, Fraction(55, 100)))[0] == 60


def test_scaled_steps_encode_length_is_the_exact_product(monkeypatch):
    monkeypatch.setattr(shared.opts, "img2img_fix_steps", False, raising=False)
    mismatches = []
    for steps in range(1, 151):
        for strength in STRENGTHS:
            result = sd_samplers_common.setup_img2img_steps(_p(steps, strength))
            expected = (min(strength, Fraction(999, 1000)) * steps).__floor__()
            if result != (steps, expected):
                mismatches.append((steps, float(strength), result[1], expected))

    assert not mismatches, mismatches[:10]


@pytest.mark.parametrize("fix_steps", [True, False])
def test_explicit_steps_and_zero_strength_keep_their_meaning(monkeypatch, fix_steps):
    monkeypatch.setattr(shared.opts, "img2img_fix_steps", fix_steps, raising=False)

    # hires fix passes its own step count and always uses the fixed-steps formula
    assert sd_samplers_common.setup_img2img_steps(_p(20, Fraction(3, 5)), steps=15) == (25, 14)
    assert sd_samplers_common.setup_img2img_steps(_p(20, Fraction(0)), steps=15) == (0, 14)
    # strengths above 0.999 are capped
    assert sd_samplers_common.setup_img2img_steps(_p(10, Fraction(1)), steps=10) == (10, 9)
