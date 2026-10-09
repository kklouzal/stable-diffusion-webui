from fractions import Fraction

from test.helpers import init_shared

shared = init_shared()

from modules.processing import StableDiffusionProcessingTxt2Img  # noqa: E402


def _target(width, height, hr_scale):
    p = StableDiffusionProcessingTxt2Img.__new__(StableDiffusionProcessingTxt2Img)
    p.__dict__.update(width=width, height=height, hr_scale=hr_scale, hr_resize_x=0, hr_resize_y=0,
                      extra_generation_params={}, applied_old_hires_behavior_to=None)
    p.calculate_target_resolution()
    return p.hr_upscale_to_x, p.hr_upscale_to_y


def test_hires_scale_target_is_the_exact_floor(monkeypatch):
    monkeypatch.setattr(shared.opts, "use_old_hires_fix_width_height", False, raising=False)
    mismatches = []
    # Oracle: floor(size * scale) in exact rational arithmetic on the decimal scale (UI slider: 1..4 in 0.05 steps).
    for size in range(64, 2049, 8):
        for twentieths in range(20, 81):
            expected = (size * Fraction(twentieths, 20)).__floor__()
            actual = _target(size, size, twentieths / 20)
            if actual != (expected, expected):
                mismatches.append((size, twentieths / 20, actual, expected))

    assert not mismatches, mismatches[:10]
    assert _target(1600, 1024, 1.15) == (1840, 1177)  # 1600 * 1.15 == 1839.9999999999998
