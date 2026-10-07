from fractions import Fraction

from modules import prompt_parser


def _switch_step(prompt, steps, hires_steps=None):
    schedule = prompt_parser.get_learned_conditioning_prompt_schedules([prompt], steps, hires_steps)[0]
    return [end for end, text in schedule if text == "a"]


def test_fractional_switch_step_is_the_exact_floor():
    # Oracle: floor(fraction * steps) in exact rational arithmetic on the decimal the user wrote.
    mismatches = []
    for hundredths in range(1, 100):
        for steps in range(1, 101):
            expected_step = (Fraction(hundredths, 100) * steps).__floor__()
            expected = [expected_step] if expected_step >= 1 else []
            actual = _switch_step(f"[a:b:0.{hundredths:02d}]", steps)
            if actual != expected:
                mismatches.append((hundredths, steps, actual, expected))

    assert not mismatches, mismatches[:10]
    assert _switch_step("[a:b:0.29]", 100) == [29]  # 0.29 * 100 == 28.999999999999996


def test_hires_fraction_offsets_keep_their_meaning():
    # With a hires pass, fractions above 1 address the hires steps and integers above the base steps do too.
    assert _switch_step("[a:b:1.29]", 20, hires_steps=100) == [29]
    assert _switch_step("[a:b:0.5]", 20, hires_steps=10) == []
    assert _switch_step("[a:b:25]", 20, hires_steps=10) == [5]
    assert _switch_step("[a:b:7]", 20) == [7]
