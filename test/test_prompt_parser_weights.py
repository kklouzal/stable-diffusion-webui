"""Prompt parsing: composable AND weights, attention brackets around BREAK, and the module's own doctests."""

import doctest

import pytest

from modules import prompt_parser


def multicond(prompt):
    indexes, texts, _ = prompt_parser.get_multicond_prompt_list([prompt])
    return [(texts[index], weight) for index, weight in indexes[0]]


@pytest.mark.parametrize("prompt", [
    "aspect ratio 16:9",
    "a clock at 10:30",
    "scene 3:2 AND poster 4:5",
    "ratio 16: 9",
    "time 10:30.5",
])
def test_colon_after_a_digit_is_text_not_a_weight(prompt):
    # Trailing whitespace is dropped, leading kept, as for every subprompt.
    assert multicond(prompt) == [(part.rstrip(), 1.0) for part in prompt.split("AND")]


@pytest.mark.parametrize(("prompt", "expected"), [
    ("text:1.2", [("text", 1.2)]),
    ("text :1.2", [("text", 1.2)]),
    ("text : 1.2", [("text", 1.2)]),
    ("a cat :1.2 AND a dog:0.5", [("a cat", 1.2), (" a dog", 0.5)]),
    ("a cat AND a dog:-.5", [("a cat", 1.0), (" a dog", -0.5)]),
    ("16:9 photo :1.5", [("16:9 photo", 1.5)]),
    ("ratio 16 :9", [("ratio 16", 9.0)]),  # whitespace before the colon: a weight
    ("(red:1.3) car", [("(red:1.3) car", 1.0)]),
])
def test_weights_keep_working(prompt, expected):
    assert multicond(prompt) == expected


def test_break_inside_brackets_stays_a_break_marker():
    parsed = prompt_parser.parse_prompt_attention("a (b BREAK c) [d BREAK e] ((f BREAK))")

    markers = [entry for entry in parsed if entry[0] == "BREAK"]
    assert markers == [["BREAK", -1]] * 3
    assert ["c", 1.1] in parsed and ["e", 1 / 1.1] in parsed
    assert parsed[1:4] == [["b", 1.1], ["BREAK", -1], ["c", 1.1]]


def test_break_marker_never_merges_with_text_weighted_minus_one():
    parsed = prompt_parser.parse_prompt_attention("(x BREAK y:-1)")

    assert parsed == [["x", -1.0], ["BREAK", -1], ["y", -1.0]]


def test_break_outside_brackets_is_unchanged():
    assert prompt_parser.parse_prompt_attention("a BREAK b") == [["a", 1.0], ["BREAK", -1], ["b", 1.0]]
    assert prompt_parser.parse_prompt_attention("(a) BREAK b") == [["a", 1.1], ["", 1.0], ["BREAK", -1], ["b", 1.0]]


def test_module_doctests():
    result = doctest.testmod(prompt_parser, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert result.attempted > 0 and result.failed == 0
