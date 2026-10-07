"""Conditioning cache keys cover every input the SDXL conditioner reads (B1, B5)."""

import contextlib
from types import SimpleNamespace

import pytest

from modules import shared, shared_init

if getattr(shared, "opts", None) is None:
    shared_init.initialize()

from modules import processing, prompt_parser  # noqa: E402
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingTxt2Img  # noqa: E402


@pytest.fixture
def p(monkeypatch):
    monkeypatch.setattr(processing.shared, "sd_model", SimpleNamespace(sd_checkpoint_info="checkpoint"), raising=False)
    monkeypatch.setattr(StableDiffusionProcessing, "active_lora_cond_signature", lambda self: ())
    monkeypatch.setattr(processing.devices, "autocast", contextlib.nullcontext)
    for name, value in {"use_old_scheduling": False, "sdxl_refiner_low_aesthetic_score": 2.5, "sdxl_refiner_high_aesthetic_score": 6.0}.items():
        monkeypatch.setattr(processing.opts, name, value, raising=False)
    p = StableDiffusionProcessingTxt2Img.__new__(StableDiffusionProcessingTxt2Img)
    p.__dict__.update(width=1024, height=1024, extra_generation_params={})
    return p


def _conds(p, cache, prompts, computed):
    def function(model, required_prompts, steps, hires_steps, use_old_scheduling):
        computed.append((list(required_prompts), required_prompts.width, required_prompts.height))
        return object()

    return p.get_conds_with_caching("hr_c", function, prompts, 20, cache, None, 10)


def test_hires_conds_recompute_when_only_the_hires_size_changes(p):
    cache = [None, None]
    computed = []

    first = _conds(p, cache, prompt_parser.SdConditioning(["a cat"], width=1536, height=1536), computed)
    assert _conds(p, cache, prompt_parser.SdConditioning(["a cat"], width=1536, height=1536), computed) is first
    resized = _conds(p, cache, prompt_parser.SdConditioning(["a cat"], width=2048, height=1536), computed)

    assert resized is not first
    assert computed == [(["a cat"], 1536, 1536), (["a cat"], 2048, 1536)]


def test_key_covers_negative_flag_and_refiner_aesthetic_scores(p, monkeypatch):
    prompts = prompt_parser.SdConditioning([""], width=1024, height=1024)
    negative = prompt_parser.SdConditioning([""], width=1024, height=1024, is_negative_prompt=True)
    base = p.cached_params("uc", negative, 20, None)

    assert p.cached_params("uc", prompts, 20, None) != base
    monkeypatch.setattr(processing.opts, "sdxl_refiner_low_aesthetic_score", 3.0, raising=False)
    changed = p.cached_params("uc", negative, 20, None)
    assert changed != base
    assert p._conditioning_cache_miss_reason(base, changed) == "changed:refiner_aesthetic_score"


def test_miss_reason_names_prompt_dimensions(p):
    small = p.cached_params("hr_c", prompt_parser.SdConditioning(["a"], width=1536, height=1536), 20, None, 10)
    large = p.cached_params("hr_c", prompt_parser.SdConditioning(["a"], width=2048, height=2048), 20, None, 10)

    assert p._conditioning_cache_miss_reason(small, large) == "changed:prompt_dimensions"
