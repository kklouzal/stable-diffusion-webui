"""Conditioning cache keys cover every input the SDXL conditioner reads (B1, B5); hits replay encoder infotext (B2)."""

import contextlib
from types import SimpleNamespace

import pytest

from test.helpers import init_shared

shared = init_shared()

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


def test_key_covers_prompt_dimensions(p):
    small = p.cached_params("hr_c", prompt_parser.SdConditioning(["a"], width=1536, height=1536), 20, None, 10)
    large = p.cached_params("hr_c", prompt_parser.SdConditioning(["a"], width=2048, height=2048), 20, None, 10)

    assert small != large


def _encoder_writes(hashes, emphasis=None):
    """What FrozenCLIPEmbedderWithCustomWords.forward writes to model_hijack.extra_generation_params."""
    params = processing.model_hijack.extra_generation_params
    if hashes:
        if params.get("TI hashes"):
            hashes = hashes + [params.get("TI hashes")]
        params["TI hashes"] = ", ".join(hashes)
    if emphasis:
        params["Emphasis"] = emphasis


def _run_request(p, caches, negative, positive, calls):
    """setup_conds order: uc, then c; each text's encoder writes depend only on that text."""
    processing.model_hijack.extra_generation_params = {}

    def encode(model, required_prompts, steps, hires_steps, use_old_scheduling):
        calls.append(required_prompts[0])
        for text in required_prompts[0].split("|"):
            _encoder_writes([f"{text}: h"] if text.startswith("ti") else [], "No norm" if "(" in text else None)
        return object()

    p.get_conds_with_caching("uc", encode, prompt_parser.SdConditioning([negative], is_negative_prompt=True), 20, caches[0], None)
    p.get_conds_with_caching("c", encode, prompt_parser.SdConditioning([positive]), 20, caches[1], None)
    return dict(processing.model_hijack.extra_generation_params)


@pytest.mark.parametrize("requests", [
    [("ti_neg", "ti_pos|(x)"), ("ti_neg", "ti_pos|(x)")],
    [("ti_neg|(y)", "ti_pos|(x)"), ("plain", "ti_pos|(x)"), ("ti_neg|(y)", "ti_pos|(x)")],
    [("ti_a|ti_b", "plain"), ("ti_c", "plain"), ("ti_a|ti_b", "ti_pos")],
])
def test_cache_hits_reproduce_encoder_infotext(p, monkeypatch, requests):
    monkeypatch.setattr(processing.model_hijack, "extra_generation_params", {}, raising=False)
    caches = ([None, None], [None, None])
    calls = []
    for negative, positive in requests:
        cached = _run_request(p, caches, negative, positive, calls)
        fresh = _run_request(p, ([None, None], [None, None]), negative, positive, [])
        assert cached == fresh
        assert list(cached) == list(fresh)
    assert len(calls) < 2 * len(requests)  # some of the cached runs were hits


def test_failed_conditioning_keeps_existing_infotext(p, monkeypatch):
    monkeypatch.setattr(processing.model_hijack, "extra_generation_params", {"TI hashes": "a: h"}, raising=False)

    def fail(*args):
        _encoder_writes(["b: h"])
        raise RuntimeError("encoder failed")

    with pytest.raises(RuntimeError, match="encoder failed"):
        p.get_conds_with_caching("c", fail, prompt_parser.SdConditioning(["x"]), 20, [None, None], None)
    assert processing.model_hijack.extra_generation_params == {"TI hashes": "a: h"}


def test_key_ignores_extra_network_tags_it_covers_otherwise(p):
    # LoRA reaches the key through the published text encoder state; hypernetworks act on the U-Net only. Tags of
    # other extra networks still key the conditioning.
    from modules.extra_networks import ExtraNetworkParams

    prompts = prompt_parser.SdConditioning(["a"], width=1024, height=1024)
    base = p.cached_params("c", prompts, 20, {})
    covered = {"lora": [ExtraNetworkParams(["detail", "1", "0.4"])], "lyco": [ExtraNetworkParams(["x"])], "hypernet": [ExtraNetworkParams(["h", "0.5"])]}
    assert p.cached_params("c", prompts, 20, covered) == base
    assert p.cached_params("c", prompts, 20, {**covered, "other": [ExtraNetworkParams(["o"])]}) != base


def test_key_covers_the_bundled_embedding_infotext_option(p, monkeypatch):
    # A LoRA-bundled embedding writes "TI hashes" only while lora_bundled_ti_to_infotext is on, and hits replay it.
    prompts = prompt_parser.SdConditioning(["a"], width=1024, height=1024)
    monkeypatch.setattr(processing.opts, "lora_bundled_ti_to_infotext", True, raising=False)
    on = p.cached_params("c", prompts, 20, None)
    monkeypatch.setattr(processing.opts, "lora_bundled_ti_to_infotext", False, raising=False)
    assert p.cached_params("c", prompts, 20, None) != on
