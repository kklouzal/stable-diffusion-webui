"""Hypertile configures the hires-fix pass for the latent the sampler actually denoises (after the hr_resize crop)."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from test.helpers import init_shared, load_source, stub_modules

shared = init_shared()

from modules import processing, script_callbacks  # noqa: E402
from modules.processing import StableDiffusionProcessingTxt2Img  # noqa: E402


class _StopAtSampling(Exception):
    pass


@pytest.fixture()
def hypertile_script(monkeypatch):
    hypertile = load_source("hypertile_for_hires", "extensions-builtin/hypertile/hypertile.py")
    monkeypatch.setattr(script_callbacks, "on_ui_settings", lambda callback: None)
    monkeypatch.setattr(script_callbacks, "on_before_ui", lambda callback: None)
    with stub_modules({"hypertile": hypertile}):
        module = load_source("hypertile_script_under_test", "extensions-builtin/hypertile/scripts/hypertile_script.py")
    return hypertile, module


def _unet_with_depth0_layer():
    unet = torch.nn.Module()
    parent = unet
    for part in "input_blocks.1.1.transformer_blocks.0".split("."):
        parent.add_module(part, torch.nn.Module())
        parent = getattr(parent, part)
    parent.add_module("attn1", torch.nn.Identity())
    unet.conditioning_key = "crossattn"
    return unet, parent.attn1


@pytest.mark.parametrize(
    ("width", "height", "hr_scale", "hr_resize_x", "hr_resize_y"),
    [
        (1024, 1024, 2.0, 1536, 1024),  # cropped to 3:2; the pre-crop 1:1 aspect transposed the token grid
        (1024, 1024, 2.0, 1024, 1536),
        (832, 1216, 2.0, 1216, 1216),
        (1000, 600, 2.0, 1600, 1600),
        (1024, 1024, 1.5, 0, 0),  # no crop: unchanged
        (1000, 600, 1.15, 0, 0),
    ],
)
def test_hires_pass_tiles_the_cropped_latent(monkeypatch, hypertile_script, width, height, hr_scale, hr_resize_x, hr_resize_y):
    hypertile, script_module = hypertile_script
    unet, attn = _unet_with_depth0_layer()
    sd_model = SimpleNamespace(first_stage_model=torch.nn.Module(), model=unet, is_sdxl=False, cond_stage_key="crossattn",
                               is_sdxl_inpaint=False)
    opts = SimpleNamespace(
        hypertile_enable_unet=True, hypertile_enable_unet_secondpass=False, hypertile_enable_vae=False,
        hypertile_swap_size_unet=1, hypertile_max_depth_unet=3, hypertile_max_tile_unet=256,
        hypertile_swap_size_vae=1, hypertile_max_depth_vae=3, hypertile_max_tile_vae=128,
    )
    monkeypatch.setattr(script_module, "shared", SimpleNamespace(opts=opts, sd_model=sd_model))
    monkeypatch.setattr(processing.shared, "sd_model", sd_model, raising=False)
    script = script_module.ScriptHypertile()

    sampled = []
    sampler = SimpleNamespace(conditioning_key="crossattn", model_wrap=SimpleNamespace(inner_model=sd_model))

    def sample_img2img(p_, x, noise, c, uc, steps=None, image_conditioning=None):
        sampled.append(x)
        raise _StopAtSampling()

    sampler.sample_img2img = sample_img2img
    monkeypatch.setattr(processing.sd_samplers, "create_sampler", lambda name, model: sampler)
    monkeypatch.setattr(processing.shared, "state", SimpleNamespace(interrupted=False, nextjob=lambda: None), raising=False)
    monkeypatch.setattr(processing.rng, "ImageRNG", lambda shape, *args, **kwargs: SimpleNamespace(next=lambda: torch.zeros(1, *shape)))
    monkeypatch.setattr(processing.devices, "autocast", lambda disable=False: contextlib.nullcontext())
    monkeypatch.setattr(processing.sd_models, "apply_token_merging", lambda model, ratio: None)

    p = StableDiffusionProcessingTxt2Img.__new__(StableDiffusionProcessingTxt2Img)
    p.__dict__.update(
        width=width, height=height, hr_scale=hr_scale, hr_resize_x=hr_resize_x, hr_resize_y=hr_resize_y,
        hr_upscale_to_x=0, hr_upscale_to_y=0, truncate_x=0, truncate_y=0, applied_old_hires_behavior_to=None,
        extra_generation_params={}, all_seeds=[1], hr_sampler_name=None, sampler_name="Euler", inpainting_mask_weight=1.0,
        latent_scale_mode={"mode": "nearest", "antialias": False}, do_not_save_samples=True, seeds=[1], subseeds=[1],
        subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0, disable_extra_networks=True, hr_c=None,
        hr_uc=None, hr_second_pass_steps=0, steps=1, calculate_hr_conds=lambda: None,
        get_token_merging_ratio=lambda for_hr=False: 0.0,
        scripts_value=SimpleNamespace(before_hr=lambda p_: script.before_hr(p_), process_before_every_sampling=lambda **kwargs: None),
    )
    p.calculate_target_resolution()

    with pytest.raises(_StopAtSampling):
        p.sample_hr_pass(torch.zeros(1, 4, 2, 2), None, [1], [1], 0.0, [""])

    rows, cols = sampled[0].shape[-2:]
    params = getattr(attn, "__webui_hypertile_params")
    assert params.enabled
    assert params.aspect_ratio == pytest.approx(cols / rows, rel=0.01)
    # The first transformer level runs on exactly this token grid: hypertile must recover it, rows first.
    assert hypertile.find_hw_candidates(rows * cols, params.aspect_ratio) == (rows, cols)
    if p.truncate_x == p.truncate_y == 0:
        assert params.aspect_ratio == p.hr_upscale_to_x / p.hr_upscale_to_y  # uncropped: the same size as before
