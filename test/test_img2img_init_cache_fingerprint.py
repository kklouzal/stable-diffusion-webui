"""img2img init cache: uint8/raw-image keys and deferred float conversion stay exact."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from test.helpers import init_shared

shared = init_shared()

from modules import images, processing  # noqa: E402
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingImg2Img  # noqa: E402


_flatten = images.flatten  # unpatched, for the oracle


def _old_batch_images(init_images, width, height, resize_mode, background):
    """Pre-change unmasked pipeline: prepare, convert to float, then batch (oracle)."""
    imgs = []
    for img in init_images:
        image = _flatten(img, background)
        if resize_mode != 3:
            image = images.resize_image(resize_mode, image, width, height)
        imgs.append(processing._image_to_chw_float32_array(image))
    return np.expand_dims(imgs[0], axis=0) if len(imgs) == 1 else np.array(imgs)


def _random_image(mode, size, seed):
    rng = np.random.default_rng(seed)
    channels = {"RGB": 3, "RGBA": 4}[mode]
    return Image.fromarray(rng.integers(0, 256, (size[1], size[0], channels), dtype=np.uint8), mode)


class _AttnBlock(torch.nn.Module):
    def forward(self, x):
        return x


class _Encoder(torch.nn.Module):
    def forward(self, x):
        return x


class _VAE(torch.nn.Module):
    """The modules the init cache key reads: the encoder (Tiled VAE overrides its forward) and the AttnBlock."""

    def __init__(self):
        super().__init__()
        self.encoder = _Encoder()
        self.decoder = torch.nn.Module()
        self.decoder.mid = torch.nn.Module()
        self.decoder.mid.attn_1 = _AttnBlock()


@pytest.fixture
def harness(monkeypatch):
    model = SimpleNamespace(cond_stage_key="crossattn", sd_checkpoint_info=SimpleNamespace(filename="m.safetensors", hash="h", sha256="s"), is_sdxl_inpaint=False, first_stage_model=_VAE())
    sampler = SimpleNamespace(conditioning_key="crossattn", model_wrap=SimpleNamespace(inner_model=model))
    encoded = []
    flattened = []

    def images_tensor_to_samples(image, approximation=None, model=None):
        encoded.append(image.clone())
        return torch.full((image.shape[0], 4, image.shape[2] // 8, image.shape[3] // 8), float(len(encoded)))

    def flatten(img, bgcolor):
        flattened.append(img)
        return _flatten(img, bgcolor)

    monkeypatch.setattr(processing.shared, "sd_model", model, raising=False)
    monkeypatch.setattr(processing.shared, "device", torch.device("cpu"), raising=False)
    monkeypatch.setattr(processing.devices, "dtype_vae", torch.float32)
    monkeypatch.setattr(processing.sd_samplers, "create_sampler", lambda name, sd_model: sampler)
    monkeypatch.setattr(processing, "images_tensor_to_samples", images_tensor_to_samples)
    monkeypatch.setattr(processing.images, "flatten", flatten)
    monkeypatch.setattr(processing.sd_vae, "get_loaded_vae_name", lambda: "vae", raising=False)
    monkeypatch.setattr(processing.sd_vae, "get_loaded_vae_hash", lambda: "vae-hash", raising=False)
    for name, value in {
        "persistent_img2img_init_cache": True, "upscaler_for_img2img": None, "img2img_color_correction": False,
        "save_init_img": False, "sd_vae_encode_method": "Full", "img2img_background_color": "#ffffff",
        "inpainting_mask_weight": 1.0,
    }.items():
        monkeypatch.setattr(processing.opts, name, value, raising=False)
    monkeypatch.setattr(StableDiffusionProcessing, "cached_img2img_init", [None, None])
    monkeypatch.setattr(StableDiffusionProcessing, "cached_img2img_init_stats", processing._cache_stats(last_hit=False, cached=False, bypass_reason=None))

    def make(init_images, *, width=64, height=48, resize_mode=0, batch_size=1, tiling=False, image_mask=None, mask_round=True, **fields):
        p = StableDiffusionProcessingImg2Img.__new__(StableDiffusionProcessingImg2Img)
        p.__dict__.update(
            extra_generation_params={}, denoising_strength=0.5, image_cfg_scale=None, sampler_name="Euler", sd_model=model,
            image_mask=image_mask, latent_mask=None, color_corrections=None, overlay_images=None, init_images=init_images,
            resize_mode=resize_mode, width=width, height=height, batch_size=batch_size, inpainting_fill=0, mask_round=mask_round,
            sd_model_name="m", sd_model_hash="h", inpainting_mask_invert=False, inpaint_full_res=False,
            inpaint_full_res_padding=0, mask_blur_x=0, mask_blur_y=0, tiling=tiling, latent_noise_fill=None,
        )
        p.__dict__.update(fields)
        p.init([""], [1], [1])
        return p

    return SimpleNamespace(make=make, encoded=encoded, flattened=flattened, model=model)


def test_raw_image_key_hits_before_flatten_and_matches_old_pipeline(harness):
    raw = _random_image("RGBA", (40, 56), 1)

    first = harness.make([raw])
    assert first.openclaw_img2img_init_cache_stats["last_hit"] is False
    oracle = torch.from_numpy(_old_batch_images([raw], 64, 48, 0, "#ffffff"))
    assert torch.equal(harness.encoded[0], oracle)
    assert harness.encoded[0].stride() == oracle.stride()
    assert len(harness.flattened) == 1

    second = harness.make([raw.copy()])
    assert second.openclaw_img2img_init_cache_stats["last_hit"] is True
    assert len(harness.encoded) == 1 and len(harness.flattened) == 1
    assert torch.equal(second.init_latent, first.init_latent)


def test_multi_image_batch_layout_matches_old_pipeline(harness):
    raws = [_random_image("RGB", (64, 48), 2), _random_image("RGB", (64, 48), 3)]

    harness.make(raws, batch_size=2)
    oracle = torch.from_numpy(_old_batch_images(raws, 64, 48, 0, "#ffffff"))
    assert torch.equal(harness.encoded[0], oracle)
    assert harness.encoded[0].stride() == oracle.stride()


def test_raw_key_distinguishes_palette_and_resize_inputs(harness):
    indices = np.random.default_rng(4).integers(0, 4, (48, 64), dtype=np.uint8)
    paletted = Image.fromarray(indices, "P")
    paletted.putpalette([0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255] * 64)
    repainted = paletted.copy()
    repainted.putpalette([255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255] * 64)

    harness.make([paletted])
    harness.make([repainted])
    harness.make([repainted], resize_mode=1)
    harness.make([repainted], resize_mode=1, width=48)
    assert len(harness.encoded) == 4
    assert torch.equal(harness.encoded[1], torch.from_numpy(_old_batch_images([repainted], 64, 48, 0, "#ffffff")))


def test_upscaler_resize_keys_on_prepared_image(harness, monkeypatch):
    monkeypatch.setattr(processing.opts, "upscaler_for_img2img", "Some upscaler", raising=False)
    raw = _random_image("RGB", (64, 48), 5)  # target size: resize_image never invokes the upscaler

    harness.make([raw])
    second = harness.make([raw.copy()])
    assert second.openclaw_img2img_init_cache_stats["last_hit"] is True
    assert len(harness.encoded) == 1
    assert len(harness.flattened) == 2  # prepared-image keys flatten/resize before the lookup


def test_color_corrections_are_computed_on_miss_and_restored_on_hit(harness, monkeypatch):
    monkeypatch.setattr(processing.opts, "img2img_color_correction", True, raising=False)
    raw = _random_image("RGBA", (40, 56), 6)

    first = harness.make([raw], batch_size=3)
    expected = processing.setup_color_correction(images.resize_image(0, _flatten(raw, "#ffffff"), 64, 48))
    assert len(first.color_corrections) == 3
    assert all(np.array_equal(item, expected) for item in first.color_corrections)

    second = harness.make([raw], batch_size=3)
    assert second.openclaw_img2img_init_cache_stats["last_hit"] is True
    assert len(second.color_corrections) == 3
    assert all(np.array_equal(item, expected) for item in second.color_corrections)


def test_caller_color_corrections_are_neither_cached_nor_replaced(harness, monkeypatch):
    # Loopback presets p.color_corrections; the option alone does not compute any then. Neither request may see the
    # other's corrections through the init cache.
    monkeypatch.setattr(processing.opts, "img2img_color_correction", True, raising=False)
    raw = _random_image("RGB", (64, 48), 10)
    preset = [processing.setup_color_correction(_random_image("RGB", (64, 48), 11))]

    def same(actual, expected):
        return len(actual) == len(expected) and all(np.array_equal(a, b) for a, b in zip(actual, expected))

    first = harness.make([raw], color_corrections=preset)
    assert same(first.color_corrections, preset)

    monkeypatch.setattr(processing.opts, "img2img_color_correction", False, raising=False)
    second = harness.make([raw.copy()])  # same key: the option is off and nothing is computed either way
    assert second.openclaw_img2img_init_cache_stats["last_hit"] is True
    assert second.color_corrections is None

    other = [processing.setup_color_correction(_random_image("RGB", (64, 48), 12))]
    third = harness.make([raw.copy()], color_corrections=other)
    assert third.openclaw_img2img_init_cache_stats["last_hit"] is True
    assert same(third.color_corrections, other)
    assert len(harness.encoded) == 1


def test_tiling_is_part_of_the_init_cache_key(harness):
    # apply_circular(p.tiling) runs before p.init and also switches the VAE encoder's convolutions to circular padding.
    raw = _random_image("RGB", (64, 48), 7)

    harness.make([raw])
    tiled = harness.make([raw.copy()], tiling=True)
    assert tiled.openclaw_img2img_init_cache_stats["last_hit"] is False
    assert len(harness.encoded) == 2

    assert harness.make([raw.copy()], tiling=True).openclaw_img2img_init_cache_stats["last_hit"] is True
    assert len(harness.encoded) == 2


def test_an_instance_encoder_forward_bypasses_the_init_cache(harness):
    # Tiled VAE (VAEHook) assigns encoder.forward in process(); its restore assigns the saved bound method back.
    raw = _random_image("RGB", (64, 48), 13)
    encoder = harness.model.first_stage_model.encoder

    encoder.forward = lambda x: x
    first = harness.make([raw])
    second = harness.make([raw.copy()])
    assert second.openclaw_img2img_init_cache_stats["bypass_reason"] == "vae_encoder_override"
    assert first.openclaw_img2img_init_cache_stats["last_hit"] is second.openclaw_img2img_init_cache_stats["last_hit"] is False
    assert len(harness.encoded) == 2

    encoder.forward = _Encoder.forward.__get__(encoder)
    harness.make([raw.copy()])
    assert harness.make([raw.copy()]).openclaw_img2img_init_cache_stats["last_hit"] is True
    assert len(harness.encoded) == 3


@pytest.mark.parametrize("change", ["upcast_attn", "sdpa_backend", "attn_block_forward", "nhwc_group_norm"])
def test_vae_execution_state_is_part_of_the_init_cache_key(harness, monkeypatch, change):
    from modules import openclaw_vae_decode_graphs, sd_hijack_optimizations

    raw = _random_image("RGB", (64, 48), 14)
    harness.make([raw])
    if change == "upcast_attn":
        monkeypatch.setattr(processing.opts, "upcast_attn", not processing.opts.upcast_attn, raising=False)
    elif change == "sdpa_backend":
        monkeypatch.setattr(sd_hijack_optimizations, "active_sdpa_backend", lambda: "another-backend")
    elif change == "attn_block_forward":  # sd_hijack installs the cross-attention optimization's forward on the class
        monkeypatch.setattr(_AttnBlock, "forward", lambda self, x: x + 0)
    else:
        monkeypatch.setattr(openclaw_vae_decode_graphs, "_nhwc_group_norm_state", lambda: ("another-scope",))

    changed = harness.make([raw.copy()])
    assert changed.openclaw_img2img_init_cache_stats["last_hit"] is False
    assert len(harness.encoded) == 2
    assert harness.make([raw.copy()]).openclaw_img2img_init_cache_stats["last_hit"] is True


def test_soft_latent_masks_stay_float32_and_complementary(harness, monkeypatch):
    # The masks blend float32 latents; in the 16-bit model dtype a soft mask was quantized and mask + nmask != 1.
    monkeypatch.setattr(processing.devices, "dtype", torch.bfloat16)
    ramp = np.tile(np.linspace(0, 255, 64).astype(np.uint8), (48, 1))
    mask_image = Image.fromarray(ramp, "L")
    expected_nmask = torch.from_numpy(processing._resize_latent_mask(mask_image, (8, 6), round=False))
    assert not torch.equal(expected_nmask.to(torch.bfloat16).float(), expected_nmask)  # the old cast was lossy here

    p = harness.make([_random_image("RGB", (64, 48), 8)], image_mask=mask_image, mask_round=False)

    assert p.nmask.dtype == torch.float32 and p.mask.dtype == torch.float32
    assert p.nmask.shape == (4, 6, 8)
    assert all(torch.equal(channel, expected_nmask) for channel in p.nmask)
    assert float((p.mask + p.nmask - 1).abs().max()) <= 2 ** -23


def test_binary_latent_masks_are_exact(harness):
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[:, 32:] = 255

    p = harness.make([_random_image("RGB", (64, 48), 9)], image_mask=Image.fromarray(mask, "L"))

    assert torch.equal(p.nmask[0], torch.cat([torch.zeros(6, 4), torch.ones(6, 4)], dim=1))
    assert torch.equal(p.mask + p.nmask, torch.ones(4, 6, 8))
