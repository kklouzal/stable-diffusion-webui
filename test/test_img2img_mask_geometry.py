"""img2img mask geometry: masks of another size, resize_mode 3, alpha masks and per-batch latent-noise fill."""

from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image

from test.test_img2img_init_cache_fingerprint import _random_image, harness  # noqa: F401 (fixture)
from modules import images, processing


def _half_mask(size, left=True):
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    mask[:, :size[0] // 2] = 255
    return Image.fromarray(mask if left else 255 - mask, "L")


def test_resize_mode_3_stretches_mask_and_builds_overlay_at_output_size(harness):  # noqa: F811
    init = _random_image("RGB", (32, 32), 20)  # another aspect than the 64x48 target: fill would pad, not stretch

    p = harness.make([init], resize_mode=3, image_mask=_half_mask((32, 32)))

    # the latent is stretched to 64x48 (bilinear), so the mask is stretched too
    stretched_mask = images.resize_image(0, _half_mask((32, 32)), 64, 48)
    assert np.array_equal(np.asarray(p.mask_for_overlay), np.clip(np.asarray(stretched_mask, dtype=np.float32) * 2, 0, 255).astype(np.uint8))
    assert torch.equal(p.nmask[0], torch.cat([torch.ones(6, 4), torch.zeros(6, 4)], dim=1))

    overlay = p.overlay_images[0]
    assert overlay.size == (64, 48)
    alpha = np.asarray(overlay)[..., 3]
    assert (alpha[:, :30] == 0).all() and (alpha[:, 40:] == 255).all()
    stretched = np.asarray(init.resize((64, 48), resample=images.LANCZOS))
    assert np.array_equal(np.asarray(overlay)[:, 40:, :3], stretched[:, 40:])

    # the composite keeps generated pixels where masked and the stretched init image elsewhere
    generated = Image.new("RGB", (64, 48), (200, 0, 0))
    out, _ = processing.apply_overlay(generated, p.paste_to, overlay)
    out = np.asarray(out)
    assert (out[:, :30] == (200, 0, 0)).all()
    assert np.array_equal(out[:, 40:], stretched[:, 40:])

    # the VAE input stays at the init size (its latent is stretched afterwards)
    assert tuple(harness.encoded[-1].shape[-2:]) == (32, 32)


def test_resize_mode_3_fill_runs_on_the_init_sized_image(harness):  # noqa: F811
    p = harness.make([_random_image("RGB", (32, 32), 21)], resize_mode=3, image_mask=_half_mask((32, 32)), inpainting_fill=0)
    assert tuple(harness.encoded[-1].shape[-2:]) == (32, 32)
    assert p.extra_generation_params["Masked content"] == "fill"


def test_only_masked_crop_uses_image_coordinates_for_a_smaller_mask(harness):  # noqa: F811
    init = _random_image("RGB", (128, 96), 22)
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[12:24, 16:32] = 255  # image region x 32..64, y 24..48

    p = harness.make([init], width=32, height=24, image_mask=Image.fromarray(mask, "L"), inpaint_full_res=True)

    # the masked region in image coordinates is x 32..64, y 24..48 (the crop then keeps the 4:3 target aspect);
    # bilinear adds at most a pixel around it. In mask coordinates the crop was x 16..32 of the 128 wide image.
    x, y, w, h = p.paste_to
    assert 30 <= x <= 32 and 22 <= y <= 24 and 64 <= x + w <= 66 and 48 <= y + h <= 50
    assert p.mask_for_overlay.size == (128, 96)
    assert p.overlay_images[0].size == (128, 96)


def test_same_size_mask_is_not_resampled(harness):  # noqa: F811
    mask = _half_mask((64, 48))
    p = harness.make([_random_image("RGB", (64, 48), 23)], image_mask=mask)
    expected = np.clip(np.asarray(mask, dtype=np.float32) * 2, 0, 255).astype(np.uint8)
    assert np.array_equal(np.asarray(p.mask_for_overlay), expected)


def test_alpha_masks_use_their_alpha_in_every_mode():
    alpha = np.zeros((8, 8), dtype=np.uint8)
    alpha[:, :3] = 255
    rgba = Image.fromarray(np.dstack([np.zeros((8, 8, 3), dtype=np.uint8), alpha]), "RGBA")
    expected = np.asarray(processing.create_binary_mask(rgba))
    assert (expected[:, :3] == 255).all() and (expected[:, 3:] == 0).all()

    la = rgba.convert("LA")
    pa = rgba.convert("PA")
    keyed = Image.fromarray(np.where(alpha > 0, 1, 0).astype(np.uint8), "P")
    keyed.putpalette([0, 0, 0, 0, 0, 0])
    keyed.info["transparency"] = 0
    for mask in (la, pa, keyed):
        assert mask.mode != "RGBA" and mask.has_transparency_data
        assert np.array_equal(np.asarray(processing.create_binary_mask(mask)), expected), mask.mode

    # opaque masks keep their luminance
    gray = Image.fromarray(np.full((8, 8), 77, dtype=np.uint8), "L")
    assert np.array_equal(np.asarray(processing.create_binary_mask(gray)), np.asarray(gray))


def test_latent_noise_fill_uses_each_batch_seeds(harness):  # noqa: F811
    p = harness.make([_random_image("RGB", (64, 48), 24)], image_mask=_half_mask((64, 48)), inpainting_fill=2)

    unfilled, seeds = p.latent_noise_fill
    assert seeds == (1,)

    def expected(batch_seeds):
        return unfilled * p.mask + processing.create_random_tensors(unfilled.shape[1:], list(batch_seeds)) * p.nmask

    first = p.init_latent
    assert torch.equal(first, expected(seeds))

    seen = []
    p.rng = SimpleNamespace(next=lambda: torch.zeros_like(unfilled))
    p.scripts = None
    p.initial_noise_multiplier = 1.0
    p.sampler.sample_img2img = lambda p_, init_latent, x, *a, **k: seen.append(init_latent) or init_latent

    p.sample(None, None, list(seeds), None, 0.0, None)
    assert seen[-1] is first  # the first batch keeps the latent init filled

    p.sample(None, None, [7], None, 0.0, None)
    assert torch.equal(seen[-1], expected((7,)))
    assert torch.equal(p.init_latent, seen[-1])
    assert not torch.equal(seen[-1], first)


def test_init_mask_is_prepare_image_mask(harness):  # noqa: F811
    """img2img init prepares its mask with processing.prepare_image_mask (ControlNet's prepare_mask uses it too):
    a mask of another size is stretched onto the init image, then inverted and blurred."""
    init = _random_image("RGB", (128, 96), 25)
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[12:24, 16:32] = 255
    mask = Image.fromarray(mask, "L")
    p = harness.make([init], width=32, height=24, image_mask=mask, inpaint_full_res=True, inpaint_full_res_padding=2,
                     inpainting_mask_invert=True, mask_blur_x=3, mask_blur_y=1)

    expected = processing.prepare_image_mask(mask, (128, 96), mask_round=True, invert=True, blur_x=3, blur_y=1)
    assert np.array_equal(np.asarray(p.mask_for_overlay), np.asarray(expected))
    # independent oracle: binary, bilinear stretch, invert, horizontal then vertical Gaussian blur
    oracle = 255 - np.asarray(mask.resize((128, 96), resample=Image.Resampling.BILINEAR))
    oracle = cv2.GaussianBlur(oracle, (2 * int(2.5 * 3 + 0.5) + 1, 1), 3)
    oracle = cv2.GaussianBlur(oracle, (1, 2 * int(2.5 * 1 + 0.5) + 1), 1)
    assert np.array_equal(np.asarray(expected), oracle)
    assert p.extra_generation_params["Mask mode"] == "Inpaint not masked"
    assert "Mask blur" in p.extra_generation_params
