"""Hires fix: no discarded full-size VAE decode (PL5) and device-resident decoded images (PL7)."""

import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test.helpers import init_shared

shared = init_shared()

from modules import devices, processing  # noqa: E402
from modules.processing import StableDiffusionProcessingTxt2Img  # noqa: E402


MODEL_KINDS = {
    "sdxl": dict(conditioning_key="crossattn"),
    "inpaint": dict(conditioning_key="hybrid"),
    "concat": dict(conditioning_key="concat"),
    "unclip": dict(conditioning_key="crossattn-adm"),
    "edit": dict(conditioning_key="crossattn", cond_stage_key="edit"),
    "sdxl_inpaint": dict(conditioning_key="crossattn", is_sdxl_inpaint=True),
}


def _old_img2img_image_conditioning(self, source_image, latent_image, image_mask=None, round_image_mask=True):
    """Pre-change implementation (oracle)."""
    source_image = devices.cond_cast_float(source_image)
    if isinstance(self.sd_model, processing.LatentDepth2ImageDiffusion):
        return self.depth2img_image_conditioning(source_image)
    if self.sd_model.cond_stage_key == "edit":
        return self.edit_image_conditioning(source_image)
    if self.sampler.conditioning_key in {'hybrid', 'concat'}:
        return self.inpainting_image_conditioning(source_image, latent_image, image_mask=image_mask, round_image_mask=round_image_mask)
    if self.sampler.conditioning_key == "crossattn-adm":
        return self.unclip_image_conditioning(source_image)
    if self.sampler.model_wrap.inner_model.is_sdxl_inpaint:
        return self.inpainting_image_conditioning(source_image, latent_image, image_mask=image_mask)
    return latent_image.new_zeros(latent_image.shape[0], 5, 1, 1)


def _make_p(monkeypatch, conditioning_key, cond_stage_key="crossattn", is_sdxl_inpaint=False):
    # StableDiffusionProcessing.sd_model reads shared.sd_model.
    model = SimpleNamespace(cond_stage_key=cond_stage_key, is_sdxl_inpaint=is_sdxl_inpaint, model=SimpleNamespace(conditioning_key=conditioning_key))
    sampler = SimpleNamespace(conditioning_key=conditioning_key, model_wrap=SimpleNamespace(inner_model=model))
    p = StableDiffusionProcessingTxt2Img.__new__(StableDiffusionProcessingTxt2Img)
    calls = []
    for name in ("depth2img_image_conditioning", "edit_image_conditioning", "unclip_image_conditioning"):
        setattr(p, name, lambda source, _name=name: calls.append((_name, source)) or (_name, source))
    p.inpainting_image_conditioning = lambda source, latent, image_mask=None, round_image_mask=True: calls.append(("inpainting", source, latent, image_mask, round_image_mask)) or ("inpainting", source)
    p.sampler = sampler
    monkeypatch.setattr(processing.shared, "sd_model", model, raising=False)
    return p, calls


@pytest.mark.parametrize("kind", sorted(MODEL_KINDS))
def test_img2img_image_conditioning_matches_old_dispatch(monkeypatch, kind):
    source = torch.randn(2, 3, 16, 16)
    latent = torch.randn(2, 4, 2, 2, dtype=torch.bfloat16)

    old_p, old_calls = _make_p(monkeypatch, **MODEL_KINDS[kind])
    old = _old_img2img_image_conditioning(old_p, source, latent)
    p, new_calls = _make_p(monkeypatch, **MODEL_KINDS[kind])
    new = p.img2img_image_conditioning(source, latent)

    assert len(new_calls) == len(old_calls)
    for new_call, old_call in zip(new_calls, old_calls):
        assert new_call[0] == old_call[0] and torch.equal(new_call[1], old_call[1])
    if isinstance(old, torch.Tensor):
        assert p.img2img_image_conditioning_reads_source() is False
        assert torch.equal(new, old) and new.dtype == old.dtype == torch.bfloat16
        # The image is never read for these models, so it need not be produced.
        assert torch.equal(p.img2img_image_conditioning(None, latent), old)
    else:
        assert p.img2img_image_conditioning_reads_source() is True


class _StopAfterConditioning(Exception):
    pass


@pytest.mark.parametrize("kind,decodes", [("sdxl", 0), ("inpaint", 1), ("sdxl_inpaint", 1)])
def test_latent_hires_decodes_only_for_models_that_read_the_image(monkeypatch, kind, decodes):
    p, calls = _make_p(monkeypatch, **MODEL_KINDS[kind])
    monkeypatch.setattr(processing.sd_samplers, "create_sampler", lambda name, model: p.sampler)
    monkeypatch.setattr(processing.shared, "state", SimpleNamespace(interrupted=False, nextjob=lambda: (_ for _ in ()).throw(_StopAfterConditioning())), raising=False)
    decoded = []
    monkeypatch.setattr(processing, "decode_first_stage", lambda model, x: decoded.append(x) or torch.full((x.shape[0], 3, x.shape[2] * 8, x.shape[3] * 8), 0.25))
    results = []
    conditioning = p.img2img_image_conditioning
    p.img2img_image_conditioning = lambda *args, **kwargs: results.append(conditioning(*args, **kwargs)) or results[-1]
    p.__dict__.update(
        hr_upscale_to_x=32, hr_upscale_to_y=32, hr_sampler_name=None, sampler_name="Euler", inpainting_mask_weight=0.5,
        latent_scale_mode={"mode": "nearest", "antialias": False}, do_not_save_samples=True,
    )
    samples = torch.randn(1, 4, 2, 2)

    with pytest.raises(_StopAfterConditioning):
        p.sample_hr_pass(samples, None, [1], [1], 0.0, [""])

    assert len(decoded) == decodes
    if decodes:
        assert calls[0][0] == "inpainting" and calls[0][1] is not None
    else:
        assert calls == []
        assert torch.equal(results[0], torch.zeros(1, 5, 1, 1))


def test_decoded_images_stay_on_device_unless_lowvram(monkeypatch):
    device = torch.device("meta")
    monkeypatch.setattr(processing.shared, "device", device, raising=False)
    monkeypatch.setattr(processing.shared, "sd_model", SimpleNamespace(lowvram=False), raising=False)
    assert processing.decoded_images_device() == device
    monkeypatch.setattr(processing.shared, "sd_model", SimpleNamespace(lowvram=True), raising=False)
    assert processing.decoded_images_device() == devices.cpu


def test_every_generation_decode_uses_decoded_images_device():
    tree = ast.parse((Path(processing.__file__)).read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "decode_latent_batch"]
    assert len(calls) == 3
    assert all(len(call.args) == 2 and not call.keywords for call in calls)
    decode = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "decode_latent_batch")
    assert [ast.unparse(arg) for arg in decode.args.args] == ["model", "batch"]
    assert any(isinstance(node, ast.Call) and getattr(node.func, "id", None) == "decoded_images_device" for node in ast.walk(decode))


@pytest.mark.parametrize("kind", ["sdxl", "sdxl_inpaint", "inpaint"])
def test_upscaler_hires_conditions_on_signed_source_image(monkeypatch, kind):
    """The upscaler path encodes [0, 1] images but must hand [-1, 1] sources to the image conditioning, like the
    latent path (decode_first_stage output) and img2img (image * 2 - 1)."""
    p, calls = _make_p(monkeypatch, **MODEL_KINDS[kind])
    monkeypatch.setattr(processing.sd_samplers, "create_sampler", lambda name, model: p.sampler)
    monkeypatch.setattr(processing.shared, "state", SimpleNamespace(interrupted=False, nextjob=lambda: (_ for _ in ()).throw(_StopAfterConditioning())), raising=False)
    monkeypatch.setattr(processing.shared, "device", torch.device("cpu"), raising=False)
    monkeypatch.setattr(processing.images, "resize_image", lambda mode, image, width, height, upscaler_name=None: image.resize((width, height)))
    encoded = []
    monkeypatch.setattr(processing, "images_tensor_to_samples", lambda image, approximation=None, model=None: encoded.append(image) or torch.zeros(image.shape[0], 4, image.shape[2] // 8, image.shape[3] // 8))
    p.__dict__.update(
        hr_upscale_to_x=16, hr_upscale_to_y=16, hr_sampler_name=None, sampler_name="Euler", inpainting_mask_weight=0.5,
        latent_scale_mode=None, hr_upscaler="Lanczos", do_not_save_samples=True,
    )
    decoded = torch.full((1, 3, 16, 16), -0.5)

    with pytest.raises(_StopAfterConditioning):
        p.sample_hr_pass(None, decoded, [1], [1], 0.0, [""])

    unit = 64 / 255  # round(255 * (-0.5 + 1) / 2)
    assert encoded[0].dtype == torch.float32  # images_tensor_to_samples casts to the VAE dtype after its affine
    assert torch.allclose(encoded[0].double(), torch.full(encoded[0].shape, unit, dtype=torch.float64), atol=1e-3)
    if kind == "sdxl":
        assert calls == []
    else:
        source = calls[0][1]
        assert source.dtype == devices.dtype_vae
        assert torch.allclose(source.double(), torch.full(source.shape, unit * 2 - 1, dtype=torch.float64), atol=1e-3)


class _StopAtSampling(Exception):
    pass


@pytest.mark.parametrize("kind", ["sdxl", "sdxl_inpaint", "inpaint"])
@pytest.mark.parametrize("truncate", [(0, 0), (2, 1)])
def test_latent_hires_conditioning_matches_the_sampled_latent(monkeypatch, kind, truncate):
    """Inpainting-model conditioning is built at the upscaled size (not the first-pass width/height) and cropped with
    the samples when hr_resize_x/y truncate the latent; otherwise the UNet concat fails on mismatched sizes."""
    p, calls = _make_p(monkeypatch, **MODEL_KINDS[kind])
    sampled = []

    def sample_img2img(p_, x, noise, c, uc, steps=None, image_conditioning=None):
        sampled.append((x, image_conditioning))
        raise _StopAtSampling()

    p.sampler.sample_img2img = sample_img2img
    monkeypatch.setattr(processing.sd_samplers, "create_sampler", lambda name, model: p.sampler)
    monkeypatch.setattr(processing.shared, "state", SimpleNamespace(interrupted=False, nextjob=lambda: None), raising=False)
    monkeypatch.setattr(processing, "images_tensor_to_samples", lambda image, approximation=None, model=None: torch.zeros(image.shape[0], 4, image.shape[2] // 8, image.shape[3] // 8))
    monkeypatch.setattr(processing.rng, "ImageRNG", lambda shape, *args, **kwargs: SimpleNamespace(next=lambda: torch.zeros(1, *shape)))
    monkeypatch.setattr(processing.devices, "autocast", lambda disable=False: contextlib.nullcontext())
    monkeypatch.setattr(processing.sd_models, "apply_token_merging", lambda model, ratio: None)
    p.__dict__.update(
        width=16, height=16, hr_upscale_to_x=48, hr_upscale_to_y=32, truncate_x=truncate[0], truncate_y=truncate[1],
        hr_sampler_name=None, sampler_name="Euler", inpainting_mask_weight=1.0, latent_scale_mode={"mode": "nearest", "antialias": False},
        do_not_save_samples=True, seeds=[1], subseeds=[1], subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0,
        disable_extra_networks=True, scripts=None, hr_c=None, hr_uc=None, hr_second_pass_steps=0, steps=1,
        calculate_hr_conds=lambda: None, get_token_merging_ratio=lambda for_hr=False: 0.0,
    )

    with pytest.raises(_StopAtSampling):
        p.sample_hr_pass(torch.randn(1, 4, 2, 2), None, [1], [1], 0.0, [""])

    x, image_conditioning = sampled[0]
    assert x.shape == (1, 4, 4 - truncate[1], 6 - truncate[0])
    if kind == "sdxl":
        assert image_conditioning.shape == (1, 5, 1, 1)
    else:
        assert image_conditioning.shape == (1, 5, *x.shape[-2:])
