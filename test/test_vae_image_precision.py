"""Decoded-image quantization: round-to-nearest uint8 codes, computed in float32 for low-precision decoder outputs.

Oracles are independent of the code under test: float64 reference values and the IEC 61966-2-1 8-bit encoding
round(255 * E').
"""

import sys

import numpy as np
import pytest
import torch

from test.helpers import init_shared

shared = init_shared()

from modules import processing, sd_samplers_common  # noqa: E402


@pytest.fixture(autouse=True)
def _real_webui_modules_package(monkeypatch):
    # Other test files leave stub "modules" packages in sys.modules.
    monkeypatch.setitem(sys.modules, "modules", processing.modules)


def _probe_values():
    codes = np.arange(256, dtype=np.float64)
    exact = codes / 255.0
    near_half = np.clip((codes + 0.5) / 255.0 + np.array([[-1e-6], [1e-6]]), 0.0, 1.0).ravel()
    random = np.random.default_rng(0).random(4096)
    return np.concatenate([exact, near_half, random, [0.0, 1.0]]).astype(np.float32)


def test_float_images_to_uint8_rounds_to_nearest_code():
    values = _probe_values()
    batch = torch.from_numpy(values[: 3 * 8 * 199].reshape(1, 3, 8, 199))

    quantized = sd_samplers_common.float_images_to_uint8(batch)

    assert quantized.dtype == torch.uint8 and quantized.shape == (1, 8, 199, 3) and quantized.is_contiguous()
    reference = batch.double().permute(0, 2, 3, 1).numpy() * 255.0
    error = quantized.numpy().astype(np.float64) - reference
    # round-to-nearest: never more than half a code away (float32 product rounding adds < 1e-4 codes)
    assert np.abs(error).max() <= 0.5 + 1e-4
    # unbiased (truncation is biased by -0.5 code)
    assert abs(error.mean()) < 0.02
    single = sd_samplers_common.float_images_to_uint8(batch[0])
    assert single.shape == (8, 199, 3) and torch.equal(single, quantized[0])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_float_images_to_uint8_quantizes_low_precision_inputs_in_float32(dtype):
    values = torch.linspace(0, 1, 3 * 16 * 16).reshape(3, 16, 16).to(dtype)

    quantized = sd_samplers_common.float_images_to_uint8(values)

    expected = np.rint(values.double().numpy() * 255.0).astype(np.uint8).transpose(1, 2, 0)
    assert np.array_equal(quantized.numpy(), expected)


@pytest.mark.parametrize("signed", [False, True])
def test_samples_to_uint8_images_round_trips_uint8_images(signed):
    """A uint8 image converted to the float layouts the pipeline uses must come back unchanged.

    Hires fix from a supplied first-pass image takes the signed path; truncation returned u - 1 for 63 of 256 codes.
    """
    from PIL import Image

    codes = np.arange(256, dtype=np.uint8)
    image = Image.fromarray(np.stack([codes, codes[::-1], codes], axis=-1).reshape(16, 16, 3))
    array = torch.from_numpy(processing._image_to_chw_float32_array(image, scale_to_signed=signed))[None]
    unsigned = torch.clamp((array + 1.0) / 2.0, min=0.0, max=1.0) if signed else array

    batch = processing.samples_to_uint8_images(unsigned)
    per_sample = processing.samples_to_uint8_images(list(unsigned))

    assert np.array_equal(batch[0], np.asarray(image))
    assert np.array_equal(per_sample[0], np.asarray(image))


def test_single_sample_to_image_accepts_bf16_decoder_output(monkeypatch):
    """Live previews and the pre-hires save decode with the bf16 VAE; bf16 tensors have no NumPy dtype."""
    decoded = torch.linspace(-1.1, 1.1, 3 * 8 * 8).reshape(1, 3, 8, 8).to(torch.bfloat16)
    monkeypatch.setattr(sd_samplers_common, "samples_to_images_tensor", lambda sample, approximation=None: decoded)

    image = sd_samplers_common.single_sample_to_image(torch.zeros(4, 1, 1))

    expected = np.rint(np.clip(decoded[0].double().numpy() * 0.5 + 0.5, 0.0, 1.0) * 255.0).astype(np.uint8)
    assert image.mode == "RGB"
    assert np.array_equal(np.asarray(image), expected.transpose(1, 2, 0))
