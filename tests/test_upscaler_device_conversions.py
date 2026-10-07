"""Differential tests: the device-side tile conversions in upscaler_utils are bitwise identical to the
former CPU numpy/float64 pipeline (kept below as the oracle)."""

from __future__ import annotations

import contextlib
import importlib.util
import math
import sys
import types
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

MODULE_PATH = Path(__file__).resolve().parents[1] / "modules" / "upscaler_utils.py"
_STUBBED = ("modules", "modules.devices", "modules.images", "modules.shared", "modules.torch_utils", "modules.upscaler_utils")
DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@pytest.fixture()
def upscaler_utils():
    previous = {name: sys.modules.get(name) for name in _STUBBED}
    modules_pkg = types.ModuleType("modules")
    modules_pkg.__path__ = []
    devices = types.ModuleType("modules.devices")
    devices.without_autocast = lambda disable=False: contextlib.nullcontext()
    shared = types.ModuleType("modules.shared")
    shared.opts = SimpleNamespace(enable_upscale_progressbar=False)
    shared.state = SimpleNamespace(interrupted=False, skipped=False)
    torch_utils = types.ModuleType("modules.torch_utils")
    torch_utils.get_param = lambda model: next(model.parameters())
    sys.modules.update({
        "modules": modules_pkg,
        "modules.devices": devices,
        "modules.images": types.ModuleType("modules.images"),
        "modules.shared": shared,
        "modules.torch_utils": torch_utils,
    })
    try:
        spec = importlib.util.spec_from_file_location("modules.upscaler_utils", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules["modules.upscaler_utils"] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


# --- oracle: the pre-change implementation -------------------------------------------------------

def oracle_pil_image_to_torch_bgr(img):
    img = np.array(img.convert("RGB"))
    img = img[:, :, ::-1]
    img = np.transpose(img, (2, 0, 1))
    img = np.ascontiguousarray(img) / 255
    return torch.from_numpy(img)


def oracle_torch_bgr_to_pil_image(tensor):
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    arr = tensor.float().cpu().clamp_(0, 1).numpy()
    arr = 255.0 * np.moveaxis(arr, 0, 2)
    arr = arr.round().astype(np.uint8)
    arr = arr[:, :, ::-1]
    return Image.fromarray(arr, "RGB")


def oracle_upscale_pil_patch(model, img):
    param = next(model.parameters())
    with torch.inference_mode():
        tensor = oracle_pil_image_to_torch_bgr(img).unsqueeze(0)
        tensor = tensor.to(device=param.device, dtype=param.dtype)
        return oracle_torch_bgr_to_pil_image(model(tensor))


# --- inputs -----------------------------------------------------------------------------------

def _all_values_image(width, height, seed):
    """Random RGB image in which every channel contains all 256 uint8 values."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    flat = pixels.reshape(-1, 3)
    for channel in range(3):
        flat[:256, channel] = rng.permutation(256).astype(np.uint8)
    return Image.fromarray(pixels, "RGB")


def _rounding_boundary_values():
    """fp32 values whose fp32 product with 255 lands exactly on k + 0.5, plus neighbours and extremes."""
    values = []
    for k in range(255):
        target = np.float32(k + 0.5)
        guess = np.float32((k + 0.5) / 255)
        for candidate in (guess, np.nextafter(guess, np.float32(0)), np.nextafter(guess, np.float32(1))):
            candidate = np.float32(candidate)
            values.append(candidate)
            if np.float32(candidate * np.float32(255.0)) == target:
                values.extend([np.nextafter(candidate, np.float32(0)), np.nextafter(candidate, np.float32(1))])
    values.extend(np.float32(v) for v in (-np.inf, -1.0, -0.0, 0.0, 1.0, 1.5, np.inf))
    return np.array(values, dtype=np.float32)


def _output_tensor(dtype, seed):
    rng = np.random.default_rng(seed)
    boundary = _rounding_boundary_values()
    random_values = rng.uniform(-0.25, 1.25, size=3 * 40 * 41 - boundary.size).astype(np.float32)
    values = np.concatenate([boundary, random_values])
    rng.shuffle(values)
    return torch.from_numpy(values.reshape(1, 3, 40, 41)).to(dtype)


# --- tests ------------------------------------------------------------------------------------

def test_boundary_values_hit_exact_half(upscaler_utils):
    boundary = _rounding_boundary_values()
    products = boundary * np.float32(255.0)
    exact_halves = np.isfinite(products) & (products - np.floor(products) == 0.5)
    assert exact_halves.sum() > 50  # the round-half-to-even cases are actually exercised


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("size", [(192, 192), (37, 53), (1, 1)])
@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P"])
def test_input_conversion_matches_float64_cpu_path(upscaler_utils, dtype, size, mode):
    if size == (1, 1):
        img = Image.new("RGB", size, (0, 128, 255))
    else:
        img = _all_values_image(*size, seed=size[0])
    img = img.convert(mode) if mode != "RGB" else img
    expected = oracle_pil_image_to_torch_bgr(img).unsqueeze(0).to(device="cpu", dtype=dtype)

    actual = upscaler_utils.pil_image_to_device_bgr(img, torch.device("cpu"), dtype)

    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.stride() == expected.stride() and actual.is_contiguous()
    assert torch.equal(actual, expected)


def _round_nearest_even(x: Fraction, significant_bits: int, min_exponent: int) -> Fraction:
    if x == 0:
        return Fraction(0)
    exponent = max(math.floor(math.log2(x)), min_exponent)
    while Fraction(2) ** exponent > x and exponent > min_exponent:
        exponent -= 1
    while Fraction(2) ** (exponent + 1) <= x:
        exponent += 1
    ulp = Fraction(2) ** (exponent - significant_bits + 1)
    whole, rest = divmod(x / ulp, 1)
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and whole % 2 == 1):
        whole += 1
    return whole * ulp


_FORMATS = {torch.float32: (24, -126), torch.float16: (11, -14), torch.bfloat16: (8, -126)}


@pytest.mark.parametrize("dtype", DTYPES)
def test_unit_lut_is_the_correctly_rounded_value_for_every_uint8(upscaler_utils, dtype):
    """Independent oracle: each entry is float64(v / 255) rounded once, to nearest-even, into dtype.

    For these 256 values that equals rounding through fp32 first, so the table does not depend on
    whether a CPU or CUDA kernel (direct or via fp32) performs the float64 -> dtype cast."""
    lut = upscaler_utils._unit_lut(torch.device("cpu"), dtype).double().tolist()
    fp32_bits, fp32_min_exponent = _FORMATS[torch.float32]
    for value in range(256):
        exact = Fraction(value / 255)
        direct = _round_nearest_even(exact, *_FORMATS[dtype])
        via_fp32 = _round_nearest_even(_round_nearest_even(exact, fp32_bits, fp32_min_exponent), *_FORMATS[dtype])
        assert Fraction(lut[value]) == direct == via_fp32, value


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_output_quantization_matches_numpy_path(upscaler_utils, dtype, seed):
    tensor = _output_tensor(dtype, seed)
    original = tensor.clone()

    expected = np.asarray(oracle_torch_bgr_to_pil_image(tensor.clone()))
    actual = np.asarray(upscaler_utils.torch_bgr_to_pil_image(tensor))

    assert actual.dtype == np.uint8 and actual.shape == expected.shape
    assert np.array_equal(actual, expected)
    assert torch.equal(tensor, original)  # the caller's tensor is not clamped in place


def test_output_quantization_on_inference_tensor_outside_inference_mode(upscaler_utils):
    with torch.inference_mode():
        tensor = _output_tensor(torch.float32, 3)
    expected = np.asarray(oracle_torch_bgr_to_pil_image(tensor.clone()))
    assert np.array_equal(np.asarray(upscaler_utils.torch_bgr_to_pil_image(tensor)), expected)


def test_output_rejects_batches(upscaler_utils):
    with pytest.raises(ValueError):
        upscaler_utils.torch_bgr_to_pil_image(torch.zeros(2, 3, 4, 4))


class _TinyUpscaler(torch.nn.Module):
    """2x nearest upscale followed by a random 3x3 conv: a cheap stand-in with real arithmetic."""

    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.conv = torch.nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x):
        return self.conv(torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest"))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_upscale_with_model_matches_oracle_end_to_end(upscaler_utils, dtype):
    model = _TinyUpscaler(seed=4).to(dtype).eval()
    img = _all_values_image(70, 45, seed=9)

    expected_patch = oracle_upscale_pil_patch(model, img)
    assert np.array_equal(np.asarray(upscaler_utils.upscale_pil_patch(model, img)), np.asarray(expected_patch))
    assert np.array_equal(
        np.asarray(upscaler_utils.upscale_with_model(model, img, tile_size=0)),
        np.asarray(expected_patch),
    )
