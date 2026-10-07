"""ESRGAN/RealESRGAN/DAT/HAT: a model that cannot be loaded fails the upscale instead of silently
falling back to a LANCZOS resize (whose result infotext would still attribute to the model), and the
shared cached descriptor is used as-is (never moved by the caller)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
_MODEL_MODULES = ("esrgan_model", "realesrgan_model", "dat_model", "hat_model")
_STUBBED = (
    "modules", "modules.shared", "modules.modelloader", "modules.devices", "modules.upscaler_utils",
    "modules.upscaler", *(f"modules.{name}" for name in _MODEL_MODULES),
)


class _LoaderStub:
    def __init__(self):
        self.calls = []
        self.error = None
        self.descriptor = object()  # no .to(): callers must not move the shared descriptor

    def __call__(self, path, **kwargs):
        self.calls.append((path, kwargs))
        if self.error is not None:
            raise self.error
        return self.descriptor


@pytest.fixture()
def upscalers(tmp_path):
    previous = {name: sys.modules.get(name) for name in _STUBBED}
    modules_pkg = types.ModuleType("modules")
    modules_pkg.__path__ = []
    shared = types.ModuleType("modules.shared")
    shared.opts = SimpleNamespace(
        ESRGAN_tile=0, ESRGAN_tile_overlap=0, DAT_tile=0, DAT_tile_overlap=0,
        realesrgan_enabled_models=[], dat_enabled_models=[],
    )
    shared.cmd_opts = SimpleNamespace(no_half=True, upcast_sampling=False)
    shared.state = SimpleNamespace(interrupted=False)
    shared.device = "cpu"
    shared.models_path = str(tmp_path)
    shared.hf_endpoint = "https://huggingface.invalid"
    loader = _LoaderStub()
    modelloader = types.ModuleType("modules.modelloader")
    modelloader.load_cached_spandrel_model = loader
    modelloader.friendly_name = lambda file: Path(file).stem
    devices = types.ModuleType("modules.devices")
    devices.device_esrgan = "cuda:0"
    upscaler_utils = types.ModuleType("modules.upscaler_utils")
    upscaled = []

    def upscale_with_model(model, img, *, tile_size, tile_overlap=0):
        upscaled.append(model)
        return img.resize((img.width * 2, img.height * 2))

    upscaler_utils.upscale_with_model = upscale_with_model
    modules_pkg.shared, modules_pkg.modelloader, modules_pkg.devices = shared, modelloader, devices
    sys.modules.update({
        "modules": modules_pkg,
        "modules.shared": shared,
        "modules.modelloader": modelloader,
        "modules.devices": devices,
        "modules.upscaler_utils": upscaler_utils,
    })
    try:
        loaded = {}
        for name in ("upscaler", *_MODEL_MODULES):
            spec = importlib.util.spec_from_file_location(f"modules.{name}", ROOT / "modules" / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"modules.{name}"] = module
            spec.loader.exec_module(module)
            loaded[name] = module
        yield SimpleNamespace(modules=loaded, loader=loader, upscaled=upscaled, tmp_path=tmp_path)
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def _instance(module, class_name, **attributes):
    upscaler = getattr(module, class_name).__new__(getattr(module, class_name))
    upscaler.scalers = []
    upscaler.device = "cpu"
    upscaler.enable = True
    upscaler.model_download_path = None
    for key, value in attributes.items():
        setattr(upscaler, key, value)
    return upscaler


def _scaler_data(env, upscaler, path):
    return env.modules["upscaler"].UpscalerData("model", path, upscaler, 4)


@pytest.mark.parametrize(("module_name", "class_name", "label"), [
    ("esrgan_model", "UpscalerESRGAN", "ESRGAN"),
    ("realesrgan_model", "UpscalerRealESRGAN", "RealESRGAN"),
    ("dat_model", "UpscalerDAT", "DAT"),
    ("hat_model", "UpscalerHAT", "HAT"),
])
def test_unloadable_model_fails_instead_of_lanczos_fallback(upscalers, module_name, class_name, label):
    module = upscalers.modules[module_name]
    model_path = upscalers.tmp_path / "broken.pth"
    model_path.write_bytes(b"not a model")
    upscaler = _instance(module, class_name, model_url="https://example.invalid/m.pth", model_name="m")
    upscaler.scalers = [_scaler_data(upscalers, upscaler, str(model_path))]
    upscalers.loader.error = ValueError("unsupported model architecture")

    with pytest.raises(RuntimeError, match=rf"Unable to load {label} model .*broken\.pth: unsupported model architecture") as raised:
        upscaler.upscale(Image.new("RGB", (16, 16)), 2, str(model_path))

    assert isinstance(raised.value.__cause__, ValueError)
    assert upscalers.upscaled == []


@pytest.mark.parametrize(("module_name", "class_name", "label"), [
    ("realesrgan_model", "UpscalerRealESRGAN", "RealESRGAN"),
    ("dat_model", "UpscalerDAT", "DAT"),
    ("hat_model", "UpscalerHAT", "HAT"),
])
def test_missing_model_fails_with_its_path(upscalers, module_name, class_name, label):
    upscaler = _instance(upscalers.modules[module_name], class_name)
    missing = str(upscalers.tmp_path / "missing.pth")

    with pytest.raises(RuntimeError, match=rf"Unable to load {label} model .*missing\.pth"):
        upscaler.upscale(Image.new("RGB", (16, 16)), 2, missing)
    assert upscalers.upscaled == []


@pytest.mark.parametrize(("module_name", "class_name", "expected_kwargs"), [
    ("esrgan_model", "UpscalerESRGAN", {"load_device": "cpu", "device": "cuda:0", "expected_architecture": "ESRGAN"}),
    ("realesrgan_model", "UpscalerRealESRGAN", {"device": "cpu", "prefer_half": False, "expected_architecture": "ESRGAN"}),
    ("dat_model", "UpscalerDAT", {"device": "cpu", "prefer_half": False, "expected_architecture": "DAT"}),
    ("hat_model", "UpscalerHAT", {"device": "cuda:0", "expected_architecture": "HAT"}),
])
def test_loaded_model_is_used_without_moving_the_shared_descriptor(upscalers, module_name, class_name, expected_kwargs):
    module = upscalers.modules[module_name]
    model_path = upscalers.tmp_path / "model.pth"
    model_path.write_bytes(b"weights")
    upscaler = _instance(module, class_name, model_url="https://example.invalid/m.pth", model_name="m")
    upscaler.scalers = [_scaler_data(upscalers, upscaler, str(model_path))]

    result = upscaler.upscale(Image.new("RGB", (16, 16)), 2, str(model_path))

    assert result.size == (32, 32)
    assert upscalers.upscaled == [upscalers.loader.descriptor]
    assert upscalers.loader.calls == [(str(model_path), expected_kwargs)]
