"""ESRGAN/RealESRGAN/DAT/HAT: a model that cannot be loaded fails the upscale instead of silently
falling back to a LANCZOS resize (whose result infotext would still attribute to the model), and the
shared cached descriptor is used as-is (never moved by the caller)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from test.helpers import load_source, module

_MODEL_MODULES = ("esrgan_model", "realesrgan_model", "dat_model", "hat_model")


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
    shared = module(
        "modules.shared",
        opts=SimpleNamespace(
            ESRGAN_tile=0, ESRGAN_tile_overlap=0, DAT_tile=0, DAT_tile_overlap=0,
            realesrgan_enabled_models=[], dat_enabled_models=[],
        ),
        cmd_opts=SimpleNamespace(no_half=True, upcast_sampling=False),
        state=SimpleNamespace(interrupted=False),
        device="cpu",
        models_path=str(tmp_path),
        hf_endpoint="https://huggingface.invalid",
    )
    loader = _LoaderStub()
    modelloader = module("modules.modelloader", load_cached_spandrel_model=loader, friendly_name=lambda file: Path(file).stem)
    devices = module("modules.devices", device_esrgan="cuda:0")
    upscaled = []

    def upscale_with_model(model, img, *, tile_size, tile_overlap=0):
        upscaled.append(model)
        return img.resize((img.width * 2, img.height * 2))

    stubs = {
        "modules": module("modules", package=True),
        "modules.shared": shared,
        "modules.modelloader": modelloader,
        "modules.devices": devices,
        "modules.upscaler_utils": module("modules.upscaler_utils", upscale_with_model=upscale_with_model),
    }
    loaded = {"upscaler": load_source("modules.upscaler", "modules/upscaler.py", stubs)}
    stubs["modules.upscaler"] = loaded["upscaler"]  # the model modules subclass the real Upscaler
    for name in _MODEL_MODULES:
        loaded[name] = load_source(f"modules.{name}", f"modules/{name}.py", stubs)
    return SimpleNamespace(modules=loaded, loader=loader, upscaled=upscaled, tmp_path=tmp_path)


def _instance(module, class_name, **attributes):
    upscaler = getattr(module, class_name).__new__(getattr(module, class_name))
    upscaler.scalers = []
    upscaler.device = "cpu"
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


@pytest.mark.parametrize(("module_name", "class_name", "sha256", "expected_downloads"), [
    # DAT checks its pinned sha256 and replaces a cached Git LFS pointer (< 200 bytes) with the weights.
    ("dat_model", "UpscalerDAT", "7760aa96", [("7760aa96", False), ("7760aa96", True)]),
    ("realesrgan_model", "UpscalerRealESRGAN", None, [(None, False)]),
])
def test_listed_url_model_is_downloaded_once(upscalers, module_name, class_name, sha256, expected_downloads):
    upscaler = _instance(upscalers.modules[module_name], class_name, model_download_path=str(upscalers.tmp_path))
    url = "https://example.invalid/model_x2.pth"
    upscaler.scalers = [upscalers.modules["upscaler"].UpscalerData("model x2", url, upscaler, 2, sha256=sha256)]
    downloaded = upscalers.tmp_path / "model_x2.pth"
    downloads = []

    def load_file_from_url(url_, *, model_dir, hash_prefix=None, re_download=False):
        assert (url_, model_dir) == (url, str(upscalers.tmp_path))
        downloads.append((hash_prefix, re_download))
        downloaded.write_bytes(b"w" * (300 if re_download else 100))
        return str(downloaded)

    upscalers.modules["upscaler"].modelloader.load_file_from_url = load_file_from_url

    upscaler.upscale(Image.new("RGB", (16, 16)), 2, url)
    upscaler.upscale(Image.new("RGB", (16, 16)), 2, url)

    assert downloads == expected_downloads
    assert [path for path, _ in upscalers.loader.calls] == [str(downloaded)] * 2


@pytest.mark.parametrize(("module_name", "class_name", "found", "expected"), [
    # With no file in the model dir, load_models lists model_url, which is named model_name.
    ("esrgan_model", "UpscalerESRGAN", ["https://example.invalid/ESRGAN.pth"], [("ESRGAN_4x", "https://example.invalid/ESRGAN.pth", 4)]),
    ("esrgan_model", "UpscalerESRGAN", ["/m/4x_foo.pth"], [("4x_foo", "/m/4x_foo.pth", 4)]),
    ("dat_model", "UpscalerDAT", ["/m/DAT_custom.pth"], [("DAT_custom", "/m/DAT_custom.pth", None)]),
    ("hat_model", "UpscalerHAT", ["/m/HAT_x4.pth"], [("HAT_x4", "/m/HAT_x4.pth", 4)]),
])
def test_model_files_are_listed_as_scalers(upscalers, module_name, class_name, found, expected):
    calls = []

    def load_models(**kwargs):
        calls.append(kwargs)
        return found

    upscalers.modules["upscaler"].modelloader.load_models = load_models
    upscaler = getattr(upscalers.modules[module_name], class_name)("/user/models")

    assert [(s.name, s.data_path, s.scale) for s in upscaler.scalers] == expected
    assert all(s.scaler is upscaler for s in upscaler.scalers)
    assert calls == [{
        "model_path": str(upscalers.tmp_path / upscaler.name),
        "model_url": upscaler.model_url,
        "command_path": "/user/models",
        "ext_filter": [".pt", ".pth"],
    }]
