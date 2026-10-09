"""Upscaler / postprocessing correctness: target sizes reached exactly, no black padding in tiles, no autocast
leak from the hires-fix sampler, device-correct untiled SwinIR/ScuNET runs, defined results on interruption,
and failing (not silently LANCZOS-resizing) upscalers whose model cannot be loaded."""

from __future__ import annotations

import contextlib
import importlib.util
import math
import random
import sys
import types
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
LANCZOS = Image.Resampling.LANCZOS


@contextlib.contextmanager
def _modules(stubs: dict):
    """Install `stubs` into sys.modules for the duration of the block, then restore every touched name."""
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _shared():
    shared = types.ModuleType("modules.shared")
    shared.opts = SimpleNamespace(
        ESRGAN_tile=192, ESRGAN_tile_overlap=8, enable_upscale_progressbar=False, upscaling_max_images_in_cache=5,
        SWIN_tile=192, SWIN_tile_overlap=8, SCUNET_tile=256, SCUNET_tile_overlap=8, ldsr_steps=1, ldsr_cached=False,
        font="", n_rows=-1,
    )
    shared.cmd_opts = SimpleNamespace(no_half=False, upcast_sampling=False, unix_filenames_sanitization=False, filenames_max_length=128)
    shared.state = SimpleNamespace(interrupted=False, skipped=False)
    shared.device = "cpu"
    shared.models_path = str(ROOT / "models")
    shared.sd_upscalers = []
    return shared


def _cpu_without_autocast():
    # modules.devices.without_autocast, for the CPU autocast that a CPU-only test can actually enable.
    return torch.autocast("cpu", enabled=False) if torch.is_autocast_enabled("cpu") else contextlib.nullcontext()


@pytest.fixture()
def env(tmp_path):
    shared = _shared()
    devices = types.ModuleType("modules.devices")
    devices.without_autocast = lambda disable=False: _cpu_without_autocast()
    devices.torch_gc = lambda: None
    devices.cpu = torch.device("cpu")
    devices.dtype = torch.bfloat16
    devices.get_device_for = lambda name: torch.device("cpu")
    loader_calls = []
    modelloader = types.ModuleType("modules.modelloader")
    modelloader.friendly_name = lambda file: Path(file).stem
    modelloader.load_models = lambda **kwargs: []

    def failing_loader(path, **kwargs):
        loader_calls.append((path, kwargs))
        raise OSError("truncated checkpoint")

    modelloader.load_spandrel_model = failing_loader
    modelloader.load_cached_spandrel_model = failing_loader
    modelloader.load_file_from_url = lambda *args, **kwargs: (_ for _ in ()).throw(OSError("offline"))
    paths_internal = types.ModuleType("modules.paths_internal")
    paths_internal.roboto_ttf_file = ""
    script_callbacks = types.ModuleType("modules.script_callbacks")
    script_callbacks.on_ui_settings = lambda fn: None
    errors = types.ModuleType("modules.errors")
    errors.report = lambda *args, **kwargs: None
    stubs = {
        "modules": _package("modules"),
        "modules.shared": shared,
        "modules.devices": devices,
        "modules.modelloader": modelloader,
        "modules.paths_internal": paths_internal,
        "modules.script_callbacks": script_callbacks,
        "modules.errors": errors,
        "modules.sd_samplers": types.ModuleType("modules.sd_samplers"),
        "modules.images": None, "modules.torch_utils": None, "modules.upscaler": None, "modules.upscaler_utils": None,
    }
    with _modules(stubs):
        for name in ("modules.images", "modules.torch_utils", "modules.upscaler", "modules.upscaler_utils"):
            sys.modules.pop(name)
        pkg = sys.modules["modules"]
        for attr in ("shared", "devices", "modelloader", "errors", "script_callbacks", "sd_samplers", "paths_internal"):
            setattr(pkg, attr, sys.modules[f"modules.{attr}"])
        for attr in ("torch_utils", "images", "upscaler", "upscaler_utils"):
            setattr(pkg, attr, _load(f"modules.{attr}", ROOT / "modules" / f"{attr}.py"))
        yield SimpleNamespace(
            shared=shared, devices=devices, modelloader=modelloader, loader_calls=loader_calls,
            images=pkg.images, upscaler=pkg.upscaler, upscaler_utils=pkg.upscaler_utils, tmp_path=tmp_path,
        )


def _random_image(width, height, seed):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, size=(height, width, 3), dtype=np.uint8), "RGB")


class _ReplicatePadUpscaler(torch.nn.Module):
    """2x nearest upscale then a 3x3 replicate-padded conv: its edges depend on what surrounds the input, so black
    padding around a tile shows up in the output (real ESRGAN convs see it through their biases/activations)."""

    def __init__(self, seed=3):
        super().__init__()
        torch.manual_seed(seed)
        self.conv = torch.nn.Conv2d(3, 3, 3, padding=1, padding_mode="replicate")

    def forward(self, x):
        return self.conv(torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")).clamp(0, 1)


# --- target sizes --------------------------------------------------------------------------------------------

def test_scaled_size_reaches_every_multiple_of_8_target(env):
    scaled_size = env.upscaler.scaled_size
    misses_before = 0
    for size in range(8, 2049, 8):
        for target in range(size + 8, 4097, 8):
            scale = target / size  # what images.resize_image and the extras "Scale to" mode pass
            assert scaled_size(size, scale) == target
            misses_before += int(size * scale) < target
    assert misses_before > 0  # e.g. 616 * (640 / 616) == 639.9999999999999 truncated to 639


def test_scaled_size_keeps_truncation_of_fractional_products(env):
    scaled_size = env.upscaler.scaled_size
    rng = random.Random(7)
    for _ in range(20000):
        size = rng.randint(1, 8192)
        steps = rng.randint(1, 160)  # slider scales 0.05 .. 8.00 in 0.05 steps
        scale = float(f"{steps * 0.05:.2f}")
        assert scaled_size(size, scale) == math.floor(Fraction(size) * Fraction(f"{steps * 0.05:.2f}"))


def test_lanczos_upscaler_reaches_target_in_one_resample(env):
    img = _random_image(616, 616, seed=1)
    upscaler = env.upscaler.UpscalerLanczos()

    result = upscaler.upscale(img, 640 / 616)  # formerly 616 -> 639 -> 632 (then resize_image -> 640 again)

    assert result.size == (640, 640)
    expected = img.resize((640, 640), resample=LANCZOS)
    assert np.array_equal(np.asarray(result), np.asarray(expected))


# --- upscale_with_model tiling ----------------------------------------------------------------------------------

@pytest.mark.parametrize("size", [(70, 45), (192, 100), (37, 191)])
def test_tile_larger_than_image_matches_untiled(env, size):
    utils = env.upscaler_utils
    model = _ReplicatePadUpscaler().eval()
    img = _random_image(*size, seed=size[0])

    untiled = utils.upscale_with_model(model, img, tile_size=0)
    tiled = utils.upscale_with_model(model, img, tile_size=192, tile_overlap=8)

    assert tiled.size == untiled.size == (size[0] * 2, size[1] * 2)
    assert np.array_equal(np.asarray(tiled), np.asarray(untiled))


def test_tile_taller_than_image_has_no_black_padding_in_rows(env):
    utils = env.upscaler_utils
    model = _ReplicatePadUpscaler().eval()
    img = _random_image(300, 45, seed=11)

    untiled = np.asarray(utils.upscale_with_model(model, img, tile_size=0))
    tiled = np.asarray(utils.upscale_with_model(model, img, tile_size=192, tile_overlap=8))

    # Columns produced by the first tile away from its right edge see the same surroundings as the untiled run.
    assert np.array_equal(tiled[:, :200], untiled[:, :200])


# --- upscale_2 (SwinIR / ScuNET) --------------------------------------------------------------------------------

def _oracle_upscale_2(img, model, *, tile_size, tile_overlap, scale):
    """The former implementation: CPU float64 -> dtype tensor, per-tile copy, ones-tensor weights, no autocast."""
    param = next(model.parameters())
    arr = np.ascontiguousarray(np.transpose(np.array(img.convert("RGB"))[:, :, ::-1], (2, 0, 1))) / 255
    tensor = torch.from_numpy(arr).to(dtype=param.dtype).unsqueeze(0)
    with torch.inference_mode():
        b, c, h, w = tensor.size()
        tile_size = min(tile_size, h, w)
        tile_overlap = max(0, min(tile_overlap, tile_size - 1))
        stride = tile_size - tile_overlap
        h_idx_list = list(range(0, h - tile_size, stride)) + [h - tile_size]
        w_idx_list = list(range(0, w - tile_size, stride)) + [w - tile_size]
        result = torch.zeros(b, c, h * scale, w * scale, dtype=tensor.dtype)
        weights = torch.zeros_like(result)
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                out_patch = model(tensor[..., h_idx:h_idx + tile_size, w_idx:w_idx + tile_size])
                region = (..., slice(h_idx * scale, (h_idx + tile_size) * scale), slice(w_idx * scale, (w_idx + tile_size) * scale))
                result[region].add_(out_patch)
                weights[region].add_(torch.ones_like(out_patch))
        output = result.div_(weights)
    arr = output.squeeze(0).float().clamp(0, 1).mul(255.0).round().to(torch.uint8).flip(0).permute(1, 2, 0).numpy()
    return Image.fromarray(np.ascontiguousarray(arr), "RGB")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_upscale_2_matches_former_cpu_path(env, dtype):
    model = _ReplicatePadUpscaler().to(dtype).eval()
    img = _random_image(53, 41, seed=5)

    result = env.upscaler_utils.upscale_2(img, model, tile_size=16, tile_overlap=5, scale=2, desc="t")

    expected = _oracle_upscale_2(img, model, tile_size=16, tile_overlap=5, scale=2)
    assert np.array_equal(np.asarray(result), np.asarray(expected))


def test_upscale_2_runs_model_outside_sampler_autocast(env):
    model = _ReplicatePadUpscaler().eval()
    img = _random_image(40, 33, seed=8)
    seen = []
    forward = model.forward
    model.forward = lambda x: (seen.append(torch.is_autocast_enabled("cpu")), forward(x))[1]

    plain = env.upscaler_utils.upscale_2(img, model, tile_size=16, tile_overlap=4, scale=2, desc="t")
    with torch.autocast("cpu", dtype=torch.bfloat16):  # what hires fix's p.sample() wraps resize_image in
        under_autocast = env.upscaler_utils.upscale_2(img, model, tile_size=16, tile_overlap=4, scale=2, desc="t")

    assert seen and not any(seen)
    assert np.array_equal(np.asarray(under_autocast), np.asarray(plain))


def test_untiled_upscale_2_runs_on_the_model_device(env):
    received = []

    def model(x):
        received.append(x.device)
        return x

    img = torch.zeros(1, 3, 8, 8)
    out = env.upscaler_utils.tiled_upscale_2(img, model, tile_size=0, tile_overlap=0, scale=1, device=torch.device("meta"))

    assert received == [torch.device("meta")] and out.device.type == "meta"


@pytest.mark.parametrize("interrupt_after", [0, 1])
def test_interrupted_upscale_2_returns_input_unchanged(env, interrupt_after):
    calls = []
    inner = _ReplicatePadUpscaler().eval()

    def model(x):
        calls.append(1)
        if len(calls) >= interrupt_after:
            env.shared.state.interrupted = True
        return inner(x)

    model.parameters = inner.parameters
    env.shared.state.interrupted = interrupt_after == 0
    img = _random_image(40, 40, seed=2)

    result = env.upscaler_utils.upscale_2(img, model, tile_size=16, tile_overlap=4, scale=2, desc="t")

    assert result is img
    assert len(calls) == interrupt_after


# --- model load failures --------------------------------------------------------------------------------------

def _load_private(name, path):
    module = _load(name, path)
    sys.modules.pop(name, None)
    return module


def _load_script(env, relative):
    path = ROOT / relative
    return _load_private(f"_c2_{path.stem}", path)


@pytest.mark.parametrize("relative, cls", [
    ("extensions-builtin/SwinIR/scripts/swinir_model.py", "UpscalerSwinIR"),
    ("extensions-builtin/ScuNET/scripts/scunet_model.py", "UpscalerScuNET"),
])
def test_unloadable_swinir_scunet_fail_instead_of_lanczos(env, relative, cls):
    module = _load_script(env, relative)
    upscaler = getattr(module, cls)(str(env.tmp_path))
    model_path = env.tmp_path / "broken.pth"
    model_path.write_bytes(b"not a model")

    with pytest.raises(RuntimeError, match="Unable to load"):
        upscaler.upscale(_random_image(16, 16, seed=0), 2, str(model_path))
    assert env.loader_calls


def test_scunet_uses_the_shared_model_cache(env):
    module = _load_script(env, "extensions-builtin/ScuNET/scripts/scunet_model.py")
    descriptor = object()
    env.modelloader.load_cached_spandrel_model = lambda path, **kwargs: (env.loader_calls.append((path, kwargs)), descriptor)[1]
    model_path = env.tmp_path / "scunet.pth"
    model_path.write_bytes(b"x")

    assert module.UpscalerScuNET(str(env.tmp_path)).load_model(str(model_path)) is descriptor
    assert env.loader_calls[-1][1]["expected_architecture"] == "SCUNet"


def test_scunet_url_model_reaches_the_loader_as_a_pth_file(env):
    # modules.util.load_file_from_url contract: saved as `file_name` if given, else as the URL's basename.
    downloads = []

    def load_file_from_url(url, *, model_dir, file_name=None, **kwargs):
        downloads.append((url, model_dir, file_name))
        return str(Path(model_dir) / (file_name or Path(url).name))

    env.modelloader.load_file_from_url = load_file_from_url
    env.modelloader.load_cached_spandrel_model = lambda path, **kwargs: (env.loader_calls.append((path, kwargs)), object())[1]
    module = _load_script(env, "extensions-builtin/ScuNET/scripts/scunet_model.py")
    upscaler = module.UpscalerScuNET(str(env.tmp_path))
    upscaler.model_download_path = str(env.tmp_path)

    upscaler.load_model(upscaler.model_url)

    assert downloads == [(upscaler.model_url, str(env.tmp_path), None)]
    # spandrel dispatches on the extension: a name without .pth raises "Unsupported model file extension".
    assert env.loader_calls[-1][0] == str(env.tmp_path / "scunet_color_real_gan.pth")


def test_unloadable_ldsr_fails_instead_of_lanczos(env):
    ldsr_arch = types.ModuleType("ldsr_model_arch")
    ldsr_arch.LDSR = lambda model, yaml: pytest.fail("LDSR must not be constructed without a model")
    stubs = {"ldsr_model_arch": ldsr_arch, "sd_hijack_autoencoder": types.ModuleType("x"), "sd_hijack_ddpm_v1": types.ModuleType("y")}
    with _modules(stubs):
        module = _load_script(env, "extensions-builtin/LDSR/scripts/ldsr_model.py")
        upscaler = module.UpscalerLDSR(str(env.tmp_path))
        upscaler.model_path = str(env.tmp_path)

        with pytest.raises(RuntimeError, match="Unable to load LDSR model"):
            upscaler.upscale(_random_image(16, 16, seed=0), 2, None)


# --- LDSR decodes once ------------------------------------------------------------------------------------------

def test_ldsr_decodes_the_sample_once(env, monkeypatch):
    ddim = types.ModuleType("ldm.models.diffusion.ddim")
    ddim.DDIMSampler = object
    util = types.ModuleType("ldm.util")
    util.instantiate_from_config = util.ismap = None
    stubs = {
        "ldm": _package("ldm"), "ldm.models": _package("ldm.models"), "ldm.models.diffusion": _package("ldm.models.diffusion"),
        "ldm.models.diffusion.ddim": ddim, "ldm.util": util, "modules.sd_hijack": types.ModuleType("modules.sd_hijack"),
    }
    with _modules(stubs):
        sys.modules["modules"].sd_hijack = stubs["modules.sd_hijack"]
        arch = _load_private("_c2_ldsr_model_arch", ROOT / "extensions-builtin/LDSR/ldsr_model_arch.py")

    decodes = []
    sample = torch.full((1, 3, 4, 4), 0.25)

    class Model:
        first_stage_key = "image"
        cond_stage_key = "LR_image"

        def get_input(self, batch, key, return_first_stage_outputs=False, force_c_encode=False, return_original_cond=False):
            assert not return_first_stage_outputs  # that would decode the 4x input just to log it
            return [torch.zeros(1, 3, 4, 4), "cond"]

        def decode_first_stage(self, z, **kwargs):
            decodes.append(kwargs)
            return z * 2

        def ema_scope(self, _context):
            return contextlib.nullcontext()

    monkeypatch.setattr(arch, "convsample_ddim", lambda model, cond, **kwargs: (sample, {}))
    log = arch.make_convolutional_sample({"image": None}, Model(), custom_steps=1)

    assert decodes == [{}]
    assert torch.equal(log["sample"], sample * 2)


# --- extras "Upscale" script -------------------------------------------------------------------------------------

@pytest.fixture()
def pp_upscale(env):
    scripts_postprocessing = types.ModuleType("modules.scripts_postprocessing")
    scripts_postprocessing.ScriptPostprocessing = type("ScriptPostprocessing", (), {})
    scripts_postprocessing.PostprocessedImage = object  # annotations only
    ui_components = types.ModuleType("modules.ui_components")
    ui_components.FormRow = ui_components.InputAccordion = None
    stubs = {
        "modules.scripts_postprocessing": scripts_postprocessing, "modules.ui_components": ui_components,
        "modules.headless_ui": types.ModuleType("modules.headless_ui"),
    }
    with _modules(stubs):
        pkg = sys.modules["modules"]
        pkg.scripts_postprocessing, pkg.headless_ui = scripts_postprocessing, stubs["modules.headless_ui"]
        module = _load_private("_c2_postprocessing_upscale", ROOT / "scripts/postprocessing_upscale.py")
    module.upscale_cache.clear()
    return module


@pytest.mark.parametrize("source, target", [((300, 200), (1001, 999)), ((640, 360), (1000, 1004)), ((616, 616), (640, 640))])
def test_crop_to_fit_fills_the_whole_target(env, pp_upscale, source, target):
    upscaler = SimpleNamespace(name="Lanczos", data_path=None, scaler=env.upscaler.UpscalerLanczos())
    # A solid image: any black pixel in the result can only be uncovered canvas.
    img = Image.new("RGB", source, (200, 150, 100))
    info = {}

    result = pp_upscale.ScriptPostprocessingUpscale().upscale(img, info, upscaler, 1, 2.0, 0, *target, True)

    assert result.size == target
    assert np.asarray(result).min() > 50
    assert info["Postprocess crop to"] == f"{target[0]}x{target[1]}"


def test_scale_by_target_is_not_a_pixel_short(env, pp_upscale):
    # 1600 * 1.15 == 1839.9999999999998: int() predicted 1839 px, the size the upscaler is asked for is 1840.
    pp = SimpleNamespace(image=Image.new("RGB", (1600, 800)), shared=SimpleNamespace(target_width=None, target_height=None))

    pp_upscale.ScriptPostprocessingUpscale().process_firstpass(pp, upscale_mode=0, upscale_by=1.15, max_side_length=0)

    assert (pp.shared.target_width, pp.shared.target_height) == (1840, 920)


def test_scale_by_without_side_limit_keeps_target_size(env, pp_upscale):
    pp = SimpleNamespace(image=Image.new("RGB", (300, 200)), shared=SimpleNamespace(target_width=None, target_height=None))
    script = pp_upscale.ScriptPostprocessingUpscale()

    script.process_firstpass(pp, upscale_mode=0, upscale_by=2.5, max_side_length=0)
    assert (pp.shared.target_width, pp.shared.target_height) == (750, 500)

    script.process_firstpass(pp, upscale_mode=0, upscale_by=2.5, max_side_length=600)
    assert (pp.shared.target_width, pp.shared.target_height) == (600, 400)
