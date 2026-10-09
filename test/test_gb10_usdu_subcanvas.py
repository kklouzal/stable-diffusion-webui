"""Ultimate SD Upscale sub-canvas patcher: fail-closed patching and a bitwise differential against the full canvas.

The differential runs the unpatched and the patched USDU passes against the real
StableDiffusionProcessingImg2Img.init (mask blur, crop region, crop/resize, overlay, latent mask, inpainting
conditioning) and the real apply_overlay; only the UNet/VAE are replaced by a deterministic per-pixel function of
the init latent, nmask and image conditioning.
"""
from __future__ import annotations

import importlib
import importlib.util
import random
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
GB10 = ROOT / "gb10"
PATCHER = GB10 / "patch-ultimate-upscale-subcanvas.py"
EXTENSION = "ultimate-upscale-for-automatic1111"
# The host deploy root, or the same checkout where run.sh mounts it inside a webui container.
INSTALLED_UU = next((path for path in (Path("/opt/gb10/stable-diffusion/Extensions") / EXTENSION, ROOT / "extensions" / EXTENSION) if path.is_dir()), None)
MODULE_NAMES = ("modules", "modules.shared", "modules.processing", "modules.images", "modules.devices", "modules.scripts", "modules.masking")
# Other tests replace these sys.modules entries with stubs and never restore them; keep the real objects we see.
_REAL_MODULES = {name: sys.modules[name] for name in MODULE_NAMES if getattr(sys.modules.get(name), "__file__", None)}


def run_patcher(target: Path, *extra: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(PATCHER), str(target), *extra], check=check, capture_output=True, text=True)


def _load_patcher():
    if str(GB10) not in sys.path:
        sys.path.insert(0, str(GB10))  # the patcher imports patchlib as a sibling module, as under run.sh
    spec = importlib.util.spec_from_file_location("gb10_patch_ultimate_upscale_subcanvas", PATCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PATCHER_MODULE = _load_patcher()


@pytest.fixture()
def usdu_source(tmp_path: Path) -> Path:
    """The installed script as run.sh hands it to this patcher: lifecycle-patched, sub-canvas not.

    An already deployed script is un-patched here, and the patcher must turn the result back into its exact bytes.
    """
    if INSTALLED_UU is None:
        pytest.skip(f"installed Ultimate Upscale fixture missing: {EXTENSION}")
    installed = (INSTALLED_UU / "scripts" / "ultimate-upscale.py").read_bytes()
    text = installed.decode("utf-8")
    if PATCHER_MODULE.MARKER in text:
        for block in reversed(PATCHER_MODULE.BLOCKS):
            assert text.count(block.patched) == block.count
            text = text.replace(block.patched, block.original)
    target = tmp_path / EXTENSION / "scripts" / "ultimate-upscale.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(text.encode("utf-8"))
    if PATCHER_MODULE.MARKER in installed.decode("utf-8"):
        probe = tmp_path / "roundtrip.py"
        probe.write_bytes(target.read_bytes())
        run_patcher(probe)
        assert probe.read_bytes() == installed
    return target


def test_patcher_is_idempotent_and_check_mode_verifies(usdu_source: Path):
    original = usdu_source.read_bytes()
    unpatched_check = run_patcher(usdu_source, "--check", check=False)
    assert unpatched_check.returncode != 0
    assert usdu_source.read_bytes() == original

    first = run_patcher(usdu_source.parents[1])
    patched = usdu_source.read_bytes()
    second = run_patcher(usdu_source.parents[1])
    run_patcher(usdu_source, "--check")

    assert "Patched Ultimate Upscale sub-canvas tiles" in first.stdout
    assert "Patched" not in second.stdout and "verified" in second.stdout
    assert usdu_source.read_bytes() == patched
    text = patched.decode("utf-8")
    assert text.count("processed = _gb10_process_tile(self, p, ") == 8
    assert text.count("self._gb10_owned = None") == 2
    compile(text, str(usdu_source), "exec")


def test_patcher_rejects_source_drift_partial_patch_and_crlf(usdu_source: Path, tmp_path: Path):
    source = usdu_source.read_text(encoding="utf-8")

    drifted = tmp_path / "drifted.py"
    drifted.write_text(source.replace("                p.init_images = [fixed_image]\n", "                p.init_images = [fixed_image, image]\n"), encoding="utf-8")
    result = run_patcher(drifted, check=False)
    assert result.returncode != 0 and "partially patched Ultimate Upscale sub-canvas source for seams-fix fixed tile" in result.stderr
    assert "_gb10_process_tile" not in drifted.read_text(encoding="utf-8")

    crlf = tmp_path / "crlf.py"
    crlf.write_bytes(source.replace("\n", "\r\n").encode("utf-8"))
    result = run_patcher(crlf, check=False)
    assert result.returncode != 0 and "line endings" in result.stderr

    run_patcher(usdu_source)
    patched = usdu_source.read_text(encoding="utf-8")
    partial = tmp_path / "partial.py"
    partial.write_text(patched.replace(
        "            processed = _gb10_process_tile(self, p, image, mask)\n",
        "            p.init_images = [image]\n            p.image_mask = mask\n            processed = processing.process_images(p)\n",
        1,
    ), encoding="utf-8")
    for extra in ((), ("--check",)):
        result = run_patcher(partial, *extra, check=False)
        assert result.returncode != 0 and "partially patched" in result.stderr


# ----------------------------------------------------------------------------------------------- differential


class _StubModel:
    """First stage = identity, so init_latent/image_conditioning are the exact crop pixels and mask."""

    cond_stage_key = "crossattn"

    def __init__(self):
        import torch
        self.dtype = torch.float32

    def encode_first_stage(self, x):
        return x

    def get_first_stage_encoding(self, x):
        return x


@pytest.fixture()
def real_modules(monkeypatch):
    monkeypatch.setenv("IGNORE_CMD_ARGS_ERRORS", "1")
    for name in MODULE_NAMES:
        if name in _REAL_MODULES:
            monkeypatch.setitem(sys.modules, name, _REAL_MODULES[name])
        elif name in sys.modules and not getattr(sys.modules[name], "__file__", None):
            monkeypatch.delitem(sys.modules, name)
    from modules import shared, shared_init
    if getattr(shared, "opts", None) is None:
        shared_init.initialize()
    from modules import processing, sd_samplers
    for name in MODULE_NAMES:
        _REAL_MODULES[name] = importlib.import_module(name)
    if "gradio" not in sys.modules and importlib.util.find_spec("gradio") is None:
        monkeypatch.setitem(sys.modules, "gradio", types.ModuleType("gradio"))
    for key, value in {
        "sd_vae_encode_method": "Full",
        "img2img_color_correction": False,
        "save_init_img": False,
        "overlay_inpaint": True,
        "inpainting_mask_weight": 1.0,
        "img2img_background_color": "#ffffff",
        "upscaler_for_img2img": None,
        "persistent_img2img_init_cache": True,
    }.items():
        monkeypatch.setitem(shared.opts.data, key, value)
    model = _StubModel()
    monkeypatch.setattr(shared, "sd_model", model, raising=False)
    monkeypatch.setattr(sd_samplers, "create_sampler", lambda name, sd_model: types.SimpleNamespace(conditioning_key="concat"))
    monkeypatch.setattr(processing, "images_tensor_to_samples", lambda image, approximation, sd_model: image.clone())
    return types.SimpleNamespace(processing=processing, shared=shared, model=model)


def load_usdu(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def usdu_pair(usdu_source: Path, tmp_path: Path, real_modules):
    patched_path = tmp_path / "patched-ultimate-upscale.py"
    shutil.copyfile(usdu_source, patched_path)
    run_patcher(patched_path)
    return load_usdu(usdu_source, "usdu_unpatched_fixture"), load_usdu(patched_path, "usdu_patched_fixture")


class Recorder:
    """Stands in for process_images: real init + deterministic sampler/VAE + the real output composite."""

    def __init__(self, env):
        self.env = env
        self.calls = []

    def process_images(self, p):
        import numpy as np
        import torch
        from PIL import Image

        processing = self.env.processing
        self.calls.append((p.init_images[0].size, p.image_mask.size))
        p.init([""], [p.seed], [p.seed])
        if p.image_mask is not None and p.paste_to is not None and p.inpaint_full_res:
            x = p.init_latent[0].float()
            n = p.nmask[0].float()
            c = p.image_conditioning[0].float()
            y = (1.0 - x) * (0.5 + 0.5 * n) + 0.25 * c[1:4] * c[:1]
        else:
            y = 1.0 - p.init_latent[0].float()
        sample = (y.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
        overlay = p.overlay_images[0] if self.env.shared.opts.overlay_inpaint and p.overlay_images else None
        image, _ = processing.apply_overlay(Image.fromarray(np.ascontiguousarray(sample)), p.paste_to, overlay)
        image.info["parameters"] = f"tile {len(self.calls)}"
        # generation_last captures p once, after the first completed tile.
        p._generation_last_captured = True
        return types.SimpleNamespace(images=[image], infotext=lambda p, index: image.info["parameters"])


def make_p(env, mask_blur, scripts=None):
    from modules.processing import StableDiffusionProcessingImg2Img

    p = StableDiffusionProcessingImg2Img.__new__(StableDiffusionProcessingImg2Img)
    for key, value in {
        "extra_generation_params": {}, "sd_model": env.model, "sampler_name": "Euler", "seed": 1,
        "inpainting_mask_invert": 0, "inpainting_fill": 1, "mask_round": True, "latent_mask": None,
        "resize_mode": 0, "batch_size": 1, "denoising_strength": 0.35, "image_cfg_scale": None,
        "color_corrections": None, "overlay_images": None, "paste_to": None, "mask_for_overlay": None,
        "sd_model_name": "fixture", "sd_model_hash": "fixture", "do_not_save_samples": True,
        "scripts_value": scripts, "width": 512, "height": 512, "inpaint_full_res": True, "inpaint_full_res_padding": 0,
    }.items():
        setattr(p, key, value)
    p.mask_blur = mask_blur
    return p


def canvas_image(width, height, seed):
    import numpy as np
    from PIL import Image

    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8))


def run_usdu(env, module, monkeypatch, case, *, scripts=None):
    recorder = Recorder(env)
    monkeypatch.setattr(env.processing, "process_images", recorder.process_images)
    width, height, tile_w, tile_h = case["canvas"] + case["tile"]
    p = make_p(env, case["mask_blur"], scripts)
    image = canvas_image(width, height, case["seed"])
    rows, cols = -(-height // tile_h), -(-width // tile_w)
    results = []
    redraw = module.USDURedraw()
    redraw.tile_width, redraw.tile_height, redraw.padding = tile_w, tile_h, case["padding"]
    redraw.mode = module.USDUMode(case["redraw"])
    if redraw.mode != module.USDUMode.NONE:
        image = redraw.start(p, image, rows, cols)
    results.append(image)
    seams = module.USDUSeamsFix()
    seams.tile_width, seams.tile_height = tile_w, tile_h
    seams.padding, seams.denoise, seams.mask_blur, seams.width = case["seams_padding"], 0.35, case["seams_blur"], case["seams_width"]
    seams.mode = module.USDUSFMode(case["seams"])
    if seams.mode != module.USDUSFMode.NONE:
        results.append(seams.start(p, image, rows, cols))
    return results, recorder.calls


def assert_same(expected, actual):
    assert len(expected) == len(actual)
    for want, got in zip(expected, actual):
        assert (got.mode, got.size) == (want.mode, want.size)
        assert got.tobytes() == want.tobytes()
        assert got.info == want.info


EDGE_CASES = [
    # redraw: 0 linear, 1 chess, 2 none; seams: 0 none, 1 band pass, 2 half tile, 3 half tile + intersections
    dict(canvas=(320, 256), tile=(64, 64), padding=32, mask_blur=8, redraw=0, seams=0, seams_padding=16, seams_blur=4, seams_width=32),
    dict(canvas=(301, 227), tile=(64, 64), padding=0, mask_blur=0, redraw=1, seams=1, seams_padding=0, seams_blur=0, seams_width=16),
    dict(canvas=(257, 193), tile=(64, 48), padding=8, mask_blur=16, redraw=0, seams=2, seams_padding=16, seams_blur=8, seams_width=32),
    dict(canvas=(200, 136), tile=(96, 64), padding=24, mask_blur=1, redraw=1, seams=3, seams_padding=8, seams_blur=4, seams_width=24),
    dict(canvas=(321, 129), tile=(64, 64), padding=4, mask_blur=32, redraw=0, seams=3, seams_padding=0, seams_blur=16, seams_width=8),
    dict(canvas=(384, 256), tile=(128, 128), padding=64, mask_blur=4, redraw=2, seams=3, seams_padding=32, seams_blur=8, seams_width=64),
]


def random_cases(count, seed=20261006):
    rng = random.Random(seed)
    cases = []
    for _ in range(count):
        tile_w = rng.choice([48, 64, 96])
        tile_h = rng.choice([tile_w, 64])
        cases.append(dict(
            canvas=(rng.randrange(tile_w + 1, 4 * tile_w + 40), rng.randrange(tile_h + 1, 4 * tile_h + 40)),
            tile=(tile_w, tile_h), padding=rng.choice([0, 1, 8, 32, 64]), mask_blur=rng.choice([0, 1, 2, 4, 8, 16]),
            redraw=rng.choice([0, 1]), seams=rng.choice([0, 1, 2, 3]), seams_padding=rng.choice([0, 8, 16]),
            seams_blur=rng.choice([0, 2, 4, 8]), seams_width=rng.choice([8, 16, 32]),
        ))
    return cases


@pytest.mark.parametrize("case", [dict(c, seed=i) for i, c in enumerate(EDGE_CASES + random_cases(64))])
def test_subcanvas_tiles_are_bitwise_identical_to_full_canvas(usdu_pair, real_modules, monkeypatch, case):
    unpatched, patched = usdu_pair
    expected, full_calls = run_usdu(real_modules, unpatched, monkeypatch, case)
    actual, window_calls = run_usdu(real_modules, patched, monkeypatch, case)

    assert_same(expected, actual)
    canvas = case["canvas"]
    assert len(window_calls) == len(full_calls) > 1
    # Only the first tile (before the last-generation snapshot) has to see the whole canvas.
    assert window_calls[0] == full_calls[0] == (canvas, canvas)
    if all(call == (canvas, canvas) for call in full_calls):
        assert any(call[0] != canvas for call in window_calls[1:])
    # else: a mask that blurs to nothing made img2img fall back to the whole image (upstream shrinks the canvas to
    # the tile size); the patched run must and does reproduce that through the full-canvas path.


class _Script:
    def __init__(self, title, filename):
        self._title, self.filename = title, f"/ext/scripts/{filename}"

    def title(self):
        return self._title


class _Runner:
    def __init__(self, scripts_with_args):
        self.alwayson_scripts = [script for script, _ in scripts_with_args]
        self._args = {id(script): args for script, args in scripts_with_args}

    def _script_args_for(self, p, script):
        return self._args[id(script)]


def _inert_runner(*extra):
    return _Runner([
        (_Script("Seed", "seed.py"), ()),
        (_Script("TeaCache", "teacache.py"), (True, 0.1)),
        (_Script("Soft Inpainting", "soft_inpainting.py"), (False, 1, 0.5)),
        (_Script("Tiled Diffusion", "tilediffusion.py"), (False, "MultiDiffusion")),
        (_Script("ControlNet", "controlnet.py"), (types.SimpleNamespace(enabled=False), {"enabled": False}, "not-a-unit")),
        *extra,
    ])


def _option(key, value):
    return lambda env, monkeypatch, p, image, mask: monkeypatch.setitem(env.shared.opts.data, key, value)


def _attr(key, value):
    return lambda env, monkeypatch, p, image, mask: setattr(p, key, value)


def _runner(factory):
    return lambda env, monkeypatch, p, image, mask: setattr(p, "scripts_value", factory())


@pytest.mark.parametrize("name, configure", [
    ("snapshot not captured yet", _attr("_generation_last_captured", False)),
    ("inpaint full res off", _attr("inpaint_full_res", False)),
    ("inverted mask", _attr("inpainting_mask_invert", 1)),
    ("latent mask", lambda env, monkeypatch, p, image, mask: setattr(p, "latent_mask", mask)),
    ("negative padding", _attr("inpaint_full_res_padding", -4)),
    ("samples saved", _attr("do_not_save_samples", False)),
    ("save init images", _option("save_init_img", True)),
    ("no inpaint overlay", _option("overlay_inpaint", False)),
    ("rgba canvas", lambda env, monkeypatch, p, image, mask: setattr(p, "_image", image.convert("RGBA"))),
    ("controlnet unit enabled", _runner(lambda: _Runner([(_Script("ControlNet", "controlnet.py"), (types.SimpleNamespace(enabled=True),))]))),
    ("controlnet unit dict without enabled", _runner(lambda: _Runner([(_Script("ControlNet", "controlnet.py"), ({"module": "tile"},))]))),
    ("controlnet legacy field", lambda env, monkeypatch, p, image, mask: (setattr(p, "scripts_value", _inert_runner()), setattr(p, "control_net_enabled", False))),
    ("soft inpainting enabled", _runner(lambda: _Runner([(_Script("Soft Inpainting", "soft_inpainting.py"), (True,))]))),
    ("demofusion enabled", _runner(lambda: _Runner([(_Script("demofusion", "tileglobal.py"), (True,))]))),
    ("unknown always-on script", _runner(lambda: _inert_runner((_Script("Something New", "new.py"), ())))),
    ("audited title from another file", _runner(lambda: _Runner([(_Script("Seed", "other.py"), ())]))),
    ("blank mask", lambda env, monkeypatch, p, image, mask: mask.paste(0, (0, 0, *mask.size))),
])
def test_plan_gate_keeps_full_canvas_when_the_canvas_is_observable(usdu_pair, real_modules, monkeypatch, name, configure):
    from PIL import Image, ImageDraw

    _, patched = usdu_pair
    p = make_p(real_modules, 4, _inert_runner())
    p.width = p.height = 128
    p.inpaint_full_res_padding = 8
    p._generation_last_captured = True
    image = canvas_image(320, 256, 1)
    mask = Image.new("L", image.size, "black")
    ImageDraw.Draw(mask).rectangle((64, 64, 128, 128), fill="white")
    assert patched._gb10_subcanvas_plan(p, image, mask) is not None

    configure(real_modules, monkeypatch, p, image, mask)
    assert patched._gb10_subcanvas_plan(p, getattr(p, "_image", image), mask) is None, name


def test_audited_and_disabled_scripts_keep_the_fast_path_exact(usdu_pair, real_modules, monkeypatch):
    unpatched, patched = usdu_pair
    case = dict(canvas=(320, 256), tile=(64, 64), padding=16, mask_blur=4, redraw=0, seams=2, seams_padding=8, seams_blur=4, seams_width=16, seed=7)
    expected, _ = run_usdu(real_modules, unpatched, monkeypatch, case, scripts=_inert_runner())
    actual, calls = run_usdu(real_modules, patched, monkeypatch, case, scripts=_inert_runner())

    assert sum(call[0] != case["canvas"] for call in calls) == len(calls) - 1
    assert_same(expected, actual)


def test_pass_input_images_are_never_mutated(usdu_pair, real_modules, monkeypatch):
    _, patched = usdu_pair
    recorder = Recorder(real_modules)
    monkeypatch.setattr(real_modules.processing, "process_images", recorder.process_images)
    p = make_p(real_modules, 4)
    p._generation_last_captured = True
    source = canvas_image(256, 192, 3)
    before = source.tobytes()
    seams = patched.USDUSeamsFix()
    seams.tile_width = seams.tile_height = 64
    seams.padding, seams.denoise, seams.mask_blur, seams.width = 8, 0.35, 4, 16
    seams.mode = patched.USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS
    result = seams.start(p, source, 3, 4)

    assert source.tobytes() == before
    assert result is not source
    assert recorder.calls and all(call[0] != source.size for call in recorder.calls)


def test_processing_drift_fails_fast_instead_of_returning_a_different_canvas(usdu_pair, real_modules, monkeypatch):
    import cv2

    _, patched = usdu_pair

    def wider_blur(src, ksize, sigma):
        return cv2.GaussianBlur(src, tuple(2 * k + 1 if k > 1 else k for k in ksize), sigma * 2)

    monkeypatch.setattr(real_modules.processing, "cv2", types.SimpleNamespace(GaussianBlur=wider_blur))
    case = dict(canvas=(320, 256), tile=(64, 64), padding=16, mask_blur=4, redraw=0, seams=0, seams_padding=0, seams_blur=0, seams_width=8, seed=7)
    with pytest.raises(RuntimeError, match="sub-canvas mismatch"):
        run_usdu(real_modules, patched, monkeypatch, case)
