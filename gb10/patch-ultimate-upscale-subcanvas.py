#!/usr/bin/env python3
"""Patch and verify the Ultimate SD Upscale per-tile sub-canvas window (B-C2).

Every USDU tile calls process_images() with the whole upscaled canvas plus a canvas-sized mask, so img2img init
and apply_overlay run about a dozen full-canvas PIL/cv2 passes per tile (mask blur, overlay premultiply/paste,
uncrop, alpha_composite, conversions) although only the "Only masked" crop region is denoised. The patch hands
process_images() a window of the canvas instead and pastes the result back. The canvas is bitwise identical:

- Blur: cv2.GaussianBlur is local with reach r = int(2.5 * blur + 0.5). The window contains the mask bbox grown
  by r + 1, so reflect-101 at an interior window edge only reflects zeros and every pixel outside the window
  blurs to 0, exactly as on the canvas.
- Crop region: the planner runs the same blur on a probe (bbox grown by r + 1 + padding, so padding is never
  clamped at an interior probe edge), then get_crop_region_v2 and expand_crop_region with the canvas size. The
  window contains that whole region, so no clamp in either function can differ between canvas and window.
- Outside the window the canvas run composites an opaque overlay of the original pixels over a transparent base,
  which returns the original pixels; inside, every operation is per pixel or works on the identical crop.
- After the tile, processing's blurred mask must equal the planner's (byte for byte inside the blur-safe box,
  nothing outside it) and p.paste_to and the result size must equal the planned region; otherwise the tile
  raises (processing.py drifted from the planner) instead of returning a different image.

The fast path runs only when nothing else can observe the canvas size: inpaint full res with a non-inverted L
mask, no latent mask, overlay_inpaint on, save_init_img off, samples not saved, the last-generation snapshot
already captured from the first full-canvas tile, and every always-on script either audited canvas-independent
or provably disabled (ControlNet units explicitly disabled and no legacy control_net_* fields; Soft Inpainting,
Tiled Diffusion and DemoFusion disabled). Anything else, including unknown scripts, takes the original path.
Each pass copies its input canvas once and then pastes tiles in place, so earlier result_images stay untouched.
"""
from __future__ import annotations

import argparse
import ast
import os
import tempfile
from pathlib import Path

TARGET_RELATIVE = Path("scripts") / "ultimate-upscale.py"
MARKER = "OPENCLAW_USDU_SUBCANVAS_V1"

IMPORTS = '''import os
import cv2
import numpy as np
from modules import masking
'''

HELPERS = '''# OPENCLAW_USDU_SUBCANVAS_V1: exact per-tile sub-canvas window, see gb10/patch-ultimate-upscale-subcanvas.py.
# Always-on scripts audited not to read p.init_images, p.image_mask, overlay/paste state or full-canvas outputs.
_GB10_SUBCANVAS_INERT_SCRIPTS = frozenset({
    ("Sampler", "sampler.py"),
    ("Seed", "seed.py"),
    ("Refiner", "refiner.py"),
    ("Comments", "comments.py"),
    ("Extra options", "extra_options_section.py"),
    ("Hypertile", "hypertile_script.py"),
    ("Incantations", "incantation_base.py"),
    ("Dynamic Thresholding (CFG Scale Fix)", "dynamic_thresholding.py"),
    ("Detail Daemon", "detail_daemon.py"),
    ("OpenClaw Denoise Ramp", "openclaw_denoise_ramp.py"),
    ("OpenClaw Multi-Sampler", "openclaw_multi_sampler.py"),
    ("TeaCache", "teacache.py"),
    ("Tiled VAE", "tilevae.py"),
})
# Always-on scripts that read the canvas when enabled; their first script argument is the enable flag.
_GB10_SUBCANVAS_SWITCHED_SCRIPTS = frozenset({
    ("Soft Inpainting", "soft_inpainting.py"),
    ("Tiled Diffusion", "tilediffusion.py"),
    ("demofusion", "tileglobal.py"),
})


def _gb10_controlnet_idle(p, args):
    # ControlNet reads p.init_images/p.image_mask for every enabled unit; a unit dict without "enabled" counts as on.
    if any(name.startswith("control_net_") for name in vars(p)):
        return False
    for arg in args:
        enabled = arg.get("enabled", True) if isinstance(arg, dict) else getattr(arg, "enabled", False)
        if enabled is not False:
            return False
    return True


def _gb10_subcanvas_scripts_inert(p):
    runner = p.scripts
    if runner is None:
        return True
    args_for = getattr(runner, "_script_args_for", None)
    if args_for is None:
        return False
    for script in runner.alwayson_scripts:
        key = (script.title(), os.path.basename(script.filename or ""))
        if key in _GB10_SUBCANVAS_INERT_SCRIPTS:
            continue
        args = args_for(p, script)
        if key in _GB10_SUBCANVAS_SWITCHED_SCRIPTS and len(args) > 0 and not args[0]:
            continue
        if key == ("ControlNet", "controlnet.py") and _gb10_controlnet_idle(p, args):
            continue
        return False
    return True


def _gb10_blur_reach(blur):
    return int(2.5 * blur + 0.5) if blur > 0 else 0


def _gb10_subcanvas_plan(p, image, mask):
    """Return (window, crop region, blur-safe box, blurred mask inside that box) or None for the full canvas.

    Boxes are (x1, y1, x2, y2) in canvas coordinates.
    """
    pad = p.inpaint_full_res_padding
    if not p.inpaint_full_res or p.inpainting_mask_invert or p.latent_mask is not None:
        return None
    if type(pad) is not int or pad < 0 or image.mode != "RGB" or mask.mode != "L" or image.size != mask.size:
        return None
    if not opts.overlay_inpaint or opts.save_init_img or not p.do_not_save_samples:
        return None
    # generation_last snapshots p.init_images/p.image_mask once per p, from the first completed (full-canvas) tile.
    if not getattr(p, "_generation_last_captured", False) or not _gb10_subcanvas_scripts_inert(p):
        return None
    box = mask.getbbox()
    if box is None:
        return None
    width, height = mask.size
    rx, ry = _gb10_blur_reach(p.mask_blur_x), _gb10_blur_reach(p.mask_blur_y)
    safe = (max(box[0] - rx - 1, 0), max(box[1] - ry - 1, 0), min(box[2] + rx + 1, width), min(box[3] + ry + 1, height))
    probe = (max(safe[0] - pad, 0), max(safe[1] - pad, 0), min(safe[2] + pad, width), min(safe[3] + pad, height))
    # Same blur as StableDiffusionProcessingImg2Img.init (an L mask is already binary there).
    np_mask = np.asarray(mask.crop(probe))
    if p.mask_blur_x > 0:
        np_mask = cv2.GaussianBlur(np_mask, (2 * rx + 1, 1), p.mask_blur_x)
    if p.mask_blur_y > 0:
        np_mask = cv2.GaussianBlur(np_mask, (1, 2 * ry + 1), p.mask_blur_y)
    blurred = Image.fromarray(np_mask)
    crop = masking.get_crop_region_v2(blurred, pad)
    if crop is None:
        return None  # blank after blur: processing falls back to whole-image img2img, which needs the canvas
    crop = (crop[0] + probe[0], crop[1] + probe[1], crop[2] + probe[0], crop[3] + probe[1])
    crop = masking.expand_crop_region(crop, p.width, p.height, width, height)
    window = (min(crop[0], safe[0]), min(crop[1], safe[1]), max(crop[2], safe[2]), max(crop[3], safe[3]))
    blurred_safe = blurred.crop((safe[0] - probe[0], safe[1] - probe[1], safe[2] - probe[0], safe[3] - probe[1]))
    return window, crop, safe, blurred_safe


def _gb10_process_tile(owner, p, image, mask):
    """process_images() for one tile; processed.images[0] is the full canvas either way."""
    plan = _gb10_subcanvas_plan(p, image, mask)
    if plan is None:
        p.init_images = [image]
        p.image_mask = mask
        return processing.process_images(p)
    (x1, y1, x2, y2), crop, safe, blurred_safe = plan
    p.init_images = [image.crop((x1, y1, x2, y2))]
    p.image_mask = mask.crop((x1, y1, x2, y2))
    processed = processing.process_images(p)
    if len(processed.images) > 0:
        result = processed.images[0]
        expected = (crop[0] - x1, crop[1] - y1, crop[2] - crop[0], crop[3] - crop[1])
        inner = (safe[0] - x1, safe[1] - y1, safe[2] - x1, safe[3] - y1)
        overlay_mask = p.mask_for_overlay
        bbox = overlay_mask.getbbox() if overlay_mask is not None else None
        if (
            not p.inpaint_full_res or p.paste_to != expected or result.size != (x2 - x1, y2 - y1) or result.mode != "RGB"
            or bbox is None or overlay_mask.size != result.size
            or not (inner[0] <= bbox[0] and inner[1] <= bbox[1] and bbox[2] <= inner[2] and bbox[3] <= inner[3])
            or overlay_mask.crop(inner).tobytes() != blurred_safe.tobytes()
        ):
            raise RuntimeError(
                f"Ultimate SD upscale sub-canvas mismatch: planned window {(x1, y1, x2, y2)} crop {expected}, "
                f"processing produced paste_to {p.paste_to} size {result.size} mode {result.mode} or a different "
                "mask blur; img2img mask handling changed, update gb10/patch-ultimate-upscale-subcanvas.py"
            )
        # The pass owns one copy of its input canvas; earlier passes' images stay untouched.
        canvas = getattr(owner, "_gb10_owned", None)
        if canvas is not image:
            canvas = owner._gb10_owned = image.copy()
        canvas.paste(result, (x1, y1))
        canvas.info = dict(result.info)
        processed.images[0] = canvas
    return processed


'''

TILE_CALL = "processed = _gb10_process_tile(self, p, {image}, mask)\n"


def tile_block(indent: str, image: str) -> str:
    return (
        f"{indent}p.init_images = [{image}]\n"
        f"{indent}p.image_mask = mask\n"
        f"{indent}processed = processing.process_images(p)\n"
    )


# (original, patched, occurrences) in the post-lifecycle-patch source. Every block starts a line and is matched
# with its preceding newline, so a 12-space block never matches inside a 16-space one.
BLOCKS = [
    (
        '\nfrom enum import Enum\n\nelem_id_prefix = "ultimateupscale"\n',
        '\nfrom enum import Enum\n' + IMPORTS + '\nelem_id_prefix = "ultimateupscale"\n',
        1,
    ),
    (
        "\n    HALF_TILE_PLUS_INTERSECTIONS = 3\n\nclass USDUpscaler():\n",
        "\n    HALF_TILE_PLUS_INTERSECTIONS = 3\n\n" + HELPERS + "class USDUpscaler():\n",
        1,
    ),
    (
        "\n    def init_draw(self, p, width, height):\n        p.inpaint_full_res = True\n",
        "\n    def init_draw(self, p, width, height):\n        self._gb10_owned = None\n        p.inpaint_full_res = True\n",
        1,
    ),
    (
        "\n    def init_draw(self, p):\n        self.initial_info = None\n",
        "\n    def init_draw(self, p):\n        self._gb10_owned = None\n        self.initial_info = None\n",
        1,
    ),
    ("\n" + tile_block(" " * 16, "image"), "\n" + " " * 16 + TILE_CALL.format(image="image"), 5),
    ("\n" + tile_block(" " * 12, "image"), "\n" + " " * 12 + TILE_CALL.format(image="image"), 2),
    ("\n" + tile_block(" " * 16, "fixed_image"), "\n" + " " * 16 + TILE_CALL.format(image="fixed_image"), 1),
]
TILE_SITES = 8  # every process_images call in USDURedraw and USDUSeamsFix


def target_for(path: Path) -> Path:
    return path / TARGET_RELATIVE if path.is_dir() else path


def verify(source: str, target: Path) -> None:
    if source.count(MARKER) != 1:
        raise SystemExit(f"Ultimate Upscale sub-canvas verification failed (marker count {source.count(MARKER)}): {target}")
    for original, patched, count in BLOCKS:
        if source.count(patched) != count or original in source:
            raise SystemExit(f"Ultimate Upscale sub-canvas verification failed (partial patch): {target}")
    if source.count("_gb10_process_tile(self, p, ") != TILE_SITES:
        raise SystemExit(f"Ultimate Upscale sub-canvas verification failed (partial patch): {target}")
    if "processing.process_images(p)" in source.replace(HELPERS, ""):
        raise SystemExit(f"Ultimate Upscale sub-canvas verification failed (unrouted process_images call): {target}")
    try:
        ast.parse(source)
    except SyntaxError as exc:
        raise SystemExit(f"Ultimate Upscale sub-canvas verification failed (invalid Python): {target}: {exc}") from exc


def patch(source: str, target: Path) -> str:
    if "\r" in source:
        raise SystemExit(f"unsupported Ultimate Upscale line endings (expected LF): {target}")
    for original, _patched, count in BLOCKS:
        if source.count(original) != count:
            raise SystemExit(f"unsupported Ultimate Upscale sub-canvas source (block found {source.count(original)}x, expected {count}x): {target}")
    for original, patched, _count in BLOCKS:
        source = source.replace(original, patched)
    return source


def replace_atomically(path: Path, text: str) -> None:
    """Write text (UTF-8, newlines untranslated) to a temporary file next to path, fsync it, give it path's mode
    (and owner when run as root, as run.sh does) and os.replace() path with it: a failure at any point leaves path
    untouched and removes the temporary file. A symlinked path is written through, as an in-place write would."""
    path = Path(os.path.realpath(path))
    stat = path.stat()
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".gb10-tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as tmp:
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.chmod(tmp_name, stat.st_mode & 0o7777)
        if os.geteuid() == 0:
            os.chown(tmp_name, stat.st_uid, stat.st_gid)
        os.replace(tmp_name, path)
    except BaseException as exc:
        try:
            os.unlink(tmp_name)
        except OSError as cleanup:
            exc.add_note(f"could not remove the temporary file {tmp_name}: {cleanup}")
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = target_for(args.path)
    if not target.is_file():
        raise SystemExit(f"Ultimate Upscale source not found: {target}")
    source = target.read_bytes().decode("utf-8")  # no newline translation: CRLF must fail closed
    if not args.check and MARKER not in source:
        source = patch(source, target)
        verify(source, target)
        replace_atomically(target, source)  # verified above; atomic, keeps mode and owner
        print(f"Patched Ultimate Upscale sub-canvas tiles: {target}")
    verify(source, target)
    print(f"Ultimate Upscale sub-canvas tiles verified: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
