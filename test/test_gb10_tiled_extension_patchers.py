"""gb10/patch-ultimate-upscale-state-lifecycle.py against the installed Ultimate SD Upscale checkout.

The shared fail-closed engine (atomic writes, CRLF, missing files) is tested in test_gb10_patchlib.py, the MultiDiffusion
patcher in test_gb10_multidiffusion_performance_patcher.py and the run.sh order in test_gb10_run_patcher_order.py.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from test.helpers import load_source, module

ROOT = Path(__file__).parents[1]
GB10 = ROOT / "gb10"
UU_PATCHER = GB10 / "patch-ultimate-upscale-state-lifecycle.py"
SUBCANVAS_PATCHER = GB10 / "patch-ultimate-upscale-subcanvas.py"
# sha256 of scripts/ultimate-upscale.py at Coyote-A/ultimate-upscale-for-automatic1111 master 2322caa, the ORIGINAL of
# both patchers.
UPSTREAM_SHA256 = "0abbb3df467ec17d852534de74484dce6710de7a4010ab25f194d2b3a089eb09"


def installed_extension(name: str) -> Path | None:
    """The host deploy root, or the same checkout where run.sh mounts it inside a webui container."""
    return next((path for path in (Path("/opt/gb10/stable-diffusion/Extensions") / name, ROOT / "extensions" / name) if path.is_dir()), None)


INSTALLED_UU = installed_extension("ultimate-upscale-for-automatic1111")


# The patchers import patchlib as a sibling module, as under run.sh.
PATCHLIB = load_source("patchlib", GB10 / "patchlib.py")
UU_MODULE = load_source("gb10_patch_ultimate_upscale_state_lifecycle", UU_PATCHER, {"patchlib": PATCHLIB})
SUBCANVAS_MODULE = load_source("gb10_patch_ultimate_upscale_subcanvas_for_lifecycle", SUBCANVAS_PATCHER, {"patchlib": PATCHLIB})


def run_patcher(patcher: Path, target: Path, *extra: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(patcher), str(target), *extra], check=check, capture_output=True, text=True)


def upstream_text(installed: str) -> str:
    """The installed script with each patcher's current or deploy10 release reverted (exactly, see patchlib)."""
    text = installed
    for releases in ((SUBCANVAS_MODULE.BLOCKS, SUBCANVAS_MODULE.DEPLOY10), (UU_MODULE.BLOCKS, UU_MODULE.DEPLOY10)):
        for blocks in releases:
            reverted = PATCHLIB._revert_previous(text, blocks)
            if reverted is not None:
                text = reverted
                break
    return text


@pytest.fixture()
def ultimate_copy(tmp_path: Path) -> Path:
    """Upstream 2322caa ultimate-upscale.py, derived from the installed script by reverting both patchers."""
    if INSTALLED_UU is None:
        pytest.skip("installed Ultimate Upscale fixture missing")
    target = tmp_path / "ultimate-upscale-for-automatic1111"
    shutil.copytree(INSTALLED_UU, target, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
    script = target / "scripts" / "ultimate-upscale.py"
    script.write_bytes(upstream_text(script.read_bytes().decode("utf-8")).encode("utf-8"))
    assert hashlib.sha256(script.read_bytes()).hexdigest() == UPSTREAM_SHA256
    return target


def test_patchers_turn_upstream_and_the_installed_release_into_the_same_bytes(ultimate_copy: Path, tmp_path: Path):
    script = ultimate_copy / "scripts" / "ultimate-upscale.py"
    upstream = script.read_bytes()
    missing = run_patcher(UU_PATCHER, script, "--check", check=False)
    assert missing.returncode != 0 and "patch missing" in missing.stderr
    first = run_patcher(UU_PATCHER, ultimate_copy)
    lifecycle_only = script.read_bytes()
    second = run_patcher(UU_PATCHER, ultimate_copy)
    run_patcher(UU_PATCHER, script, "--check")
    assert "Patched Ultimate Upscale state lifecycle" in first.stdout
    assert "Patched" not in second.stdout and "lifecycle verified" in second.stdout
    assert script.read_bytes() == lifecycle_only != upstream
    text = lifecycle_only.decode("utf-8")
    assert "state.begin()\n" not in text and "state.end()\n" not in text
    run_patcher(SUBCANVAS_PATCHER, script)
    run_patcher(UU_PATCHER, script, "--check")  # the sub-canvas blocks leave the lifecycle blocks intact
    both = script.read_bytes()

    installed = (INSTALLED_UU / "scripts" / "ultimate-upscale.py").read_bytes()
    upgraded = tmp_path / "installed.py"
    upgraded.write_bytes(installed)
    for patcher in (UU_PATCHER, SUBCANVAS_PATCHER):
        run_patcher(patcher, upgraded)
    assert upgraded.read_bytes() == both

    deploy10 = tmp_path / "deploy10.py"
    deploy10.write_bytes(upstream_text(installed.decode("utf-8")).encode("utf-8"))
    for blocks in (UU_MODULE.DEPLOY10, SUBCANVAS_MODULE.DEPLOY10):
        text = deploy10.read_text(encoding="utf-8")
        for block in blocks:
            text = text.replace(block.original, block.patched)
        deploy10.write_text(text, encoding="utf-8")
    assert installed in (upstream, deploy10.read_bytes(), both)  # a fresh install, the deploy10 release or this one
    for patcher in (UU_PATCHER, SUBCANVAS_PATCHER):
        outdated = run_patcher(patcher, deploy10, "--check", check=False)
        assert outdated.returncode != 0 and "patch outdated" in outdated.stderr
    for patcher in (UU_PATCHER, SUBCANVAS_PATCHER):
        run_patcher(patcher, deploy10)
    assert deploy10.read_bytes() == both


def test_lifecycle_patcher_rejects_source_drift_without_writing(ultimate_copy: Path, tmp_path: Path):
    source = (ultimate_copy / "scripts" / "ultimate-upscale.py").read_text(encoding="utf-8")
    for name, text in (
        ("drift", source.replace("        state.begin()\n", "        state.begin(job='usdu')\n")),
        ("crlf", source.replace("\n", "\r\n")),
        ("partial", source.replace(UU_MODULE.BLOCKS[1].original, UU_MODULE.BLOCKS[1].patched)),
    ):
        target = tmp_path / f"{name}.py"
        target.write_bytes(text.encode("utf-8"))
        result = run_patcher(UU_PATCHER, target, check=False)
        assert result.returncode != 0, name
        assert ("line endings" if name == "crlf" else "partially patched") in result.stderr, name
        assert target.read_bytes() == text.encode("utf-8"), name


@pytest.mark.parametrize("patcher", [UU_PATCHER, SUBCANVAS_PATCHER], ids=["lifecycle", "subcanvas"])
def test_ultimate_upscale_patchers_fail_closed_on_a_missing_script(tmp_path: Path, patcher: Path):
    # run.sh has no separate existence check: a missing (or never installed) script must abort the deploy here.
    for target in (tmp_path / "ultimate-upscale-for-automatic1111", tmp_path):
        result = run_patcher(patcher, target, check=False)
        assert result.returncode != 0 and "source not found" in result.stderr


# ------------------------------------------------------------------------------------------------- behaviour


class _State:
    """modules.shared.state: an API job already began it (api.py), possibly interrupted while the upscaler ran."""

    def __init__(self, interrupted: bool = False) -> None:
        self.interrupted = interrupted
        self.job_count = -1
        self.events: list[str] = []

    def begin(self, job: str = "(unknown)") -> None:
        self.events.append("begin")
        self.interrupted = False

    def end(self) -> None:
        self.events.append("end")


class _Processing:
    """modules.processing: process_images records each tile; override settings record apply/restore."""

    def __init__(self, fail_on_tile: int | None = None) -> None:
        self.tiles = 0
        self.events: list[str] = []
        self.fail_on_tile = fail_on_tile

    def store(self, p):
        self.events.append("store")
        return {"sd_vae": "Automatic"}

    def apply(self, p):
        self.events.append("apply")

    def restore(self, stored):
        self.events.append(f"restore {stored}")

    def process_images(self, p):
        self.tiles += 1
        if self.tiles == self.fail_on_tile:
            raise RuntimeError("tile boom")
        image = p.init_images[0].copy()
        return types.SimpleNamespace(images=[image], infotext=lambda p, index, n=self.tiles: f"tile {n}")


def load_usdu(path: Path, state: _State, recorder: _Processing):
    processing = module(
        "modules.processing", StableDiffusionProcessing=object, Processed=object, process_images=recorder.process_images,
        store_processing_override_settings=recorder.store, apply_processing_override_settings=recorder.apply,
        restore_processing_override_settings=recorder.restore,
    )
    return load_source("ultimate_upscale_fixture", path, {
        "modules": module("modules", package=True, devices=types.SimpleNamespace(device="cpu"), scripts=types.SimpleNamespace(Script=object), processing=processing),
        "modules.shared": module("modules.shared", state=state, opts=types.SimpleNamespace(), sd_upscalers=[]),
        "modules.processing": processing,
        "modules.images": module("modules.images", save_image=lambda *args, **kwargs: None),
        "gradio": module("gradio"),
    })


def make_usdu(usdu, *, canvas=(128, 128), tile=64, redraw=0, seams=0, restore_afterwards=True):
    """A USDUpscaler over a black canvas; redraw: 0 linear, 1 chess, 2 none; seams: USDUSFMode value."""
    from PIL import Image

    p = types.SimpleNamespace(width=canvas[0], height=canvas[1], override_settings={"sd_vae": "fixture.safetensors"},
                              override_settings_restore_afterwards=restore_afterwards, extra_generation_params={})
    upscaler = object.__new__(usdu.USDUpscaler)
    upscaler.p, upscaler.image, upscaler.initial_info = p, Image.new("RGB", canvas), None
    upscaler.redraw, upscaler.seams_fix = usdu.USDURedraw(), usdu.USDUSeamsFix()
    for part, save in ((upscaler.redraw, False), (upscaler.seams_fix, False)):
        part.save, part.tile_width, part.tile_height = save, tile, tile
    upscaler.rows, upscaler.cols = -(-canvas[1] // tile), -(-canvas[0] // tile)
    upscaler.setup_redraw(redraw, 8, 4)
    upscaler.setup_seams_fix(8, 0.35, 4, 16, seams)
    return upscaler


@pytest.fixture()
def patched_script(ultimate_copy: Path) -> Path:
    run_patcher(UU_PATCHER, ultimate_copy)
    return ultimate_copy / "scripts" / "ultimate-upscale.py"


@pytest.mark.parametrize("redraw", [0, 1])
def test_upstream_clears_an_interrupt_sent_during_the_upscaler(ultimate_copy: Path, redraw: int):
    state, recorder = _State(interrupted=True), _Processing()
    usdu = load_usdu(ultimate_copy / "scripts" / "ultimate-upscale.py", state, recorder)
    make_usdu(usdu, redraw=redraw).process()
    assert state.events == ["begin", "end"] and recorder.tiles > 0


@pytest.mark.parametrize("redraw", [0, 1])
@pytest.mark.parametrize("seams", [0, 1, 2, 3])
def test_interrupt_during_the_upscaler_runs_no_tile_and_reports_no_infotext(patched_script: Path, redraw: int, seams: int):
    state, recorder = _State(interrupted=True), _Processing()
    usdu = load_usdu(patched_script, state, recorder)
    upscaler = make_usdu(usdu, redraw=redraw, seams=seams)
    image = upscaler.image
    upscaler.process()
    assert state.events == [] and state.interrupted  # the caller's job state is untouched; no extra torch_gc
    assert recorder.tiles == 0
    assert upscaler.result_images == [image] and upscaler.initial_info is None
    assert recorder.events == ["store", "apply", "restore {'sd_vae': 'Automatic'}"]


@pytest.mark.parametrize("restore_afterwards", [True, False])
def test_override_settings_apply_once_around_every_tile_and_restore_on_failure(patched_script: Path, restore_afterwards: bool):
    state, recorder = _State(), _Processing()
    usdu = load_usdu(patched_script, state, recorder)
    usdu.processing.process_images = lambda p: (recorder.events.append("tile"), recorder.process_images(p))[1]
    make_usdu(usdu, canvas=(128, 64), seams=2, restore_afterwards=restore_afterwards).process()
    restore = ["restore {'sd_vae': 'Automatic'}"] if restore_afterwards else []
    assert recorder.events == ["store", "apply", "tile", "tile", "tile"] + restore  # 2 redraw + 1 seams tile

    failing = _Processing(fail_on_tile=2)
    usdu = load_usdu(patched_script, state, failing)
    with pytest.raises(RuntimeError, match="tile boom"):
        make_usdu(usdu, canvas=(128, 64), restore_afterwards=restore_afterwards).process()
    assert failing.events == ["store", "apply"] + restore and state.events == []


@pytest.mark.parametrize("canvas, seams, seams_tiles", [
    ((64, 64), 1, 0),    # band pass, one tile: no seam
    ((64, 64), 2, 0),    # half tile, one tile
    ((64, 64), 3, 0),    # half tile + intersections, one tile
    ((128, 64), 3, 1),   # one half-tile seam, no intersection
    ((128, 128), 3, 5),  # 4 half-tile seams + 1 intersection
])
def test_seams_pass_reports_its_image_and_infotext_only_when_it_ran_a_tile(patched_script: Path, canvas, seams, seams_tiles):
    recorder = _Processing()
    usdu = load_usdu(patched_script, _State(), recorder)
    upscaler = make_usdu(usdu, canvas=canvas, seams=seams)
    upscaler.process()
    redraw_tiles = upscaler.rows * upscaler.cols
    assert recorder.tiles == redraw_tiles + seams_tiles
    assert len(upscaler.result_images) == (2 if seams_tiles else 1)
    assert upscaler.image is upscaler.result_images[-1]
    assert upscaler.initial_info == f"tile {recorder.tiles}"


def test_upstream_blanks_the_infotext_and_duplicates_the_image_when_no_seams_tile_runs(ultimate_copy: Path):
    usdu = load_usdu(ultimate_copy / "scripts" / "ultimate-upscale.py", _State(), _Processing())
    upscaler = make_usdu(usdu, canvas=(128, 64), seams=3)
    upscaler.process()
    assert upscaler.initial_info is None
    assert len(upscaler.result_images) == 2


class _VaeProcessing(_Processing):
    """modules.processing's override handling as process_images() runs it per tile, with sd_vae reloads counted:
    a reload happens whenever the selected VAE differs from the loaded one (sd_vae.reload_vae_weights)."""

    def __init__(self) -> None:
        super().__init__()
        self.opts = {"sd_vae": "Automatic"}
        self.loaded = "Automatic"
        self.reloads = 0
        self.tile_vaes: list[str] = []

    def _reload(self):
        if self.loaded != self.opts["sd_vae"]:
            self.loaded = self.opts["sd_vae"]
            self.reloads += 1

    def store(self, p):
        return {key: self.opts[key] for key in p.override_settings}

    def apply(self, p):
        self.opts.update(p.override_settings)
        self._reload()

    def restore(self, stored):
        self.opts.update(stored)
        self._reload()

    def process_images(self, p):
        stored = self.store(p)
        self.apply(p)
        try:
            self.tile_vaes.append(self.loaded)
            return super().process_images(p)
        finally:
            self.restore(stored)


def test_vae_override_loads_once_per_run_instead_of_twice_per_tile(ultimate_copy: Path, tmp_path: Path):
    upstream = ultimate_copy / "scripts" / "ultimate-upscale.py"
    patched = tmp_path / "patched.py"
    shutil.copyfile(upstream, patched)
    run_patcher(UU_PATCHER, patched)
    for path, reloads in ((upstream, 2 * 8), (patched, 2)):
        recorder = _VaeProcessing()
        usdu = load_usdu(path, _State(), recorder)
        make_usdu(usdu, canvas=(128, 128), seams=2).process()
        assert recorder.tiles == 8  # 4 redraw + 4 half-tile seams
        assert recorder.tile_vaes == ["fixture.safetensors"] * 8 and recorder.loaded == "Automatic"
        assert recorder.reloads == reloads
