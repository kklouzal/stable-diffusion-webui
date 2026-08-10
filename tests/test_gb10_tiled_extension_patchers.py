from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
MD_PATCHER = ROOT / "gb10" / "patch-multidiffusion-terminal-tiles.py"
UU_PATCHER = ROOT / "gb10" / "patch-ultimate-upscale-state-lifecycle.py"
INSTALLED_MD = Path("/opt/gb10/stable-diffusion/Extensions/multidiffusion-upscaler-for-automatic1111")
INSTALLED_UU = Path("/opt/gb10/stable-diffusion/Extensions/ultimate-upscale-for-automatic1111")


def run_patcher(patcher: Path, target: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(patcher), str(target)],
        check=check,
        capture_output=True,
        text=True,
    )


@pytest.fixture()
def multidiffusion_copy(tmp_path: Path) -> Path:
    if not INSTALLED_MD.exists():
        pytest.skip(f"installed MultiDiffusion fixture missing: {INSTALLED_MD}")
    target = tmp_path / "multidiffusion-upscaler-for-automatic1111"
    shutil.copytree(
        INSTALLED_MD,
        target,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
    )
    return target


@pytest.fixture()
def ultimate_copy(tmp_path: Path) -> Path:
    if not INSTALLED_UU.exists():
        pytest.skip(f"installed Ultimate Upscale fixture missing: {INSTALLED_UU}")
    target = tmp_path / "ultimate-upscale-for-automatic1111"
    shutil.copytree(
        INSTALLED_UU,
        target,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
    )
    return target


def multidiffusion_origins(module_path: Path, extent: int, tile: int, overlap: int) -> list[int]:
    source = module_path.read_text(encoding="utf-8")
    if "def _gb10_terminal_tile_origins" in source:
        match = re.search(
            r"def _gb10_terminal_tile_origins\(extent:int, tile:int, overlap:int\).*?(?=\n\ndef split_bboxes)",
            source,
            flags=re.S,
        )
        assert match, "patched terminal-origin helper not found"
        namespace: dict[str, object] = {}
        exec("from typing import List\n" + match.group(0), namespace)
        return namespace["_gb10_terminal_tile_origins"](extent, tile, overlap)  # type: ignore[index,operator]

    match = re.search(
        r"cols = math\.ceil\(\(w - overlap\) / \(tile_w - overlap\)\).*?x = min\(int\(col \* dx\), w - tile_w\)",
        source,
        flags=re.S,
    )
    assert match, "legacy MultiDiffusion origin calculation not found"
    import math

    cols = math.ceil((extent - overlap) / (tile - overlap))
    dx = (extent - tile) / (cols - 1) if cols > 1 else 0
    return [min(int(col * dx), extent - tile) for col in range(cols)]


def assert_axis_coverage(origins: list[int], extent: int, tile: int) -> None:
    assert origins == sorted(origins)
    assert len(origins) == len(set(origins))
    assert all(origin >= 0 for origin in origins)
    assert origins[0] == 0
    assert origins[-1] == max(0, extent - tile)
    covered = [False] * extent
    for origin in origins:
        for idx in range(origin, min(origin + tile, extent)):
            covered[idx] = True
    assert all(covered)


def test_multidiffusion_fixture_still_contains_proven_floor_rounding_gap(multidiffusion_copy: Path):
    utils = multidiffusion_copy / "tile_utils" / "utils.py"
    origins = multidiffusion_origins(utils, extent=555, tile=64, overlap=16)

    assert origins[-3:] == [401, 446, 490]
    assert 491 not in origins
    assert origins[-1] != 555 - 64


def test_multidiffusion_patcher_is_idempotent_and_covers_edges(multidiffusion_copy: Path):
    utils = multidiffusion_copy / "tile_utils" / "utils.py"
    first = run_patcher(MD_PATCHER, multidiffusion_copy)
    patched = utils.read_text(encoding="utf-8")
    second = run_patcher(MD_PATCHER, multidiffusion_copy)

    assert "Patched MultiDiffusion terminal tile origins" in first.stdout
    assert "already patched" in second.stdout
    assert utils.read_text(encoding="utf-8") == patched

    x_origins = multidiffusion_origins(utils, extent=555, tile=64, overlap=16)
    y_origins = multidiffusion_origins(utils, extent=427, tile=64, overlap=16)
    assert x_origins[-2:] == [480, 491]
    assert y_origins[-2:] == [336, 363]
    assert (x_origins[0], y_origins[0]) == (0, 0)
    assert (x_origins[-1], y_origins[-1]) == (491, 363)
    assert_axis_coverage(x_origins, 555, 64)
    assert_axis_coverage(y_origins, 427, 64)


def test_multidiffusion_patcher_preserves_normal_stride_grid(multidiffusion_copy: Path):
    utils = multidiffusion_copy / "tile_utils" / "utils.py"
    before = multidiffusion_origins(utils, extent=160, tile=64, overlap=16)
    run_patcher(MD_PATCHER, multidiffusion_copy)
    after = multidiffusion_origins(utils, extent=160, tile=64, overlap=16)

    assert before == [0, 48, 96]
    assert after == before


def test_multidiffusion_patcher_rejects_source_drift(tmp_path: Path):
    target = tmp_path / "multidiffusion-upscaler-for-automatic1111" / "tile_utils"
    target.mkdir(parents=True)
    source = target / "utils.py"
    source.write_text("def split_bboxes():\n    return []\n", encoding="utf-8")

    result = run_patcher(MD_PATCHER, source, check=False)

    assert result.returncode != 0
    assert "unsupported MultiDiffusion split_bboxes implementation" in result.stderr


class _State:
    def __init__(self) -> None:
        self.events: list[str] = []

    def begin(self) -> None:
        self.events.append("begin")

    def end(self) -> None:
        self.events.append("end")


def load_usdu_module(path: Path, state: _State):
    modules_pkg = types.ModuleType("modules")
    modules_pkg.__path__ = []
    modules_pkg.devices = types.SimpleNamespace(device="cpu")
    modules_pkg.scripts = types.SimpleNamespace(Script=object)
    shared_mod = types.ModuleType("modules.shared")
    shared_mod.state = state
    shared_mod.opts = types.SimpleNamespace()
    shared_mod.sd_upscalers = [types.SimpleNamespace(name="fixture", scaler=types.SimpleNamespace(upscale=lambda image, scale, data_path: image), data_path="")]
    processing_mod = types.ModuleType("modules.processing")
    processing_mod.StableDiffusionProcessing = object
    processing_mod.Processed = object
    images_mod = types.ModuleType("modules.images")
    images_mod.save_image = lambda *args, **kwargs: None
    gradio_mod = types.ModuleType("gradio")
    gradio_mod.Blocks = object
    gradio_mod.Dropdown = object
    gradio_mod.Slider = object
    gradio_mod.Accordion = object
    gradio_mod.Checkbox = object
    gradio_mod.HTML = object
    gradio_mod.Row = object
    gradio_mod.Radio = object

    sys.modules["modules"] = modules_pkg
    sys.modules["modules.shared"] = shared_mod
    sys.modules["modules.processing"] = processing_mod
    sys.modules["modules.images"] = images_mod
    sys.modules["gradio"] = gradio_mod

    spec = importlib.util.spec_from_file_location("ultimate_upscale_fixture", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["ultimate_upscale_fixture"] = module
    spec.loader.exec_module(module)
    return module


def make_usdu(module, *, redraw_enabled: bool = False, seams_enabled: bool = False, fail: bool = False):
    usdu = object.__new__(module.USDUpscaler)
    usdu.p = types.SimpleNamespace(extra_generation_params={})
    usdu.image = object()
    usdu.rows = 1
    usdu.cols = 1
    usdu.upscaler = types.SimpleNamespace(name="fixture")
    usdu.calc_jobs_count = lambda: None
    usdu.result_images = []
    usdu.initial_info = None
    usdu.save_image = lambda: None

    def redraw_start(*_args):
        if fail:
            raise RuntimeError("redraw boom")
        return object()

    usdu.redraw = types.SimpleNamespace(
        enabled=redraw_enabled,
        save=False,
        start=redraw_start,
        initial_info="redraw-info",
        tile_width=64,
        tile_height=64,
        padding=8,
    )
    usdu.seams_fix = types.SimpleNamespace(
        enabled=seams_enabled,
        save=False,
        start=lambda *_args: object(),
        initial_info="seams-info",
        mode=types.SimpleNamespace(name="NONE"),
    )
    return usdu


def test_ultimate_upscale_fixture_leaves_state_open_on_exception(ultimate_copy: Path):
    module_path = ultimate_copy / "scripts" / "ultimate-upscale.py"
    state = _State()
    module = load_usdu_module(module_path, state)
    usdu = make_usdu(module, redraw_enabled=True, fail=True)

    with pytest.raises(RuntimeError, match="redraw boom"):
        usdu.process()

    assert state.events == ["begin"]


def test_ultimate_upscale_patcher_idempotent_success_and_exception_lifecycle(ultimate_copy: Path):
    module_path = ultimate_copy / "scripts" / "ultimate-upscale.py"
    first = run_patcher(UU_PATCHER, ultimate_copy)
    patched = module_path.read_text(encoding="utf-8")
    second = run_patcher(UU_PATCHER, ultimate_copy)

    assert "Patched Ultimate Upscale state lifecycle" in first.stdout
    assert "already patched" in second.stdout
    assert module_path.read_text(encoding="utf-8") == patched
    assert patched.count("state.end()") == 1
    assert "finally:\n            state.end()" in patched

    success_state = _State()
    module = load_usdu_module(module_path, success_state)
    make_usdu(module).process()
    assert success_state.events == ["begin", "end"]

    exception_state = _State()
    module = load_usdu_module(module_path, exception_state)
    usdu = make_usdu(module, redraw_enabled=True, fail=True)
    with pytest.raises(RuntimeError, match="redraw boom"):
        usdu.process()
    assert exception_state.events == ["begin", "end"]


def test_ultimate_upscale_patcher_rejects_source_drift(tmp_path: Path):
    target = tmp_path / "ultimate-upscale-for-automatic1111" / "scripts"
    target.mkdir(parents=True)
    source = target / "ultimate-upscale.py"
    source.write_text("class USDUpscaler:\n    def process(self):\n        pass\n", encoding="utf-8")

    result = run_patcher(UU_PATCHER, source, check=False)

    assert result.returncode != 0
    assert "unsupported Ultimate Upscale process lifecycle implementation" in result.stderr


def test_run_sh_patches_optional_tiled_extensions_after_extension_sync_before_container_start():
    run_sh = (ROOT / "gb10" / "run.sh").read_text(encoding="utf-8")

    assert 'MULTIDIFFUSION_ROOT="${HOST_ROOT}/Extensions/multidiffusion-upscaler-for-automatic1111"' in run_sh
    assert 'if [[ -d "${MULTIDIFFUSION_ROOT}" ]]; then' in run_sh
    assert "patch-multidiffusion-terminal-tiles.py" in run_sh
    assert 'ULTIMATE_UPSCALE_ROOT="${HOST_ROOT}/Extensions/ultimate-upscale-for-automatic1111"' in run_sh
    assert 'if [[ -d "${ULTIMATE_UPSCALE_ROOT}" ]]; then' in run_sh
    assert "patch-ultimate-upscale-state-lifecycle.py" in run_sh

    extension_sync = run_sh.index('sudo rsync -a --delete --delete-excluded')
    multidiffusion_patch = run_sh.index("patch-multidiffusion-terminal-tiles.py")
    ultimate_patch = run_sh.index("patch-ultimate-upscale-state-lifecycle.py")
    container_start = run_sh.index('sudo "$DOCKER_BIN" run "${DOCKER_ARGS[@]}"')
    assert extension_sync < multidiffusion_patch < ultimate_patch < container_start
