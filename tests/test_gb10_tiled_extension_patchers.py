from __future__ import annotations

import importlib.util
import re
import resource
import shutil
import signal
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
MD_PATCHER = ROOT / "gb10" / "patch-multidiffusion-terminal-tiles.py"
UU_PATCHER = ROOT / "gb10" / "patch-ultimate-upscale-state-lifecycle.py"
SUBCANVAS_PATCHER = ROOT / "gb10" / "patch-ultimate-upscale-subcanvas.py"


def installed_extension(name: str) -> Path | None:
    """The host deploy root, or the same checkout where run.sh mounts it inside a webui container."""
    return next((path for path in (Path("/opt/gb10/stable-diffusion/Extensions") / name, ROOT / "extensions" / name) if path.is_dir()), None)


INSTALLED_MD = installed_extension("multidiffusion-upscaler-for-automatic1111")
INSTALLED_UU = installed_extension("ultimate-upscale-for-automatic1111")


def load_patcher(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MD_MODULE = load_patcher(MD_PATCHER, "gb10_patch_multidiffusion_terminal_tiles")
UU_MODULE = load_patcher(UU_PATCHER, "gb10_patch_ultimate_upscale_state_lifecycle")
SUBCANVAS_MODULE = load_patcher(SUBCANVAS_PATCHER, "gb10_patch_ultimate_upscale_subcanvas_for_lifecycle")


def run_patcher(patcher: Path, target: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(patcher), str(target)],
        check=check,
        capture_output=True,
        text=True,
    )


def unpatch_lifecycle(source: str) -> str:
    """Exact inverse of patch-ultimate-upscale-state-lifecycle.py's re-indenting try/finally rewrite."""
    lines = source.splitlines(keepends=True)
    marker = next(i for i, line in enumerate(lines) if line.strip() == f"# {UU_MODULE.MARKER}: one begin owns one end.")
    start = marker - 1
    indent = lines[start][: len(lines[start]) - len(lines[start].lstrip(" "))]
    assert lines[start] == f"{indent}try:\n"
    end = lines.index(f"{indent}finally:\n", marker)
    assert lines[end + 1] == f"{indent}    state.end()\n"
    body = [indent + line[len(indent) + 4 :] if line.strip() else line for line in lines[marker + 1 : end]]
    return "".join(lines[:start] + body + [f"{indent}state.end()\n"] + lines[end + 2 :])


@pytest.fixture()
def multidiffusion_copy(tmp_path: Path) -> Path:
    """Upstream split_bboxes: the installed checkout with the terminal-tiles patch reversed (round trip asserted)."""
    if INSTALLED_MD is None:
        pytest.skip("installed MultiDiffusion fixture missing")
    target = tmp_path / "multidiffusion-upscaler-for-automatic1111"
    shutil.copytree(INSTALLED_MD, target, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
    utils = target / "tile_utils" / "utils.py"
    installed = utils.read_bytes()
    if MD_MODULE.PATCHED in installed.decode("utf-8"):
        utils.write_bytes(installed.decode("utf-8").replace(MD_MODULE.PATCHED, MD_MODULE.ORIGINAL, 1).encode("utf-8"))
        probe = tmp_path / "roundtrip" / "tile_utils" / "utils.py"
        probe.parent.mkdir(parents=True)
        probe.write_bytes(utils.read_bytes())
        run_patcher(MD_PATCHER, probe)
        assert probe.read_bytes() == installed
    return target


@pytest.fixture()
def ultimate_copy(tmp_path: Path) -> Path:
    """Upstream ultimate-upscale.py: the installed script with the sub-canvas and lifecycle patches reversed."""
    if INSTALLED_UU is None:
        pytest.skip("installed Ultimate Upscale fixture missing")
    target = tmp_path / "ultimate-upscale-for-automatic1111"
    shutil.copytree(INSTALLED_UU, target, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
    script = target / "scripts" / "ultimate-upscale.py"
    installed = script.read_bytes()
    text = installed.decode("utf-8")
    if SUBCANVAS_MODULE.MARKER in text:
        for original, patched, count in reversed(SUBCANVAS_MODULE.BLOCKS):
            assert text.count(patched) == count
            text = text.replace(patched, original)
    if UU_MODULE.MARKER in text:
        text = unpatch_lifecycle(text)
    script.write_bytes(text.encode("utf-8"))
    if script.read_bytes() != installed:
        probe = tmp_path / "roundtrip.py"
        probe.write_bytes(script.read_bytes())
        run_patcher(UU_PATCHER, probe)
        if SUBCANVAS_MODULE.MARKER in installed.decode("utf-8"):
            run_patcher(SUBCANVAS_PATCHER, probe)
        assert probe.read_bytes() == installed
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


def load_usdu_module(path: Path, state: _State, monkeypatch):
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

    # monkeypatch restores the real webui modules for later tests.
    monkeypatch.setitem(sys.modules, "modules", modules_pkg)
    monkeypatch.setitem(sys.modules, "modules.shared", shared_mod)
    monkeypatch.setitem(sys.modules, "modules.processing", processing_mod)
    monkeypatch.setitem(sys.modules, "modules.images", images_mod)
    monkeypatch.setitem(sys.modules, "gradio", gradio_mod)

    spec = importlib.util.spec_from_file_location("ultimate_upscale_fixture", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "ultimate_upscale_fixture", module)
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


def test_ultimate_upscale_fixture_leaves_state_open_on_exception(ultimate_copy: Path, monkeypatch):
    module_path = ultimate_copy / "scripts" / "ultimate-upscale.py"
    state = _State()
    module = load_usdu_module(module_path, state, monkeypatch)
    usdu = make_usdu(module, redraw_enabled=True, fail=True)

    with pytest.raises(RuntimeError, match="redraw boom"):
        usdu.process()

    assert state.events == ["begin"]


def test_ultimate_upscale_patcher_idempotent_success_and_exception_lifecycle(ultimate_copy: Path, monkeypatch):
    module_path = ultimate_copy / "scripts" / "ultimate-upscale.py"
    first = run_patcher(UU_PATCHER, ultimate_copy)
    patched = module_path.read_text(encoding="utf-8")
    second = run_patcher(UU_PATCHER, ultimate_copy)

    assert "Patched Ultimate Upscale state lifecycle" in first.stdout
    assert "Patched" not in second.stdout and "lifecycle verified" in second.stdout
    assert module_path.read_text(encoding="utf-8") == patched
    assert patched.count("state.end()") == 1
    assert "finally:\n            state.end()" in patched

    success_state = _State()
    module = load_usdu_module(module_path, success_state, monkeypatch)
    make_usdu(module).process()
    assert success_state.events == ["begin", "end"]

    exception_state = _State()
    module = load_usdu_module(module_path, exception_state, monkeypatch)
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
    assert "unsupported or partial Ultimate Upscale state lifecycle" in result.stderr


def test_ultimate_upscale_patcher_check_mode_and_failure_injection(tmp_path: Path):
    original = '''class Fixture:\n    def process(self):\n        state.begin()\n        if self.redraw.enabled:\n            self.image = self.redraw.start(self.p, self.image, self.rows, self.cols)\n        state.end()\n'''
    target = tmp_path / "ultimate-upscale.py"
    target.write_text(original)
    run_patcher(UU_PATCHER, target)
    first = target.read_bytes()
    run_patcher(UU_PATCHER, target)
    assert target.read_bytes() == first
    subprocess.run([sys.executable, str(UU_PATCHER), str(target), "--check"], check=True, capture_output=True, text=True)

    source = target.read_text()
    events = []
    namespace = {
        "state": types.SimpleNamespace(begin=lambda: events.append("begin"), end=lambda: events.append("end")),
        "USDURedrawMode": types.SimpleNamespace(LINEAR=1, CHESS=2, NONE=3),
        "USDUSFMode": types.SimpleNamespace(NONE=0),
    }
    exec(compile(source, "<fixture>", "exec"), namespace)
    redraw = types.SimpleNamespace(enabled=True, mode=1, start=lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))
    obj = types.SimpleNamespace(redraw=redraw, seams_fix=types.SimpleNamespace(enabled=False), p=None, image=None, rows=1, cols=1)
    with pytest.raises(RuntimeError, match="boom"):
        namespace["Fixture"].process(obj)
    assert events == ["begin", "end"]

    partial = tmp_path / "partial.py"
    partial.write_text(source.replace("finally:\n            state.end()", "state.end()"))
    result = subprocess.run([sys.executable, str(UU_PATCHER), str(partial), "--check"], capture_output=True, text=True)
    assert result.returncode != 0


def test_run_sh_patches_optional_tiled_extensions_after_extension_sync_before_container_start():
    run_sh = (ROOT / "gb10" / "run.sh").read_text(encoding="utf-8")

    assert 'MULTIDIFFUSION_ROOT="${HOST_ROOT}/Extensions/multidiffusion-upscaler-for-automatic1111"' in run_sh
    assert 'if [[ -d "${MULTIDIFFUSION_ROOT}" ]]; then' in run_sh
    assert "patch-multidiffusion-terminal-tiles.py" in run_sh
    assert 'ULTIMATE_UPSCALE_ROOT="${HOST_ROOT}/Extensions/ultimate-upscale-for-automatic1111"' in run_sh
    assert 'if [[ ! -f "${ULTIMATE_UPSCALE_ROOT}/scripts/ultimate-upscale.py" ]]; then' in run_sh
    assert "patch-ultimate-upscale-state-lifecycle.py" in run_sh

    extension_sync = run_sh.index('sudo rsync -a --checksum --delete --delete-excluded')
    multidiffusion_patch = run_sh.index("patch-multidiffusion-terminal-tiles.py")
    ultimate_patch = run_sh.index("patch-ultimate-upscale-state-lifecycle.py")
    container_start = run_sh.index('sudo "$DOCKER_BIN" run "${DOCKER_ARGS[@]}"')
    assert extension_sync < multidiffusion_patch < ultimate_patch < container_start


def test_lifecycle_patcher_fails_closed_on_reindent_hazards_and_crlf_without_writing(tmp_path: Path):
    # A continuation line shallower than the body (here inside a multi-line string) cannot be re-indented as text.
    hazard = (
        "class Fixture:\n    def process(self):\n        state.begin()\n"
        "        self.note = \"\"\"first\n        second\"\"\"\n        state.end()\n"
    )
    for name, payload in (("hazard.py", hazard.encode("utf-8")), ("crlf.py", hazard.replace("\n", "\r\n").encode("utf-8"))):
        target = tmp_path / name
        target.write_bytes(payload)
        result = run_patcher(UU_PATCHER, target, check=False)
        assert result.returncode != 0, name
        assert ("re-indentation changed its statements" if name == "hazard.py" else "line endings") in result.stderr
        assert target.read_bytes() == payload, name


def test_terminal_tiles_patcher_rejects_crlf_and_partial_helper_without_writing(tmp_path: Path):
    utils = tmp_path / "tile_utils" / "utils.py"
    utils.parent.mkdir()
    for payload, message in (
        (MD_MODULE.ORIGINAL.replace("\n", "\r\n").encode("utf-8"), "CRLF"),
        (("def _gb10_terminal_tile_origins():\n    pass\n\n" + MD_MODULE.ORIGINAL).encode("utf-8"), "unsupported MultiDiffusion split_bboxes"),
        ((MD_MODULE.PATCHED + MD_MODULE.PATCHED).encode("utf-8"), "ambiguous"),
    ):
        utils.write_bytes(payload)
        result = run_patcher(MD_PATCHER, utils, check=False)
        assert result.returncode != 0 and message in result.stderr
        assert utils.read_bytes() == payload

    utils.write_bytes(MD_MODULE.ORIGINAL.encode("utf-8"))
    run_patcher(MD_PATCHER, utils)
    assert utils.read_bytes() == MD_MODULE.PATCHED.encode("utf-8")


def run_patcher_with_file_size_limit(patcher: Path, target: Path, limit: int) -> subprocess.CompletedProcess[str]:
    """Run a patcher with RLIMIT_FSIZE=limit and SIGXFSZ ignored: a write past `limit` bytes fails with EFBIG
    partway through, as on a full disk."""
    def limit_file_size():
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))

    return subprocess.run([sys.executable, str(patcher), str(target)], capture_output=True, text=True, preexec_fn=limit_file_size)


def directory_snapshot(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


LIFECYCLE_FIXTURE = (
    "class Fixture:\n    def process(self):\n        state.begin()\n"
    "        if self.redraw.enabled:\n            self.image = self.redraw.start(self.p, self.image, self.rows, self.cols)\n"
    "        state.end()\n"
)


@pytest.mark.parametrize("patcher, relative, payload", [
    (MD_PATCHER, "tile_utils/utils.py", MD_MODULE.ORIGINAL),
    (UU_PATCHER, "scripts/ultimate-upscale.py", LIFECYCLE_FIXTURE),
], ids=["terminal-tiles", "lifecycle"])
def test_patcher_write_failing_midway_leaves_the_target_intact(tmp_path: Path, patcher: Path, relative: str, payload: str):
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(payload.encode("utf-8"))
    target.chmod(0o640)
    before = directory_snapshot(tmp_path)

    result = run_patcher_with_file_size_limit(patcher, target, 64)

    assert result.returncode != 0 and "File too large" in result.stderr
    assert directory_snapshot(tmp_path) == before  # no truncated target, no leftover temporary file
    run_patcher(patcher, target)
    assert target.read_bytes() != payload.encode("utf-8")
    assert target.stat().st_mode & 0o777 == 0o640
    assert not list(tmp_path.rglob("*.gb10-tmp"))


def test_terminal_tiles_patcher_verifies_the_patched_text_before_writing(tmp_path: Path):
    utils = tmp_path / "tile_utils" / "utils.py"
    utils.parent.mkdir()
    payload = (MD_MODULE.ORIGINAL + "\ndef broken(:\n").encode("utf-8")
    utils.write_bytes(payload)

    result = run_patcher(MD_PATCHER, utils, check=False)

    assert result.returncode != 0 and "verification failed (invalid Python)" in result.stderr
    assert utils.read_bytes() == payload
