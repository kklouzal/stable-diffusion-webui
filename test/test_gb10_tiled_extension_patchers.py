"""gb10/patch-ultimate-upscale-state-lifecycle.py against the installed Ultimate SD Upscale checkout.

The shared fail-closed engine (atomic writes, CRLF, missing files) is tested in test_gb10_patchlib.py, the MultiDiffusion
patcher in test_gb10_multidiffusion_performance_patcher.py and the run.sh order in test_gb10_run_patcher_order.py.
"""
from __future__ import annotations

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


def installed_extension(name: str) -> Path | None:
    """The host deploy root, or the same checkout where run.sh mounts it inside a webui container."""
    return next((path for path in (Path("/opt/gb10/stable-diffusion/Extensions") / name, ROOT / "extensions" / name) if path.is_dir()), None)


INSTALLED_UU = installed_extension("ultimate-upscale-for-automatic1111")


# The patchers import patchlib as a sibling module, as under run.sh.
PATCHLIB = {"patchlib": load_source("patchlib", GB10 / "patchlib.py")}
UU_MODULE = load_source("gb10_patch_ultimate_upscale_state_lifecycle", UU_PATCHER, PATCHLIB)
SUBCANVAS_MODULE = load_source("gb10_patch_ultimate_upscale_subcanvas_for_lifecycle", SUBCANVAS_PATCHER, PATCHLIB)


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
        for block in reversed(SUBCANVAS_MODULE.BLOCKS):
            assert text.count(block.patched) == block.count
            text = text.replace(block.patched, block.original)
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


class _State:
    def __init__(self) -> None:
        self.events: list[str] = []

    def begin(self) -> None:
        self.events.append("begin")

    def end(self) -> None:
        self.events.append("end")


def load_usdu_module(path: Path, state: _State):
    return load_source("ultimate_upscale_fixture", path, {
        "modules": module("modules", package=True, devices=types.SimpleNamespace(device="cpu"), scripts=types.SimpleNamespace(Script=object)),
        "modules.shared": module(
            "modules.shared",
            state=state,
            opts=types.SimpleNamespace(),
            sd_upscalers=[types.SimpleNamespace(name="fixture", scaler=types.SimpleNamespace(upscale=lambda image, scale, data_path: image), data_path="")],
        ),
        "modules.processing": module("modules.processing", StableDiffusionProcessing=object, Processed=object),
        "modules.images": module("modules.images", save_image=lambda *args, **kwargs: None),
        "gradio": module(
            "gradio", Blocks=object, Dropdown=object, Slider=object, Accordion=object, Checkbox=object, HTML=object, Row=object, Radio=object),
    })


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
    assert "Patched" not in second.stdout and "lifecycle verified" in second.stdout
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


@pytest.mark.parametrize("patcher", [UU_PATCHER, SUBCANVAS_PATCHER], ids=["lifecycle", "subcanvas"])
def test_ultimate_upscale_patchers_fail_closed_on_a_missing_script(tmp_path: Path, patcher: Path):
    # run.sh has no separate existence check: a missing (or never installed) script must abort the deploy here.
    for target in (tmp_path / "ultimate-upscale-for-automatic1111", tmp_path):
        result = run_patcher(patcher, target, check=False)
        assert result.returncode != 0 and "source not found" in result.stderr
