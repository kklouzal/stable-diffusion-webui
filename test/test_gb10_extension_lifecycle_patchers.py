from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTROLNET_PATCHER = ROOT / "gb10" / "patch-controlnet-hook-restore.py"
UPSCALE_PATCHER = ROOT / "gb10" / "patch-ultimate-upscale-state-lifecycle.py"


def load_constants(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def run_patcher(patcher: Path, target: Path, *args: str, success: bool = True):
    result = subprocess.run(
        [sys.executable, str(patcher), str(target), *args],
        text=True,
        capture_output=True,
    )
    assert (result.returncode == 0) is success, result.stdout + result.stderr
    return result


def test_controlnet_patcher_unpatched_patched_idempotent_and_partial(tmp_path):
    c = load_constants(CONTROLNET_PATCHER)
    target = tmp_path / "hook.py"
    target.write_text("\n".join((c.INIT_ORIGINAL, c.FORWARD_ORIGINAL, c.INSTALL_ORIGINAL, c.RESTORE_ORIGINAL)))
    run_patcher(CONTROLNET_PATCHER, target)
    first = target.read_bytes()
    run_patcher(CONTROLNET_PATCHER, target)
    assert target.read_bytes() == first
    run_patcher(CONTROLNET_PATCHER, target, "--check")

    partial = tmp_path / "partial.py"
    partial.write_text("\n".join((c.INIT_PATCHED, c.FORWARD_ORIGINAL, c.INSTALL_ORIGINAL, c.RESTORE_ORIGINAL)))
    run_patcher(CONTROLNET_PATCHER, partial, "--check", success=False)


def compile_controlnet_methods(source: str):
    module = ast.parse("class Fixture:\n" + "\n".join("    " + line for line in source.splitlines()))
    namespace = {
        "UNetModel": object,
        "scripts": SimpleNamespace(script_callbacks=SimpleNamespace(remove_callbacks_for_function=lambda _: None)),
    }
    exec(compile(module, "<fixture>", "exec"), namespace)
    return namespace["Fixture"]


def test_controlnet_owner_install_restore_and_baseline_identity():
    c = load_constants(CONTROLNET_PATCHER)
    install_body = "outer = self\n" + textwrap.dedent(c.INSTALL_PATCHED)
    install = textwrap.indent(install_body, "    ")
    restore = textwrap.dedent(c.RESTORE_PATCHED)
    fixture = compile_controlnet_methods(
        "def install(self, model, forward_webui):\n" + install + "\n" + restore
    )

    def baseline(*args, **kwargs):
        return args, kwargs

    model = SimpleNamespace(forward=baseline)
    owner = SimpleNamespace(_forward_hook_owner_token=object(), _forward_hook_wrapper=None)
    fixture.install(owner, model, lambda *args, **kwargs: None)
    wrapper = model.forward
    fixture.install(owner, model, lambda *args, **kwargs: None)
    assert model.forward is wrapper
    assert owner.original_forward is baseline

    other = SimpleNamespace(_forward_hook_owner_token=object(), _forward_hook_wrapper=None)
    with pytest.raises(RuntimeError):
        fixture.install(other, model, lambda *args, **kwargs: None)
    other.model = model
    other.guidance_schedule_handler = lambda _: None
    other.control_params = []
    fixture.restore(other)
    assert model.forward is wrapper

    owner.model = model
    owner.guidance_schedule_handler = lambda _: None
    owner.control_params = []
    fixture.restore(owner)
    assert model.forward is baseline


def test_ultimate_upscale_patcher_and_failure_injection(tmp_path):
    original = '''class Fixture:\n    def process(self):\n        state.begin()\n        if self.redraw.enabled:\n            self.image = self.redraw.start(self.p, self.image, self.rows, self.cols)\n        state.end()\n'''
    target = tmp_path / "ultimate-upscale.py"
    target.write_text(original)
    run_patcher(UPSCALE_PATCHER, target)
    first = target.read_bytes()
    run_patcher(UPSCALE_PATCHER, target)
    assert target.read_bytes() == first
    run_patcher(UPSCALE_PATCHER, target, "--check")

    source = target.read_text()
    module = ast.parse(source)
    events = []
    state = SimpleNamespace(begin=lambda: events.append("begin"), end=lambda: events.append("end"))
    namespace = {
        "state": state,
        "USDURedrawMode": SimpleNamespace(LINEAR=1, CHESS=2, NONE=3),
        "USDUSFMode": SimpleNamespace(NONE=0),
    }
    exec(compile(module, "<fixture>", "exec"), namespace)
    redraw = SimpleNamespace(enabled=True, mode=1, start=lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))
    obj = SimpleNamespace(redraw=redraw, seams_fix=SimpleNamespace(enabled=False), p=None, image=None, rows=1, cols=1)
    with pytest.raises(RuntimeError, match="boom"):
        namespace["Fixture"].process(obj)
    assert events == ["begin", "end"]

    partial = tmp_path / "partial.py"
    partial.write_text(source.replace("finally:\n            state.end()", "state.end()"))
    run_patcher(UPSCALE_PATCHER, partial, "--check", success=False)
