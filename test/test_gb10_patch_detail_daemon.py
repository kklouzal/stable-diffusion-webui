"""gb10/patch-detail-daemon.py against the installed Detail Daemon checkout, and the patched sigma guard's behaviour.

The shared fail-closed engine (atomic writes, CRLF, missing files) is tested in test_gb10_patchlib.py and the run.sh
order in test_gb10_run_patcher_order.py.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from test.helpers import load_source, module

ROOT = Path(__file__).parents[1]
PATCHER = ROOT / "gb10" / "patch-detail-daemon.py"
# sha256 of scripts/detail_daemon.py at muerrilla/sd-webui-detail-daemon master 1947999, the patcher's ORIGINAL.
UPSTREAM_SHA256 = "76354478f021a69ce3e4831f821fe6747c4d69afe3108a36a8172e7817d87453"
INSTALLED = next((path for path in (Path("/opt/gb10/stable-diffusion/Extensions/sd-webui-detail-daemon"), ROOT / "extensions" / "sd-webui-detail-daemon") if path.is_dir()), None)

PATCHLIB = load_source("patchlib", ROOT / "gb10" / "patchlib.py")
PATCH_MODULE = load_source("gb10_patch_detail_daemon", PATCHER, {"patchlib": PATCHLIB})


def run_patcher(target: Path, *extra: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(PATCHER), str(target), *extra], check=check, capture_output=True, text=True)


@pytest.fixture()
def daemon_copy(tmp_path: Path) -> Path:
    """Upstream 1947999 detail_daemon.py, derived from the installed script (reverting this patch if deployed)."""
    if INSTALLED is None:
        pytest.skip("installed Detail Daemon fixture missing")
    target = tmp_path / "sd-webui-detail-daemon"
    shutil.copytree(INSTALLED, target, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
    script = target / "scripts" / "detail_daemon.py"
    text = script.read_bytes().decode("utf-8")
    script.write_bytes((PATCHLIB._revert_previous(text, PATCH_MODULE.BLOCKS) or text).encode("utf-8"))
    assert hashlib.sha256(script.read_bytes()).hexdigest() == UPSTREAM_SHA256
    return target


def test_patcher_patches_upstream_once_and_verifies_the_patched_text(daemon_copy: Path):
    script = daemon_copy / "scripts" / "detail_daemon.py"
    upstream = script.read_bytes()
    missing = run_patcher(daemon_copy, "--check", check=False)
    assert missing.returncode != 0 and "patch missing" in missing.stderr and script.read_bytes() == upstream
    first = run_patcher(daemon_copy)
    patched = script.read_bytes()
    second = run_patcher(daemon_copy)
    run_patcher(script, "--check")
    assert "Patched Detail Daemon sigma guard" in first.stdout
    assert "Patched" not in second.stdout and "sigma guard verified" in second.stdout
    assert script.read_bytes() == patched != upstream
    assert patched.count(PATCH_MODULE.MARKER.encode()) == 1
    installed = (INSTALLED / "scripts" / "detail_daemon.py").read_bytes()
    assert installed in (upstream, patched)  # a fresh install or this release


def test_patcher_rejects_drift_crlf_partial_and_missing_sources_without_writing(daemon_copy: Path, tmp_path: Path):
    source = (daemon_copy / "scripts" / "detail_daemon.py").read_text(encoding="utf-8")
    block = PATCH_MODULE.BLOCKS[0]
    for name, text in (
        ("drift", source.replace("self.make_schedule(actual_steps, ", "self.make_schedule(steps, ")),
        ("crlf", source.replace("\n", "\r\n")),
        ("stray-marker", source + f"# {PATCH_MODULE.MARKER}\n"),
        ("twice", source.replace(block.original, block.patched) + block.patched),
    ):
        target = tmp_path / f"{name}.py"
        target.write_bytes(text.encode("utf-8"))
        result = run_patcher(target, check=False)
        assert result.returncode != 0, name
        assert ("line endings" if name == "crlf" else "partially patched") in result.stderr, name
        assert target.read_bytes() == text.encode("utf-8"), name
    for target in (tmp_path / "missing" / "sd-webui-detail-daemon", tmp_path / "empty"):
        target.mkdir(parents=True)
        result = run_patcher(target, check=False)
        assert result.returncode != 0 and "source not found" in result.stderr


# ------------------------------------------------------------------------------------------------- behaviour


def load_daemon(path: Path):
    noop = lambda *args, **kwargs: None  # noqa: E731
    scripts = module("modules.scripts", Script=object, AlwaysVisible=object(), scripts_data=[])
    callbacks = module("modules.script_callbacks", on_cfg_denoiser=noop, remove_callbacks_for_function=noop,
                       on_infotext_pasted=noop, on_ui_settings=noop)
    shared = module("modules.shared", opts=types.SimpleNamespace(data={}))
    return load_source("detail_daemon_fixture", path, {
        "modules": module("modules", package=True, scripts=scripts, script_callbacks=callbacks, shared=shared),
        "modules.scripts": scripts,
        "modules.script_callbacks": callbacks,
        "modules.ui_components": module("modules.ui_components", InputAccordion=object),
        "modules.shared": shared,
        "gradio": module("gradio"),
    })


def run_daemon(path: Path, *, mode: str, amount: float, cfg: float, steps: int = 12, batch: int = 1):
    """process() one daemon (UI defaults besides mode/amount), then one denoiser call per step; returns each step's
    sigma as the model would see it (cond rows first, then uncond rows)."""
    dd = load_daemon(path)
    script = dd.Script()
    script.tab_param_count = 12
    p = types.SimpleNamespace(sampler_name="Euler", extra_generation_params={}, cfg_scale=cfg, batch_size=batch)
    script.process(p, True, True, False, mode, 0.2, 0.8, 0.5, amount, 1.0, 0.0, 0.0, 0.0, True)
    sigmas = []
    for step in range(steps):
        denoiser = types.SimpleNamespace(step=step, total_steps=steps, steps=steps)
        params = types.SimpleNamespace(sigma=torch.full((2 * batch,), 14.6 - step, dtype=torch.float32),
                                       sampling_step=step, total_sampling_steps=steps, denoiser=denoiser)
        script.denoiser_callback(params)
        sigmas.append(params.sigma.clone())
    return torch.stack(sigmas)


@pytest.fixture()
def scripts_pair(daemon_copy: Path, tmp_path: Path) -> tuple[Path, Path]:
    upstream = daemon_copy / "scripts" / "detail_daemon.py"
    patched = tmp_path / "patched.py"
    shutil.copyfile(upstream, patched)
    run_patcher(patched)
    return upstream, patched


@pytest.mark.parametrize("mode, amount, cfg", [
    ("both", 0.1, 7.0), ("both", 0.5, 5.0), ("both", -0.5, 7.0), ("both", 1.99, 5.0),
    ("cond", 0.5, 7.0), ("cond", 9.99, 7.0), ("uncond", 0.5, 7.0), ("uncond", -9.99, 7.0), ("uncond", 50.0, 7.0),
])
def test_valid_requests_get_upstream_sigmas_bit_for_bit(scripts_pair, mode, amount, cfg):
    upstream, patched = scripts_pair
    expected = run_daemon(upstream, mode=mode, amount=amount, cfg=cfg, batch=2)
    assert (expected > 0).all()
    assert torch.equal(run_daemon(patched, mode=mode, amount=amount, cfg=cfg, batch=2), expected)


@pytest.mark.parametrize("mode, amount, cfg", [
    ("both", 2.0, 5.0),     # factor 1 - 0.2*5 == 0 at the peak step
    ("both", 1.5, 7.0),     # the production CFG range: amount 1.5 already reverses the sigma
    ("both", -2.0, -5.0),   # a negative CFG flips the sign of the bound
    ("cond", 10.0, 7.0),
    ("uncond", -10.0, 7.0),
    ("uncond", -25.0, 7.0),
])
def test_a_schedule_that_drives_sigma_non_positive_fails_before_any_sigma_changes(scripts_pair, mode, amount, cfg):
    upstream, patched = scripts_pair
    assert (run_daemon(upstream, mode=mode, amount=amount, cfg=cfg) <= 0).any()  # upstream feeds the model sigma <= 0
    dd = load_daemon(patched)
    script = dd.Script()
    script.tab_param_count = 12
    p = types.SimpleNamespace(sampler_name="Euler", extra_generation_params={}, cfg_scale=cfg, batch_size=1)
    script.process(p, True, True, False, mode, 0.2, 0.8, 0.5, amount, 1.0, 0.0, 0.0, 0.0, True)
    sigma = torch.tensor([14.6, 14.6])
    params = types.SimpleNamespace(sigma=sigma.clone(), sampling_step=0, total_sampling_steps=12,
                                   denoiser=types.SimpleNamespace(step=0, total_steps=12, steps=12))
    with pytest.raises(ValueError, match=rf"Detail Daemon: Daemon 1 \({mode} mode, amount {amount}\).*sigma would be <= 0"):
        script.denoiser_callback(params)
    assert torch.equal(params.sigma, sigma)
    assert script.daemon_data[0]["schedule"] is None  # an invalid schedule is never stored
