"""initialize.imports(), the first step of startup: it fails without a usable CUDA device unless the command line chose
another device."""
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from modules import initialize
from test.helpers import ROOT


def _args(**overrides):
    return SimpleNamespace(**{"skip_torch_cuda_test": False, "use_ipex": False, "use_cpu": [], **overrides})


@pytest.mark.parametrize("overrides", [{"skip_torch_cuda_test": True}, {"use_ipex": True}, {"use_cpu": ["all"]}, {"use_cpu": ["sd", "all"]}])
def test_an_explicit_device_choice_starts_without_cuda(monkeypatch, overrides):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    initialize.require_torch_device(_args(**overrides))


@pytest.mark.parametrize("overrides", [{}, {"use_cpu": ["sd", "interrogate"]}])
def test_startup_without_cuda_fails_unless_every_task_runs_on_the_cpu(monkeypatch, overrides):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="--use-cpu all"):
        initialize.require_torch_device(_args(**overrides))


def test_startup_with_cuda_needs_no_flag(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    initialize.require_torch_device(_args())


def _run_imports(*flags):
    """initialize.imports() in a fresh interpreter, the way webui.py runs it, with `flags` as the command line."""
    code = (
        "from modules import initialize\n"
        "initialize.imports()\n"
        "print('imports ok')\n"
    )
    env = {**os.environ, "IGNORE_CMD_ARGS_ERRORS": "1", "COMMANDLINE_ARGS": ""}
    return subprocess.run([sys.executable, "-c", code, *flags], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the startup failure on a host without CUDA")
def test_real_startup_imports_fail_without_cuda_and_a_device_flag():
    result = _run_imports()

    assert result.returncode != 0
    assert "RuntimeError: Torch cannot use a CUDA device" in result.stderr
    assert "imports ok" not in result.stdout


def test_real_startup_imports_complete_with_a_device_flag_and_no_cuda():
    result = _run_imports("--skip-torch-cuda-test")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "imports ok" in result.stdout
