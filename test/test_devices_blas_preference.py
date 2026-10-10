"""OPENCLAW_PREFER_CUBLASLT (modules/devices.py): read once when devices is imported; 1 selects cuBLASLt for torch's
CUDA GEMMs, unset/0 leaves torch's default, a malformed value fails the import (startup)."""
import os
import subprocess
import sys

from test.helpers import ROOT

_PROBE = "from modules import devices; import torch; print(torch.backends.cuda.preferred_blas_library())"


def _import_devices(value):
    env = {key: item for key, item in os.environ.items() if key not in ("OPENCLAW_PREFER_CUBLASLT", "TORCH_BLAS_PREFER_CUBLASLT")}
    env["IGNORE_CMD_ARGS_ERRORS"] = "1"
    if value is not None:
        env["OPENCLAW_PREFER_CUBLASLT"] = value
    return subprocess.run([sys.executable, "-c", _PROBE], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)


def test_switch_selects_cublaslt_only_when_on():
    for value, expected in ((None, "_BlasBackend.Cublas"), ("0", "_BlasBackend.Cublas"), ("1", "_BlasBackend.Cublaslt"), ("on", "_BlasBackend.Cublaslt")):
        process = _import_devices(value)
        assert process.returncode == 0, process.stderr[-3000:]
        assert process.stdout.strip().splitlines()[-1] == expected, (value, process.stdout)


def test_malformed_value_fails_the_import():
    process = _import_devices("cublaslt")
    assert process.returncode != 0 and "OPENCLAW_PREFER_CUBLASLT='cublaslt' is not a boolean" in process.stderr
