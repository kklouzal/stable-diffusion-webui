#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import pathlib
import py_compile
import subprocess
import sys


def main() -> int:
    spec = importlib.util.find_spec("torch")
    if spec is None or spec.origin is None:
        raise SystemExit("torch is not installed; cannot patch torch.utils.mkldnn")

    torch_root = pathlib.Path(spec.origin).resolve().parent
    target = torch_root / "utils" / "mkldnn.py"
    if not target.exists():
        raise SystemExit(f"torch MKLDNN helper is missing: {target}")

    original = target.read_text(encoding="utf-8")
    patched = original.replace("@torch.jit.script_method", "@torch.jit.export")
    if patched == original:
        if "@torch.jit.export" in original:
            print(f"Torch MKLDNN compatibility patch already applied: {target}")
        else:
            raise SystemExit(f"expected torch.jit.script_method decorators not found in {target}")
    else:
        target.write_text(patched, encoding="utf-8")
        # A stale timestamp pyc would be recompiled in memory on every start (the runtime user cannot rewrite it).
        py_compile.compile(str(target), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
        print(f"Patched torch.jit script_method decorators in {target}")

    subprocess.run(
        [
            sys.executable,
            "-W",
            "error::DeprecationWarning",
            "-c",
            "import torch.utils.mkldnn; print('torch.utils.mkldnn import clean')",
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
