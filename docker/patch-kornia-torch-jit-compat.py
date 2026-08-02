#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys


def main() -> int:
    spec = importlib.util.find_spec("kornia")
    if spec is None or spec.submodule_search_locations is None:
        raise SystemExit("kornia is not installed; cannot apply torch.jit compatibility patch")

    root = pathlib.Path(next(iter(spec.submodule_search_locations))).resolve()
    changed: list[str] = []
    for path in root.rglob("*.py"):
        original = path.read_text(encoding="utf-8")
        patched = original.replace("@torch.jit.script\n", "")
        if patched != original:
            path.write_text(patched, encoding="utf-8")
            changed.append(str(path.relative_to(root)))

    print(f"kornia torch.jit compatibility patch: files={len(changed)}")
    if changed:
        for rel in changed:
            print(f"  - {rel}")

    subprocess.run(
        [
            sys.executable,
            "-W",
            "error::DeprecationWarning",
            "-c",
            "import kornia; print('kornia import clean')",
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
