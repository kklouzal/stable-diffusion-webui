#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "gb10/controlnet-cache-correctness.patch"
HELPER = ROOT / "gb10/controlnet_cache_contract.py"
FILES = ["scripts/controlnet.py", "scripts/supported_preprocessor.py", "scripts/preprocessor/model_free_preprocessors.py"]


def run(args, cwd):
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("controlnet_root", type=Path)
    args = parser.parse_args()
    target = args.controlnet_root.resolve()
    for rel in FILES:
        if not (target / rel).is_file():
            raise SystemExit(f"missing ControlNet source: {target / rel}")
    helper_target = target / "scripts/cache_contract.py"
    with tempfile.TemporaryDirectory(prefix="controlnet-cache-patch-") as tmp:
        stage = Path(tmp)
        for rel in FILES:
            (stage / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target / rel, stage / rel)
        helper_stage = stage / "scripts/cache_contract.py"
        helper_stage.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HELPER, helper_stage)
        applied = run(["patch", "-p1", "--forward", "--batch", "-i", str(PATCH)], stage)
        if applied.returncode != 0:
            reverse = run(["patch", "-p1", "--reverse", "--dry-run", "--batch", "-i", str(PATCH)], target)
            helper_same = helper_target.is_file() and helper_target.read_bytes() == HELPER.read_bytes()
            if reverse.returncode == 0 and helper_same:
                return 0
            sys.stderr.write(applied.stdout + applied.stderr)
            raise SystemExit("ControlNet sources do not match the audited baseline or patched result")
        if args.check:
            return 0
        for rel in FILES:
            os.replace(stage / rel, target / rel)
        helper_target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(helper_stage, helper_target)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
