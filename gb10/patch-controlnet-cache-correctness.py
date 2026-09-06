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
FILES = ["scripts/controlnet.py", "scripts/controlnet_lllite.py", "scripts/hook.py", "scripts/ipadapter/plugable_ipadapter.py", "scripts/supported_preprocessor.py", "scripts/preprocessor/model_free_preprocessors.py", "annotator/openpose/body.py"]
PATCHED_MARKERS = {
    "scripts/controlnet.py": (
        "from internal_controlnet.cache_contract import AtomicLRU, callable_identity, freeze, runtime_identity",
        'model_load_cache = AtomicLRU(2, "controlnet-model")',
        "def _model_cache_key(p, unet, model):",
        "def clear_controlnet_caches_for_reload():",
    ),
    "scripts/controlnet_lllite.py": (
        "_all_hack_lock = RLock()",
        "def clear_all_lllite():",
        'getattr(k, "_controlnet_lllite_owner", None) is owned',
    ),
    "scripts/hook.py": (
        "from scripts.controlnet_lllite import clear_all_lllite",
        "clear_all_ip_adapter()",
        'release_request_state',
    ),
    "scripts/ipadapter/plugable_ipadapter.py": (
        "_all_hacks_lock = RLock()",
        'getattr(k, "_controlnet_ipadapter_owner", None) is owned',
        "def release_request_state(self):",
    ),
    "scripts/supported_preprocessor.py": (
        "from internal_controlnet.cache_contract import AtomicLRU, callable_identity, freeze, runtime_identity",
        "def _cache_identity(self, args, kwargs):",
        "def _cached_call(self, *args, **kwargs):",
    ),
    "scripts/preprocessor/model_free_preprocessors.py": (
        "class PreprocessorNone(Preprocessor):\n    cacheable = True",
        "class PreprocessorCanny(Preprocessor):\n    cacheable = True",
        "class PreprocessorScribbleXdog(Preprocessor):\n    cacheable = True",
    ),
    "annotator/openpose/body.py": (
        "heatmap_avg += heatmap / len(multiplier)",
    ),
}


def run(args, cwd):
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True)


def is_audited_result(target: Path, helper_target: Path) -> bool:
    if not helper_target.is_file() or helper_target.read_bytes() != HELPER.read_bytes():
        return False
    return all(
        all(marker in (target / rel).read_text(encoding="utf-8") for marker in markers)
        for rel, markers in PATCHED_MARKERS.items()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("controlnet_root", type=Path)
    args = parser.parse_args()
    target = args.controlnet_root.resolve()
    for rel in FILES:
        if not (target / rel).is_file():
            raise SystemExit(f"missing ControlNet source: {target / rel}")
    helper_target = target / "internal_controlnet/cache_contract.py"
    # Other compatibility patches may legitimately change the same files after
    # this cache patch. Validate its complete semantic result before attempting
    # a context-sensitive reverse dry run.
    if is_audited_result(target, helper_target):
        return 0
    with tempfile.TemporaryDirectory(prefix="controlnet-cache-patch-") as tmp:
        stage = Path(tmp)
        for rel in FILES:
            (stage / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target / rel, stage / rel)
        helper_stage = stage / "internal_controlnet/cache_contract.py"
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
