#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import re
from pathlib import Path

EXACT_PROTECTED = {"torch", "torchvision", "torchaudio", "triton"}
PREFIX_PROTECTED = ("nvidia-", "cuda-")
REQUIRED_PRESENT = {"torch", "torchvision", "triton"}
OPTIONAL_ABSENT_OK = {"torchaudio"}
DEFAULT_PROTECTED_NAMES_FILE = Path("/opt/base-python-protected-names.txt")


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def load_protected_names(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        normalize(line)
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def is_protected(name: str, base_protected_names: set[str]) -> bool:
    norm = normalize(name)
    return norm in EXACT_PROTECTED or norm in base_protected_names or norm.startswith(PREFIX_PROTECTED)


def dist_fingerprint(dist: md.Distribution) -> str:
    h = hashlib.sha256()
    h.update((dist.metadata.get("Name") or "").encode())
    h.update(b"\0")
    h.update(dist.version.encode())
    h.update(b"\0")
    for rel in sorted(str(f) for f in (dist.files or ())):
        h.update(rel.encode())
        h.update(b"\0")
    return h.hexdigest()


def snapshot(base_protected_names: set[str], protected_names_file: Path) -> dict:
    packages: dict[str, dict] = {}
    for dist in md.distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        norm = normalize(name)
        if not is_protected(norm, base_protected_names):
            continue
        packages[norm] = {
            "name": name,
            "normalized": norm,
            "version": dist.version,
            "metadata_fingerprint": dist_fingerprint(dist),
        }
    exact = {}
    for name in sorted(EXACT_PROTECTED):
        exact[name] = packages.get(name) or {
            "name": name,
            "normalized": name,
            "version": None,
            "metadata_fingerprint": None,
            "present": False,
        }
        exact[name]["present"] = bool(packages.get(name))
        if name in OPTIONAL_ABSENT_OK and not exact[name]["present"]:
            exact[name]["optional_absent"] = True
    return {
        "schema": "gb10-a1111-protected-stack-v2",
        "base_protected_names_file": str(protected_names_file),
        "base_protected_names_count": len(base_protected_names),
        "base_protected_names": sorted(base_protected_names),
        "protected_exact": sorted(EXACT_PROTECTED),
        "protected_prefixes": list(PREFIX_PROTECTED),
        "required_present": sorted(REQUIRED_PRESENT),
        "optional_absent_ok": sorted(OPTIONAL_ABSENT_OK),
        "exact": exact,
        "packages": dict(sorted(packages.items())),
    }


def validate_baseline(data: dict) -> list[str]:
    problems = []
    exact = data.get("exact", {})
    for name in sorted(REQUIRED_PRESENT):
        if not exact.get(name, {}).get("present"):
            problems.append(f"required protected package is absent: {name}")
    packages = data.get("packages", {})
    for name in data.get("base_protected_names", []):
        if name not in packages:
            problems.append(f"NVIDIA base package is absent: {name}")
    return problems


def compare(before: dict, after: dict) -> list[str]:
    problems = validate_baseline(after)
    before_pkgs = before.get("packages", {})
    after_pkgs = after.get("packages", {})
    for name, before_item in sorted(before_pkgs.items()):
        after_item = after_pkgs.get(name)
        if after_item is None:
            problems.append(f"protected package removed by app deps: {name}")
            continue
        for key in ("version", "metadata_fingerprint"):
            if before_item.get(key) != after_item.get(key):
                problems.append(
                    f"protected package changed by app deps: {name} {key} "
                    f"{before_item.get(key)!r} -> {after_item.get(key)!r}"
                )
    for name in sorted(set(after_pkgs) - set(before_pkgs)):
        problems.append(f"protected package introduced by app deps: {name}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="Snapshot/compare GB10 protected CUDA/PyTorch package boundary.")
    ap.add_argument("--snapshot", help="write a baseline snapshot JSON")
    ap.add_argument("--compare", help="compare the current environment to a previous snapshot JSON")
    ap.add_argument("--out", help="write current snapshot/compare result JSON here")
    ap.add_argument(
        "--protected-names-file",
        default=str(DEFAULT_PROTECTED_NAMES_FILE),
        help="newline-delimited package names inherited from the NVIDIA base image",
    )
    args = ap.parse_args()

    protected_names_file = Path(args.protected_names_file)
    base_protected_names = load_protected_names(protected_names_file)
    if not base_protected_names:
        raise SystemExit(f"NVIDIA base protected names file is absent or empty: {protected_names_file}")
    current = snapshot(base_protected_names, protected_names_file)
    problems = validate_baseline(current)
    result = {"current": current, "problems": problems}

    if args.compare:
        before = json.loads(Path(args.compare).read_text())
        before_data = before.get("current", before)
        problems = compare(before_data, current)
        result = {"before": before_data, "current": current, "problems": problems}

    out_path = Path(args.out or args.snapshot or "-")
    if str(out_path) != "-":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))

    if args.snapshot:
        Path(args.snapshot).parent.mkdir(parents=True, exist_ok=True)
        Path(args.snapshot).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    if problems:
        for problem in problems:
            print(problem)
        return 1
    absent = [n for n, item in current["exact"].items() if item.get("optional_absent")]
    if absent:
        print("optional protected package absent by policy: " + ", ".join(sorted(absent)))
    print(f"protected NVIDIA base package boundary: ok ({len(base_protected_names)} packages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
