#!/usr/bin/env python3
"""Snapshot the NGC base Python package set into protected pins and released floors.

Every distribution importable in the base stage is protected (exact pin) unless it
is named in the released list. A released distribution is resolved by the A1111
application closure instead, with the NGC version as its floor. The release rule
is enforced here so a changed base image cannot silently release a package that
NVIDIA builds or that the NVIDIA-built runtime stack declares as a dependency.

With --pristine, the build fails if a step in this stage removed or changed an inherited
package (for example a --force-reinstall pulling a newer dependency) before the snapshot; only
packages named with --rebuilt may differ.

Only the effective (first on sys.path) distribution of each name is considered;
shadowed copies, such as dpkg-owned ones under /usr/lib/python3/dist-packages,
are not importable and are ignored.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import re
from pathlib import Path

from packaging.requirements import Requirement

# NVIDIA-built packages that A1111 imports at runtime. Their declared requirement
# closure stays protected: NVIDIA tests these builds against the exact versions
# NGC ships, and those dependencies steer torch compile/codegen behavior.
RUNTIME_STACK_ROOTS = ("torch", "torchvision", "triton", "torchao", "mslk")
PROTECTED_PREFIXES = ("nvidia-", "cuda-")


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def effective_distributions() -> dict[str, md.Distribution]:
    dists: dict[str, md.Distribution] = {}
    for dist in md.distributions():
        name = dist.metadata.get("Name")
        if name:
            dists.setdefault(normalize(name), dist)
    return dists


def active_requirements(dist: md.Distribution) -> list[Requirement]:
    reqs = []
    for raw in dist.requires or ():
        req = Requirement(raw)
        if req.marker is None or req.marker.evaluate({"extra": ""}):
            reqs.append(req)
    return reqs


def requirement_closure(dists: dict[str, md.Distribution], roots: tuple[str, ...]) -> set[str]:
    closure: set[str] = set()
    stack = [root for root in roots if root in dists]
    while stack:
        name = stack.pop()
        if name in closure:
            continue
        closure.add(name)
        stack.extend(normalize(req.name) for req in active_requirements(dists[name]) if normalize(req.name) in dists)
    return closure


def custom_build_reason(dist: md.Distribution) -> str | None:
    installer = (dist.read_text("INSTALLER") or "").strip()
    if installer != "pip":
        return f"installed by {installer or 'unknown installer'}, not a pip wheel"
    if "+" in dist.version:
        return f"local version label {dist.version}"
    direct_url = dist.read_text("direct_url.json")
    if direct_url:
        return f"installed from {json.loads(direct_url).get('url', 'a direct URL')}"
    return None


def pristine_drift(dists: dict[str, md.Distribution], pristine: Path, rebuilt: set[str]) -> list[str]:
    """Inherited packages that a build step removed or changed, except those rebuilt on purpose.
    Packages that build steps added (build tools, source-built wheels) are not drift."""
    problems = []
    for raw in pristine.read_text().splitlines():
        if not raw.strip():
            continue
        name, version = raw.split("==", 1)
        name = normalize(name)
        if name in rebuilt:
            continue
        current = dists.get(name)
        if current is None:
            problems.append(f"{name}: NGC ships {version} but a build step removed it")
        elif current.version != version:
            problems.append(f"{name}: NGC ships {version} but a build step changed it to {current.version}")
    return problems


def load_names(path: Path) -> list[str]:
    names = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            names.append(normalize(line))
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--released", required=True, help="newline-delimited released package names")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pristine", help="name==version list recorded before any build step ran in this stage")
    ap.add_argument("--rebuilt", action="append", default=[], help="package intentionally replaced by a build step")
    args = ap.parse_args()

    dists = effective_distributions()
    released = load_names(Path(args.released))
    stack = requirement_closure(dists, RUNTIME_STACK_ROOTS)

    problems = []
    if args.pristine:
        drift = pristine_drift(dists, Path(args.pristine), {normalize(name) for name in args.rebuilt})
        if drift:
            raise SystemExit("inherited NGC packages changed before protection:\n  " + "\n  ".join(drift))
    duplicates = sorted({name for name in released if released.count(name) > 1})
    if duplicates:
        problems.append(f"duplicate released names: {', '.join(duplicates)}")
    for name in sorted(set(released)):
        dist = dists.get(name)
        if dist is None:
            problems.append(f"{name}: released but not installed in the base image; remove it from the list")
            continue
        if name.startswith(PROTECTED_PREFIXES):
            problems.append(f"{name}: CUDA/NVIDIA package prefixes are never released")
        if name in stack:
            problems.append(f"{name}: in the requirement closure of {', '.join(RUNTIME_STACK_ROOTS)}")
        reason = custom_build_reason(dist)
        if reason:
            problems.append(f"{name}: not a stock PyPI wheel ({reason})")
    if problems:
        raise SystemExit("released package rule violated:\n  " + "\n  ".join(problems))

    released_set = set(released)
    protected = {name: dist for name, dist in sorted(dists.items()) if name not in released_set}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "base-python-protected-constraints.txt").write_text(
        "".join(f"{name}=={dist.version}\n" for name, dist in protected.items())
    )
    (out / "base-python-protected-names.txt").write_text("".join(f"{name}\n" for name in protected))
    (out / "base-python-released-floors.txt").write_text(
        "".join(f"{name}>={dists[name].version}\n" for name in sorted(released_set))
    )

    def version(name: str) -> str | None:
        return dists[name].version if name in dists else None

    print(json.dumps({
        "protected_count": len(protected),
        "released_count": len(released_set),
        "runtime_stack_closure": sorted(stack),
        "torch": version("torch"),
        "torchvision": version("torchvision"),
        "torchaudio": version("torchaudio"),
        "torchaudio_optional_absent": version("torchaudio") is None,
        "torchao": version("torchao"),
        "mslk": version("mslk"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
