#!/usr/bin/env python3
"""Build the resolver input: the app requirements plus the --include supplements, minus every package the
NVIDIA base owns (the protected constraints file plus the fixed torch/CUDA names), which the resolver sees as stubs.

A requirement on a protected package is dropped, so a version specifier on one would never be resolved: the build
fails unless the protected (installed) version satisfies it."""
import argparse
import re
from pathlib import Path

from packaging.requirements import Requirement

PROTECTED_NAMES = ("torch", "torchvision", "torchaudio", "triton")
PROTECTED_PREFIXES = ("nvidia-", "cuda-")


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def parse_req_name(line: str) -> str | None:
    line = line.strip()
    if not line or line.startswith("#") or line.startswith("-"):
        return None
    name = re.split(r"[<>=!~ ;\[]", line, maxsplit=1)[0].strip()
    return normalize(name) if name else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--wheel-dir", required=True)
    ap.add_argument(
        "--include",
        action="append",
        default=[],
        help="Additional requirements files to append to the resolver input.",
    )
    ap.add_argument(
        "--protected-constraints-file",
        required=True,
        help="name==version lines of the packages inherited from and owned by the base image.",
    )
    args = ap.parse_args()

    source = Path(args.source)
    target = Path(args.target)
    Path(args.wheel_dir).mkdir(parents=True, exist_ok=True)

    protected_versions = {}
    for line in Path(args.protected_constraints_file).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            name, version = line.split("==", 1)
            protected_versions[normalize(name)] = version.strip()
    protected_names = {normalize(name) for name in PROTECTED_NAMES} | protected_versions.keys()

    emitted = []
    removed = set()
    conflicts = []
    sources = [(source, source.read_text().splitlines())]
    for include in args.include:
        include_path = Path(include)
        if not include_path.exists():
            raise FileNotFoundError(f"included requirements file not found: {include_path}")
        sources.append((include_path, include_path.read_text().splitlines()))

    for src, lines in sources:
        if emitted:
            emitted.append(f"# requirements from {src}")
        for raw in lines:
            name = parse_req_name(raw)
            if name and (name in protected_names or name.startswith(PROTECTED_PREFIXES)):
                requirement = Requirement(re.sub(r"\s+#.*", "", raw).strip())
                version = protected_versions.get(name)
                if requirement.url or (requirement.specifier and (version is None or not requirement.specifier.contains(version, prereleases=True))):
                    conflicts.append(f"{src}: {raw.strip()} (protected base version: {version or 'not installed'})")
                removed.add(name)
                continue
            emitted.append(raw.rstrip())

    if conflicts:
        raise SystemExit("requirements on protected base packages that the base version does not satisfy:\n  " + "\n  ".join(conflicts))

    target.write_text("\n".join(line for line in emitted if line.strip()) + "\n")
    included = [str(src) for src, _lines in sources[1:]]
    suffix = f" plus {included}" if included else ""
    print(f"wrote resolver input {target} from {source}{suffix}")
    if removed:
        print(f"removed protected resolver inputs: {', '.join(sorted(removed))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
