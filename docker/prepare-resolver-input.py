#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

DEFAULT_PROTECTED_NAMES = ("torch", "torchvision", "torchaudio", "triton")
DEFAULT_PROTECTED_PREFIXES = ("nvidia-", "cuda-")


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
    ap.add_argument("--audit", default="/opt/build/requirements-resolver-audit.json")
    ap.add_argument("--protected-name", action="append", default=list(DEFAULT_PROTECTED_NAMES))
    ap.add_argument("--protected-prefix", action="append", default=list(DEFAULT_PROTECTED_PREFIXES))
    args = ap.parse_args()

    source = Path(args.source)
    target = Path(args.target)
    Path(args.wheel_dir).mkdir(parents=True, exist_ok=True)

    protected_names = {normalize(name) for name in args.protected_name}
    protected_prefixes = tuple(normalize(prefix) for prefix in args.protected_prefix)

    emitted = []
    removed = []
    sources = [(source, source.read_text().splitlines())]
    included = []
    for include in args.include:
        include_path = Path(include)
        if not include_path.exists():
            raise FileNotFoundError(f"included requirements file not found: {include_path}")
        sources.append((include_path, include_path.read_text().splitlines()))
        included.append(str(include_path))

    for src, lines in sources:
        if emitted:
            emitted.append(f"# requirements from {src}")
        for raw in lines:
            name = parse_req_name(raw)
            if name and (name in protected_names or name.startswith(protected_prefixes)):
                removed.append({"source": str(src), "name": name, "line": raw.strip(), "reason": "protected CUDA/PyTorch package boundary"})
                continue
            emitted.append(raw.rstrip())

    target.write_text("\n".join(line for line in emitted if line.strip()) + "\n")
    audit = {
        "source": str(source),
        "included": included,
        "target": str(target),
        "protected_names": sorted(protected_names),
        "protected_prefixes": list(protected_prefixes),
        "removed": removed,
    }
    Path(args.audit).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    suffix = f" plus {included}" if included else ""
    print(f"wrote resolver input {target} from {source}{suffix}")
    if removed:
        removed_names = ", ".join(sorted({item["name"] for item in removed}))
        print(f"removed protected resolver inputs: {removed_names}; audit={args.audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
