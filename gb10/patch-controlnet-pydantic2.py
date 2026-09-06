#!/usr/bin/env python3
"""Make legacy ControlNet root validators load under NVIDIA's Pydantic 2."""

from __future__ import annotations

import argparse
from pathlib import Path


VALIDATORS = ("bound_check_params", "guidance_check")


def patch(path: Path) -> bool:
    source = path.read_text()
    updated = source
    for function in VALIDATORS:
        legacy = f"    @root_validator\n    def {function}"
        compatible = f"    @root_validator(skip_on_failure=True)\n    def {function}"
        if legacy in updated:
            updated = updated.replace(legacy, compatible, 1)
        elif compatible not in updated:
            raise SystemExit(f"ControlNet validator shape changed; cannot patch {function}: {path}")
    changed = updated != source
    if changed:
        path.write_text(updated)
    return changed


def verify(path: Path) -> None:
    source = path.read_text()
    missing = [
        function
        for function in VALIDATORS
        if f"    @root_validator(skip_on_failure=True)\n    def {function}" not in source
    ]
    if missing:
        raise SystemExit(f"ControlNet Pydantic 2 validator patch missing for: {', '.join(missing)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = Path(args.path)
    if not path.is_file():
        raise SystemExit(f"ControlNet args module missing: {path}")
    if not args.check:
        changed = patch(path)
        print(f"ControlNet Pydantic 2 compatibility {'patched' if changed else 'already present'}: {path}")
    verify(path)
    if args.check:
        print(f"ControlNet Pydantic 2 compatibility verified: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
