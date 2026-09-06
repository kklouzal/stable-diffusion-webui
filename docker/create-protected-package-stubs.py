#!/usr/bin/env python3
"""Create dependency-free wheel stubs for packages owned by the base image.

The resolver runs in an isolated virtual environment populated with these stubs.
That lets pip enforce the exact versions shipped by NVIDIA without recursively
re-evaluating occasionally stale/internally inconsistent dependency metadata in
the already-tested NGC environment.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import re
import zipfile
from pathlib import Path


EXACT_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)$")
REAL_BOOTSTRAP_PACKAGES = {"pip"}


def wheel_component(value: str) -> str:
    return re.sub(r"[-_.]+", "_", value)


def wheel_version_component(value: str) -> str:
    return value.replace("-", "_")


def record_hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def build_stub(target: Path, name: str, version: str) -> Path:
    distribution = wheel_component(name)
    wheel_version = wheel_version_component(version)
    dist_info = f"{distribution}-{wheel_version}.dist-info"
    entries = {
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.1\n"
            f"Name: {name}\n"
            f"Version: {version}\n"
            "Summary: Resolver-only stand-in for an NVIDIA base-image package\n"
            "\n"
        ).encode(),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: gb10-a1111-protected-package-stubs\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
            "\n"
        ).encode(),
    }
    record_path = f"{dist_info}/RECORD"
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for path, data in sorted(entries.items()):
        writer.writerow((path, record_hash(data), len(data)))
    writer.writerow((record_path, "", ""))
    entries[record_path] = record.getvalue().encode()

    wheel = target / f"{distribution}-{wheel_version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, data in entries.items():
            archive.writestr(path, data)
    return wheel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--constraints", required=True)
    parser.add_argument("--wheel-dir", required=True)
    parser.add_argument("--requirements-out", required=True)
    args = parser.parse_args()

    target = Path(args.wheel_dir)
    target.mkdir(parents=True, exist_ok=True)
    pins = []
    for raw in Path(args.constraints).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = EXACT_PIN.fullmatch(line)
        if not match:
            raise SystemExit(f"protected constraint is not an exact package pin: {line}")
        pins.append(match.groups())

    wheels = [
        build_stub(target, name, version)
        for name, version in pins
        if name.lower() not in REAL_BOOTSTRAP_PACKAGES
    ]
    Path(args.requirements_out).write_text("\n".join(str(wheel.resolve()) for wheel in wheels) + "\n")
    print(
        f"created {len(wheels)} protected resolver stubs in {target}; "
        f"kept real bootstrap packages: {', '.join(sorted(REAL_BOOTSTRAP_PACKAGES))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
