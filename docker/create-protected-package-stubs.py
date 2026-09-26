#!/usr/bin/env python3
"""Create resolver wheel stubs for packages owned by the base image.

The resolver runs in an isolated virtual environment populated with these stubs.
That lets pip enforce the exact versions shipped by NVIDIA without recursively
re-evaluating occasionally stale/internally inconsistent dependency metadata in
the already-tested NGC environment.

Stubs declare only the requirements that target packages released from base
protection (--released-floors). Those requirements are real ceilings for the
application resolver: the dependents list names every stub that carries one, so
the dry-run can request them and pip must satisfy their released-package
requirements when it picks newer released versions.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata as md
import io
import re
import zipfile
from pathlib import Path

from packaging.requirements import Requirement


EXACT_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)$")
REAL_BOOTSTRAP_PACKAGES = {"pip"}


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def load_released_names(path: Path) -> set[str]:
    names = set()
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            names.add(normalize(Requirement(line).name))
    return names


def released_requirements(dist: md.Distribution, released: set[str]) -> list[str]:
    """Return the dist's active requirements on released packages, without extras or markers."""
    out = []
    for raw in dist.requires or ():
        req = Requirement(raw)
        if normalize(req.name) not in released:
            continue
        if req.marker is not None and not req.marker.evaluate({"extra": ""}):
            continue
        if req.url:
            raise SystemExit(f"{dist.metadata['Name']}: URL requirement on released package: {raw}")
        out.append(f"{req.name}{req.specifier}")
    return sorted(set(out))


def wheel_component(value: str) -> str:
    return re.sub(r"[-_.]+", "_", value)


def wheel_version_component(value: str) -> str:
    return value.replace("-", "_")


def record_hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def build_stub(target: Path, name: str, version: str, requires: list[str] = ()) -> Path:
    distribution = wheel_component(name)
    wheel_version = wheel_version_component(version)
    dist_info = f"{distribution}-{wheel_version}.dist-info"
    entries = {
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.1\n"
            f"Name: {name}\n"
            f"Version: {version}\n"
            "Summary: Resolver-only stand-in for an NVIDIA base-image package\n"
            + "".join(f"Requires-Dist: {req}\n" for req in requires)
            + "\n"
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
    parser.add_argument("--released-floors", help="released package floors; stubs then declare requirements on them")
    parser.add_argument("--dependents-out", help="write name==version for every stub that declares a requirement")
    args = parser.parse_args()
    if bool(args.released_floors) != bool(args.dependents_out):
        raise SystemExit("--released-floors and --dependents-out must be given together")

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

    released: set[str] = set()
    installed: dict[str, md.Distribution] = {}
    if args.released_floors:
        released = load_released_names(Path(args.released_floors))
        for dist in md.distributions():
            if dist.metadata.get("Name"):
                installed.setdefault(normalize(dist.metadata["Name"]), dist)

    wheels = []
    dependents = []
    for name, version in pins:
        if name.lower() in REAL_BOOTSTRAP_PACKAGES:
            continue
        requires: list[str] = []
        if released:
            if normalize(name) in released:
                raise SystemExit(f"{name}: released package is also pinned as protected")
            dist = installed.get(normalize(name))
            if dist is None or dist.version != version:
                found = dist.version if dist else "absent"
                raise SystemExit(f"{name}: protected pin {version} does not match installed {found}")
            requires = released_requirements(dist, released)
            if requires:
                dependents.append(f"{name}=={version}")
        wheels.append(build_stub(target, name, version, requires))
    Path(args.requirements_out).write_text("\n".join(str(wheel.resolve()) for wheel in wheels) + "\n")
    if args.dependents_out:
        Path(args.dependents_out).write_text("".join(f"{line}\n" for line in dependents))
    print(
        f"created {len(wheels)} protected resolver stubs in {target}; "
        f"{len(dependents)} declare requirements on {len(released)} released packages; "
        f"kept real bootstrap packages: {', '.join(sorted(REAL_BOOTSTRAP_PACKAGES))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
