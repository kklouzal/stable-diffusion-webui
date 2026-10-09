#!/usr/bin/env python3
"""Make the listed pure-Python packages depend on OpenCV headless in this headless container image.

facexlib and the Depth Anything v1/v2 ControlNet preprocessor packages declare `Requires-Dist: opencv-python`
(the GUI build). The image installs opencv-python-headless; a second distribution owning the same cv2 package
would overwrite it. For each listed package present in the resolver input, the requirement line is downloaded
as a wheel (a `name @ URL#sha256=...` line is hash-checked by pip), its METADATA requirement is rewritten to
opencv-python-headless (exactly one match, or the build fails), RECORD is regenerated, and the wheel is saved
with the build tag 1gb10opencvheadless. The resolver input line is replaced by the patched wheel's path, and the
wheel build later finds it through --find-links.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

PACKAGES = ("facexlib", "depth-anything", "depth-anything-v2")
BUILD_TAG = "1gb10opencvheadless"


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def requirement_name(line: str) -> str | None:
    line = line.strip()
    if not line or line.startswith(("#", "-")):
        return None
    return normalize(re.split(r"[<>=!~ ;@\[]", line, maxsplit=1)[0])


def record_hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def patched_wheel_name(wheel: Path) -> str:
    parts = wheel.name.removesuffix(".whl").split("-")
    if len(parts) != 5:  # distribution-version-python-abi-platform, no build tag yet
        raise SystemExit(f"unexpected wheel filename: {wheel.name}")
    distribution, version, python_tag, abi_tag, platform_tag = parts
    return f"{distribution}-{version}-{BUILD_TAG}-{python_tag}-{abi_tag}-{platform_tag}.whl"


def patch_wheel(wheel: Path, target: Path) -> None:
    entries: dict[str, bytes] = {}
    infos: dict[str, zipfile.ZipInfo] = {}
    record_path = ""
    replaced = 0
    with zipfile.ZipFile(wheel) as source:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename.endswith(".dist-info/METADATA"):
                text, count = re.subn(
                    r"(?m)^Requires-Dist: opencv-python[ \t]*$",
                    "Requires-Dist: opencv-python-headless",
                    data.decode(),
                )
                replaced += count
                data = text.encode()
            if info.filename.endswith(".dist-info/RECORD"):
                record_path = info.filename
            entries[info.filename] = data
            infos[info.filename] = info
    if replaced != 1 or not record_path:
        raise SystemExit(f"could not patch the opencv-python requirement of {wheel.name} (matches: {replaced})")

    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for name, data in sorted(entries.items()):
        if name == record_path:
            writer.writerow((name, "", ""))
        else:
            writer.writerow((name, record_hash(data), len(data)))
    entries[record_path] = record.getvalue().encode()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as out:
        for name, data in entries.items():
            out.writestr(infos[name], data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements", required=True)
    parser.add_argument("--wheel-dir", required=True)
    args = parser.parse_args()

    requirements = Path(args.requirements)
    wheel_dir = Path(args.wheel_dir)
    wheel_dir.mkdir(parents=True, exist_ok=True)
    lines = requirements.read_text().splitlines()
    for package in PACKAGES:
        matches = [index for index, line in enumerate(lines) if requirement_name(line) == package]
        if not matches:
            print(f"{package} is absent; no wheel override needed")
            continue
        if len(matches) != 1:
            raise SystemExit(f"expected exactly one {package} requirement, found {len(matches)}")
        with tempfile.TemporaryDirectory(dir=wheel_dir) as download_dir:
            subprocess.run(
                [sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:", "--dest", download_dir, lines[matches[0]].strip()],
                check=True,
            )
            wheels = sorted(Path(download_dir).glob("*.whl"))
            if len(wheels) != 1 or normalize(wheels[0].name.split("-", 1)[0]) != package:
                raise SystemExit(f"expected one downloaded {package} wheel, found {[wheel.name for wheel in wheels]}")
            patched = (wheel_dir / patched_wheel_name(wheels[0])).resolve()
            patch_wheel(wheels[0], patched)
        lines[matches[0]] = str(patched)
        print(f"using {package} wheel with headless OpenCV: {patched.name}")
    requirements.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
