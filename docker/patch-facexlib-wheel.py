#!/usr/bin/env python3
"""Use OpenCV headless for facexlib in this headless container image."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import re
import subprocess
import sys
import zipfile
from pathlib import Path


def record_hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def patched_wheel_path(wheel: Path) -> Path:
    parts = wheel.name.removesuffix(".whl").rsplit("-", 4)
    if len(parts) != 5:
        raise SystemExit(f"unexpected wheel filename: {wheel.name}")
    distribution, version, python_tag, abi_tag, platform_tag = parts
    return wheel.with_name(
        f"{distribution}-{version}-1gb10opencvheadless-{python_tag}-{abi_tag}-{platform_tag}.whl"
    )


def patch_wheel(wheel: Path) -> Path:
    patched = patched_wheel_path(wheel)
    entries: dict[str, bytes] = {}
    infos: dict[str, zipfile.ZipInfo] = {}
    record_path = ""
    replaced = False
    with zipfile.ZipFile(wheel) as source:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename.endswith("/METADATA"):
                text = data.decode()
                text, count = re.subn(
                    r"(?m)^Requires-Dist: opencv-python\s*$",
                    "Requires-Dist: opencv-python-headless",
                    text,
                )
                replaced = replaced or count == 1
                data = text.encode()
            if info.filename.endswith("/RECORD"):
                record_path = info.filename
            entries[info.filename] = data
            infos[info.filename] = info
    if not replaced or not record_path:
        raise SystemExit(f"could not patch facexlib metadata in {wheel}")

    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for name, data in sorted(entries.items()):
        if name == record_path:
            writer.writerow((name, "", ""))
        else:
            writer.writerow((name, record_hash(data), len(data)))
    entries[record_path] = record.getvalue().encode()
    with zipfile.ZipFile(patched, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for name, data in entries.items():
            target.writestr(infos[name], data)
    wheel.unlink()
    return patched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements", required=True)
    parser.add_argument("--wheel-dir", required=True)
    args = parser.parse_args()

    requirements = Path(args.requirements)
    lines = requirements.read_text().splitlines()
    matches = [index for index, line in enumerate(lines) if re.match(r"^\s*facexlib(?:\s|[<>=!~;]|$)", line, re.I)]
    if not matches:
        print("facexlib is absent; no wheel override needed")
        return 0
    if len(matches) != 1:
        raise SystemExit("expected exactly one facexlib requirement")

    wheel_dir = Path(args.wheel_dir)
    wheel_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--no-deps",
            "--only-binary=:all:",
            "--dest",
            str(wheel_dir),
            lines[matches[0]].strip(),
        ],
        check=True,
    )
    wheels = sorted(wheel_dir.glob("facexlib-*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected one downloaded facexlib wheel, found {len(wheels)}")
    patched = patch_wheel(wheels[0]).resolve()
    lines[matches[0]] = str(patched)
    requirements.write_text("\n".join(lines) + "\n")
    print(f"using patched facexlib wheel: {patched.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
