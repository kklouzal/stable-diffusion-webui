from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

SCHEMA_VERSION = 1


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strong_file_identity(path: str | os.PathLike[str]) -> dict:
    resolved = os.path.abspath(os.fspath(path))
    stat = os.stat(resolved)
    return {
        "path": resolved,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "sha256": sha256_file(resolved),
    }


def atomic_write(path: str | os.PathLike[str], data: bytes) -> None:
    path = os.fspath(path)
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile("wb", delete=False, dir=parent, prefix=".partial-", suffix="-" + uuid.uuid4().hex) as stream:
            temporary = stream.name
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)


@contextlib.contextmanager
def exclusive_lock(path: str | os.PathLike[str]):
    import fcntl

    lock_path = os.fspath(path) + ".lock"
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    with open(lock_path, "a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


@contextlib.contextmanager
def active_lease(path: str | os.PathLike[str]):
    lease = os.fspath(path) + ".lease." + str(os.getpid()) + "." + uuid.uuid4().hex
    fd = os.open(lease, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(fd, str(time.time_ns()).encode("ascii"))
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        yield lease
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(lease)


def has_active_lease(path: str | os.PathLike[str]) -> bool:
    base = Path(path)
    return any(base.parent.glob(base.name + ".lease.*"))


def cleanup_partial_files(root: str | os.PathLike[str], *, older_than_seconds: int = 3600, dry_run: bool = True) -> list[str]:
    cutoff = time.time() - older_than_seconds
    selected = []
    for path in Path(root).rglob(".partial-*"):
        if path.is_file() and path.stat().st_mtime <= cutoff:
            selected.append(str(path))
            if not dry_run:
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()
    return sorted(selected)


def enforce_quota(
    artifacts: Iterable[str | os.PathLike[str]],
    *,
    max_bytes: int,
    dry_run: bool = True,
) -> dict:
    """Evict oldest inactive artifact groups until total bytes fit max_bytes.

    Each input is the primary artifact. Its sidecars (same path plus ``.json``)
    are counted/deleted with it. Active leases make a group ineligible.
    """
    groups = []
    for value in artifacts:
        primary = Path(value)
        if not primary.is_file():
            continue
        members = [primary]
        members.extend(sorted(path for path in primary.parent.glob(primary.name + "*.json") if path.is_file()))
        size = sum(member.stat().st_size for member in members)
        atime = max(member.stat().st_atime_ns for member in members)
        groups.append({"primary": str(primary), "members": [str(x) for x in members], "bytes": size, "atime_ns": atime, "active": has_active_lease(primary)})
    total = sum(group["bytes"] for group in groups)
    evicted = []
    remaining = total
    for group in sorted(groups, key=lambda item: (item["atime_ns"], item["primary"])):
        if remaining <= max_bytes:
            break
        if group["active"]:
            continue
        evicted.append(group)
        remaining -= group["bytes"]
        if not dry_run:
            for member in group["members"]:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(member)
    return {"schema_version": SCHEMA_VERSION, "dry_run": dry_run, "max_bytes": max_bytes, "before_bytes": total, "after_bytes": remaining, "evicted": evicted, "protected": [g["primary"] for g in groups if g["active"]], "within_quota": remaining <= max_bytes}


def write_quota_report(path: str | os.PathLike[str], report: dict) -> None:
    atomic_write(path, (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf8"))


def enforce_directory_quota(root: str | os.PathLike[str], *, max_bytes: int, dry_run: bool = True) -> dict:
    root = Path(root)
    return enforce_quota(root.rglob("*.pt") if root.is_dir() else (), max_bytes=max_bytes, dry_run=dry_run)
