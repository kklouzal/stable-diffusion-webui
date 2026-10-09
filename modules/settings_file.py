"""The settings file (config.json): reading it, setting an unreadable one aside, and writing it durably.

gb10/run.sh bind-mounts config.json into the container as a single file, and rename cannot replace a single-file bind
mount (EBUSY). The file is therefore always rewritten in place, which cannot be atomic: a crash during a write can
leave it truncated. write() makes a completed write durable before it returns, and read() turns a file it cannot use
into the default settings, keeping a copy of the bad bytes, so the server starts instead of crash-looping.
"""
import datetime
import json
import os

from modules import errors
from modules.paths_internal import cache_dir


def _write_durably(filename, data: bytes, mode):
    with open(filename, mode) as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write(filename, text: str):
    """Rewrites `filename` in place with `text` (UTF-8) and fsyncs it before returning."""
    _write_durably(filename, text.encode("utf8"), "wb")


def read(filename) -> dict:
    """The settings stored in `filename`, or {} when the file does not exist.

    A file that is not a UTF-8 JSON object, including an empty file, is copied to
    <cache_dir>/config-recovery/<name>.corrupt-<UTC time> (the app cache, a host mount in the gb10 deployment, so the
    copy survives the container's replacement or rollback) and then rewritten in place as {}, and the error is
    reported: the settings revert to their defaults. The copy and its directory entry are fsynced before the file is
    reset, and any OS error (an unreadable file, a full disk) propagates instead, leaving the file untouched.
    """
    try:
        with open(filename, "rb") as file:
            raw = file.read()
    except FileNotFoundError:
        return {}

    try:
        settings = json.loads(raw.decode("utf8"))
        if not isinstance(settings, dict):
            raise ValueError(f"expected a JSON object, got {type(settings).__name__}")
    except ValueError:
        recovery_dir = os.path.join(cache_dir, "config-recovery")
        os.makedirs(recovery_dir, exist_ok=True)
        stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S%fZ")
        backup = os.path.join(recovery_dir, f"{os.path.basename(filename)}.corrupt-{stamp}")
        _write_durably(backup, raw, "xb")
        _fsync_directory(recovery_dir)
        _fsync_directory(cache_dir)  # the entry of a just-created config-recovery/
        write(filename, "{}")
        errors.report(f'\nCould not load settings\nThe config file "{filename}" is likely corrupted\nIts content was copied to "{backup}" and the file was reset to {{}}\nReverting config to default\n\n', exc_info=True)
        return {}

    return settings
