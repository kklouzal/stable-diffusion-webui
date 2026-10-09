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
from modules.paths_internal import script_path


def _write_durably(filename, data: bytes, mode):
    with open(filename, mode) as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())


def write(filename, text: str):
    """Rewrites `filename` in place with `text` (UTF-8) and fsyncs it before returning."""
    _write_durably(filename, text.encode("utf8"), "wb")


def read(filename) -> dict:
    """The settings stored in `filename`, or {} when the file does not exist.

    A file that is not a UTF-8 JSON object, including an empty file, is copied to tmp/<name>.corrupt-<UTC time> and
    then rewritten in place as {}, and the error is reported: the settings revert to their defaults. The copy is
    written before the file is reset, and any OS error (an unreadable file, a full disk) propagates instead.
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
        tmp_dir = os.path.join(script_path, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S%fZ")
        backup = os.path.join(tmp_dir, f"{os.path.basename(filename)}.corrupt-{stamp}")
        _write_durably(backup, raw, "xb")
        write(filename, "{}")
        errors.report(f'\nCould not load settings\nThe config file "{filename}" is likely corrupted\nIts content was copied to "{backup}" and the file was reset to {{}}\nReverting config to default\n\n', exc_info=True)
        return {}

    return settings
