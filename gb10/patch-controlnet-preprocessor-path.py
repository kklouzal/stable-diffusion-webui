#!/usr/bin/env python3
"""Add the control_net_preprocessor_models_path setting to the mounted ControlNet extension.

Each target must be either exactly original (it is patched) or exactly patched (it is left alone); anything else,
including CRLF line endings, fails closed. Every target is validated and its patched text verified (blocks, valid
Python) before any file is written, and each file is replaced atomically, keeping its mode and owner.
"""
from __future__ import annotations

import os
import pathlib
import sys
import tempfile

OLD_ANNOTATOR = """models_path = shared.opts.data.get('control_net_modules_path', None)
if not models_path:
    models_path = getattr(shared.cmd_opts, 'controlnet_annotator_models_path', None)
"""
NEW_ANNOTATOR = """models_path = shared.opts.data.get('control_net_preprocessor_models_path', None)
if not models_path:
    models_path = shared.opts.data.get('control_net_modules_path', None)
if not models_path:
    models_path = getattr(shared.cmd_opts, 'controlnet_annotator_models_path', None)
"""
OLD_CONTROLNET = """    shared.opts.add_option(\"control_net_modules_path\", shared.OptionInfo(
        \"\", \"Path to directory containing annotator model directories (overrides corresponding command line flag)\", section=section).needs_reload_ui())
"""
NEW_CONTROLNET = """    shared.opts.add_option(\"control_net_modules_path\", shared.OptionInfo(
        \"\", \"Legacy path to directory containing annotator/preprocessor model directories (overrides corresponding command line flag when no preprocessor-specific path is set)\", section=section).needs_reload_ui())
    shared.opts.add_option(\"control_net_preprocessor_models_path\", shared.OptionInfo(
        \"\", \"Path to directory containing annotator/preprocessor model directories (overrides legacy setting and corresponding command line flag)\", section=section).needs_reload_ui())
"""


TARGETS = (
    (pathlib.Path("annotator") / "annotator_path.py", OLD_ANNOTATOR, NEW_ANNOTATOR),
    (pathlib.Path("scripts") / "controlnet.py", OLD_CONTROLNET, NEW_CONTROLNET),
)


def block_counts(text: str, old: str, new: str) -> tuple[int, int]:
    """(old blocks outside a new block, new blocks). The old annotator block is a suffix of the new one."""
    return text.replace(new, "").count(old), text.count(new)


def verify(text: str, path: pathlib.Path, old: str, new: str) -> None:
    """Post-condition of a patch: the new block exactly once, no old block outside it, valid Python."""
    if block_counts(text, old, new) != (0, 1):
        raise SystemExit(f"ControlNet preprocessor path verification failed (partial patch): {path}")
    try:
        compile(text, str(path), "exec", dont_inherit=True)
    except SyntaxError as exc:
        raise SystemExit(f"ControlNet preprocessor path verification failed (invalid Python): {path}: {exc}") from exc


def patched_text(path: pathlib.Path, old: str, new: str) -> str | None:
    """The verified patched text of path, or None when it is already patched. Anything but exactly-original or
    exactly-patched fails closed, as do CRLF line endings (read without newline translation: rewriting the file
    would otherwise turn every CRLF into LF)."""
    if not path.is_file():
        raise SystemExit(f"ControlNet source not found: {path}")
    text = path.read_bytes().decode("utf-8")
    if "\r" in text:
        raise SystemExit(f"unsupported ControlNet source (CRLF line endings): {path}")
    counts = block_counts(text, old, new)
    if counts == (0, 1):
        return None
    if counts != (1, 0):
        raise SystemExit(f"unsupported ControlNet source (old block x{counts[0]}, new block x{counts[1]}): {path}")
    text = text.replace(old, new, 1)
    verify(text, path, old, new)
    return text


def replace_atomically(path: pathlib.Path, text: str) -> None:
    """Write text (UTF-8, newlines untranslated) to a temporary file next to path, fsync it, give it path's mode
    (and owner when run as root, as run.sh does) and os.replace() path with it: a failure at any point leaves path
    untouched and removes the temporary file. A symlinked path is written through, as an in-place write would."""
    path = pathlib.Path(os.path.realpath(path))
    stat = path.stat()
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".gb10-tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as tmp:
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.chmod(tmp_name, stat.st_mode & 0o7777)
        if os.geteuid() == 0:
            os.chown(tmp_name, stat.st_uid, stat.st_gid)
        os.replace(tmp_name, path)
    except BaseException as exc:
        try:
            os.unlink(tmp_name)
        except OSError as cleanup:
            exc.add_note(f"could not remove the temporary file {tmp_name}: {cleanup}")
        raise


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit("usage: patch-controlnet-preprocessor-path.py /path/to/sd-webui-controlnet")
    root = pathlib.Path(argv[1]).resolve()
    # Every target is validated and its patched text verified before any file is written.
    pending = {}
    for relative, old, new in TARGETS:
        text = patched_text(root / relative, old, new)
        if text is not None:
            pending[root / relative] = text
    for path, text in pending.items():
        replace_atomically(path, text)
    if pending:
        print(f"Patched ControlNet preprocessor path plumbing under {root}")
    else:
        print(f"ControlNet preprocessor path plumbing already patched under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
