"""Shared contract of the gb10/patch-*.py deploy patchers for host-mounted third-party extensions.

gb10/run.sh runs each patcher as `sudo python3 gb10/patch-<name>.py ROOT`; Python puts the script's directory first
on sys.path, so this sibling module imports without any path setup.

Every patcher holds to the same rules:
- Sources are UTF-8 with LF line endings and are read without newline translation; CRLF fails closed instead of being
  rewritten.
- A target is either exactly original (it is patched) or exactly patched (it is verified and left alone). Anything
  else (unknown upstream text, a partial patch) raises SystemExit and aborts the deploy.
- Every target is validated and its patched text verified before any file is written; each file is then replaced
  atomically, keeping its mode and owner.
- --check writes nothing and fails unless every target is patched.
"""
from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple


class Block(NamedTuple):
    """One exact ORIGINAL -> PATCHED text replacement.

    A file is original when it holds `original` exactly `count` times and `patched` never, and patched when the
    reverse holds. `sentinel`, when set, is text unique to the patched block (for example a helper's `def` line): it
    must occur `count` times in a patched file and never in an original one, so a stray copy fails closed.
    """

    name: str
    original: str
    patched: str
    count: int = 1
    sentinel: str | None = None


def parse_cli(description: str) -> argparse.Namespace:
    """The patchers' shared command line: `[--check] PATH` (PATH: the extension checkout or the target file)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("path", type=Path, help="extension checkout (or the target file itself)")
    parser.add_argument("--check", action="store_true", help="verify that every target is patched; write nothing")
    return parser.parse_args()


def read_lf(path: Path, label: str) -> str:
    """The UTF-8 text of path without newline translation; a missing file or a CR fails closed."""
    if not path.is_file():
        raise SystemExit(f"{label} source not found: {path}")
    text = path.read_bytes().decode("utf-8")
    if "\r" in text:
        raise SystemExit(f"unsupported {label} source (CRLF line endings): {path}")
    return text


def verify_python(text: str, path: Path, label: str) -> None:
    try:
        compile(text, str(path), "exec", dont_inherit=True)
    except SyntaxError as exc:
        raise SystemExit(f"{label} patch verification failed (invalid Python): {path}: {exc}") from exc


def replace_atomically(path: Path, text: str) -> None:
    """Write text (UTF-8, newlines untranslated) to a temporary file next to path, fsync it, give it path's mode
    (and owner when run as root, as run.sh does) and os.replace() path with it: a failure at any point leaves path
    untouched and removes the temporary file. A symlinked path is written through, as an in-place write would."""
    path = Path(os.path.realpath(path))
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


def _block_state(source: str, block: Block) -> str | None:
    """The block's state in source: original, patched, or None for any other combination of its texts."""
    sentinel = source.count(block.sentinel) if block.sentinel is not None else None
    counts = (source.count(block.original), source.count(block.patched))
    if counts == (block.count, 0) and sentinel in (None, 0):
        return "original"
    if counts == (0, block.count) and sentinel in (None, block.count):
        return "patched"
    return None


def file_state(source: str, blocks: Sequence[Block], path: Path, label: str) -> str:
    """Return "original" or "patched" for a target's text; raise SystemExit for anything else."""
    states = set()
    for block in blocks:
        state = _block_state(source, block)
        if state is None:
            sentinel = f", sentinel x{source.count(block.sentinel)}" if block.sentinel is not None else ""
            raise SystemExit(
                f"unsupported or partially patched {label} source for {block.name} (original x{source.count(block.original)}, "
                f"patched x{source.count(block.patched)}{sentinel}, expected x{block.count}): {path}"
            )
        states.add(state)
    if len(states) != 1:
        raise SystemExit(f"partially patched {label} source: {path}")
    return states.pop()


def apply_blocks(
    targets: Mapping[Path, Sequence[Block]],
    *,
    label: str,
    check: bool,
    verify: Callable[[Path, str], None] | None = None,
) -> list[Path]:
    """Patch every original target and verify every target, writing nothing until all of them verify.

    `verify(path, text)` adds a patcher-specific post-condition on the patched text (raise SystemExit to fail). Each
    patched text must also be fully patched (a PATCHED text that re-introduces another block's ORIGINAL fails here)
    and valid Python. Returns the written paths; with check=True nothing is written and an original target fails.
    """
    pending: dict[Path, str] = {}
    for path, blocks in targets.items():
        source = read_lf(path, label)
        if file_state(source, blocks, path, label) == "original":
            if check:
                raise SystemExit(f"{label} patch missing: {path}")
            for block in blocks:
                source = source.replace(block.original, block.patched)
            pending[path] = source
        if any(_block_state(source, block) != "patched" for block in blocks):
            raise SystemExit(f"{label} patch verification failed (not fully patched): {path}")
        verify_python(source, path, label)
        if verify is not None:
            verify(path, source)
    for path, text in pending.items():
        replace_atomically(path, text)
    return list(pending)
