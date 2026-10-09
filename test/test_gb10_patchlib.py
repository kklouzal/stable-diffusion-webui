"""gb10/patchlib.py: the shared fail-closed contract of the gb10/patch-*.py deploy patchers.

The per-patcher suites (test_gb10_multidiffusion_performance_patcher.py, test_gb10_usdu_subcanvas.py,
test_gb10_tiled_extension_patchers.py) cover each patcher's blocks against the installed extensions.
"""
from __future__ import annotations

import os
import re
import resource
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from test.helpers import load_source

GB10 = Path(__file__).parents[1] / "gb10"
patchlib = load_source("patchlib", GB10 / "patchlib.py")
Block, apply_blocks = patchlib.Block, patchlib.apply_blocks

LABEL = "Fixture"
BLOCKS = [Block("A", "a = 1\n", "a = 2\n"), Block("B", "b = 1\n", "b = 2  # gb10\n", sentinel="# gb10")]
ORIGINAL = "a = 1\nb = 1\n"
PATCHED = "a = 2\nb = 2  # gb10\n"


def tree(tmp_path: Path, **files: str) -> dict[Path, list[Block]]:
    targets = {}
    for name, text in files.items():
        path = tmp_path / f"{name}.py"
        path.write_bytes(text.encode("utf-8"))
        targets[path] = BLOCKS
    return targets


def snapshot(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


@pytest.mark.parametrize("source,state", [(ORIGINAL, "original"), (PATCHED, "patched")])
def test_file_state_accepts_only_exact_original_or_patched(source: str, state: str):
    assert patchlib.file_state(source, BLOCKS, Path("x.py"), LABEL) == state


@pytest.mark.parametrize("source,message", [
    ("a = 3\nb = 1\n", "unsupported or partially patched Fixture source for A (original x0, patched x0"),
    (ORIGINAL + "a = 1\n", "for A (original x2, patched x0"),
    ("a = 2\nb = 1\n", "partially patched Fixture source"),
    (ORIGINAL + "# gb10\n", "for B (original x1, patched x0, sentinel x1"),  # stray copy of patched-only text
    (PATCHED + "# gb10\n", "for B (original x0, patched x1, sentinel x2"),
])
def test_file_state_fails_closed_on_drift_partial_patches_and_stray_sentinels(source: str, message: str):
    with pytest.raises(SystemExit, match=re.escape(message)):
        patchlib.file_state(source, BLOCKS, Path("x.py"), LABEL)


def test_occurrence_counts_are_exact():
    blocks = [Block("tiles", "\n    run(x)\n", "\n    run_tile(x)\n", count=2)]
    assert patchlib.file_state("\n    run(x)\n\n    run(x)\n", blocks, Path("x.py"), LABEL) == "original"
    with pytest.raises(SystemExit, match="expected x2"):
        patchlib.file_state("\n    run(x)\n\n    run_tile(x)\n", blocks, Path("x.py"), LABEL)


def test_apply_patches_then_verifies_idempotently_and_check_writes_nothing(tmp_path: Path):
    targets = tree(tmp_path, one=ORIGINAL, two=PATCHED)
    (tmp_path / "one.py").chmod(0o640)

    with pytest.raises(SystemExit, match="Fixture patch missing"):
        apply_blocks(targets, label=LABEL, check=True)
    assert (tmp_path / "one.py").read_text() == ORIGINAL

    assert apply_blocks(targets, label=LABEL, check=False) == [tmp_path / "one.py"]
    assert (tmp_path / "one.py").read_text() == PATCHED
    assert (tmp_path / "one.py").stat().st_mode & 0o777 == 0o640
    assert apply_blocks(targets, label=LABEL, check=False) == []
    assert apply_blocks(targets, label=LABEL, check=True) == []
    assert not list(tmp_path.glob("*.gb10-tmp"))


@pytest.mark.parametrize("second,message", [
    ("a = 3\nb = 1\n", "unsupported or partially patched"),
    (ORIGINAL.replace("\n", "\r\n"), "CRLF line endings"),
    (ORIGINAL + "def broken(:\n", "verification failed (invalid Python)"),
    (None, "Fixture source not found"),
])
def test_every_target_is_validated_before_any_is_written(tmp_path: Path, second: str | None, message: str):
    targets = tree(tmp_path, one=ORIGINAL) | ({} if second is None else tree(tmp_path, two=second))
    if second is None:
        targets[tmp_path / "missing.py"] = BLOCKS
    before = snapshot(tmp_path)

    with pytest.raises(SystemExit, match=re.escape(message)):
        apply_blocks(targets, label=LABEL, check=False)
    assert snapshot(tmp_path) == before


def test_patched_text_must_verify_before_it_is_written(tmp_path: Path):
    # A PATCHED text that re-introduces another block's ORIGINAL leaves a file that is not fully patched.
    broken = [Block("A", "a = 1\n", "a = 2\n"), Block("B", "b = 1\n", "b = 2\na = 1\n")]
    target = tmp_path / "one.py"
    target.write_text(ORIGINAL)
    with pytest.raises(SystemExit, match=r"verification failed \(not fully patched\)"):
        apply_blocks({target: broken}, label=LABEL, check=False)

    def reject(path: Path, text: str) -> None:
        raise SystemExit(f"rejected {path.name}")

    with pytest.raises(SystemExit, match="rejected one.py"):
        apply_blocks({target: BLOCKS}, label=LABEL, check=False, verify=reject)
    assert target.read_text() == ORIGINAL


# The release deployed before block A changed (A then patched to "a = 9"); B is unchanged.
PREVIOUS = [Block("A", "a = 1\n", "a = 9  # v1\n"), BLOCKS[1]]
PREVIOUS_PATCHED = "a = 9  # v1\nb = 2  # gb10\n"


def test_previous_release_is_reverted_and_patched_and_check_reports_it(tmp_path: Path):
    target = tmp_path / "one.py"
    target.write_text(PREVIOUS_PATCHED)
    previous = {target: PREVIOUS}

    with pytest.raises(SystemExit, match="Fixture patch outdated"):
        apply_blocks({target: BLOCKS}, label=LABEL, check=True, previous=previous)
    assert target.read_text() == PREVIOUS_PATCHED
    with pytest.raises(SystemExit, match="unsupported or partially patched Fixture source for A"):
        apply_blocks({target: BLOCKS}, label=LABEL, check=False)  # without `previous` it stays unsupported

    assert apply_blocks({target: BLOCKS}, label=LABEL, check=False, previous=previous) == [target]
    assert target.read_text() == PATCHED
    assert apply_blocks({target: BLOCKS}, label=LABEL, check=True, previous=previous) == []
    target.write_text(ORIGINAL)
    assert apply_blocks({target: BLOCKS}, label=LABEL, check=False, previous=previous) == [target]
    assert target.read_text() == PATCHED


@pytest.mark.parametrize("source", [
    "a = 9  # v1\nb = 1\n",  # partially patched by the previous release
    PREVIOUS_PATCHED + "a = 1\n",  # previous release plus a stray upstream copy: the revert would not round-trip
    PREVIOUS_PATCHED + "a = 9  # v1\n",
])
def test_previous_release_must_match_exactly(tmp_path: Path, source: str):
    target = tmp_path / "one.py"
    target.write_text(source)
    with pytest.raises(SystemExit, match="partially patched Fixture source"):
        apply_blocks({target: BLOCKS}, label=LABEL, check=False, previous={target: PREVIOUS})
    assert target.read_text() == source


def test_symlinked_target_is_written_through(tmp_path: Path):
    real = tmp_path / "real.py"
    real.write_text(ORIGINAL)
    link = tmp_path / "link.py"
    link.symlink_to(real)
    apply_blocks({link: BLOCKS}, label=LABEL, check=False)
    assert link.is_symlink() and real.read_text() == PATCHED


def test_write_failing_midway_leaves_the_target_intact(tmp_path: Path):
    """RLIMIT_FSIZE with SIGXFSZ ignored makes the temporary-file write fail with EFBIG partway through, as on a full
    disk: the target keeps its bytes and mode and no temporary file is left behind."""
    target = tmp_path / "one.py"
    target.write_bytes((ORIGINAL + "#" * 200 + "\n").encode("utf-8"))
    target.chmod(0o640)
    before = snapshot(tmp_path)
    script = (
        "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); import patchlib\n"
        "patchlib.replace_atomically(Path(sys.argv[2]), Path(sys.argv[2]).read_text().replace('a = 1', 'a = 2'))\n"
    )

    def limit_file_size():
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        resource.setrlimit(resource.RLIMIT_FSIZE, (64, 64))

    result = subprocess.run([sys.executable, "-c", script, str(GB10), str(target)], capture_output=True, text=True, preexec_fn=limit_file_size)

    assert result.returncode != 0 and "File too large" in result.stderr
    assert snapshot(tmp_path) == before
    assert os.stat(target).st_mode & 0o777 == 0o640
