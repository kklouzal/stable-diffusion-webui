"""gb10/build.sh's provenance label, run for real in a scratch checkout against stand-in sudo/docker: the revision is
marked -dirty only for changes to what the build reads (.dockerignore's allowlist, .dockerignore and the Dockerfile),
including untracked files there, and never for changes outside the build context."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

FAKE_DOCKER = r'''
import json, os, sys
argv = sys.argv[1:]
if argv[:3] == ["buildx", "imagetools", "inspect"]:
    print('"sha256:' + "a" * 64 + '"')
elif argv[:2] == ["image", "inspect"]:
    sys.exit(1)
elif argv[0] == "build":
    labels = dict(argv[i + 1].split("=", 1) for i, a in enumerate(argv) if a == "--label")
    with open(os.environ["FAKE_BUILD_LABELS"], "w") as f:
        json.dump(labels, f)
else:
    sys.exit(f"unsupported fake docker command {argv}")
'''


def _git(checkout, *args):
    subprocess.run(["git", "-C", str(checkout), *args], check=True, capture_output=True)


@pytest.fixture
def checkout(tmp_path):
    checkout = tmp_path / "checkout"
    (checkout / "gb10").mkdir(parents=True)
    shutil.copy2(ROOT / "gb10" / "build.sh", checkout / "gb10" / "build.sh")
    shutil.copy2(ROOT / ".dockerignore", checkout / ".dockerignore")
    (checkout / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (checkout / "modules").mkdir()
    (checkout / "modules" / "a.py").write_text("a\n", encoding="utf-8")
    (checkout / "test").mkdir()
    (checkout / "test" / "t.py").write_text("t\n", encoding="utf-8")
    (checkout / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    _git(checkout, "init", "-q")
    _git(checkout, "add", "-A")
    _git(checkout, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c")
    _git(checkout, "tag", "v1")
    return checkout


def _revision(tmp_path, checkout, **env):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "sudo").write_text('#!/bin/sh\nexec "$@"\n', encoding="utf-8")
    (bin_dir / "docker").write_text(f"#!{sys.executable}\n" + FAKE_DOCKER, encoding="utf-8")
    for tool in ("sudo", "docker"):
        (bin_dir / tool).chmod(0o755)
    labels = tmp_path / "labels.json"
    result = subprocess.run(
        ["bash", str(checkout / "gb10" / "build.sh")],
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "BUILD_CGROUP_PARENT": "", "FAKE_BUILD_LABELS": str(labels), **env},
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    head = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    revision = json.loads(labels.read_text(encoding="utf-8"))["org.opencontainers.image.revision"]
    assert revision in (head, head + "-dirty")
    return revision.removeprefix(head) or "clean"


def test_a_clean_build_context_is_clean_whatever_else_changed(tmp_path, checkout):
    (checkout / ".claude").mkdir()
    (checkout / ".claude" / "notes").write_text("x\n", encoding="utf-8")
    (checkout / "test" / "t.py").write_text("changed\n", encoding="utf-8")
    (checkout / "scratch.txt").write_text("x\n", encoding="utf-8")
    (checkout / "modules" / "__pycache__").mkdir()
    (checkout / "modules" / "__pycache__" / "a.pyc").write_text("x\n", encoding="utf-8")

    assert _revision(tmp_path, checkout) == "clean"


@pytest.mark.parametrize("change", [
    "modified tracked source", "untracked source", "deleted source", "Dockerfile", ".dockerignore", "external Dockerfile",
])
def test_a_change_the_build_reads_marks_the_revision_dirty(tmp_path, checkout, change):
    env = {}
    if change == "modified tracked source":
        (checkout / "modules" / "a.py").write_text("changed\n", encoding="utf-8")
    elif change == "untracked source":
        (checkout / "modules" / "new.py").write_text("new\n", encoding="utf-8")
    elif change == "deleted source":
        (checkout / "modules" / "a.py").unlink()
    elif change == "Dockerfile":
        (checkout / "Dockerfile").write_text("FROM scratch\nRUN true\n", encoding="utf-8")
    elif change == ".dockerignore":
        with (checkout / ".dockerignore").open("a", encoding="utf-8") as f:
            f.write("!test\n")
    else:
        external = tmp_path / "Dockerfile.external"
        external.write_text("FROM scratch\n", encoding="utf-8")
        env["DOCKERFILE"] = str(external)

    assert _revision(tmp_path, checkout, **env) == "-dirty"


def test_an_allowlist_pattern_is_refused(tmp_path, checkout):
    with (checkout / ".dockerignore").open("a", encoding="utf-8") as f:
        f.write("!scripts/*.py\n")

    result = subprocess.run(["bash", str(checkout / "gb10" / "build.sh")], env={**os.environ, "BUILD_CGROUP_PARENT": ""}, capture_output=True, text=True, timeout=60)

    assert result.returncode == 1
    assert "allows 'scripts/*.py', a pattern" in result.stderr
