"""launch_utils.git_tag(): the version tag shown in infotext and /internal/sysinfo.

Contract, in order: a non-blank A1111_VERSION_TAG (the image build bakes it), else `git describe --tags` in a git
checkout, else "<none>".
"""
import subprocess
from types import SimpleNamespace

import pytest

from modules import launch_utils


@pytest.fixture
def git(monkeypatch, tmp_path):
    """A webui root in tmp_path without A1111_VERSION_TAG, and a recording subprocess.check_output that returns
    `result` (raises it, if it is an exception). git_tag() is uncached around the test."""
    fake = SimpleNamespace(calls=[], result="v1.10.1-123-gabcdef0\n")

    def check_output(args, **kwargs):
        fake.calls.append(args)
        if isinstance(fake.result, BaseException):
            raise fake.result
        return fake.result

    monkeypatch.setattr(launch_utils, "script_path", str(tmp_path))
    monkeypatch.delenv("A1111_VERSION_TAG", raising=False)
    monkeypatch.setattr(launch_utils.subprocess, "check_output", check_output)
    launch_utils.git_tag.cache_clear()
    yield fake
    launch_utils.git_tag.cache_clear()


def test_version_tag_env_wins_without_running_git(git, monkeypatch):
    monkeypatch.setattr(launch_utils, "_script_path_is_git_repo", lambda: True)
    monkeypatch.setenv("A1111_VERSION_TAG", "  deploy9-75a94f59 \n")

    assert launch_utils.git_tag() == "deploy9-75a94f59"
    assert git.calls == []


def test_blank_version_tag_env_falls_through_to_git_describe(git, monkeypatch, tmp_path):
    monkeypatch.setattr(launch_utils, "_script_path_is_git_repo", lambda: True)
    monkeypatch.setenv("A1111_VERSION_TAG", "   ")

    assert launch_utils.git_tag() == "v1.10.1-123-gabcdef0"
    assert git.calls == [[launch_utils.git, "-C", str(tmp_path), "describe", "--tags"]]


def test_failing_git_describe_is_none(git, monkeypatch):
    monkeypatch.setattr(launch_utils, "_script_path_is_git_repo", lambda: True)
    git.result = subprocess.CalledProcessError(128, ["git", "describe", "--tags"])

    assert launch_utils.git_tag() == "<none>"
    assert len(git.calls) == 1


def test_outside_a_git_checkout_is_none_without_running_git(git, monkeypatch):
    monkeypatch.setattr(launch_utils, "_script_path_is_git_repo", lambda: False)

    assert launch_utils.git_tag() == "<none>"
    assert git.calls == []


@pytest.mark.xfail(strict=True, reason="needs the docs stream's git_tag() without the CHANGELOG.md fallback; "
                                       "drop this marker when merging that branch")
def test_a_changelog_is_never_read_as_the_version(git, monkeypatch, tmp_path):
    monkeypatch.setattr(launch_utils, "_script_path_is_git_repo", lambda: False)
    (tmp_path / "CHANGELOG.md").write_text("## v1.10.1\n\n### Bug Fixes:\n", encoding="utf-8")

    assert launch_utils.git_tag() == "<none>"
    assert git.calls == []
