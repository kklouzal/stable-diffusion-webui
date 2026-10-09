import os
from types import SimpleNamespace

import pytest

from test.helpers import load_source, module


@pytest.fixture
def cache_module(tmp_path):
    return load_source("cache_under_test", "modules/cache.py", {"modules.paths": module("modules.paths", data_path=str(tmp_path), script_path=str(tmp_path))})


@pytest.fixture
def hashes_module(cache_module):
    return load_source("hashes_under_test", "modules/hashes.py", {
        "modules": module("modules", package=True, cache=cache_module),
        "modules.cache": cache_module,
        "modules.shared": module("modules.shared", cmd_opts=SimpleNamespace(no_hashing=True)),
        "modules.errors": module("modules.errors", report=lambda *args, **kwargs: None),
        "modules.persistent_artifact_cache": load_source("modules.persistent_artifact_cache", "modules/persistent_artifact_cache.py"),
    })


def replace_keeping_size_and_mtime(path, content):
    """Replaces the file by rename (as cp/rsync/tar extraction do) with equal-size content and the old mtime."""
    old = os.stat(path)
    replacement = path.with_name(path.name + ".tmp")
    replacement.write_bytes(content)
    os.utime(replacement, ns=(old.st_atime_ns, old.st_mtime_ns))
    os.replace(replacement, path)
    new = os.stat(path)
    assert (new.st_size, new.st_mtime_ns) == (old.st_size, old.st_mtime_ns)


def test_file_cache_key_changes_when_the_file_is_replaced_with_equal_size_and_mtime(cache_module, tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"weights-A")
    loaded = {}
    first_key = cache_module.file_cache_key(str(checkpoint), "sha")
    loaded[first_key] = "state dict of A"

    replace_keeping_size_and_mtime(checkpoint, b"weights-B")
    second_key = cache_module.file_cache_key(str(checkpoint), "sha")
    cache_module.drop_stale_file_entries(loaded, second_key)

    assert second_key != first_key
    assert second_key not in loaded and loaded == {}


def test_file_cache_key_is_stable_for_an_unchanged_file_and_tolerates_a_missing_one(cache_module, tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"weights")

    assert cache_module.file_cache_key(str(checkpoint)) == cache_module.file_cache_key(str(checkpoint))
    missing = cache_module.file_cache_key(str(tmp_path / "missing.safetensors"), "x")
    assert missing[0] == str(tmp_path / "missing.safetensors") and missing[2:] == ("x",)


class _CulledEntryCache(dict):
    """A disk cache that culls the entry between a membership test and the read."""

    def __contains__(self, key):
        return True

    def __getitem__(self, key):
        raise KeyError(key)


def test_sha256_from_cache_survives_an_entry_culled_while_it_is_read(hashes_module, monkeypatch, tmp_path):
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"weights")
    monkeypatch.setattr(hashes_module, "cache", lambda subsection: _CulledEntryCache())

    assert hashes_module.sha256_from_cache(str(model), "checkpoint/model") is None


def test_sha256_from_cache_returns_the_hash_of_the_current_revision_only(hashes_module, cache_module, monkeypatch, tmp_path):
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"weights")
    entries = {"checkpoint/model": {"source_revision": cache_module.file_revision(os.stat(model)), "sha256": "abc"}}
    monkeypatch.setattr(hashes_module, "cache", lambda subsection: entries)

    assert hashes_module.sha256_from_cache(str(model), "checkpoint/model") == "abc"
    replace_keeping_size_and_mtime(model, b"weightz")
    assert hashes_module.sha256_from_cache(str(model), "checkpoint/model") is None
