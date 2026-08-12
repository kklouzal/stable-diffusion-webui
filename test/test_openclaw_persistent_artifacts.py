import json
import multiprocessing
import os
import time
from pathlib import Path

from modules import compile_cache_namespace, persistent_artifact_cache


def _writer(path, payload):
    with persistent_artifact_cache.exclusive_lock(path):
        persistent_artifact_cache.atomic_write(path, payload)


def test_strong_identity_separates_same_size_same_mtime_replacement(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"a" * 64)
    identity_a = persistent_artifact_cache.strong_file_identity(path)
    path.write_bytes(b"b" * 64)
    os.utime(path, ns=(identity_a["mtime_ns"], identity_a["mtime_ns"]))
    identity_b = persistent_artifact_cache.strong_file_identity(path)
    assert identity_a["size"] == identity_b["size"]
    assert identity_a["mtime_ns"] == identity_b["mtime_ns"]
    assert identity_a["sha256"] != identity_b["sha256"]


def test_atomic_publication_leaves_no_partial_and_concurrent_writers(tmp_path):
    path = tmp_path / "artifact.pt"
    payloads = [bytes([index]) * 8192 for index in range(1, 5)]
    processes = [multiprocessing.Process(target=_writer, args=(str(path), payload)) for payload in payloads]
    for process in processes:
        process.start()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0
    assert path.read_bytes() in payloads
    assert not list(tmp_path.glob(".partial-*"))


def test_partial_cleanup_is_bounded_and_dry_runnable(tmp_path):
    partial = tmp_path / ".partial-crash"
    partial.write_bytes(b"incomplete")
    old = time.time() - 7200
    os.utime(partial, (old, old))
    assert persistent_artifact_cache.cleanup_partial_files(tmp_path, older_than_seconds=3600, dry_run=True) == [str(partial)]
    assert partial.exists()
    persistent_artifact_cache.cleanup_partial_files(tmp_path, older_than_seconds=3600, dry_run=False)
    assert not partial.exists()


def test_quota_order_bounds_dry_run_and_active_lease(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"artifact-{index}.pt"
        path.write_bytes(bytes([index]) * 100)
        timestamp = time.time() - (300 - index)
        os.utime(path, (timestamp, timestamp))
        paths.append(path)
    with persistent_artifact_cache.active_lease(paths[0]):
        report = persistent_artifact_cache.enforce_quota(paths, max_bytes=100, dry_run=True)
        assert report["protected"] == [str(paths[0])]
        assert [Path(item["primary"]).name for item in report["evicted"]] == ["artifact-1.pt", "artifact-2.pt"]
        assert all(path.exists() for path in paths)
        applied = persistent_artifact_cache.enforce_quota(paths, max_bytes=100, dry_run=False)
        assert applied["within_quota"]
        assert paths[0].exists() and not paths[1].exists() and not paths[2].exists()


def test_compile_namespace_changes_with_dependency_runtime_and_options(tmp_path):
    base = dict(app_revision="abc", dependencies={"torch": "1"}, runtime={"sm": [12, 1]}, options={"dtype": "bf16"})
    first = compile_cache_namespace.namespace_identity(**base)
    for field, value in (("dependencies", {"torch": "2"}), ("runtime", {"sm": [10, 0]}), ("options", {"dtype": "fp16"})):
        changed = dict(base)
        changed[field] = value
        assert compile_cache_namespace.namespace_identity(**changed)["digest"] != first["digest"]
    environ = {}
    configured = compile_cache_namespace.configure_environment(tmp_path, first, environ)
    assert json.loads(Path(configured["manifest"]).read_text())["digest"] == first["digest"]
    assert set(environ) == {"TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH"}


def test_compile_cleanup_is_non_destructive_plan_for_app_namespaces(tmp_path):
    for provider in ("torchinductor", "triton", "cuda"):
        (tmp_path / provider / "keep").mkdir(parents=True)
        (tmp_path / provider / "stale").mkdir()
    plan = compile_cache_namespace.plan_namespace_cleanup(tmp_path, keep={"keep"})
    assert len(plan) == 3 and all(Path(path).name == "stale" for path in plan)
    assert all(Path(path).is_dir() for path in plan)
