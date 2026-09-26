import multiprocessing
import os
import time
from pathlib import Path

from modules import persistent_artifact_cache


def _writer(path, payload):
    with persistent_artifact_cache.exclusive_lock(path):
        persistent_artifact_cache.atomic_write(path, payload)


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
