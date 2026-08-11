from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

from modules import openclaw_cache_epochs


HEX_DIGEST = re.compile(r"^[0-9a-f]{16}$")
FORBIDDEN = (
    "ultra secret prompt", "negative secret", "image/png", "base64",
    "/home/private/models/secret-model.safetensors", "secret-model", "private-lora",
    "authorization", "api_token", "user@example.com",
)


def setup_function():
    openclaw_cache_epochs.reset_for_tests()


def family(snapshot, family_id):
    return next(item for item in snapshot["families"] if item["id"] == family_id)


def test_counter_accuracy_and_snapshot_reset_behavior():
    openclaw_cache_epochs.observe("E05", "hit", reason="cache_hit", semantic_key=("opaque", 1), count=2)
    openclaw_cache_epochs.observe("E05", "miss", reason="cache_miss", semantic_key=("opaque", 2))
    openclaw_cache_epochs.observe("E05", "publish", reason="published", semantic_key=("opaque", 2))
    before = openclaw_cache_epochs.snapshot()
    item = family(before, "E05")
    assert item["events"]["hit"] == 2
    assert item["events"]["miss"] == 1
    assert item["events"]["publish"] == 1
    assert item["reason_counts"] == {"cache_hit": 2, "cache_miss": 1, "published": 1}

    openclaw_cache_epochs.reset_for_tests()
    after = openclaw_cache_epochs.snapshot()
    assert family(after, "E05")["events"]["hit"] == 0
    assert before == before  # an earlier snapshot is immutable by later registry reset


def test_thread_safety_and_counter_accuracy():
    workers = 16
    iterations = 500

    def record(worker):
        for offset in range(iterations):
            openclaw_cache_epochs.observe("E11", "hit", reason="cache_hit", semantic_key=(worker, offset % 8))
            openclaw_cache_epochs.observe("E11", "miss", reason="cache_miss")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(record, range(workers)))

    item = family(openclaw_cache_epochs.snapshot(), "E11")
    assert item["events"]["hit"] == workers * iterations
    assert item["events"]["miss"] == workers * iterations
    assert len(item["semantic_key_digests"]) <= openclaw_cache_epochs.MAX_KEY_SUMMARIES


def test_bounded_reason_key_and_dependency_retention():
    for index in range(100):
        openclaw_cache_epochs.observe("E12", "reject", reason=f"not-allowed-{index}", semantic_key=("key", index))
        openclaw_cache_epochs.observe_dependency("E12", {"lora_source_epoch": index, "unknown_secret": f"secret-{index}"})

    item = family(openclaw_cache_epochs.snapshot(), "E12")
    assert item["reason_counts"] == {"other": 100, "dependency_snapshot": 100}
    assert len(item["semantic_key_digests"]) == openclaw_cache_epochs.MAX_KEY_SUMMARIES
    assert len(item["dependency_observation_digests"]) == openclaw_cache_epochs.MAX_KEY_SUMMARIES
    assert item["semantic_key_observations"] == 100
    assert item["dependency_observations"] == 100
    assert all(HEX_DIGEST.fullmatch(value) for value in item["semantic_key_digests"])
    assert all(HEX_DIGEST.fullmatch(value) for value in item["dependency_observation_digests"])


def test_telemetry_is_sanitized():
    semantic_key = {
        "prompt": "ultra secret prompt",
        "negative": "negative secret",
        "image": "data:image/png;base64,DEADBEEF",
        "path": "/home/private/models/secret-model.safetensors",
        "model": "secret-model",
        "lora": "private-lora",
        "authorization": "Bearer api_token",
        "user": "user@example.com",
    }
    openclaw_cache_epochs.observe("E05", "miss", reason="cache_miss", semantic_key=semantic_key)
    payload = json.dumps(openclaw_cache_epochs.snapshot(), sort_keys=True).lower()
    for forbidden in FORBIDDEN:
        assert forbidden.lower() not in payload
    assert "semantic_key_digests" in payload
    assert re.search(r"[0-9a-f]{16}", payload)


def test_all_caches_declare_dependency_contract():
    snapshot = openclaw_cache_epochs.snapshot()
    assert [item["id"] for item in snapshot["families"]] == [f"E{index:02d}" for index in range(1, 35)]
    for item in snapshot["families"]:
        assert item["name"]
        assert item["contract"]["dirty_on"]
        assert item["contract"]["stable_on"]
        assert item["source_cache_ids"]
        assert isinstance(item["instrumented"], bool)
    source_ids = [source_id for item in snapshot["families"] for source_id in item["source_cache_ids"]]
    assert len(source_ids) == 70
    assert len(set(source_ids)) == 70


def test_endpoint_schema_is_read_only_and_bounded():
    snapshot = openclaw_cache_epochs.snapshot()
    assert snapshot["schema_version"] == 1
    assert snapshot["read_only"] is True
    assert set(snapshot) == {
        "schema_version", "read_only", "limits", "reason_code_vocabulary",
        "dependency_dimensions", "totals", "families",
    }
    assert set(snapshot["totals"]) == openclaw_cache_epochs.EVENTS
    assert set(snapshot["reason_code_vocabulary"]) == openclaw_cache_epochs.REASON_CODES
    for item in snapshot["families"]:
        assert set(item) == {
            "id", "name", "instrumented", "events", "reason_counts",
            "semantic_key_digests", "semantic_key_observations",
            "dependency_observation_digests", "dependency_observations",
            "current_size", "capacity", "source_cache_ids", "contract",
        }
        assert set(item["events"]) == openclaw_cache_epochs.EVENTS


def test_size_provider_failures_do_not_break_snapshot():
    openclaw_cache_epochs.register_size_provider("E08", lambda: (_ for _ in ()).throw(RuntimeError("private path")))
    item = family(openclaw_cache_epochs.snapshot(), "E08")
    assert item["current_size"] is None
    assert item["capacity"] is None
