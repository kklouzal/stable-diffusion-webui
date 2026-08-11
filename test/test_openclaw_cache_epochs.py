from __future__ import annotations

import ast
import asyncio
import contextvars
import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from modules import openclaw_cache_epochs

HEX_DIGEST = re.compile(r"^[0-9a-f]{16}$")


@pytest.fixture(autouse=True)
def reset_registries():
    openclaw_cache_epochs.reset_for_tests()
    yield
    openclaw_cache_epochs.reset_for_tests()


def digest(epochs: dict[str, int]) -> str:
    payload = json.dumps(epochs, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def test_epoch_dimensions_are_declared_dependencies_except_request_owner():
    assert openclaw_cache_epochs.EPOCH_DIMENSIONS == tuple(
        dimension
        for dimension in openclaw_cache_epochs.DEPENDENCY_DIMENSIONS
        if dimension != "request_owner"
    )
    assert "request_owner" not in openclaw_cache_epochs.EPOCH_DIMENSIONS


def test_epoch_bumps_are_monotonic_and_reasons_are_validated():
    dimension = "checkpoint_object_epoch"
    assert openclaw_cache_epochs.bump_epoch(dimension, reason="manual") == 1
    assert openclaw_cache_epochs.bump_epoch(dimension, reason="dependency_changed") == 2

    with pytest.raises(ValueError, match="unknown epoch dimension"):
        openclaw_cache_epochs.bump_epoch("request_owner", reason="manual")
    with pytest.raises(ValueError, match="invalid epoch bump reason"):
        openclaw_cache_epochs.bump_epoch(dimension, reason="private/raw/reason")

    assert openclaw_cache_epochs.epoch_snapshot()["epochs"][dimension] == 2


def test_epoch_snapshot_is_atomic_with_stable_opaque_digest():
    openclaw_cache_epochs.bump_epoch("vae_object_epoch", reason="vae_loaded")
    first = openclaw_cache_epochs.epoch_snapshot()
    second = openclaw_cache_epochs.epoch_snapshot()

    assert first == second
    assert HEX_DIGEST.fullmatch(first["digest"])
    assert first["digest"] == digest(first["epochs"])

    openclaw_cache_epochs.bump_epoch("vae_object_epoch", reason="vae_unloaded")
    third = openclaw_cache_epochs.epoch_snapshot()
    assert third["digest"] == digest(third["epochs"])
    assert third["digest"] != first["digest"]


def test_concurrent_epoch_bumps_are_not_lost_and_snapshots_are_consistent():
    workers = 12
    bumps_per_worker = 400
    dimension = "precision_epoch"
    stop = threading.Event()
    observed: list[dict[str, object]] = []

    def sample_snapshots():
        while not stop.is_set():
            observed.append(openclaw_cache_epochs.epoch_snapshot())

    sampler = threading.Thread(target=sample_snapshots)
    sampler.start()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(
                lambda _: [
                    openclaw_cache_epochs.bump_epoch(
                        dimension, reason="precision_changed"
                    )
                    for _ in range(bumps_per_worker)
                ],
                range(workers),
            ))
    finally:
        stop.set()
        sampler.join()

    final = openclaw_cache_epochs.epoch_snapshot()
    assert final["epochs"][dimension] == workers * bumps_per_worker
    assert observed
    assert all(item["digest"] == digest(item["epochs"]) for item in observed)


def test_generation_owner_supports_same_context_nested_reentry():
    with openclaw_cache_epochs.generation_owner():
        assert openclaw_cache_epochs.generation_owner_public_summary() == {
            "active": True,
            "depth": 1,
            "owner_generation": 1,
            "acquire_count": 1,
            "release_count": 0,
            "reject_count": 0,
        }
        with openclaw_cache_epochs.generation_owner():
            assert openclaw_cache_epochs.generation_owner_public_summary()["depth"] == 2

    assert openclaw_cache_epochs.generation_owner_public_summary() == {
        "active": False,
        "depth": 0,
        "owner_generation": 1,
        "acquire_count": 2,
        "release_count": 2,
        "reject_count": 0,
    }


def test_foreign_concurrent_generation_owner_is_rejected():
    entered = threading.Event()
    release = threading.Event()

    def active_owner():
        with openclaw_cache_epochs.generation_owner():
            entered.set()
            assert release.wait(timeout=5)

    thread = threading.Thread(target=active_owner)
    thread.start()
    assert entered.wait(timeout=5)
    try:
        with pytest.raises(
            openclaw_cache_epochs.GenerationOwnerError,
            match="already active",
        ):
            with openclaw_cache_epochs.generation_owner():
                pass
    finally:
        release.set()
        thread.join(timeout=5)

    summary = openclaw_cache_epochs.generation_owner_public_summary()
    assert not summary["active"]
    assert summary["reject_count"] == 1


def test_copied_context_cannot_reenter_owner_from_a_foreign_thread():
    outcome = []
    copied_context = None
    release = threading.Event()

    with openclaw_cache_epochs.generation_owner():
        copied_context = contextvars.copy_context()

        def foreign_reentry():
            try:
                copied_context.run(
                    lambda: openclaw_cache_epochs.generation_owner().__enter__()
                )
            except openclaw_cache_epochs.GenerationOwnerError:
                outcome.append("rejected")
            finally:
                release.set()

        thread = threading.Thread(target=foreign_reentry)
        thread.start()
        assert release.wait(timeout=5)
        thread.join(timeout=5)

    assert outcome == ["rejected"]
    assert openclaw_cache_epochs.generation_owner_public_summary()["reject_count"] == 1


def test_copied_context_cannot_reenter_owner_from_a_foreign_async_task():
    async def exercise():
        entered = asyncio.Event()
        release = asyncio.Event()

        async def active_owner():
            with openclaw_cache_epochs.generation_owner():
                entered.set()
                await release.wait()

        task = asyncio.create_task(active_owner())
        await entered.wait()
        try:
            with pytest.raises(
                openclaw_cache_epochs.GenerationOwnerError,
                match="already active or mismatched",
            ):
                with openclaw_cache_epochs.generation_owner():
                    pass
        finally:
            release.set()
            await task

    asyncio.run(exercise())
    assert openclaw_cache_epochs.generation_owner_public_summary()["reject_count"] == 1


def test_generation_owner_mismatch_fails_closed():
    registry = openclaw_cache_epochs.generation_owner_registry
    with pytest.raises(
        openclaw_cache_epochs.GenerationOwnerError,
        match="state mismatch",
    ):
        with openclaw_cache_epochs.generation_owner():
            with registry._lock:
                registry._active_owner = object()

    summary = openclaw_cache_epochs.generation_owner_public_summary()
    assert summary["active"] is False
    assert summary["depth"] == 0
    assert summary["reject_count"] == 1


def test_generation_owner_exception_cleanup():
    with pytest.raises(RuntimeError, match="generation failed"):
        with openclaw_cache_epochs.generation_owner():
            raise RuntimeError("generation failed")

    summary = openclaw_cache_epochs.generation_owner_public_summary()
    assert summary["active"] is False
    assert summary["depth"] == 0
    assert summary["acquire_count"] == summary["release_count"] == 1


def test_public_summaries_are_sanitized_and_bounded():
    secret = "raw-task-id-and-owner-token-must-not-leak"
    with pytest.raises(ValueError):
        openclaw_cache_epochs.bump_epoch("device_epoch", reason=secret)
    with openclaw_cache_epochs.generation_owner():
        public = openclaw_cache_epochs.snapshot()

    serialized = json.dumps(public, sort_keys=True)
    assert secret not in serialized
    assert set(public["epochs"]) == {
        "digest", "epochs", "bump_counts", "reason_counts", "reason_vocabulary",
    }
    assert set(public["epochs"]["epochs"]) == set(openclaw_cache_epochs.EPOCH_DIMENSIONS)
    assert set(public["generation_owner"]) == {
        "active", "depth", "owner_generation", "acquire_count", "release_count",
        "reject_count",
    }
    assert len(public["epochs"]["reason_vocabulary"]) == len(
        openclaw_cache_epochs.EPOCH_BUMP_REASONS
    )


def attribute_name(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def with_name(item: ast.withitem) -> str | None:
    expression = item.context_expr
    if isinstance(expression, ast.Call):
        expression = expression.func
    return attribute_name(expression)


def test_api_generation_regions_lock_queue_before_opaque_owner():
    source = Path("modules/api/api.py").read_text()
    tree = ast.parse(source)
    api = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Api"
    )
    owner_contexts = [
        node
        for node in ast.walk(api)
        if isinstance(node, ast.With)
        for item in node.items
        if with_name(item) == "openclaw_cache_epochs.generation_owner"
    ]
    assert len(owner_contexts) == 2

    for method_name, processing_class in (
        ("text2imgapi", "StableDiffusionProcessingTxt2Img"),
        ("img2imgapi", "StableDiffusionProcessingImg2Img"),
    ):
        method = next(
            node
            for node in api.body
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        )
        queue_contexts = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.With)
            and any(with_name(item) == "self.queue_lock" for item in node.items)
        ]
        assert len(queue_contexts) == 1
        queue_context = queue_contexts[0]
        owner_context = next(
            node
            for node in queue_context.body
            if isinstance(node, ast.With)
            and any(
                with_name(item) == "openclaw_cache_epochs.generation_owner"
                for item in node.items
            )
        )
        assert any(
            isinstance(node, ast.Name) and node.id == processing_class
            for node in ast.walk(owner_context)
        )
