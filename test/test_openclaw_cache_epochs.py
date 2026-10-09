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

    assert openclaw_cache_epochs.epoch_subset((dimension,)) == ((dimension, 2),)


def test_public_summary_digest_is_stable_and_opaque():
    openclaw_cache_epochs.bump_epoch("vae_object_epoch", reason="vae_loaded")
    first = openclaw_cache_epochs.epoch_public_summary()
    second = openclaw_cache_epochs.epoch_public_summary()

    assert first == second
    assert HEX_DIGEST.fullmatch(first["digest"])
    assert first["digest"] == digest(first["epochs"])

    openclaw_cache_epochs.bump_epoch("vae_object_epoch", reason="vae_unloaded")
    third = openclaw_cache_epochs.epoch_public_summary()
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
            observed.append(openclaw_cache_epochs.epoch_public_summary())

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

    final = openclaw_cache_epochs.epoch_public_summary()
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
    assert len(owner_contexts) == 1

    methods = {node.name: node for node in api.body if isinstance(node, ast.FunctionDef)}
    run_task = methods["_run_generation_task"]
    queue_contexts = [
        node
        for node in ast.walk(run_task)
        if isinstance(node, ast.With)
        and any(with_name(item) == "self.queue_lock" for item in node.items)
    ]
    assert len(queue_contexts) == 1
    owner_context = next(
        node
        for node in queue_contexts[0].body
        if isinstance(node, ast.With)
        and any(
            with_name(item) == "openclaw_cache_epochs.generation_owner"
            for item in node.items
        )
    )
    assert any(
        isinstance(node, ast.Name) and node.id == "processing_class"
        for node in ast.walk(owner_context)
    )
    names = {node.id for node in ast.walk(run_task) if isinstance(node, ast.Name)}
    assert {"StableDiffusionProcessingTxt2Img", "StableDiffusionProcessingImg2Img"} <= names

    for method_name in ("text2imgapi", "img2imgapi"):
        calls = [
            node
            for node in ast.walk(methods[method_name])
            if isinstance(node, ast.Call) and attribute_name(node.func) == "self._run_generation_task"
        ]
        assert len(calls) == 1


def test_epoch_subset_changes_only_when_a_member_dimension_is_bumped():
    import modules.openclaw_cache_epochs as epochs
    relevant = (
        'checkpoint_object_epoch', 'conditioner_epoch', 'textual_inversion_epoch',
        'tokenizer_epoch', 'conditioning_hook_epoch', 'lora_applied_epoch',
        'device_epoch',
    )
    epochs.epoch_registry.reset()
    baseline = epochs.epoch_subset(relevant)
    for dimension in relevant:
        epochs.epoch_registry.reset()
        epochs.bump_epoch(dimension, reason='manual')
        assert epochs.epoch_subset(relevant) != baseline
    for dimension in ('vae_object_epoch', 'vae_bytes_epoch'):
        epochs.epoch_registry.reset()
        epochs.bump_epoch(dimension, reason='manual')
        assert epochs.epoch_subset(relevant) == baseline


def test_epoch_transaction_blocks_generic_bump_until_cache_publication():
    import threading

    import modules.openclaw_cache_epochs as epochs

    epochs.epoch_registry.reset()
    compute_entered = threading.Event()
    allow_publish = threading.Event()
    bump_started = threading.Event()
    bump_finished = threading.Event()
    cache = [None, None]
    captured = []

    def cache_transaction():
        with epochs.epoch_transaction():
            snapshot = epochs.epoch_subset(('conditioner_epoch',))
            compute_entered.set()
            assert allow_publish.wait(2)
            cache[:] = [snapshot, 'computed']
            captured.append(epochs.epoch_subset(('conditioner_epoch',)))

    def mutate_dependency():
        assert compute_entered.wait(2)
        bump_started.set()
        epochs.bump_epoch('conditioner_epoch', reason='conditioning_cleared')
        bump_finished.set()

    cache_thread = threading.Thread(target=cache_transaction)
    bump_thread = threading.Thread(target=mutate_dependency)
    cache_thread.start()
    bump_thread.start()
    assert bump_started.wait(2)
    assert not bump_finished.wait(0.05)
    allow_publish.set()
    cache_thread.join(2)
    bump_thread.join(2)

    assert not cache_thread.is_alive() and not bump_thread.is_alive()
    assert cache[0] == captured[0] == (('conditioner_epoch', 0),)
    assert epochs.epoch_subset(('conditioner_epoch',)) == (('conditioner_epoch', 1),)


def test_epoch_transaction_exposes_only_coherent_ti_and_clear_states():
    import threading

    import modules.openclaw_cache_epochs as epochs

    epochs.epoch_registry.reset()
    state = {'maps': 'old', 'cache': 'populated'}
    ti_maps_published = threading.Event()
    allow_ti_epochs = threading.Event()
    clear_cache_published = threading.Event()
    allow_clear_epochs = threading.Event()
    observations = []

    def observe(label):
        with epochs.epoch_transaction():
            observations.append((label, state.copy(), dict(epochs.epoch_subset((
                'textual_inversion_epoch', 'tokenizer_epoch',
                'conditioner_epoch', 'conditioning_hook_epoch',
            )))))

    def publish_ti():
        with epochs.epoch_transaction():
            state['maps'] = 'new'
            ti_maps_published.set()
            assert allow_ti_epochs.wait(2)
            epochs.bump_epoch('textual_inversion_epoch', reason='textual_inversion_reloaded')
            epochs.bump_epoch('tokenizer_epoch', reason='textual_inversion_reloaded')

    ti_thread = threading.Thread(target=publish_ti)
    ti_thread.start()
    assert ti_maps_published.wait(2)
    ti_observer = threading.Thread(target=observe, args=('ti',))
    ti_observer.start()
    ti_observer.join(0.05)
    assert ti_observer.is_alive()
    allow_ti_epochs.set()
    ti_thread.join(2)
    ti_observer.join(2)

    def clear_conditioning():
        with epochs.epoch_transaction():
            state['cache'] = 'empty'
            clear_cache_published.set()
            assert allow_clear_epochs.wait(2)
            epochs.bump_epoch('conditioner_epoch', reason='conditioning_cleared')
            epochs.bump_epoch('conditioning_hook_epoch', reason='conditioning_hook_changed')

    clear_thread = threading.Thread(target=clear_conditioning)
    clear_thread.start()
    assert clear_cache_published.wait(2)
    clear_observer = threading.Thread(target=observe, args=('clear',))
    clear_observer.start()
    clear_observer.join(0.05)
    assert clear_observer.is_alive()
    allow_clear_epochs.set()
    clear_thread.join(2)
    clear_observer.join(2)

    by_label = {label: (values, snapshot) for label, values, snapshot in observations}
    assert by_label['ti'][0]['maps'] == 'new'
    assert by_label['ti'][1]['textual_inversion_epoch'] == 1
    assert by_label['ti'][1]['tokenizer_epoch'] == 1
    assert by_label['clear'][0]['cache'] == 'empty'
    assert by_label['clear'][1]['conditioner_epoch'] == 1
    assert by_label['clear'][1]['conditioning_hook_epoch'] == 1
