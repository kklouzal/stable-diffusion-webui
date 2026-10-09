"""TeaCacheSession math and cache-decision state (scripts/teacache.py); the `teacache` fixture is in conftest.py."""

import ast
import math
from pathlib import Path

import pytest
import torch


def test_normalize_args_clamps_and_orders_range(teacache):
    assert teacache.normalize_args([1, "2.5", "12.8", 0.9, -0.2]) == (True, 1.0, 12, 0.0, 0.9)
    assert teacache.normalize_args(["true", "2", "999", "0.9", "0.1"]) == (True, 1.0, 150, 0.1, 0.9)
    assert teacache.normalize_args([]) == (False, 0.25, 4, 0.35, 0.9)
    assert teacache.normalize_args([True]) == (True, 0.25, 4, 0.35, 0.9)
    assert teacache.normalize_args(["false"]) == (False, 0.25, 4, 0.35, 0.9)
    assert teacache.normalize_args([True, "bad", None, "bad", "bad"]) == (True, 0.25, 4, 0.35, 0.9)


def test_relative_l1_distance_handles_zero_baseline(teacache):
    prev = torch.zeros((2, 2), dtype=torch.float32)
    curr = torch.ones((2, 2), dtype=torch.float32)

    distance = teacache.relative_l1_distance(prev, curr)

    assert isinstance(distance, torch.Tensor)
    assert distance.shape == torch.Size([])
    assert distance.device == prev.device
    assert math.isfinite(distance.item())
    assert distance.item() > 0


def test_sdxl_polynomial_distance_matches_coefficients_on_tensor_device(teacache):
    relative = torch.tensor(0.125, dtype=torch.float32)
    coeffs = relative.new_tensor(teacache.SDXL_POLYNOMIAL_COEFFICIENTS)

    distance = teacache.sdxl_polynomial_distance(relative, coeffs)
    expected = sum(float(coeff) * (float(relative) ** power) for power, coeff in enumerate(teacache.SDXL_POLYNOMIAL_COEFFICIENTS))

    assert isinstance(distance, torch.Tensor)
    assert distance.device == relative.device
    assert math.isclose(distance.item(), expected, rel_tol=1e-6, abs_tol=1e-6)


def test_session_caches_device_tensors_for_hot_path_constants(teacache):
    session = teacache.TeaCacheSession(threshold=1.0, max_consecutive=0, start=0.0, end=1.0, steps=10)
    signature = ((1, 2), "torch.float32", "cpu")
    old = torch.ones((1, 2), dtype=torch.float32)
    session.previous_fb[0] = old
    session.residuals[0] = (signature, torch.zeros((1, 2), dtype=torch.float32))

    session.update_condition(old * 1.001, signature)

    assert session.use_cache
    assert session.distances[0].device == old.device
    assert session._threshold_tensors[(str(old.device), torch.float32)].device == old.device
    assert session._coefficient_tensors[(str(old.device), torch.float32)].device == old.device


def test_hot_path_sync_constructs_are_explicitly_allowlisted(teacache):
    source = Path(teacache.__file__).read_text()
    hot_source = source[source.index("def relative_l1_distance"):]
    tree = ast.parse(hot_source)
    banned_attrs = {"cpu", "numpy", "tolist", "item"}
    banned_calls = []
    bool_calls = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in banned_attrs:
                banned_calls.append((func.attr, node.lineno))
            elif isinstance(func, ast.Name) and func.id in {"float", "int", "bool"}:
                segment = ast.get_source_segment(hot_source, node) or ""
                if segment == "bool(should_refresh)":
                    bool_calls.append(node.lineno)
                else:
                    banned_calls.append((func.id, node.lineno))

    assert banned_calls == []
    assert len(bool_calls) == 1
    assert "Intentional sync point" in source


def test_progress_start_is_exclusive_and_end_is_inclusive(teacache):
    session = teacache.TeaCacheSession(threshold=1.0, max_consecutive=4, start=0.2, end=0.8, steps=10, initial_step=2)
    signature = ((1,),)
    session.previous_fb[0] = torch.zeros(1)
    session.residuals[0] = (signature, torch.ones(1))
    session.update_condition(torch.zeros(1), signature)
    assert not session.use_cache

    session.current_step = 8
    session.update_condition(torch.zeros(1), signature)
    assert session.use_cache


def test_session_requires_residual_for_current_call_index(teacache):
    session = teacache.TeaCacheSession(threshold=1.0, max_consecutive=0, start=0.0, end=1.0, steps=10)
    session.use_cache = True
    signature = ((1, 1), "torch.float32", "cpu")
    session.residuals[0] = (signature, torch.ones((1, 1), dtype=torch.float32))

    assert session.current_residual(signature) is not None
    assert session.current_residual(((2, 1), "torch.float32", "cpu")) is None
    session.call_index = 1
    assert session.current_residual(signature) is None


def test_session_isolates_cache_by_call_signature(teacache):
    h = torch.zeros(2, 4, 8, 8)
    timesteps = torch.zeros(2)
    context = torch.zeros(2, 77, 2048)
    y = torch.zeros(2, 2816)
    signature = teacache._call_signature(h, timesteps, context, y)
    other_signature = teacache._call_signature(h, timesteps, context[:1], y)
    assert signature != other_signature

    session = teacache.TeaCacheSession(threshold=1.0, max_consecutive=0, start=0.0, end=1.0, steps=10)
    old = torch.ones((1, 2), dtype=torch.float32)
    session.previous_fb[0] = old
    session.residuals[0] = (other_signature, torch.ones((1, 2), dtype=torch.float32))
    session.update_condition(old * 1.01, signature)

    assert not session.use_cache
    session.store_current_residual(signature, torch.full((1, 2), 3.0))
    session.use_cache = True
    assert torch.equal(session.current_residual(signature), torch.full((1, 2), 3.0))
    assert session.current_residual(other_signature) is None


def test_session_window_and_max_consecutive_are_quality_guards(teacache):
    signature = ((1, 2), "torch.float32", "cpu")
    session = teacache.TeaCacheSession(threshold=1.0, max_consecutive=1, start=0.25, end=0.75, steps=4, initial_step=1)
    session.previous_fb[0] = torch.ones((1, 2), dtype=torch.float32)
    session.residuals[0] = (signature, torch.zeros((1, 2), dtype=torch.float32))

    session.update_condition(torch.ones((1, 2), dtype=torch.float32), signature)
    assert not session.use_cache

    session.next_step()
    session.update_condition(torch.ones((1, 2), dtype=torch.float32), signature)
    assert session.use_cache
    assert session.consecutive_hits == {0: 1}

    session.next_step()
    session.update_condition(torch.ones((1, 2), dtype=torch.float32), signature)
    assert not session.use_cache


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_session_refreshes_on_nonfinite_distance_and_resets_accumulator(teacache, value):
    signature = ((1, 2), "torch.float32", "cpu")
    session = teacache.TeaCacheSession(threshold=1.0, max_consecutive=0, start=0.0, end=1.0, steps=10)
    session.previous_fb[0] = torch.ones((1, 2), dtype=torch.float32)
    session.residuals[0] = (signature, torch.zeros((1, 2), dtype=torch.float32))

    session.update_condition(torch.full((1, 2), value, dtype=torch.float32), signature)

    assert not session.use_cache
    assert torch.equal(session.distances[0], torch.zeros((), dtype=torch.float32))


def test_session_detaches_first_block_residual_before_distance_math(teacache):
    signature = ((1, 2), "torch.float32", "cpu")
    session = teacache.TeaCacheSession(threshold=1.0, max_consecutive=0, start=0.0, end=1.0, steps=10)
    session.previous_fb[0] = torch.ones((1, 2), dtype=torch.float32)
    session.residuals[0] = (signature, torch.zeros((1, 2), dtype=torch.float32))

    session.update_condition(torch.ones((1, 2), dtype=torch.float32, requires_grad=True), signature)

    assert session.use_cache
    assert not session.previous_fb[0].requires_grad
    assert not session.distances[0].requires_grad


def test_storing_fresh_residual_resets_accumulated_distance(teacache):
    signature = ((1, 2), "torch.float32", "cpu")
    session = teacache.TeaCacheSession(threshold=1.0, max_consecutive=0, start=0.0, end=1.0, steps=10)
    session.distances[0] = torch.tensor(0.75)

    session.store_current_residual(signature, torch.ones((1, 2), dtype=torch.float32))

    torch.testing.assert_close(session.distances[0], torch.zeros(()))


def test_cached_residual_is_cloned_not_mutable_alias(teacache):
    session = teacache.TeaCacheSession(threshold=1.0, max_consecutive=4, start=0.0, end=1.0, steps=10)
    signature = ((1,),)
    residual = torch.ones(2)
    session.store_current_residual(signature, residual)
    residual.add_(10)
    torch.testing.assert_close(session.current_residual(signature), torch.ones(2))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_previous_first_block_residual_is_an_owned_fp32_copy_with_unchanged_distances(teacache, dtype):
    # The lane keeps its previous residual in fp32 (converted once, no extra clone); the distance must
    # equal the bf16-stored formulation bitwise, and the stored copy must not alias the producer's tensor.
    gen = torch.Generator().manual_seed(3)
    signature = ((2, 8, 4, 4),)
    session = teacache.TeaCacheSession(threshold=10.0, max_consecutive=0, start=0.0, end=1.0, steps=10)
    base = torch.randn(2, 8, 4, 4, generator=gen)
    residuals = [(base + 0.02 * torch.randn(2, 8, 4, 4, generator=gen)).to(dtype) for _ in range(4)]
    session.update_condition(residuals[0], signature)
    session.store_current_residual(signature, residuals[0])
    previous = residuals[0].clone()
    residuals[0].add_(100)
    expected_distance = torch.zeros(())
    for current in residuals[1:]:
        session.next_step()
        session.update_condition(current, signature)
        prev_f, curr_f = previous.float(), current.float()
        rel = (prev_f - curr_f).abs().mean() / prev_f.abs().mean().clamp_min(torch.finfo(torch.float32).eps)
        expected_distance = expected_distance + teacache.sdxl_polynomial_distance(
            rel, torch.tensor(teacache.SDXL_POLYNOMIAL_COEFFICIENTS)
        )
        assert session.use_cache
        assert session.previous_fb[0].dtype == torch.float32
        assert session.previous_fb[0].data_ptr() != current.data_ptr()
        assert torch.equal(session.distances[0], expected_distance)
        previous = current.clone()
        current.add_(100)


class _GuardLock:
    def __init__(self):
        self.depth = 0
        self.entries = 0

    def __enter__(self):
        self.depth += 1
        self.entries += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        self.depth -= 1


def test_global_session_accessors_are_lock_guarded(teacache):
    guard = _GuardLock()
    teacache._cache_lock = guard
    session = teacache.TeaCacheSession(threshold=1.0, max_consecutive=0, start=0.0, end=1.0, steps=10)

    teacache._set_cache(session)
    assert teacache._get_cache() is session
    teacache._set_cache(None)

    assert guard.entries == 3
    assert guard.depth == 0
