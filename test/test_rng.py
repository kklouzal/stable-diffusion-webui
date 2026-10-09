import types

import pytest
import torch

from modules import rng, rng_philox


@pytest.mark.parametrize("source", ["GPU", "CPU", "NV"])
def test_randn_reseeds_then_draws_the_seeds_stream(monkeypatch, source):
    """Oracle: the per-source draw spelled out with torch and the philox generator directly (CPU device)."""
    monkeypatch.setattr(rng.shared, "opts", types.SimpleNamespace(randn_source=source), raising=False)
    monkeypatch.setattr(rng.devices, "device", torch.device("cpu"))
    shape = (2, 4, 4)

    with torch.random.fork_rng(devices=[]):  # randn reseeds the global generator
        first, second = rng.randn(1234, shape), rng.randn_without_seed(shape)
        with_generator = rng.randn(1234, shape, generator=rng.create_generator(77))

        if source == "NV":
            stream = rng_philox.Generator(1234)
            expected = [torch.asarray(stream.randn(shape)), torch.asarray(stream.randn(shape))]
            expected_with_generator = torch.asarray(rng_philox.Generator(77).randn(shape))
        else:
            torch.manual_seed(1234)
            expected = [torch.randn(shape), torch.randn(shape)]
            expected_with_generator = torch.randn(shape, generator=torch.Generator().manual_seed(77))

    assert torch.equal(first, expected[0]) and torch.equal(second, expected[1])
    assert torch.equal(with_generator, expected_with_generator)


def test_slerp_linear_fallback_preserves_endpoint_order():
    low = torch.ones(2, 4)
    high = low * 2

    torch.testing.assert_close(rng.slerp(0.0, low, high), low)
    torch.testing.assert_close(rng.slerp(1.0, low, high), high)

    val = 0.25
    expected = low * (1 - val) + high * val
    torch.testing.assert_close(rng.slerp(val, low, high), expected)


def test_slerp_curved_path_preserves_endpoint_order():
    low = torch.tensor([[1.0, 0.0]])
    high = torch.tensor([[0.0, 1.0]])

    torch.testing.assert_close(rng.slerp(0.0, low, high), low)
    torch.testing.assert_close(rng.slerp(1.0, low, high), high)
