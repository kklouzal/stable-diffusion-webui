import torch

from modules import rng


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
