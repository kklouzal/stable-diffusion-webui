"""devices.test_for_nans raises for a NaN anywhere in the tensor, not only in element [0, ..., 0]."""

import pytest
import torch

from test.helpers import init_shared

shared = init_shared()

from modules import devices  # noqa: E402


@pytest.fixture(autouse=True)
def nan_check_enabled(monkeypatch):
    monkeypatch.setattr(shared.cmd_opts, "disable_nan_check", False)


def with_nan(shape, index):
    x = torch.zeros(shape)
    x[index] = float("nan")
    return x


@pytest.mark.parametrize(("where", "message"), [("unet", "Unet"), ("vae", "VAE"), ("other", "A tensor with NaNs")])
@pytest.mark.parametrize("x", [
    with_nan((1, 4, 8, 8), (0, 0, 0, 0)),
    with_nan((2, 4, 8, 8), (1,)),  # one image of the batch
    with_nan((1, 4, 8, 8), (0, 3, 5, 7)),  # one region
    with_nan((3, 64, 64), (2, 63, 63)).to(torch.bfloat16),
], ids=["first", "second-image", "region", "bf16-last"])
def test_any_nan_raises(x, where, message):
    with pytest.raises(devices.NansException, match=message):
        devices.test_for_nans(x, where)


@pytest.mark.parametrize("x", [torch.zeros(2, 4, 8, 8), torch.full((1, 3, 4, 4), float("inf"))])
def test_nan_free_tensors_pass(x):
    devices.test_for_nans(x, "vae")


def test_disabled_check_never_raises(monkeypatch):
    monkeypatch.setattr(shared.cmd_opts, "disable_nan_check", True)
    devices.test_for_nans(with_nan((2, 4, 8, 8), (1,)), "unet")
