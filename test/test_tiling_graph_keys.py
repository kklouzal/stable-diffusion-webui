"""Tiling (sd_hijack.model_hijack.apply_circular) is part of both CUDA graph keys.

apply_circular(p.tiling) switches every hijacked Conv2d of the UNet, VAE and text encoders to circular padding
without touching a tensor. A graph captured with one padding must never replay for a request with the other.
"""

import contextlib
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from test.helpers import init_shared

shared = init_shared()

from modules import sd_models  # noqa: E402,F401  (imports sd_hijack in the order webui startup does)
from modules import openclaw_cuda_graphs, openclaw_vae_decode_graphs, sd_hijack  # noqa: E402


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.diffusion_model = torch.nn.Sequential(torch.nn.Conv2d(4, 4, 3, padding=1))
        self.first_stage_model = torch.nn.Sequential(torch.nn.Conv2d(4, 3, 3, padding=1)).eval()
        self.first_stage_model.dtype = torch.float32  # the sgm/ldm autoencoders expose their dtype

    def decode_first_stage(self, x):
        return self.first_stage_model(x)


@pytest.fixture
def hijacked(monkeypatch):
    model = TinyModel()
    model_hijack = sd_hijack.StableDiffusionModelHijack()
    model_hijack.layers = list(model.modules())  # what hijack() collects with flatten(m)
    monkeypatch.setattr(sd_hijack, "model_hijack", model_hijack)
    yield model, model_hijack
    model_hijack.apply_circular(False)


def test_unet_runtime_branch_key_follows_apply_circular(hijacked):
    model, model_hijack = hijacked
    zeros = openclaw_cuda_graphs._runtime_branch_key()

    model_hijack.apply_circular(True)
    assert model.model.diffusion_model[0].padding_mode == "circular"
    assert openclaw_cuda_graphs._runtime_branch_key() != zeros

    model_hijack.apply_circular(False)
    assert openclaw_cuda_graphs._runtime_branch_key() == zeros


def test_vae_key_follows_apply_circular(hijacked):
    model, model_hijack = hijacked
    x = torch.zeros(1, 4, 8, 8)
    zeros = openclaw_vae_decode_graphs._key(model, x)

    model_hijack.apply_circular(True)
    assert model.first_stage_model[0].padding_mode == "circular"
    assert openclaw_vae_decode_graphs._key(model, x) != zeros

    model_hijack.apply_circular(False)
    assert openclaw_vae_decode_graphs._key(model, x) == zeros


@contextlib.contextmanager
def _fake_cuda_capture(graphs):
    """CPU stand-ins for the CUDA capture API: the "graph" freezes the output its capture computed, as a CUDA graph
    replays the kernels (and padding) it recorded."""

    class Context:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False

    stream = SimpleNamespace(wait_stream=lambda other: None)
    cuda = graphs.torch.cuda
    with mock.patch.object(graphs, "_bypass_reason", return_value=None), \
         mock.patch.object(cuda, "Stream", return_value=stream), \
         mock.patch.object(cuda, "current_stream", return_value=stream), \
         mock.patch.object(cuda, "stream", return_value=Context()), \
         mock.patch.object(cuda, "CUDAGraph", side_effect=lambda: SimpleNamespace(replay=lambda: None)), \
         mock.patch.object(cuda, "graph", side_effect=lambda graph, pool=None: Context()), \
         mock.patch.object(cuda, "graph_pool_handle", side_effect=object), \
         mock.patch.object(cuda, "synchronize"):
        graphs.set_enabled(True, clear_cache=True)
        try:
            yield
        finally:
            graphs.set_enabled(False, clear_cache=True)


def test_vae_graph_replay_never_crosses_tiling_states(hijacked):
    model, model_hijack = hijacked
    x = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(0))
    graphs = openclaw_vae_decode_graphs
    with torch.no_grad(), _fake_cuda_capture(graphs):
        before = graphs.status()  # the counters are cumulative across cache resets
        for circular in (False, True, False, True):
            model_hijack.apply_circular(circular)
            decoded = graphs.run(model, x)
            assert decoded is not None, graphs.status()
            assert torch.equal(decoded, model.decode_first_stage(x)), f"circular={circular}"
        after = graphs.status()
    assert after["captures"] - before["captures"] == 2
    assert after["replays"] - before["replays"] == 2
