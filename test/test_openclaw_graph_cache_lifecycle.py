"""UNet and VAE decode CUDA graph caches on a CPU host, with the CUDA calls faked: the first request returns the
replayed capture, the VAE capture synchronizes once before publication and never on replay, and a manual reset clears
retained state without touching enablement, cumulative counters or lifecycle epochs."""

from collections import OrderedDict
from types import SimpleNamespace

import pytest


class _Context:
    def __init__(self, events, name):
        self.events, self.name = events, name

    def __enter__(self):
        self.events.append(f"enter {self.name}")

    def __exit__(self, *exc):
        self.events.append(f"exit {self.name}")
        return False


class _DeviceValue:
    """Stands in for a CUDA tensor: copy_/clone move the payload in `value`."""

    device = SimpleNamespace(type="cuda")

    def __init__(self, events, value):
        self.events, self.value = events, value

    def detach(self):
        return self

    def contiguous(self):
        return self

    def copy_(self, other, non_blocking=False):
        self.events.append("copy")
        self.value = other.value
        return self

    def clone(self):
        self.events.append("clone")
        return _DeviceValue(self.events, self.value)


def _fake_cuda(monkeypatch, cuda, events, replay):
    """Fake the CUDA stream/graph API; a graph replay runs replay()."""
    stream = SimpleNamespace(wait_stream=lambda other: None)

    class Graph:
        def replay(self):
            events.append("replay")
            replay()

    monkeypatch.setattr(cuda, "Stream", lambda *args, **kwargs: stream)
    monkeypatch.setattr(cuda, "current_stream", lambda *args, **kwargs: stream)
    monkeypatch.setattr(cuda, "stream", lambda _stream: _Context(events, "side stream"))
    monkeypatch.setattr(cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(cuda, "graph", lambda graph, pool=None: _Context(events, "capture"))
    monkeypatch.setattr(cuda, "graph_pool_handle", lambda: ("pool",))


@pytest.fixture
def vae_graphs(monkeypatch):
    from modules import openclaw_vae_decode_graphs as graphs

    monkeypatch.setattr(graphs, "_ENABLED", False)
    monkeypatch.setattr(graphs, "_CACHE", OrderedDict())
    monkeypatch.setattr(graphs, "_FAILED_KEYS", OrderedDict())
    monkeypatch.setattr(graphs, "_COUNTERS", dict.fromkeys(graphs._COUNTERS, 0))
    monkeypatch.setattr(graphs, "_BYPASS_REASONS", {})
    monkeypatch.setattr(graphs, "_INVALIDATION_REASONS", {})
    monkeypatch.setattr(graphs, "_GRAPH_POOL", None)
    return graphs


def test_vae_graph_first_request_returns_the_replayed_capture_and_synchronizes_only_before_publish(vae_graphs, monkeypatch):
    graphs = vae_graphs
    graphs.set_enabled(True)
    events, captured = [], {}
    key = ("vae", "key")

    def execute(model, static_input):
        events.append("execute")
        if "input" not in captured:  # the side-stream warm-up
            captured["input"] = static_input
            return _DeviceValue(events, "warm-up output")
        captured["output"] = _DeviceValue(events, "capture-time output")  # not computed until a replay
        return captured["output"]

    def replay():
        captured["output"].value = f"decoded:{captured['input'].value}"

    _fake_cuda(monkeypatch, graphs.torch.cuda, events, replay)
    monkeypatch.setattr(graphs.torch.cuda, "synchronize", lambda *args: events.append(("synchronize", key in graphs._CACHE)))
    monkeypatch.setattr(graphs, "_bypass_reason", lambda model, x, approximation: None)
    monkeypatch.setattr(graphs, "_key", lambda model, x: key)
    monkeypatch.setattr(graphs, "_execute", execute)

    first = graphs.run(object(), _DeviceValue(events, "latent-1"))
    assert first.value == "decoded:latent-1" and first is not captured["output"]
    assert events == [
        "clone",  # static input
        "enter side stream", "execute", "exit side stream",
        "enter capture", "execute", "exit capture",
        ("synchronize", False),  # one barrier, before the entry is published
        "replay", "clone",
    ]

    events.clear()
    hit = graphs.run(object(), _DeviceValue(events, "latent-2"))
    assert hit.value == "decoded:latent-2"
    assert events == ["copy", "replay", "clone"]  # no device barrier on a replay
    status = graphs.status()
    assert (status["captures"], status["replays"], status["cache_size"]) == (1, 1, 1)


def test_vae_graph_reset_keeps_enablement_cumulative_counters_and_epochs(vae_graphs):
    graphs = vae_graphs
    epochs = graphs.openclaw_cache_epochs
    graphs.set_enabled(True)
    graphs._COUNTERS.update(captures=3, replays=5, failures=1)
    graphs._CACHE[("cached",)] = {"graph": object(), "input": object(), "output": object()}
    graphs._FAILED_KEYS[("failed",)] = None
    graphs._GRAPH_POOL = ("pool",)
    lifecycle = epochs.epoch_subset(epochs.EPOCH_DIMENSIONS)

    status = graphs.set_enabled(None, clear_cache=True)  # the API's {"clear": true} without "enabled"

    assert status["enabled"] is True
    assert (status["cache_size"], status["failed_key_count"], graphs._GRAPH_POOL) == (0, 0, None)
    assert (status["captures"], status["replays"], status["failures"]) == (3, 5, 1)
    assert (status["invalidations"], status["invalidation_reasons"]) == (1, {"manual_reset": 1})
    assert epochs.epoch_subset(epochs.EPOCH_DIMENSIONS) == lifecycle  # a cache reset is not a lifecycle change


@pytest.fixture
def without_autograd():
    """Autograd off, as on the generation path (graph capture bypasses while grad is enabled). Patching
    torch.is_grad_enabled instead would make every torch.no_grad() entered meanwhile restore grad mode to off."""
    import torch

    with torch.no_grad():
        yield


def test_unet_graph_first_request_returns_the_replayed_capture(monkeypatch, without_autograd):
    from modules import openclaw_cuda_graphs as graphs

    torch = graphs.torch
    real_is_tensor = torch.is_tensor
    events, captured = [], {}

    def fn(x, sigma, cond=None):
        events.append("forward")
        if "x" not in captured:  # the side-stream warm-up
            captured["x"] = x
            return _DeviceValue(events, "warm-up output")
        captured["out"] = _DeviceValue(events, "capture-time output")  # not computed until a replay
        return captured["out"]

    def replay():
        captured["out"].value = f"denoised:{captured['x'].value}"

    _fake_cuda(monkeypatch, torch.cuda, events, replay)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "is_tensor", lambda value: isinstance(value, _DeviceValue) or real_is_tensor(value))
    monkeypatch.setattr(graphs, "on_default_stream", lambda device: True)
    monkeypatch.setattr(graphs, "_graph_denoiser_bypass_reason", lambda denoiser, fn: None)
    monkeypatch.setattr(graphs, "_cache_key", lambda fn, x, sigma, cond, denoiser: ("unet", "key"))
    monkeypatch.setattr(graphs, "_schedule_tensors", lambda fn: ())
    monkeypatch.setattr(graphs, "_MAX_CACHE_SIZE", 1)
    enabled = graphs._ENABLED
    graphs.set_enabled(True, clear=True)
    try:
        first = graphs.run(fn, _DeviceValue(events, "latent-1"), _DeviceValue(events, "sigma"), {})
        assert first.value == "denoised:latent-1" and first is not captured["out"]
        assert events.count("forward") == 2  # one warm-up, one capture: no eager run is returned
        assert events[-2:] == ["replay", "clone"]
        status = graphs.status()
        assert (status["captures"], status["replays"], status["bypasses"], status["fallbacks"]) == (1, 0, 0, 0)

        events.clear()
        hit = graphs.run(fn, _DeviceValue(events, "latent-2"), _DeviceValue(events, "sigma"), {})
        assert hit.value == "denoised:latent-2"
        assert events == ["copy", "copy", "replay", "clone"]
    finally:
        graphs.set_enabled(enabled, clear=True)
