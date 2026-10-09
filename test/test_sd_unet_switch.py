"""sd_unet.apply_unet: swapping in an alternative UNet runs inside the CUDA graph mutation boundary."""

from types import SimpleNamespace

import pytest
import torch

from test.helpers import init_shared

shared = init_shared()

from modules import openclaw_cuda_graphs, sd_unet  # noqa: E402


class StubUnet(sd_unet.SdUnet):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def activate(self):
        self.events.append(("activate", openclaw_cuda_graphs._RUNTIME_LOCK._is_owned()))

    def deactivate(self):
        self.events.append(("deactivate", openclaw_cuda_graphs._RUNTIME_LOCK._is_owned()))


class StubOption(sd_unet.SdUnetOption):
    label = "stub"
    model_name = "stub-model"

    def __init__(self, events):
        self.events = events

    def create_unet(self):
        return StubUnet(self.events)


class RecordingUnet(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def to(self, device):
        self.events.append(("to", str(device), openclaw_cuda_graphs._RUNTIME_LOCK._is_owned()))
        return self


@pytest.fixture
def environment(monkeypatch):
    events = []
    option = StubOption(events)
    model = SimpleNamespace(lowvram=False, model=SimpleNamespace(diffusion_model=RecordingUnet(events)))
    monkeypatch.setattr(shared, "sd_model", model, raising=False)
    monkeypatch.setattr(sd_unet, "unet_options", [option])
    monkeypatch.setattr(sd_unet, "current_unet_option", None)
    monkeypatch.setattr(sd_unet, "current_unet", None)
    monkeypatch.setattr(sd_unet.devices, "device", torch.device("cpu"))
    monkeypatch.setattr(sd_unet.devices, "torch_gc", lambda: None)

    def invalidate(reason, details=None):
        events.append(("invalidate", reason, details))

    monkeypatch.setattr(openclaw_cuda_graphs, "invalidate", invalidate)
    return events


def test_activation_and_deactivation_run_inside_the_graph_boundary(environment):
    events = environment

    sd_unet.apply_unet("stub")

    assert isinstance(sd_unet.current_unet, StubUnet)
    assert events == [("invalidate", "alternative_unet", "stub"), ("to", "cpu", True), ("activate", True)]

    events.clear()
    sd_unet.apply_unet("None")

    assert sd_unet.current_unet is None
    assert events == [("invalidate", "alternative_unet", None), ("deactivate", True), ("to", "cpu", True)]


def test_unchanged_option_leaves_graphs_alone(environment):
    sd_unet.apply_unet("None")

    assert environment == []
