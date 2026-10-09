import pytest
import torch

from modules import devices


@pytest.fixture
def half_precision_cuda(monkeypatch):
    """devices state of a CUDA fp16/bf16 autocast run, with a fake card and a recorded manual_cast."""
    monkeypatch.setattr(devices, "force_fp16", False)
    monkeypatch.setattr(devices, "fp8", False)
    monkeypatch.setattr(devices, "dtype", torch.float16)
    monkeypatch.setattr(devices, "dtype_inference", torch.float16)
    monkeypatch.setattr(devices, "has_xpu", lambda: False)
    monkeypatch.setattr(devices, "get_cuda_device_id", lambda: 0)
    monkeypatch.setattr(devices, "manual_cast", lambda dtype: ("manual_cast", dtype))
    devices._autocast_needs_manual_cast.cache_clear()
    yield monkeypatch
    devices._autocast_needs_manual_cast.cache_clear()


def fake_card(monkeypatch, capability, name):
    probes = []

    def get_device_capability(device_id):
        probes.append(device_id)
        return capability

    monkeypatch.setattr(torch.cuda, "get_device_capability", get_device_capability)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device_id: name)
    return probes


@pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda'")
@pytest.mark.parametrize("capability,name,manual", [
    ((7, 5), "NVIDIA GeForce GTX 1660 SUPER", True),
    ((7, 5), "NVIDIA GeForce RTX 2080", False),
    ((12, 1), "NVIDIA GB10", False),
])
def test_autocast_probes_the_card_once_and_keeps_its_path(half_precision_cuda, capability, name, manual):
    probes = fake_card(half_precision_cuda, capability, name)

    contexts = [devices.autocast() for _ in range(3)]

    assert probes == [0]
    for context in contexts:
        if manual:
            assert context == ("manual_cast", torch.float16)
        else:
            assert isinstance(context, torch.autocast) and context.fast_dtype == torch.float16


def test_autocast_xpu_takes_manual_cast_without_probing_the_card(half_precision_cuda):
    probes = fake_card(half_precision_cuda, (7, 5), "NVIDIA GeForce GTX 1660")
    half_precision_cuda.setattr(devices, "has_xpu", lambda: True)

    assert devices.autocast() == ("manual_cast", torch.float16)
    assert probes == []


def test_autocast_does_not_cache_a_failed_probe(half_precision_cuda):
    def no_device():
        raise RuntimeError("no CUDA device")

    half_precision_cuda.setattr(devices, "get_cuda_device_id", no_device)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="no CUDA device"):
            devices.autocast()
    assert devices._autocast_needs_manual_cast.cache_info().currsize == 0
