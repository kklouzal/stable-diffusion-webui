"""Checkpoint switch on unified memory: in-place device load versus the CPU round trip.

Both paths drive the real reload_model_weights/reuse_model_from_already_loaded/load_model_weights code on a tiny
SD1-shaped model with mixed-dtype source tensors, a buffer-registered alphas_cumprod, a VAE carve-out and
channels_last. The CPU round trip is the oracle. On a CUDA host the same differential also runs with the model
resident on the GPU, which is the case the in-place path exists for.
"""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from test.helpers import init_shared

shared = init_shared()

from modules import devices, openclaw_lifecycle_epochs, sd_models, sd_vae  # noqa: E402


DEVICES = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])


class TinySD(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.diffusion_model = torch.nn.Sequential(torch.nn.Conv2d(4, 8, 3), torch.nn.GroupNorm(2, 8), torch.nn.Conv2d(8, 4, 1))
        self.cond_stage_model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.LayerNorm(32))
        self.first_stage_model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.Conv2d(4, 3, 3))
        self.register_buffer("alphas_cumprod", torch.linspace(0.999, 0.01, 10))
        self.register_buffer("betas", torch.linspace(1e-4, 2e-2, 10))
        self.lowvram = False
        self.used_config = "tiny.yaml"


def _checkpoint(name, seed):
    generator = torch.Generator().manual_seed(seed)
    state_dict = {}
    for key, value in TinySD().state_dict().items():
        tensor = torch.randn(value.shape, generator=generator, dtype=torch.float64) * (1 + 10 * torch.rand(value.shape, generator=generator, dtype=torch.float64))
        # Mixed source dtypes, like real checkpoints: fp32 UNet/VAE, fp16 text encoder, fp64 schedule.
        state_dict[key] = tensor.to(torch.float16 if key.startswith("cond_stage_model.") else torch.float64 if key in ("alphas_cumprod", "betas") else torch.float32)
    state_dict["alphas_cumprod"] = torch.linspace(0.9991, 0.0047, 10, dtype=torch.float64) + seed * 1e-5
    info = SimpleNamespace(filename=f"{name}.safetensors", title=name, sha256=f"{seed:064x}", calculate_shorthash=lambda: f"{seed:010x}")
    return info, state_dict


class Recorder:
    def __init__(self):
        self.events = []

    def __call__(self, name):
        return lambda *args, **kwargs: self.events.append(name)


def _environment(monkeypatch, device, *, unified, limit=1, keep_in_cpu=True):
    recorder = Recorder()
    monkeypatch.setattr(sd_models, "_device_has_unified_memory", lambda _device: unified)
    monkeypatch.setattr(devices, "device", device)
    monkeypatch.setattr(shared, "device", device)
    for name, value in (("dtype", torch.bfloat16), ("dtype_unet", torch.bfloat16), ("dtype_vae", torch.bfloat16), ("unet_needs_upcast", False), ("fp8", False), ("mxfp8", False), ("nvfp4", False)):
        monkeypatch.setattr(devices, name, value, raising=False)
    monkeypatch.setattr(devices, "torch_gc", recorder("torch_gc"))
    monkeypatch.setattr(shared, "parallel_processing_allowed", True)
    monkeypatch.setattr(shared.cmd_opts, "opt_channelslast", True)
    monkeypatch.setattr(shared.cmd_opts, "upcast_sampling", False)
    monkeypatch.setattr(shared.cmd_opts, "no_half_vae", False)
    for flag in ("lowvram", "medvram", "medvram_sdxl"):
        monkeypatch.setattr(shared.cmd_opts, flag, False)
    monkeypatch.setitem(shared.opts.data, "sd_checkpoints_limit", limit)
    monkeypatch.setitem(shared.opts.data, "sd_checkpoints_keep_in_cpu", keep_in_cpu)
    monkeypatch.setitem(shared.opts.data, "sd_checkpoint_cache", 0)
    monkeypatch.setitem(shared.opts.data, "sd_checkpoint_hash", None)
    monkeypatch.setattr(sd_models.SkipWritingToConfig, "skip", True)
    monkeypatch.setattr(sd_models, "model_data", sd_models.SdModelData())
    for name in ("base_vae", "checkpoint_info", "loaded_vae_file"):
        monkeypatch.setattr(sd_vae, name, None)
    monkeypatch.setattr(sd_vae, "resolve_vae", lambda _filename: SimpleNamespace(tuple=lambda: (None, "test")))
    monkeypatch.setattr(sd_vae, "load_vae", recorder("load_vae"))
    monkeypatch.setattr(sd_models.sd_unet, "apply_unet", recorder("apply_unet"))
    monkeypatch.setattr(sd_models.sd_hijack.model_hijack, "undo_hijack", recorder("undo_hijack"))
    monkeypatch.setattr(sd_models.sd_hijack.model_hijack, "hijack", recorder("hijack"))
    monkeypatch.setattr(sd_models.script_callbacks, "model_loaded_callback", recorder("model_loaded_callback"))
    monkeypatch.setattr(sd_models.openclaw_cuda_graphs, "note_model_loaded", recorder("note_model_loaded"))
    monkeypatch.setattr(sd_models.sd_models_config, "find_checkpoint_config", lambda _state_dict, _info: "tiny.yaml")

    def get_empty_cond(model):
        recorder.events.append("get_empty_cond")
        assert not torch.is_grad_enabled()
        assert sd_models.model_data.sd_model is model  # extra_networks.activate in the real one reads shared.sd_model
        return empty_prompt_of(model)

    monkeypatch.setattr(sd_models, "get_empty_cond", get_empty_cond)
    monkeypatch.setattr(devices, "autocast", contextlib.nullcontext)  # CUDA autocast; the CPU stand-in needs none
    monkeypatch.setattr(openclaw_lifecycle_epochs, "publish_checkpoint_commit", recorder("publish_checkpoint_commit"))

    original_send_model_to_cpu = sd_models.send_model_to_cpu

    def send_model_to_cpu(model):
        recorder.events.append("send_model_to_cpu")
        original_send_model_to_cpu(model)

    monkeypatch.setattr(sd_models, "send_model_to_cpu", send_model_to_cpu)

    original_boundary = sd_models._model_acceleration_boundary

    def boundary(reason, model):
        recorder.events.append(f"boundary:{reason}")
        return original_boundary(reason, model)

    monkeypatch.setattr(sd_models, "_model_acceleration_boundary", boundary)
    return recorder


def empty_prompt_of(model):
    """A stand-in for the text encoder's empty-prompt encoding: a function of its current weights."""
    return model.cond_stage_model[0].weight.detach().float().sum().reshape(1, 1, 1)


def _loaded_model():
    """A model as a previous load leaves it: base checkpoint weights converted, on the device, registered."""
    torch.manual_seed(0)
    model = TinySD()
    base_info, base_state_dict = _checkpoint("base", 1)
    sd_models.load_model_weights(model, base_info, base_state_dict, sd_models.Timer())
    sd_models.send_model_to_device(model)
    sd_models.model_data.set_sd_model(model)
    return model, base_info


def _switch(monkeypatch, device, *, unified, **options):
    recorder = _environment(monkeypatch, device, unified=unified, **options)
    model, _ = _loaded_model()
    alternate_info, alternate_state_dict = _checkpoint("alternate", 2)
    monkeypatch.setattr(sd_models, "get_checkpoint_state_dict", lambda info, _timer: {key: value.clone() for key, value in alternate_state_dict.items()})
    pointers = {name: tensor.data_ptr() for name, tensor in model.state_dict().items()}
    recorder.events.clear()
    result = sd_models.reload_model_weights(model, alternate_info)
    assert result is model
    assert result.sd_checkpoint_info is alternate_info
    return model, recorder.events, pointers


def _assert_same_model_state(actual, expected):
    actual_state, expected_state = actual.state_dict(), expected.state_dict()
    assert list(actual_state) == list(expected_state)
    for key, expected_tensor in expected_state.items():
        tensor = actual_state[key]
        assert tensor.dtype == expected_tensor.dtype, key
        assert tensor.device == expected_tensor.device, key
        assert tensor.stride() == expected_tensor.stride(), key
        assert torch.equal(tensor.cpu(), expected_tensor.cpu()), key
    for attribute in ("alphas_cumprod", "alphas_cumprod_original"):
        assert getattr(actual, attribute).dtype == getattr(expected, attribute).dtype
        assert torch.equal(getattr(actual, attribute).cpu(), getattr(expected, attribute).cpu())


@pytest.mark.parametrize("device", DEVICES, ids=str)
def test_in_place_switch_matches_cpu_round_trip_bitwise(monkeypatch, device):
    with monkeypatch.context() as patch:
        oracle, oracle_events, _ = _switch(patch, device, unified=False)
    with monkeypatch.context() as patch:
        model, events, pointers = _switch(patch, device, unified=True)

    assert oracle_events.count("send_model_to_cpu") == 2  # keep_in_cpu in reuse_model_from_already_loaded + reload
    assert "send_model_to_cpu" not in events
    _assert_same_model_state(model, oracle)
    # Weights were overwritten in their existing (device-resident) storage, not reallocated by a move.
    assert {name: tensor.data_ptr() for name, tensor in model.state_dict().items()} == pointers
    first_stage = model.first_stage_model[0].weight
    unet = model.model.diffusion_model[0].weight
    assert unet.dtype == torch.bfloat16 and unet.is_contiguous(memory_format=torch.channels_last)
    assert first_stage.dtype == torch.bfloat16 and first_stage.device.type == device.type
    assert model.alphas_cumprod.dtype == torch.float32  # schedule buffer is excluded from the bf16 cast


def test_in_place_switch_keeps_lifecycle_order(monkeypatch):
    _, events, _ = _switch(monkeypatch, torch.device("cpu"), unified=True)

    assert events[:5] == ["apply_unet", "boundary:model_reload_in_place", "torch_gc", "undo_hijack", "load_vae"]
    assert events[5:] == ["hijack", "boundary:model_to_device", "model_loaded_callback", "get_empty_cond", "apply_unet", "note_model_loaded", "publish_checkpoint_commit"]


@pytest.mark.parametrize("unified", [False, True], ids=["cpu-round-trip", "in-place"])
def test_switch_recomputes_the_empty_prompt_padding(monkeypatch, unified):
    """pad_cond_uncond pads with cond_stage_model_empty_prompt: after a same-config switch it must be the new text
    encoder's, not the previous checkpoint's."""
    _environment(monkeypatch, torch.device("cpu"), unified=unified)
    model, _ = _loaded_model()
    model.cond_stage_model_empty_prompt = empty_prompt_of(model)
    before = model.cond_stage_model_empty_prompt
    alternate_info, alternate_state_dict = _checkpoint("alternate", 2)
    monkeypatch.setattr(sd_models, "get_checkpoint_state_dict", lambda info, _timer: {key: value.clone() for key, value in alternate_state_dict.items()})

    sd_models.reload_model_weights(model, alternate_info)

    assert torch.equal(model.cond_stage_model_empty_prompt, empty_prompt_of(model))
    assert not torch.equal(model.cond_stage_model_empty_prompt, before)


@pytest.mark.parametrize(("unified", "limit", "lowvram", "torchao"), [
    (False, 1, False, False),
    (True, 2, False, False),
    (True, 1, True, False),
    (True, 1, False, True),
])
def test_cpu_round_trip_is_kept_outside_the_in_place_contract(monkeypatch, unified, limit, lowvram, torchao):
    _environment(monkeypatch, torch.device("cpu"), unified=unified, limit=limit)
    model = TinySD()
    model.lowvram = lowvram
    monkeypatch.setattr(sd_models, "model_has_torchao_quantization", lambda _model: torchao)

    assert sd_models.checkpoint_switch_in_place_on_device(model) is False
    assert sd_models.checkpoint_switch_in_place_on_device(None) is False


def test_in_place_contract_holds_for_plain_unified_single_checkpoint(monkeypatch):
    _environment(monkeypatch, torch.device("cpu"), unified=True)

    assert sd_models.checkpoint_switch_in_place_on_device(TinySD()) is True


def test_unified_memory_probe_never_touches_cuda_for_other_devices():
    assert sd_models._device_has_unified_memory(torch.device("cpu")) is False
    assert sd_models._device_has_unified_memory(torch.device("meta")) is False


def test_limit_two_keeps_outgoing_model_cached_on_cpu(monkeypatch):
    recorder = _environment(monkeypatch, torch.device("cpu"), unified=True, limit=2)
    model, _ = _loaded_model()
    alternate_info, _ = _checkpoint("alternate", 2)
    fresh_model = TinySD()
    fresh_model.sd_checkpoint_info = alternate_info

    def load_model(checkpoint_info):
        assert checkpoint_info is alternate_info
        sd_models.model_data.set_sd_model(fresh_model)

    monkeypatch.setattr(sd_models, "load_model", load_model)
    recorder.events.clear()

    assert sd_models.reload_model_weights(model, alternate_info) is fresh_model
    assert recorder.events[:3] == ["send_model_to_cpu", "boundary:model_to_cpu", "torch_gc"]
    assert "boundary:model_reload_in_place" not in recorder.events
    assert sd_models.model_data.loaded_sd_models == [fresh_model, model]


def test_in_place_failure_rolls_back_and_finalizes_like_the_cpu_path(monkeypatch):
    recorder = _environment(monkeypatch, torch.device("cpu"), unified=True)
    model, base_info = _loaded_model()
    alternate_info, _ = _checkpoint("alternate", 2)
    monkeypatch.setattr(sd_models, "get_checkpoint_state_dict", lambda _info, _timer: {"alternate": True})
    loads = []

    def load_model_weights(_model, checkpoint_info, state_dict, _timer):
        loads.append((checkpoint_info, state_dict))
        if checkpoint_info is alternate_info:
            raise RuntimeError("alternate checkpoint is corrupt")

    monkeypatch.setattr(sd_models, "load_model_weights", load_model_weights)
    recorder.events.clear()

    with pytest.raises(RuntimeError, match="alternate checkpoint is corrupt"):
        sd_models.reload_model_weights(model, alternate_info)

    assert loads == [(alternate_info, {"alternate": True}), (base_info, None)]
    assert recorder.events == [
        "apply_unet", "boundary:model_reload_in_place", "torch_gc", "undo_hijack",
        "hijack", "boundary:model_to_device", "model_loaded_callback",
    ]  # no CPU move; checkpoint commits stay success-only, as on the CPU path


def test_reload_waits_for_a_model_load_in_progress(monkeypatch):
    import threading

    _environment(monkeypatch, torch.device("cpu"), unified=False)
    model, base_info = _loaded_model()
    loading, finish_loading = threading.Event(), threading.Event()

    def load_in_progress():  # holds model_data.lock as SdModelData.get_sd_model does for the startup load
        with sd_models.model_data.lock:
            loading.set()
            finish_loading.wait(10)

    loader = threading.Thread(target=load_in_progress)
    loader.start()
    assert loading.wait(10)
    results = []
    reload = threading.Thread(target=lambda: results.append(sd_models.reload_model_weights(model, base_info)))
    reload.start()
    reload.join(0.5)
    still_waiting = reload.is_alive()
    finish_loading.set()
    reload.join(10)
    loader.join(10)

    # Before, the reload ran next to the load in progress and could replace model_data.sd_model under it.
    assert still_waiting
    assert results == [model]


def test_reload_inside_a_model_load_on_the_same_thread_does_not_deadlock(monkeypatch):
    _environment(monkeypatch, torch.device("cpu"), unified=False)
    model, base_info = _loaded_model()

    with sd_models.model_data.lock:  # e.g. a model_loaded callback of a load reloading the checkpoint
        assert sd_models.reload_model_weights(model, base_info) is model
