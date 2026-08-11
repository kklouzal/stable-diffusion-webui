from types import SimpleNamespace

import pytest
import torch

from modules import shared, shared_init

if getattr(shared, "opts", None) is None:
    shared_init.initialize()

from modules import sd_models, sd_vae


class NoGenericToModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.register_buffer("ordinary_buffer", torch.ones(1))
        self.lowvram = False
        self.used_config = object()
        self.sd_checkpoint_info = SimpleNamespace(filename="base.safetensors")
        self.sd_model_checkpoint = self.sd_checkpoint_info.filename
        self.mxfp8_quantization_stats = {"selected_linear_coverage": []}
        self.generic_to_calls = []

    def to(self, *args, **kwargs):  # pragma: no cover - failures prove the regression
        self.generic_to_calls.append((args, kwargs))
        raise AssertionError("generic Module.to() must not be used for TorchAO reload lifecycle")


class ModelDataStub:
    def __init__(self, model):
        self.sd_model = model
        self.loaded_sd_models = [model]
        self.set_calls = []

    def set_sd_model(self, model):
        self.set_calls.append(model)
        self.sd_model = model


def _patch_torchao_reload_path(monkeypatch, model, *, target_device=torch.device("cpu"), callback=None, fresh_checkpoint_reload=False):
    checkpoint_config = model.used_config
    model_data = ModelDataStub(model)

    monkeypatch.setattr(sd_models, "model_data", model_data)
    monkeypatch.setattr(sd_models.devices, "fp8", False, raising=False)
    model.mxfp8_quantization_stats = {"selected_linear_coverage": []} if fresh_checkpoint_reload else {}
    monkeypatch.setattr(sd_models.devices, "mxfp8", fresh_checkpoint_reload, raising=False)
    monkeypatch.setattr(sd_models.devices, "nvfp4", False, raising=False)
    monkeypatch.setattr(sd_models.devices, "cpu", torch.device("cpu"), raising=False)
    monkeypatch.setattr(sd_models.devices, "device", target_device, raising=False)
    monkeypatch.setattr(sd_models.devices, "dtype", torch.float32, raising=False)
    monkeypatch.setattr(sd_models.devices, "torch_gc", lambda: None)
    monkeypatch.setattr(sd_models.shared, "device", target_device, raising=False)
    monkeypatch.setattr(sd_models.lowvram, "apply", lambda _model: None)
    monkeypatch.setattr(sd_models, "check_fp8", lambda _model: False)
    monkeypatch.setattr(sd_models, "check_mxfp8", lambda _model: fresh_checkpoint_reload)
    monkeypatch.setattr(sd_models, "check_nvfp4", lambda _model: False)
    if not fresh_checkpoint_reload:
        monkeypatch.setattr(sd_models, "model_has_torchao_quantization", lambda _model: True)
    monkeypatch.setattr(sd_models, "mxfp8_selected_linear_coverage", lambda: [])
    monkeypatch.setattr(sd_models, "nvfp4_selected_linear_coverage", lambda: [])
    monkeypatch.setattr(sd_models, "reuse_model_from_already_loaded", lambda sd_model, _info, _timer: sd_model)
    monkeypatch.setattr(sd_models, "get_checkpoint_state_dict", lambda _info, _timer: {"state_dict": "alternate"})
    monkeypatch.setattr(sd_models.sd_models_config, "find_checkpoint_config", lambda _state_dict, _info: checkpoint_config)
    monkeypatch.setattr(sd_models.sd_unet, "apply_unet", lambda *args, **kwargs: None)
    monkeypatch.setattr(sd_models.sd_hijack.model_hijack, "undo_hijack", lambda _model: None)
    monkeypatch.setattr(sd_models.sd_hijack.model_hijack, "hijack", lambda _model: None)
    monkeypatch.setattr(sd_models.script_callbacks, "model_loaded_callback", callback or (lambda _model: None))

    return model_data


def test_torchao_alternate_checkpoint_uses_fresh_uncached_load(monkeypatch):
    old_model = NoGenericToModel()
    fresh_model = NoGenericToModel()
    fresh_model.sd_checkpoint_info = SimpleNamespace(filename="alternate.safetensors")
    model_data = _patch_torchao_reload_path(monkeypatch, old_model, fresh_checkpoint_reload=True)
    alternate = SimpleNamespace(filename="alternate.safetensors")
    state_dict = {"state_dict": "fresh alternate"}
    load_calls = []
    undo_calls = []

    monkeypatch.setattr(sd_models, "read_state_dict", lambda filename: state_dict if filename == "alternate.safetensors" else (_ for _ in ()).throw(AssertionError(filename)))
    monkeypatch.setattr(sd_models, "load_model_weights", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("generic reload path must not be used for TorchAO checkpoint switches")))
    monkeypatch.setattr(sd_models, "send_model_to_cpu", lambda _model: (_ for _ in ()).throw(AssertionError("old TorchAO tree should be detached, not generically reloaded")))
    monkeypatch.setattr(sd_models.sd_hijack.model_hijack, "undo_hijack", lambda model: undo_calls.append(model))

    def load_model(checkpoint_info, *, already_loaded_state_dict, checkpoint_config):
        load_calls.append((checkpoint_info.filename, already_loaded_state_dict, checkpoint_config))
        model_data.sd_model = fresh_model

    monkeypatch.setattr(sd_models, "load_model", load_model)

    result = sd_models.reload_model_weights(old_model, alternate)

    assert result is fresh_model
    assert load_calls == [("alternate.safetensors", state_dict, old_model.used_config)]
    assert undo_calls == [old_model]
    assert old_model not in model_data.loaded_sd_models
    assert old_model.generic_to_calls == []


def test_torchao_alternate_reload_rollback_uses_safe_moves_and_restores_target_placement(monkeypatch):
    model = NoGenericToModel()
    target = torch.device("meta")
    _patch_torchao_reload_path(monkeypatch, model, target_device=target)
    alternate = SimpleNamespace(filename="alternate.safetensors")
    calls = []

    def load_model_weights(_model, checkpoint_info, state_dict, _timer):
        calls.append((checkpoint_info.filename, state_dict))
        if checkpoint_info is alternate:
            raise RuntimeError("alternate quantization cache miss")

    monkeypatch.setattr(sd_models, "load_model_weights", load_model_weights)

    with pytest.raises(RuntimeError, match="alternate quantization cache miss"):
        sd_models.reload_model_weights(model, alternate)

    assert calls == [("alternate.safetensors", {"state_dict": "alternate"}), ("base.safetensors", None)]
    assert model.generic_to_calls == []
    assert model.linear.weight.device.type == "meta"
    assert model.ordinary_buffer.device.type == "meta"


def test_torchao_alternate_reload_preserves_original_error_when_finalization_fails(monkeypatch):
    model = NoGenericToModel()
    _patch_torchao_reload_path(
        monkeypatch,
        model,
        callback=lambda _model: (_ for _ in ()).throw(RuntimeError("callback finalization failed")),
    )
    alternate = SimpleNamespace(filename="alternate.safetensors")

    def load_model_weights(_model, checkpoint_info, _state_dict, _timer):
        if checkpoint_info is alternate:
            raise RuntimeError("alternate quantization cache miss")

    monkeypatch.setattr(sd_models, "load_model_weights", load_model_weights)

    with pytest.raises(RuntimeError, match="alternate quantization cache miss") as excinfo:
        sd_models.reload_model_weights(model, alternate)

    assert model.generic_to_calls == []
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "callback finalization failed" in str(excinfo.value.__cause__)


def test_torchao_device_helper_moves_ordinary_params_and_buffers_without_module_to(monkeypatch):
    model = NoGenericToModel()
    monkeypatch.setattr(sd_models, "model_has_torchao_quantization", lambda _model: True)
    monkeypatch.setattr(sd_models.lowvram, "apply", lambda _model: None)
    monkeypatch.setattr(sd_models.shared, "device", torch.device("meta"), raising=False)

    sd_models.send_model_to_device(model)

    assert model.generic_to_calls == []
    assert model.linear.weight.device.type == "meta"
    assert model.ordinary_buffer.device.type == "meta"


def test_mxfp8_cache_miss_pre_moves_model_and_does_not_pass_device_to_quantize(monkeypatch):
    import sys

    class QuantizeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
            self.first_stage_model = object()
            self.to_calls = []

        def to(self, *args, **kwargs):
            self.to_calls.append((args, kwargs))
            return super().to(*args, **kwargs)

    class Timer:
        def __init__(self):
            self.records = []

        def record(self, item):
            self.records.append(item)

    model = QuantizeModel()
    quantize_calls = []

    def quantize_(quant_model, config, **kwargs):
        quantize_calls.append((quant_model, config, kwargs))
        assert "device" not in kwargs

    quantization_module = SimpleNamespace(quantize_=quantize_)
    monkeypatch.setitem(sys.modules, "torchao.quantization", quantization_module)
    monkeypatch.setattr(sd_models.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sd_models.devices, "dtype", torch.bfloat16, raising=False)
    monkeypatch.setattr(sd_models.devices, "device", torch.device("cpu"), raising=False)
    monkeypatch.setattr(sd_models.devices, "cpu", torch.device("cpu"), raising=False)
    monkeypatch.setattr(sd_models.shared, "device", torch.device("cpu"), raising=False)
    monkeypatch.setattr(sd_models, "mxfp8_selected_linear_coverage", lambda: [])
    monkeypatch.setattr(sd_models, "mxfp8_linear_policy_skip_reason", lambda _module, _fqn: None)
    monkeypatch.setattr(sd_models.mxfp8_config, "technical_linear_skip_reason", lambda _module: None)
    monkeypatch.setattr(sd_models.mxfp8_config, "get_mxfp8_config", lambda: "mxfp8-config")
    monkeypatch.setattr(sd_models.mxfp8_config, "validate_kernel_preference", lambda _config: None)
    monkeypatch.setattr(sd_models.mxfp8_model_cache, "load_into_model", lambda *args, **kwargs: False)
    save_calls = []
    monkeypatch.setattr(sd_models.mxfp8_model_cache, "save_from_model", lambda *args, **kwargs: save_calls.append((args, kwargs)))

    timer = Timer()
    sd_models.apply_mxfp8_weight_quantization(model, timer, source_path="cache-miss.safetensors")

    assert model.to_calls == [((torch.device("cpu"),), {})]
    assert quantize_calls == [(model, "mxfp8-config", {"filter_fn": quantize_calls[0][2]["filter_fn"]})]
    assert save_calls
    assert model.first_stage_model is not None
    assert model.mxfp8_quantization_stats["cache_loaded"] is False

class FakeFirstStage:
    def load_state_dict(self, state):
        pass

    def to(self, dtype):
        self.dtype = dtype
        return self


class VaeReloadModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lowvram = False
        self.first_stage_model = FakeFirstStage()
        self.sd_checkpoint_info = SimpleNamespace(filename="base.safetensors")
        self.to_calls = []
        self.mxfp8_quantization_stats = {"selected_linear_coverage": []}

    def to(self, *args, **kwargs):  # pragma: no cover - any call is the regression
        self.to_calls.append((args, kwargs))
        raise AssertionError("generic model.to() must not be used for TorchAO VAE reload")


def test_reload_vae_uses_torchao_safe_model_movers(monkeypatch):
    from modules import sd_vae

    model = VaeReloadModel()
    calls = []

    monkeypatch.setattr(sd_vae, "loaded_vae_file", "old.vae")
    monkeypatch.setattr(sd_vae.sd_models, "send_model_to_cpu", lambda m: calls.append(("cpu", m)))
    monkeypatch.setattr(sd_vae.sd_models, "send_model_to_device", lambda m: calls.append(("device", m)))
    monkeypatch.setattr(sd_vae.sd_hijack.model_hijack, "undo_hijack", lambda m: calls.append(("undo", m)))
    monkeypatch.setattr(sd_vae.sd_hijack.model_hijack, "hijack", lambda m: calls.append(("hijack", m)))
    monkeypatch.setattr(sd_vae.script_callbacks, "model_loaded_callback", lambda m: calls.append(("callback", m)))
    monkeypatch.setattr(sd_vae, "load_vae", lambda *args, **kwargs: calls.append(("load", args[0])))

    sd_vae.reload_vae_weights(model, vae_file="new.vae")

    assert [name for name, _ in calls] == ["cpu", "undo", "load", "hijack", "device", "callback"]
    assert model.to_calls == []


def test_reload_vae_finalizes_after_load_failure(monkeypatch):
    from modules import sd_vae

    model = VaeReloadModel()
    calls = []

    monkeypatch.setattr(sd_vae, "loaded_vae_file", "old.vae")
    monkeypatch.setattr(sd_vae.sd_models, "send_model_to_cpu", lambda m: calls.append(("cpu", m)))
    monkeypatch.setattr(sd_vae.sd_models, "send_model_to_device", lambda m: calls.append(("device", m)))
    monkeypatch.setattr(sd_vae.sd_hijack.model_hijack, "undo_hijack", lambda m: calls.append(("undo", m)))
    monkeypatch.setattr(sd_vae.sd_hijack.model_hijack, "hijack", lambda m: calls.append(("hijack", m)))
    monkeypatch.setattr(sd_vae.script_callbacks, "model_loaded_callback", lambda m: calls.append(("callback", m)))

    def fail_load(*args, **kwargs):
        calls.append(("load", args[0]))
        raise RuntimeError("vae load failed")

    monkeypatch.setattr(sd_vae, "load_vae", fail_load)

    with pytest.raises(RuntimeError, match="vae load failed"):
        sd_vae.reload_vae_weights(model, vae_file="bad.vae")

    assert [name for name, _ in calls] == ["cpu", "undo", "load", "hijack", "device", "callback"]
    assert model.to_calls == []



def test_torchao_fresh_load_bypasses_meta_state_dict_loader(monkeypatch):
    model = NoGenericToModel()
    model.is_sdxl = True
    state_dict = {"state_dict": "weights"}
    calls = []

    monkeypatch.setattr(sd_models, "instantiate_from_config", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(sd_models, "repair_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sd_models.OmegaConf, "load", lambda _config: SimpleNamespace(model="model-config"))
    monkeypatch.setattr(sd_models, "set_model_type", lambda _model, _state_dict: setattr(_model, "is_sdxl", True))
    monkeypatch.setattr(sd_models, "set_model_fields", lambda _model: None)
    monkeypatch.setattr(sd_models, "check_mxfp8", lambda _model: True)
    monkeypatch.setattr(sd_models, "check_nvfp4", lambda _model: False)
    monkeypatch.setattr(sd_models, "load_model_weights", lambda *_args, **_kwargs: calls.append("load"))
    monkeypatch.setattr(sd_models, "get_empty_cond", lambda _model: "empty-cond")
    monkeypatch.setattr(sd_models, "send_model_to_device", lambda _model: calls.append("device"))
    monkeypatch.setattr(sd_models.devices, "autocast", lambda: __import__("contextlib").nullcontext())
    monkeypatch.setattr(sd_models.sd_hijack.model_hijack, "hijack", lambda _model: calls.append("hijack"))
    monkeypatch.setattr(sd_models.script_callbacks, "model_loaded_callback", lambda _model: calls.append("callback"))
    monkeypatch.setattr(sd_models.model_data, "set_sd_model", lambda _model: calls.append("set"))
    monkeypatch.setattr(sd_models.sd_hijack.model_hijack.embedding_db, "load_textual_inversion_embeddings", lambda **_kwargs: calls.append("embeddings"))

    class ForbiddenMetaLoader:
        def __init__(self, *args, **kwargs):
            raise AssertionError("TorchAO fresh loads must not enter LoadStateDictOnMeta")

    monkeypatch.setattr(sd_models.sd_disable_initialization, "LoadStateDictOnMeta", ForbiddenMetaLoader)

    result = sd_models.load_model(SimpleNamespace(filename="torchao.safetensors"), already_loaded_state_dict=state_dict, checkpoint_config="cfg")

    assert result is model
    assert calls == ["load", "device", "hijack", "set", "embeddings", "callback"]


def test_model_move_mutation_is_serialized_with_unet_graph_runtime(monkeypatch):
    model = NoGenericToModel()
    entered = []
    invalidations = []

    class Boundary:
        def __enter__(self):
            entered.append("enter")

        def __exit__(self, exc_type, exc, tb):
            entered.append("exit")

    def boundary(reason, details=None):
        invalidations.append(("unet-boundary", reason, details))
        return Boundary()

    monkeypatch.setattr(sd_models, "model_has_torchao_quantization", lambda _model: True)
    monkeypatch.setattr(sd_models.lowvram, "apply", lambda _model: entered.append("lowvram"))
    monkeypatch.setattr(sd_models.shared, "device", torch.device("cpu"), raising=False)
    monkeypatch.setattr(sd_models.openclaw_cuda_graphs, "mutable_runtime_boundary", boundary)
    monkeypatch.setattr(sd_models.openclaw_cuda_graphs, "invalidate", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("direct invalidate should be inside mutable_runtime_boundary")))
    monkeypatch.setattr(sd_models.openclaw_vae_decode_graphs, "invalidate_if_changed", lambda boundary, state, reason: invalidations.append(("vae", boundary, reason)))

    sd_models.send_model_to_device(model)

    assert invalidations == [
        ("unet-boundary", "model_to_device", "base.safetensors"),
        ("vae", "model_acceleration", "model_to_device"),
    ]
    assert entered[0] == "enter"
    assert entered[-1] == "exit"
    assert "lowvram" in entered
    assert model.generic_to_calls == []


def test_model_moves_invalidate_unet_and_vae_graph_caches(monkeypatch):
    model = NoGenericToModel()
    invalidations = []

    monkeypatch.setattr(sd_models, "model_has_torchao_quantization", lambda _model: True)
    monkeypatch.setattr(sd_models.lowvram, "apply", lambda _model: None)
    monkeypatch.setattr(sd_models.devices, "torch_gc", lambda: None)
    monkeypatch.setattr(sd_models.shared, "device", torch.device("cpu"), raising=False)
    monkeypatch.setattr(sd_models.openclaw_cuda_graphs, "invalidate", lambda reason, details=None: invalidations.append(("unet", reason, details)))
    monkeypatch.setattr(sd_models.openclaw_vae_decode_graphs, "invalidate_if_changed", lambda boundary, state, reason: invalidations.append(("vae", boundary, reason)))

    sd_models.send_model_to_device(model)
    sd_models.send_model_to_cpu(model)

    assert ("unet", "model_to_device", "base.safetensors") in invalidations
    assert ("vae", "model_acceleration", "model_to_device") in invalidations
    assert ("unet", "model_to_cpu", "base.safetensors") in invalidations
    assert ("vae", "model_acceleration", "model_to_cpu") in invalidations


def test_vae_reload_finalization_failure_discards_pending_and_next_success_is_fresh(monkeypatch):
    model = VaeReloadModel()
    notes = []
    pending = []

    monkeypatch.setattr(sd_vae, "loaded_vae_file", "old.vae")
    monkeypatch.setattr(sd_vae, "load_vae", lambda target, *_: pending.append("new"))
    monkeypatch.setattr(sd_vae.openclaw_lifecycle_epochs, "discard_pending_vae_commit", lambda target: pending.clear())
    monkeypatch.setattr(sd_vae.openclaw_lifecycle_epochs, "take_pending_vae_commit", lambda target: (bool(pending.pop()) if pending else False, False))
    monkeypatch.setattr(sd_vae.openclaw_lifecycle_epochs, "note_vae_commit", lambda target, **kwargs: notes.append(kwargs))
    monkeypatch.setattr(sd_vae.sd_models, "send_model_to_cpu", lambda target: None)
    monkeypatch.setattr(sd_vae.sd_models, "send_model_to_device", lambda target: None)
    monkeypatch.setattr(sd_vae.sd_hijack.model_hijack, "undo_hijack", lambda target: None)
    monkeypatch.setattr(sd_vae.sd_hijack.model_hijack, "hijack", lambda target: None)
    monkeypatch.setattr(sd_vae.script_callbacks, "model_loaded_callback", lambda target: (_ for _ in ()).throw(RuntimeError("finalize")))

    with pytest.raises(RuntimeError, match="finalize"):
        sd_vae.reload_vae_weights(model, vae_file="new.vae")
    assert pending == []
    assert notes == []

    monkeypatch.setattr(sd_vae.script_callbacks, "model_loaded_callback", lambda target: None)
    sd_vae.reload_vae_weights(model, vae_file="new.vae")
    assert notes == [{"bytes_changed": True, "object_changed": False, "publish": True}]


def test_vae_load_has_single_pending_note_contract():
    source = open("modules/sd_vae.py", encoding="utf-8").read()
    load_body = source[source.index("def load_vae("):source.index("# don't call this from outside")]
    assert load_body.count("note_vae_commit(") == 1
