from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from modules import shared, shared_init

if getattr(shared, "opts", None) is None:
    shared_init.initialize()

from modules import sd_models, torchao_weight_quant
from modules.torchao_weight_quant import BACKENDS, MXFP8, NVFP4


def _opts(monkeypatch, **values):
    monkeypatch.setattr(sd_models, "shared", SimpleNamespace(opts=SimpleNamespace(**values)))


def test_persisted_backend_identity_is_stable():
    # Cache sidecars record these values; changing any of them silently invalidates every cached artifact.
    assert {(b.name, b.config_name, b.cache_dir_name, b.cache_version, b.sidecar_suffix, b.block_size) for b in BACKENDS.values()} == {
        ("mxfp8", "MXDynamicActivationMXWeightConfig_e4m3fn_AUTO_RCEIL_DynamicLinearCoverage_v4", "mxfp8", 2, ".mxfp8-cache.json", 32),
        ("nvfp4", "NVFP4DynamicActivationNVFP4WeightConfig_DynamicPerTensor_Triton_DynamicLinearCoverage_v1", "nvfp4", 2, ".nvfp4-cache.json", 16),
    }
    assert list(BACKENDS) == ["mxfp8", "nvfp4"]  # load order and policy-signature layout


@pytest.mark.parametrize("backend", BACKENDS.values(), ids=list(BACKENDS))
def test_technical_linear_skip_reasons(backend):
    bf16 = torch.bfloat16
    assert backend.technical_linear_skip_reason(torch.nn.Linear(backend.block_size * 2, 32, dtype=bf16)) is None
    assert backend.technical_linear_skip_reason(torch.nn.Conv2d(4, 4, 3)) == "not_2d_weight"
    assert backend.technical_linear_skip_reason(torch.nn.Linear(64, 32)) == "weight_not_bfloat16"
    transposed = torch.nn.Linear(32, 64, dtype=bf16)
    transposed.weight = torch.nn.Parameter(torch.zeros(32, 64, dtype=bf16).t())
    assert backend.technical_linear_skip_reason(transposed) == "weight_not_contiguous"
    assert backend.technical_linear_skip_reason(torch.nn.Linear(backend.block_size + 8, 32, dtype=bf16)) == f"in_features_not_multiple_of_{backend.block_size}"
    assert backend.technical_linear_skip_reason(torch.nn.Linear(64, 24, dtype=bf16)) == "out_features_not_multiple_of_16"


def test_selected_linear_coverage_normalizes_option_values(monkeypatch):
    _opts(monkeypatch, mxfp8_linear_coverage="self_attention", nvfp4_linear_coverage=None)
    assert sd_models.selected_linear_coverage(MXFP8) == {"self_attention"}
    assert sd_models.selected_linear_coverage(NVFP4) == set(torchao_weight_quant.LINEAR_COVERAGE_DEFAULT)
    _opts(monkeypatch, mxfp8_linear_coverage=["conditioner", "bogus", "cross_attention"])
    assert sd_models.selected_linear_coverage(MXFP8) == {"conditioner", "cross_attention"}


def test_linear_skip_reason_applies_region_policy_before_technical_checks(monkeypatch):
    _opts(monkeypatch, nvfp4_linear_coverage=["self_attention"])
    linear = torch.nn.Linear(64, 32, dtype=torch.bfloat16)
    unet = "model.diffusion_model.input_blocks.1.1.transformer_blocks.0"
    assert sd_models.linear_skip_reason(NVFP4, linear, f"{unet}.attn1.to_q") is None
    assert sd_models.linear_skip_reason(NVFP4, linear, f"{unet}.attn2.to_q") == "cross_attention"
    assert sd_models.linear_skip_reason(NVFP4, linear, "first_stage_model.decoder.mid.attn_1.q") == "vae"
    assert sd_models.linear_skip_reason(NVFP4, linear, "conditioner.embedders.0.self_attn.out_proj") == "conditioner"
    assert sd_models.linear_skip_reason(NVFP4, torch.nn.Conv2d(4, 4, 3), f"{unet}.attn1.to_q") == "not_linear"


def test_storage_gating_and_policy_signature_layout(monkeypatch):
    monkeypatch.setattr(sd_models.devices, "get_optimal_device_name", lambda: "cuda")
    monkeypatch.setattr(sd_models.devices, "dtype", torch.bfloat16, raising=False)
    sdxl, sd15 = SimpleNamespace(is_sdxl=True), SimpleNamespace(is_sdxl=False)
    _opts(monkeypatch, mxfp8_storage="Enable for SDXL", nvfp4_storage="Disable", mxfp8_linear_coverage=["unet_other", "self_attention"])

    assert sd_models.weight_quant_storage_enabled(MXFP8, None) is None
    assert sd_models.weight_quant_storage_enabled(MXFP8, sdxl) is True
    assert sd_models.weight_quant_storage_enabled(MXFP8, sd15) is False
    assert sd_models.weight_quant_storage_enabled(NVFP4, sdxl) is False
    assert sd_models.torchao_weight_quant_requested(sdxl) and not sd_models.torchao_weight_quant_requested(sd15)
    assert sd_models.torchao_quant_policy_signature(sdxl) == (True, ("self_attention", "unet_other"), MXFP8.config_name, False, (), None, torch.bfloat16)
    assert sd_models.torchao_quant_policy_signature(sd15) == (False, (), None, False, (), None, torch.bfloat16)


def test_backend_status_hooks_target_live_functions():
    # _wrap_backend_function skips missing attributes silently, so a rename would drop a status hook unnoticed.
    root = Path(__file__).resolve().parents[1]
    source = (root / "extensions/openclaw-clear-cond-cache/scripts/openclaw_clear_cond_cache.py").read_text()
    assert '_wrap_backend_function(_sd_models, "apply_weight_quantization"' in source
    assert callable(sd_models.apply_weight_quantization)
    assert '_wrap_backend_function(_lora_networks, "prepare_quant_active_config"' in source
    assert "\ndef prepare_quant_active_config(backend):" in (root / "extensions-builtin/Lora/networks.py").read_text().replace("\r\n", "\n")
