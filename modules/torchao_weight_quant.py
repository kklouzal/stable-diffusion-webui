"""TorchAO Linear weight-quantization backends (MXFP8 and NVFP4).

A backend's persisted identity must stay stable: `config_name`, `cache_dir_name`, `cache_version` and
`sidecar_suffix` are recorded in cached-artifact sidecars, so changing any of them invalidates every cached
artifact. `name` is the prefix of the backend's options (`<name>_storage`, `<name>_linear_coverage`) and
model/module attributes (`<name>_quantization_stats`, `network_<name>_*`); `label` appears in logs and timers.

torchao is imported lazily, so option definitions can import this module before torchao loads.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable

import torch

LINEAR_COVERAGE_UNET_OTHER = "unet_other"
LINEAR_COVERAGE_SELF_ATTENTION = "self_attention"
LINEAR_COVERAGE_CROSS_ATTENTION = "cross_attention"
LINEAR_COVERAGE_CONDITIONER = "conditioner"
LINEAR_COVERAGE_CHOICES = [
    LINEAR_COVERAGE_UNET_OTHER,
    LINEAR_COVERAGE_SELF_ATTENTION,
    LINEAR_COVERAGE_CROSS_ATTENTION,
    LINEAR_COVERAGE_CONDITIONER,
]
LINEAR_COVERAGE_DEFAULT = [LINEAR_COVERAGE_UNET_OTHER]
OUT_FEATURE_MULTIPLE = 16


@dataclasses.dataclass(frozen=True)
class Backend:
    name: str
    label: str
    config_name: str
    block_size: int
    cache_dir_name: str
    cache_version: int
    sidecar_suffix: str
    make_config: Callable[[], Any]
    validate_config: Callable[[Any], None]
    tensor_type: Callable[[], type]

    def is_quant_tensor(self, tensor) -> bool:
        return isinstance(tensor, self.tensor_type())

    def register_safe_globals(self) -> None:
        # Caches are produced locally from already-trusted model files. Keep weights_only=True, but allow
        # TorchAO's tensor subclass through PyTorch's safe unpickler instead of unrestricted pickle loading.
        torch.serialization.add_safe_globals([self.tensor_type()])

    def technical_linear_skip_reason(self, module) -> str | None:
        weight = getattr(module, "weight", None)
        if weight is None or getattr(weight, "ndim", None) != 2:
            return "not_2d_weight"
        already_quantized = self.is_quant_tensor(weight)
        if not already_quantized and getattr(weight, "dtype", None) != torch.bfloat16:
            return "weight_not_bfloat16"
        if not already_quantized and hasattr(weight, "is_contiguous") and not weight.is_contiguous():
            return "weight_not_contiguous"
        out_features, in_features = weight.shape
        if in_features % self.block_size != 0:
            return f"in_features_not_multiple_of_{self.block_size}"
        if out_features % OUT_FEATURE_MULTIPLE != 0:
            return f"out_features_not_multiple_of_{OUT_FEATURE_MULTIPLE}"
        return None


@functools.cache
def _mxfp8_tensor_type() -> type:
    from torchao.prototype.mx_formats.mx_tensor import MXTensor

    return MXTensor


@functools.cache
def _nvfp4_tensor_type() -> type:
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    return NVFP4Tensor


def _mxfp8_config():
    from torchao.prototype.mx_formats.inference_workflow import MXDynamicActivationMXWeightConfig
    from torchao.prototype.mx_formats.config import ScaleCalculationMode
    from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

    return MXDynamicActivationMXWeightConfig(
        activation_dtype=torch.float8_e4m3fn,
        weight_dtype=torch.float8_e4m3fn,
        kernel_preference=KernelPreference.AUTO,
        scaling_mode=ScaleCalculationMode.RCEIL,
    )


def _validate_mxfp8_config(config) -> None:
    from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

    if config.kernel_preference not in (KernelPreference.AUTO, KernelPreference.EMULATED):
        raise RuntimeError(
            f"Unsupported MXFP8 kernel preference for TorchAO MXTensor Linear path: {config.kernel_preference}. "
            "Use AUTO for native torch._scaled_mm/cuBLASLt or EMULATED for diagnostics."
        )


def _nvfp4_config():
    from torchao.prototype.mx_formats.inference_workflow import NVFP4DynamicActivationNVFP4WeightConfig

    return NVFP4DynamicActivationNVFP4WeightConfig(
        use_dynamic_per_tensor_scale=True,
        use_triton_kernel=True,
    )


def _validate_nvfp4_config(config) -> None:
    if not getattr(config, "use_dynamic_per_tensor_scale", False):
        raise RuntimeError("NVFP4 A1111 path currently requires dynamic per-tensor scale")
    if not getattr(config, "use_triton_kernel", False):
        raise RuntimeError("NVFP4 A1111 path currently expects the MSLK/Triton kernel path")


MXFP8 = Backend(
    name="mxfp8",
    label="MXFP8",
    config_name="MXDynamicActivationMXWeightConfig_e4m3fn_AUTO_RCEIL_DynamicLinearCoverage_v4",
    block_size=32,
    cache_dir_name="mxfp8",
    cache_version=2,
    sidecar_suffix=".mxfp8-cache.json",
    make_config=_mxfp8_config,
    validate_config=_validate_mxfp8_config,
    tensor_type=_mxfp8_tensor_type,
)
NVFP4 = Backend(
    name="nvfp4",
    label="NVFP4",
    config_name="NVFP4DynamicActivationNVFP4WeightConfig_DynamicPerTensor_Triton_DynamicLinearCoverage_v1",
    block_size=16,
    cache_dir_name="nvfp4",
    cache_version=2,
    sidecar_suffix=".nvfp4-cache.json",
    make_config=_nvfp4_config,
    validate_config=_validate_nvfp4_config,
    tensor_type=_nvfp4_tensor_type,
)
BACKENDS = {backend.name: backend for backend in (MXFP8, NVFP4)}
