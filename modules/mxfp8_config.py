from __future__ import annotations

import torch

CONFIG_NAME = "MXDynamicActivationMXWeightConfig_e4m3fn_AUTO_RCEIL_DynamicLinearCoverage_v4"
BLOCK_SIZE = 32
OUT_FEATURE_MULTIPLE = 16

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


def get_mxfp8_config():
    from torchao.prototype.mx_formats.inference_workflow import MXDynamicActivationMXWeightConfig
    from torchao.prototype.mx_formats.config import ScaleCalculationMode
    from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

    return MXDynamicActivationMXWeightConfig(
        activation_dtype=torch.float8_e4m3fn,
        weight_dtype=torch.float8_e4m3fn,
        kernel_preference=KernelPreference.AUTO,
        scaling_mode=ScaleCalculationMode.RCEIL,
    )


def technical_linear_skip_reason(module) -> str | None:
    weight = getattr(module, "weight", None)
    if weight is None or getattr(weight, "ndim", None) != 2:
        return "not_2d_weight"
    is_mxfp8_tensor = type(weight).__name__ == "MXTensor" and type(weight).__module__.startswith("torchao.")
    if not is_mxfp8_tensor and getattr(weight, "dtype", None) != torch.bfloat16:
        return "weight_not_bfloat16"
    if not is_mxfp8_tensor and hasattr(weight, "is_contiguous") and not weight.is_contiguous():
        return "weight_not_contiguous"
    out_features, in_features = weight.shape
    if in_features % BLOCK_SIZE != 0:
        return "in_features_not_multiple_of_32"
    if out_features % OUT_FEATURE_MULTIPLE != 0:
        return "out_features_not_multiple_of_16"
    return None


def validate_kernel_preference(config) -> None:
    from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

    if config.kernel_preference not in (KernelPreference.AUTO, KernelPreference.EMULATED):
        raise RuntimeError(
            f"Unsupported MXFP8 kernel preference for TorchAO MXTensor Linear path: {config.kernel_preference}. "
            "Use AUTO for native torch._scaled_mm/cuBLASLt or EMULATED for diagnostics."
        )
