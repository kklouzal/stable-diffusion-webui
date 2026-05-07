from __future__ import annotations

import torch

CONFIG_NAME = "NVFP4DynamicActivationNVFP4WeightConfig_DynamicPerTensor_Triton_DynamicLinearCoverage_v1"
BLOCK_SIZE = 16
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


def is_nvfp4_tensor(weight) -> bool:
    return type(weight).__name__ == "NVFP4Tensor" and type(weight).__module__.startswith("torchao.")


def get_nvfp4_config():
    from torchao.prototype.mx_formats.inference_workflow import NVFP4DynamicActivationNVFP4WeightConfig

    return NVFP4DynamicActivationNVFP4WeightConfig(
        use_dynamic_per_tensor_scale=True,
        use_triton_kernel=True,
    )


def technical_linear_skip_reason(module) -> str | None:
    weight = getattr(module, "weight", None)
    if weight is None or getattr(weight, "ndim", None) != 2:
        return "not_2d_weight"
    already_nvfp4 = is_nvfp4_tensor(weight)
    if not already_nvfp4 and getattr(weight, "dtype", None) != torch.bfloat16:
        return "weight_not_bfloat16"
    if not already_nvfp4 and hasattr(weight, "is_contiguous") and not weight.is_contiguous():
        return "weight_not_contiguous"
    out_features, in_features = weight.shape
    if in_features % BLOCK_SIZE != 0:
        return "in_features_not_multiple_of_16"
    if out_features % OUT_FEATURE_MULTIPLE != 0:
        return "out_features_not_multiple_of_16"
    return None


def validate_config(config) -> None:
    if not getattr(config, "use_dynamic_per_tensor_scale", False):
        raise RuntimeError("NVFP4 A1111 path currently requires dynamic per-tensor scale")
    if not getattr(config, "use_triton_kernel", False):
        raise RuntimeError("NVFP4 A1111 path currently expects the MSLK/Triton kernel path")
