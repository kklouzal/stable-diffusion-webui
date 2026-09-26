from __future__ import annotations

from typing import Callable, Optional

import torch
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

from modules import nvfp4_config, torchao_model_cache

CACHE_DIR_NAME = "nvfp4"
CACHE_VERSION = 2
CONFIG_NAME = nvfp4_config.CONFIG_NAME
SIDECAR_SUFFIX = ".nvfp4-cache.json"
LABEL = "NVFP4"


def is_nvfp4_cache_path(filename: str) -> bool:
    return torchao_model_cache.is_cache_path(filename, CACHE_DIR_NAME)


def _register_nvfp4_safe_globals() -> None:
    # NVFP4 caches are produced locally from already-trusted model files. Keep
    # weights_only=True, but allow TorchAO's tensor subclass through PyTorch's
    # safe unpickler instead of falling back to unrestricted pickle loading.
    torch.serialization.add_safe_globals([NVFP4Tensor])


def _is_nvfp4_tensor(tensor) -> bool:
    return isinstance(tensor, NVFP4Tensor)


def load_into_model(model, source_path: Optional[str], filter_fn: Callable, device: torch.device | str, coverage=None) -> bool:
    return torchao_model_cache.load_into_model(
        model=model,
        source_path=source_path,
        filter_fn=filter_fn,
        device=device,
        coverage=coverage,
        cache_dir_name=CACHE_DIR_NAME,
        cache_version=CACHE_VERSION,
        config_name=CONFIG_NAME,
        sidecar_suffix=SIDECAR_SUFFIX,
        label=LABEL,
        is_quant_tensor=_is_nvfp4_tensor,
        register_safe_globals=_register_nvfp4_safe_globals,
    )


def save_from_model(model, source_path: Optional[str], filter_fn: Callable, eligible: int, skipped_linear: int, skipped_reasons: dict, coverage=None) -> Optional[str]:
    return torchao_model_cache.save_from_model(
        model=model,
        source_path=source_path,
        filter_fn=filter_fn,
        eligible=eligible,
        skipped_linear=skipped_linear,
        skipped_reasons=skipped_reasons,
        coverage=coverage,
        cache_dir_name=CACHE_DIR_NAME,
        cache_version=CACHE_VERSION,
        config_name=CONFIG_NAME,
        sidecar_suffix=SIDECAR_SUFFIX,
        label=LABEL,
        is_quant_tensor=_is_nvfp4_tensor,
    )
