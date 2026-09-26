from __future__ import annotations

from typing import Callable, Optional

import torch
from torchao.prototype.mx_formats.mx_tensor import MXTensor

from modules import mxfp8_config, torchao_model_cache

CACHE_DIR_NAME = "mxfp8"
CACHE_VERSION = 2
CONFIG_NAME = mxfp8_config.CONFIG_NAME
SIDECAR_SUFFIX = ".mxfp8-cache.json"
LABEL = "MXFP8"


def is_mxfp8_cache_path(filename: str) -> bool:
    return torchao_model_cache.is_cache_path(filename, CACHE_DIR_NAME)


def _register_mxfp8_safe_globals() -> None:
    # MXFP8 caches are produced locally from already-trusted model files. Keep
    # weights_only=True, but allow TorchAO's tensor subclass through PyTorch's
    # safe unpickler instead of falling back to unrestricted pickle loading.
    torch.serialization.add_safe_globals([MXTensor])


def _is_mxfp8_tensor(tensor) -> bool:
    return isinstance(tensor, MXTensor)


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
        is_quant_tensor=_is_mxfp8_tensor,
        register_safe_globals=_register_mxfp8_safe_globals,
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
        is_quant_tensor=_is_mxfp8_tensor,
    )
