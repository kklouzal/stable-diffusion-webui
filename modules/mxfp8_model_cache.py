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

_is_safetensors = torchao_model_cache.is_safetensors
_sha256_file = torchao_model_cache.sha256_file
_stat_source = torchao_model_cache.stat_source
_tensor_meta = torchao_model_cache.tensor_meta
_device_matches = torchao_model_cache.device_matches
_metadata_matches = torchao_model_cache.metadata_matches
_bias_metadata_matches = torchao_model_cache.bias_metadata_matches
_cached_bias_matches = torchao_model_cache.cached_bias_matches
_parameter_on_device = torchao_model_cache.parameter_on_device
_write_atomic_bytes = torchao_model_cache.write_atomic_bytes
_iter_eligible_linear_modules = torchao_model_cache.iter_eligible_linear_modules


def is_mxfp8_cache_path(filename: str) -> bool:
    return torchao_model_cache.is_cache_path(filename, CACHE_DIR_NAME)


def _cache_path_for(filename: Optional[str]) -> Optional[str]:
    return torchao_model_cache.cache_path_for(filename, CACHE_DIR_NAME)


def _sidecar_path(cache_path: str) -> str:
    return torchao_model_cache.sidecar_path(cache_path, SIDECAR_SUFFIX)


def _load_sidecar(cache_path: str) -> Optional[dict]:
    return torchao_model_cache.load_sidecar(cache_path, SIDECAR_SUFFIX)


def _expected_cache_metadata(filename: str, coverage=None) -> dict:
    return torchao_model_cache.expected_cache_metadata(filename, CACHE_VERSION, CONFIG_NAME, coverage)


def _sidecar_matches(filename: str, cache_path: str, coverage=None) -> bool:
    return torchao_model_cache.sidecar_matches(filename, cache_path, CACHE_VERSION, CONFIG_NAME, SIDECAR_SUFFIX, coverage)


def _register_mxfp8_safe_globals() -> None:
    # MXFP8 caches are produced locally from already-trusted model files. Keep
    # weights_only=True, but allow TorchAO's tensor subclass through PyTorch's
    # safe unpickler instead of falling back to unrestricted pickle loading.
    torch.serialization.add_safe_globals([MXTensor])


def _torch_load_cache(cache_path: str, device: torch.device | str):
    return torchao_model_cache.torch_load_cache(cache_path, device, _register_mxfp8_safe_globals)


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
