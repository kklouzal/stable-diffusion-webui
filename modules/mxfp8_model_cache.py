from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Optional

import torch
from torchao.prototype.mx_formats.mx_tensor import MXTensor

from modules import mxfp8_config

CACHE_DIR_NAME = "mxfp8"
SUPPORTED_ROOT_NAMES = ("Stable-diffusion", "Lora", "VAE")
CACHE_VERSION = 1
CONFIG_NAME = mxfp8_config.CONFIG_NAME


def _is_safetensors(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() == ".safetensors"


def _stat_source(filename: str) -> dict:
    stat = os.stat(filename)
    return {
        "path": os.path.abspath(filename),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def is_mxfp8_cache_path(filename: str) -> bool:
    return CACHE_DIR_NAME in Path(filename).parts


def _cache_path_for(filename: Optional[str]) -> Optional[str]:
    if not filename or not _is_safetensors(filename):
        return None

    path = Path(filename).resolve()
    parts = path.parts
    if CACHE_DIR_NAME in parts:
        return None

    root_index = None
    for root_name in SUPPORTED_ROOT_NAMES:
        try:
            root_index = parts.index(root_name)
            break
        except ValueError:
            continue

    if root_index is None:
        return None

    root = Path(*parts[:root_index + 1])
    relative = Path(*parts[root_index + 1:])
    if not relative.parts:
        return None

    return str(root / CACHE_DIR_NAME / relative.with_suffix(relative.suffix + ".pt"))


def _sidecar_path(cache_path: str) -> str:
    return cache_path + ".mxfp8-cache.json"


def _load_sidecar(cache_path: str) -> Optional[dict]:
    try:
        with open(_sidecar_path(cache_path), "r", encoding="utf8") as f:
            return json.load(f)
    except Exception:
        return None


def _sidecar_matches(filename: str, cache_path: str, coverage=None) -> bool:
    if not os.path.exists(cache_path):
        return False

    sidecar = _load_sidecar(cache_path)
    if not sidecar:
        return False

    return (
        sidecar.get("cache_version") == CACHE_VERSION
        and sidecar.get("config") == CONFIG_NAME
        and sidecar.get("source") == _stat_source(filename)
        and (coverage is None or sidecar.get("coverage") in (None, sorted(coverage)))
    )


def _write_atomic_bytes(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with NamedTemporaryFile("wb", delete=False, dir=os.path.dirname(path), prefix=".tmp-", suffix=".json") as f:
        tmp = f.name
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _register_mxfp8_safe_globals() -> None:
    # MXFP8 caches are produced locally from already-trusted model files. Keep
    # weights_only=True, but allow TorchAO's tensor subclass through PyTorch's
    # safe unpickler instead of falling back to unrestricted pickle loading.
    torch.serialization.add_safe_globals([MXTensor])


def _torch_load_cache(cache_path: str, device: torch.device | str):
    # A1111 monkey-patches torch.load with a legacy checkpoint pre-check that
    # rejects TorchAO tensor subclasses before PyTorch's weights_only safe
    # unpickler gets a chance to apply add_safe_globals(). Bypass only that
    # outer A1111 pre-check for our own sidecar-validated cache while keeping
    # weights_only=True.
    try:
        from modules import safe
        torch_load = safe.unsafe_torch_load
    except Exception:
        torch_load = torch.load

    _register_mxfp8_safe_globals()
    return torch_load(cache_path, map_location=device, weights_only=True)


def _is_mxfp8_tensor(tensor) -> bool:
    return isinstance(tensor, MXTensor)


def _iter_eligible_linear_modules(model, filter_fn: Callable):
    for fqn, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and filter_fn(module, fqn):
            yield fqn, module


def load_into_model(model, source_path: Optional[str], filter_fn: Callable, device: torch.device | str, coverage=None) -> bool:
    cache_path = _cache_path_for(source_path)
    if cache_path is None or source_path is None or not _sidecar_matches(source_path, cache_path, coverage):
        return False

    print(f"Loading MXFP8 cache for {source_path} from {cache_path}", flush=True)
    try:
        payload = _torch_load_cache(cache_path, device)
    except Exception as e:
        print(f"Ignoring unreadable MXFP8 cache {cache_path}: {e}")
        return False

    if not isinstance(payload, dict):
        print(f"Ignoring unreadable MXFP8 cache {cache_path}: payload is not a dict")
        return False

    tensors = payload.get("tensors", {})
    eligible_modules = list(_iter_eligible_linear_modules(model, filter_fn))
    print(f"Validating MXFP8 cache for {source_path}: expected {len(eligible_modules)} Linear modules", flush=True)
    missing = []
    for fqn, module in eligible_modules:
        entry = tensors.get(fqn)
        weight = entry.get("weight") if entry is not None else None
        if not _is_mxfp8_tensor(weight):
            missing.append(fqn)

    if missing:
        print(f"Ignoring incomplete MXFP8 cache {cache_path}: expected {len(eligible_modules)}, missing {len(missing)}")
        return False

    print(f"Assigning MXFP8 cache for {source_path}: {len(eligible_modules)} Linear modules", flush=True)
    with torch.no_grad():
        for fqn, module in eligible_modules:
            entry = tensors[fqn]
            module._parameters["weight"] = entry["weight"]
            bias = entry.get("bias")
            if bias is not None:
                module._parameters["bias"] = torch.nn.Parameter(bias.to(device=device), requires_grad=False)
            elif module.bias is not None:
                module._parameters["bias"] = None

    print(f"Loaded MXFP8 cache for {source_path} from {cache_path}", flush=True)
    return True


def save_from_model(model, source_path: Optional[str], filter_fn: Callable, eligible: int, skipped_linear: int, skipped_reasons: dict, coverage=None) -> Optional[str]:
    cache_path = _cache_path_for(source_path)
    if cache_path is None or source_path is None or eligible == 0:
        return None

    tensors = {}
    for fqn, module in _iter_eligible_linear_modules(model, filter_fn):
        if not _is_mxfp8_tensor(module.weight):
            return None
        tensors[fqn] = {
            "weight": module.weight.detach(),
            "bias": module.bias.detach() if module.bias is not None else None,
        }

    source_stat = _stat_source(source_path)
    payload = {
        "cache_version": CACHE_VERSION,
        "config": CONFIG_NAME,
        "source": source_stat,
        "coverage": sorted(coverage) if coverage is not None else None,
        "eligible_linear": eligible,
        "skipped_linear": skipped_linear,
        "skipped_reasons": skipped_reasons,
        "tensors": tensors,
    }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = cache_path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, cache_path)

    sidecar = {
        "cache_version": CACHE_VERSION,
        "config": CONFIG_NAME,
        "source": source_stat,
        "coverage": sorted(coverage) if coverage is not None else None,
        "cache": _stat_source(cache_path),
        "eligible_linear": eligible,
        "skipped_linear": skipped_linear,
        "skipped_reasons": skipped_reasons,
    }
    _write_atomic_bytes(_sidecar_path(cache_path), json.dumps(sidecar, indent=2, sort_keys=True).encode("utf8"))
    print(f"Created MXFP8 cache for {source_path} -> {cache_path}")
    return cache_path
