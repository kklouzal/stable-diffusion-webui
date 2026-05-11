from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Optional

import torch
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

from modules import nvfp4_config

CACHE_DIR_NAME = "nvfp4"
SUPPORTED_ROOT_NAMES = ("Stable-diffusion",)
CACHE_VERSION = 2
CONFIG_NAME = nvfp4_config.CONFIG_NAME


def _is_safetensors(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() == ".safetensors"


def _sha256_file(filename: str) -> str:
    h = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stat_source(filename: str) -> dict:
    stat = os.stat(filename)
    return {
        "path": os.path.abspath(filename),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(filename),
    }


def _tensor_meta(tensor) -> dict:
    return {
        "shape": list(getattr(tensor, "shape", []) or []),
        "dtype": str(getattr(tensor, "dtype", None)),
        "device": str(getattr(tensor, "device", None)),
        "tensor_type": type(tensor).__module__ + "." + type(tensor).__name__,
    }


def _device_matches(actual, expected) -> bool:
    actual_device = torch.device(str(actual))
    expected_device = torch.device(expected)
    if expected_device.index is None:
        return actual_device.type == expected_device.type
    return actual_device == expected_device


def _metadata_matches(tensor, metadata: dict | None, expected_device) -> bool:
    if metadata:
        actual = _tensor_meta(tensor)
        for key in ("shape", "dtype", "tensor_type"):
            if actual.get(key) != metadata.get(key):
                return False
    return _device_matches(getattr(tensor, "device", None), expected_device)


def is_nvfp4_cache_path(filename: str) -> bool:
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
    return cache_path + ".nvfp4-cache.json"


def _load_sidecar(cache_path: str) -> Optional[dict]:
    try:
        with open(_sidecar_path(cache_path), "r", encoding="utf8") as f:
            return json.load(f)
    except Exception:
        return None


def _expected_cache_metadata(filename: str, coverage=None) -> dict:
    return {
        "cache_version": CACHE_VERSION,
        "config": CONFIG_NAME,
        "source": _stat_source(filename),
        "coverage": sorted(coverage) if coverage is not None else None,
    }


def _sidecar_matches(filename: str, cache_path: str, coverage=None) -> bool:
    if not os.path.exists(cache_path):
        return False

    sidecar = _load_sidecar(cache_path)
    if not sidecar:
        return False

    expected = _expected_cache_metadata(filename, coverage)
    coverage_matches = coverage is None or sidecar.get("coverage") in (None, expected["coverage"])
    return (
        sidecar.get("cache_version") == expected["cache_version"]
        and sidecar.get("config") == expected["config"]
        and sidecar.get("source") == expected["source"]
        and coverage_matches
        and sidecar.get("cache") == _stat_source(cache_path)
    )


def _write_atomic_bytes(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with NamedTemporaryFile("wb", delete=False, dir=os.path.dirname(path), prefix=".tmp-", suffix=".json") as f:
        tmp = f.name
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _register_nvfp4_safe_globals() -> None:
    # NVFP4 caches are produced locally from already-trusted model files. Keep
    # weights_only=True, but allow TorchAO's tensor subclass through PyTorch's
    # safe unpickler instead of falling back to unrestricted pickle loading.
    torch.serialization.add_safe_globals([NVFP4Tensor])


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

    _register_nvfp4_safe_globals()
    return torch_load(cache_path, map_location=device, weights_only=True)


def _is_nvfp4_tensor(tensor) -> bool:
    return isinstance(tensor, NVFP4Tensor)


def _iter_eligible_linear_modules(model, filter_fn: Callable):
    for fqn, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and filter_fn(module, fqn):
            yield fqn, module


def load_into_model(model, source_path: Optional[str], filter_fn: Callable, device: torch.device | str, coverage=None) -> bool:
    cache_path = _cache_path_for(source_path)
    if cache_path is None or source_path is None or not _sidecar_matches(source_path, cache_path, coverage):
        return False

    print(f"Loading NVFP4 cache for {source_path} from {cache_path}", flush=True)
    try:
        payload = _torch_load_cache(cache_path, device)
    except Exception as e:
        print(f"Ignoring unreadable NVFP4 cache {cache_path}: {e}")
        return False

    if not isinstance(payload, dict):
        print(f"Ignoring unreadable NVFP4 cache {cache_path}: payload is not a dict")
        return False

    expected = _expected_cache_metadata(source_path, coverage)
    payload_coverage_matches = coverage is None or payload.get("coverage") in (None, expected["coverage"])
    if not (
        payload.get("cache_version") == expected["cache_version"]
        and payload.get("config") == expected["config"]
        and payload.get("source") == expected["source"]
        and payload_coverage_matches
    ):
        print(f"Ignoring stale NVFP4 cache {cache_path}: payload metadata does not match requested source/config/coverage")
        return False

    tensors = payload.get("tensors", {})
    metadata = payload.get("metadata", {})
    eligible_modules = list(_iter_eligible_linear_modules(model, filter_fn))
    print(f"Validating NVFP4 cache for {source_path}: expected {len(eligible_modules)} Linear modules", flush=True)
    missing = []
    incompatible = []
    for fqn, module in eligible_modules:
        entry = tensors.get(fqn)
        weight = entry.get("weight") if entry is not None else None
        if not _is_nvfp4_tensor(weight):
            missing.append(fqn)
            continue
        expected_shape = list(module.weight.shape) if module.weight is not None else []
        weight_meta = metadata.get(fqn, {}).get("weight", {})
        cached_shape = list(getattr(weight, "shape", []) or weight_meta.get("shape", []))
        if cached_shape != expected_shape or not _metadata_matches(weight, weight_meta, device):
            incompatible.append({"name": fqn, "expected": expected_shape, "cached": _tensor_meta(weight)})
        bias = entry.get("bias")
        bias_meta = metadata.get(fqn, {}).get("bias")
        if bias is not None and (module.bias is None or list(bias.shape) != list(module.bias.shape) or not _metadata_matches(bias, bias_meta, device)):
            expected_bias = None if module.bias is None else list(module.bias.shape)
            incompatible.append({"name": fqn + ".bias", "expected": expected_bias, "cached": _tensor_meta(bias)})

    if missing:
        print(f"Ignoring incomplete NVFP4 cache {cache_path}: expected {len(eligible_modules)}, missing {len(missing)}")
        return False
    if incompatible:
        print(f"Ignoring incompatible NVFP4 cache {cache_path}: {incompatible[:5]}")
        return False

    print(f"Assigning NVFP4 cache for {source_path}: {len(eligible_modules)} Linear modules", flush=True)
    with torch.no_grad():
        for fqn, module in eligible_modules:
            entry = tensors[fqn]
            module._parameters["weight"] = entry["weight"]
            bias = entry.get("bias")
            if bias is not None:
                module._parameters["bias"] = torch.nn.Parameter(bias.to(device=device), requires_grad=False)
            elif module.bias is not None:
                module._parameters["bias"] = None

    print(f"Loaded NVFP4 cache for {source_path} from {cache_path}", flush=True)
    return True


def save_from_model(model, source_path: Optional[str], filter_fn: Callable, eligible: int, skipped_linear: int, skipped_reasons: dict, coverage=None) -> Optional[str]:
    cache_path = _cache_path_for(source_path)
    if cache_path is None or source_path is None or eligible == 0:
        return None

    tensors = {}
    metadata = {}
    for fqn, module in _iter_eligible_linear_modules(model, filter_fn):
        if not _is_nvfp4_tensor(module.weight):
            return None
        tensors[fqn] = {
            "weight": module.weight.detach(),
            "bias": module.bias.detach() if module.bias is not None else None,
        }
        metadata[fqn] = {
            "weight": _tensor_meta(module.weight),
            "bias": _tensor_meta(module.bias) if module.bias is not None else None,
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
        "metadata": metadata,
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
    print(f"Created NVFP4 cache for {source_path} -> {cache_path}")
    return cache_path
