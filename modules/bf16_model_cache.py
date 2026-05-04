from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

import safetensors.torch
from safetensors import safe_open
import torch

CACHE_DIR_NAME = "bf16"
SUPPORTED_ROOT_NAMES = ("Stable-diffusion", "Lora", "VAE")
CACHE_VERSION = 1


def _is_safetensors(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() == ".safetensors"


def _stat_source(filename: str) -> dict:
    stat = os.stat(filename)
    return {
        "path": os.path.abspath(filename),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _sha256(filename: str) -> str:
    h = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_safetensors_metadata(filename: str) -> dict[str, str]:
    try:
        with safe_open(filename, framework="pt", device="cpu") as f:
            return dict(f.metadata() or {})
    except Exception:
        return {}


def is_bf16_cache_path(filename: str) -> bool:
    return CACHE_DIR_NAME in Path(filename).parts


def _cache_path_for(filename: str) -> Optional[str]:
    if not _is_safetensors(filename):
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

    return str(root / CACHE_DIR_NAME / relative)


def _sidecar_path(cache_path: str) -> str:
    return cache_path + ".bf16-cache.json"


def _load_sidecar(cache_path: str) -> Optional[dict]:
    try:
        with open(_sidecar_path(cache_path), "r", encoding="utf8") as f:
            return json.load(f)
    except Exception:
        return None


def _sidecar_matches(filename: str, cache_path: str) -> bool:
    if not os.path.exists(cache_path):
        return False

    sidecar = _load_sidecar(cache_path)
    if not sidecar:
        return False

    return (
        sidecar.get("cache_version") == CACHE_VERSION
        and sidecar.get("dtype") == "bfloat16"
        and sidecar.get("source") == _stat_source(filename)
    )


def _tensor_to_bf16(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_floating_point() and tensor.dtype != torch.bfloat16:
        return tensor.to(dtype=torch.bfloat16)
    return tensor


def _write_atomic_bytes(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with NamedTemporaryFile("wb", delete=False, dir=os.path.dirname(path), prefix=".tmp-", suffix=".json") as f:
        tmp = f.name
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def ensure_bf16_cache(filename: str) -> Optional[str]:
    """Return a bf16 safetensors cache for supported model/LoRA files, creating it when stale/missing."""
    cache_path = _cache_path_for(filename)
    if cache_path is None:
        return None

    if _sidecar_matches(filename, cache_path):
        return cache_path

    source_stat = _stat_source(filename)
    source_sha256 = _sha256(filename)
    print(f"Creating bf16 cache for {filename} -> {cache_path}")
    tensors = safetensors.torch.load_file(filename, device="cpu")
    tensors = {key: _tensor_to_bf16(value) for key, value in tensors.items()}

    metadata = _read_safetensors_metadata(filename)
    metadata.update({
        "bf16_cache_version": str(CACHE_VERSION),
        "bf16_cache_source_path": source_stat["path"],
        "bf16_cache_source_size": str(source_stat["size"]),
        "bf16_cache_source_mtime_ns": str(source_stat["mtime_ns"]),
        "bf16_cache_source_sha256": source_sha256,
    })

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = cache_path + ".tmp"
    safetensors.torch.save_file(tensors, tmp_path, metadata=metadata)
    os.replace(tmp_path, cache_path)

    sidecar = {
        "cache_version": CACHE_VERSION,
        "dtype": "bfloat16",
        "source": source_stat,
        "source_sha256": source_sha256,
        "cache": _stat_source(cache_path),
    }
    _write_atomic_bytes(_sidecar_path(cache_path), json.dumps(sidecar, indent=2, sort_keys=True).encode("utf8"))
    return cache_path


def load_file(filename: str, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    cache_path = ensure_bf16_cache(filename)
    if cache_path is not None:
        print(f"Loading bf16 cache for {filename} from {cache_path}")
        return safetensors.torch.load_file(cache_path, device=device)

    return safetensors.torch.load_file(filename, device=device)
