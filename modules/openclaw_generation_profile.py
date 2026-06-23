from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class GenerationProfileKey:
    kind: str
    sampler: str | None
    scheduler: str | None
    steps: int
    device: str
    dtype: str
    params: tuple[Any, ...] = ()


_LOCK = threading.RLock()
_TENSOR_CACHE: OrderedDict[GenerationProfileKey, torch.Tensor] = OrderedDict()
_STATS = {
    "hits": 0,
    "misses": 0,
    "stores": 0,
    "evictions": 0,
    "bypasses": 0,
    "last_key": None,
    "last_bypass_reason": None,
}


def enabled() -> bool:
    return True


def _max_size() -> int:
    try:
        return max(0, int(os.environ.get("OPENCLAW_GENERATION_PROFILE_CACHE_MAX", "16") or 0))
    except ValueError:
        return 16


def _jsonable_key(key: GenerationProfileKey | None) -> dict[str, Any] | None:
    if key is None:
        return None
    return {
        "kind": key.kind,
        "sampler": key.sampler,
        "scheduler": key.scheduler,
        "steps": key.steps,
        "device": key.device,
        "dtype": key.dtype,
        "params": repr(key.params),
    }


def status() -> dict[str, Any]:
    with _LOCK:
        return {
            "enabled": enabled(),
            "cache_size": len(_TENSOR_CACHE),
            "max_cache_size": _max_size(),
            **_STATS,
            "last_key": _jsonable_key(_STATS.get("last_key")),
        }


def clear() -> dict[str, Any]:
    with _LOCK:
        _TENSOR_CACHE.clear()
        _STATS.update({
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
            "bypasses": 0,
            "last_key": None,
            "last_bypass_reason": None,
        })
        return status()


def bypass(reason: str) -> None:
    with _LOCK:
        _STATS["bypasses"] += 1
        _STATS["last_bypass_reason"] = reason


def make_key(
    kind: str,
    sampler: str | None,
    scheduler: str | None,
    steps: int,
    device: torch.device | str,
    dtype: torch.dtype | str,
    params: tuple[Any, ...] = (),
) -> GenerationProfileKey:
    return GenerationProfileKey(
        kind=kind,
        sampler=sampler,
        scheduler=scheduler,
        steps=int(steps),
        device=str(device),
        dtype=str(dtype),
        params=params,
    )


def tensor_for_key(key: GenerationProfileKey, factory: Callable[[], torch.Tensor]) -> torch.Tensor:
    max_size = _max_size()
    if max_size <= 0:
        bypass("max_size_zero")
        return factory()

    with _LOCK:
        cached = _TENSOR_CACHE.get(key)
        if cached is not None:
            _TENSOR_CACHE.move_to_end(key)
            _STATS["hits"] += 1
            _STATS["last_key"] = key
            _STATS["last_bypass_reason"] = None
            return cached

        _STATS["misses"] += 1

    tensor = factory()
    if not torch.is_tensor(tensor):
        return tensor

    tensor = tensor.detach()

    with _LOCK:
        existing = _TENSOR_CACHE.get(key)
        if existing is not None:
            _TENSOR_CACHE.move_to_end(key)
            _STATS["hits"] += 1
            _STATS["last_key"] = key
            _STATS["last_bypass_reason"] = None
            return existing

        while len(_TENSOR_CACHE) >= max_size:
            _TENSOR_CACHE.popitem(last=False)
            _STATS["evictions"] += 1
        _TENSOR_CACHE[key] = tensor
        _STATS["stores"] += 1
        _STATS["last_key"] = key
        _STATS["last_bypass_reason"] = None

    return tensor


def cache_tensor(kind: str, sampler: str | None, scheduler: str | None, steps: int, tensor: torch.Tensor, params: tuple[Any, ...] = ()) -> torch.Tensor:
    key = make_key(kind, sampler, scheduler, steps, tensor.device, tensor.dtype, params)
    return tensor_for_key(key, lambda: tensor)


def cached_tensor(
    kind: str,
    sampler: str | None,
    scheduler: str | None,
    steps: int,
    device: torch.device | str,
    dtype: torch.dtype | str,
    factory: Callable[[], torch.Tensor],
    params: tuple[Any, ...] = (),
) -> torch.Tensor:
    key = make_key(kind, sampler, scheduler, steps, device, dtype, params)
    return tensor_for_key(key, factory)
