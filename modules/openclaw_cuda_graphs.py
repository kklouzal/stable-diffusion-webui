from __future__ import annotations

import threading
import traceback
from typing import Any

import torch

_ENABLED = False
_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_LOCK = threading.RLock()
_STATS = {"captures": 0, "replays": 0, "fallbacks": 0, "bypasses": 0, "failures": 0, "last_error": None, "last_key": None}
_FAILED_KEYS: set[tuple[Any, ...]] = set()


def _reset_stats() -> None:
    _STATS.update({"captures": 0, "replays": 0, "fallbacks": 0, "bypasses": 0, "failures": 0, "last_error": None, "last_key": None})


def status() -> dict[str, Any]:
    with _LOCK:
        return {"enabled": _ENABLED, "cache_size": len(_CACHE), **_STATS}


def set_enabled(enabled: bool, clear: bool = False) -> dict[str, Any]:
    global _ENABLED
    with _LOCK:
        _ENABLED = bool(enabled)
        if clear or not _ENABLED:
            _CACHE.clear()
            _FAILED_KEYS.clear()
            _reset_stats()
        return status()


def clear() -> dict[str, Any]:
    with _LOCK:
        _CACHE.clear()
        _FAILED_KEYS.clear()
        _reset_stats()
        return status()


def _tensor_signature(t: torch.Tensor) -> tuple[Any, ...]:
    return ("tensor", tuple(t.shape), str(t.dtype), str(t.device), bool(t.requires_grad), tuple(t.stride()))


def _structure_signature(value: Any) -> Any:
    if torch.is_tensor(value):
        return _tensor_signature(value)
    if isinstance(value, dict):
        return ("dict", tuple((key, _structure_signature(value[key])) for key in sorted(value)))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_structure_signature(item) for item in value))
    return (type(value).__name__, repr(value))


def _clone_static(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_static(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_static(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_static(item) for item in value)
    return value


def _copy_into_static(static: Any, current: Any) -> None:
    if torch.is_tensor(static) and torch.is_tensor(current):
        static.copy_(current, non_blocking=True)
    elif isinstance(static, dict) and isinstance(current, dict):
        for key in static:
            _copy_into_static(static[key], current[key])
    elif isinstance(static, list) and isinstance(current, list):
        for dst, src in zip(static, current):
            _copy_into_static(dst, src)
    elif isinstance(static, tuple) and isinstance(current, tuple):
        for dst, src in zip(static, current):
            _copy_into_static(dst, src)


def _model_signature(fn: Any) -> tuple[Any, ...]:
    try:
        from modules import shared

        checkpoint_info = getattr(getattr(shared, "sd_model", None), "sd_checkpoint_info", None)
        checkpoint_key = (
            getattr(checkpoint_info, "filename", None),
            getattr(checkpoint_info, "hash", None),
            getattr(checkpoint_info, "sha256", None),
        )
    except Exception:
        checkpoint_key = None

    try:
        import networks

        lora_key = tuple(
            (
                getattr(net, "name", None),
                getattr(net, "mentioned_name", None),
                getattr(net, "te_multiplier", None),
                getattr(net, "unet_multiplier", None),
                getattr(net, "dyn_dim", None),
                networks.network_lora_source_signature(getattr(net, "network_on_disk", None), net)
                if hasattr(networks, "network_lora_source_signature")
                else None,
            )
            for net in getattr(networks, "loaded_networks", [])
        )
    except Exception:
        lora_key = None

    return (type(fn).__module__, type(fn).__qualname__, checkpoint_key, lora_key)


def _cache_key(fn: Any, x: torch.Tensor, sigma: torch.Tensor, cond: Any) -> tuple[Any, ...]:
    try:
        from modules import sd_hijack_optimizations
        attention = sd_hijack_optimizations.attention_backend_status()
        attention_key = (attention.get("active"), attention.get("sdpa_backend"))
    except Exception:
        attention_key = None
    return (_model_signature(fn), _tensor_signature(x), _tensor_signature(sigma), _structure_signature(cond), attention_key)


def _graph_safe_denoiser_context(denoiser: Any | None) -> bool:
    if denoiser is None:
        return True

    # Inpaint/masked blending depends on mutable latent-mask state outside the
    # wrapped UNet call. Keep that path eager unless it is audited separately.
    if getattr(denoiser, "mask", None) is not None or getattr(denoiser, "nmask", None) is not None:
        return False

    p = getattr(denoiser, "p", None)
    if p is not None:
        if getattr(p, "mask", None) is not None or getattr(p, "nmask", None) is not None:
            return False

        incant_cfg = getattr(p, "incant_cfg_params", None)
        if isinstance(incant_cfg, dict):
            seg_params = incant_cfg.get("seg_params")
            # SEG uses Python forward hooks in the UNet attention path. CUDA graph
            # replay replays captured kernels and does not re-enter those hooks.
            if bool(getattr(seg_params, "seg_active", False)):
                return False

    return True


def run(fn: Any, x: torch.Tensor, sigma: torch.Tensor, cond: Any, *, denoiser: Any | None = None):
    if not _ENABLED:
        return fn(x, sigma, cond=cond)
    if not _graph_safe_denoiser_context(denoiser):
        with _LOCK:
            _STATS["bypasses"] += 1
        return fn(x, sigma, cond=cond)
    if not torch.cuda.is_available() or not torch.is_tensor(x) or x.device.type != "cuda" or torch.is_grad_enabled():
        with _LOCK:
            _STATS["fallbacks"] += 1
        return fn(x, sigma, cond=cond)

    key = _cache_key(fn, x, sigma, cond)
    with _LOCK:
        entry = _CACHE.get(key)
        failed_before = key in _FAILED_KEYS
    if failed_before:
        with _LOCK:
            _STATS["fallbacks"] += 1
            _STATS["last_key"] = repr(key)
        return fn(x, sigma, cond=cond)
    if entry is not None:
        _copy_into_static(entry["x"], x)
        _copy_into_static(entry["sigma"], sigma)
        _copy_into_static(entry["cond"], cond)
        entry["graph"].replay()
        with _LOCK:
            _STATS["replays"] += 1
            _STATS["last_key"] = repr(key)
        return entry["out"].clone()

    try:
        static_x = _clone_static(x)
        static_sigma = _clone_static(sigma)
        static_cond = _clone_static(cond)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            static_out = fn(static_x, static_sigma, cond=static_cond)
        torch.cuda.current_stream().wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = fn(static_x, static_sigma, cond=static_cond)
        entry = {"graph": graph, "x": static_x, "sigma": static_sigma, "cond": static_cond, "out": static_out}
        with _LOCK:
            _CACHE[key] = entry
            _STATS["captures"] += 1
            _STATS["last_error"] = None
            _STATS["last_key"] = repr(key)
        return static_out.clone()
    except Exception as exc:
        with _LOCK:
            _STATS["failures"] += 1
            _FAILED_KEYS.add(key)
            _STATS["last_error"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-8000:]
            _STATS["last_key"] = repr(key)
        return fn(x, sigma, cond=cond)
