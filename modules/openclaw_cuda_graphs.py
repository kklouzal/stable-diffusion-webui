from __future__ import annotations

import os
import threading
import traceback
from typing import Any

import torch

_ENABLED = False
_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_LOCK = threading.RLock()
_STATS = {
    "captures": 0,
    "replays": 0,
    "fallbacks": 0,
    "bypasses": 0,
    "bypass_reasons": {},
    "last_bypass_reason": None,
    "failures": 0,
    "last_error": None,
    "last_key": None,
}
_FAILED_KEYS: set[tuple[Any, ...]] = set()


def _read_max_cache_size() -> int:
    try:
        return max(0, int(os.environ.get("OPENCLAW_CUDA_GRAPH_CACHE_MAX", "8") or 0))
    except ValueError:
        return 8


_MAX_CACHE_SIZE = _read_max_cache_size()


def _reset_stats() -> None:
    _STATS.update({
        "captures": 0,
        "replays": 0,
        "fallbacks": 0,
        "bypasses": 0,
        "bypass_reasons": {},
        "last_bypass_reason": None,
        "failures": 0,
        "last_error": None,
        "last_key": None,
    })


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default

    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _allow_seg_graphs() -> bool:
    return _env_flag("OPENCLAW_CUDA_GRAPH_ALLOW_SEG", False)


def status() -> dict[str, Any]:
    with _LOCK:
        return {
            "enabled": _ENABLED,
            "cache_size": len(_CACHE),
            "max_cache_size": _MAX_CACHE_SIZE,
            "allow_seg": _allow_seg_graphs(),
            **_STATS,
        }


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


def _evict_if_needed_locked() -> None:
    if _MAX_CACHE_SIZE <= 0:
        _CACHE.clear()
        return
    while len(_CACHE) >= _MAX_CACHE_SIZE:
        _CACHE.pop(next(iter(_CACHE)))


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


def _cache_key(fn: Any, x: torch.Tensor, sigma: torch.Tensor, cond: Any, denoiser: Any | None = None) -> tuple[Any, ...]:
    try:
        from modules import sd_hijack_optimizations
        attention = sd_hijack_optimizations.sdpa_backend_status()
        attention_key = attention.get("sdpa_backend")
    except Exception:
        attention_key = None
    return (_model_signature(fn), _tensor_signature(x), _tensor_signature(sigma), _structure_signature(cond), attention_key, _denoiser_graph_key(denoiser))


def _seg_params(denoiser: Any | None) -> Any | None:
    p = getattr(denoiser, "p", None) if denoiser is not None else None
    incant_cfg = getattr(p, "incant_cfg_params", None)
    return incant_cfg.get("seg_params") if isinstance(incant_cfg, dict) else None


def _seg_active_for_all_graph_steps(denoiser: Any, seg_params: Any) -> bool:
    # SEG toggles Python attention hooks per step. Replay is allowed only when
    # SEG is active for the whole sampling window, so the captured hook path
    # does not need to change between denoiser calls.
    total_steps = getattr(denoiser, "total_steps", None) or getattr(denoiser, "steps", None)
    if total_steps is None:
        try:
            from modules.shared import state

            total_steps = state.sampling_steps
        except Exception:
            total_steps = None
    if not total_steps:
        return False

    start_step = int(getattr(seg_params, "seg_start_step", 0) or 0)
    end_step = int(getattr(seg_params, "seg_end_step", -1) or -1)
    return start_step <= 0 and end_step >= int(total_steps) - 1


def _denoiser_graph_key(denoiser: Any | None) -> Any:
    seg_params = _seg_params(denoiser)
    if seg_params is None or not bool(getattr(seg_params, "seg_active", False)):
        return None

    p = getattr(denoiser, "p", None)
    try:
        import modules.shared as shared

        batch_cond_uncond = bool(getattr(shared.opts, "batch_cond_uncond", False))
    except Exception:
        batch_cond_uncond = None

    return (
        "seg",
        _allow_seg_graphs(),
        bool(getattr(seg_params, "seg_active", False)),
        float(getattr(seg_params, "seg_blur_sigma", 0.0) or 0.0),
        float(getattr(seg_params, "seg_blur_threshold", 0.0) or 0.0),
        int(getattr(seg_params, "seg_start_step", 0) or 0),
        int(getattr(seg_params, "seg_end_step", 0) or 0),
        int(getattr(p, "height", 0) or 0),
        int(getattr(p, "width", 0) or 0),
        batch_cond_uncond,
    )


def _graph_denoiser_bypass_reason(denoiser: Any | None) -> str | None:
    if denoiser is None:
        return None

    # Inpaint/masked blending depends on mutable latent-mask state outside the
    # wrapped UNet call. Keep that path eager unless it is audited separately.
    if getattr(denoiser, "mask", None) is not None or getattr(denoiser, "nmask", None) is not None:
        return "denoiser_mask"

    p = getattr(denoiser, "p", None)
    if p is not None:
        if getattr(p, "mask", None) is not None or getattr(p, "nmask", None) is not None:
            return "processing_mask"

        seg_params = _seg_params(denoiser)
        if bool(getattr(seg_params, "seg_active", False)):
            # SEG mutates Python attention hooks and module fields during sampling.
            # Full-window SEG keeps the same hook path for every denoiser call, so
            # allow it only behind the explicit opt-in. Partial-window SEG still has
            # changing Python hook state and must stay eager.
            if not (_allow_seg_graphs() and _seg_active_for_all_graph_steps(denoiser, seg_params)):
                return "seg_active"

    return None


def _record_bypass(reason: str) -> None:
    with _LOCK:
        _STATS["bypasses"] += 1
        reasons = dict(_STATS.get("bypass_reasons") or {})
        reasons[reason] = reasons.get(reason, 0) + 1
        _STATS["bypass_reasons"] = reasons
        _STATS["last_bypass_reason"] = reason


def run(fn: Any, x: torch.Tensor, sigma: torch.Tensor, cond: Any, *, denoiser: Any | None = None):
    if not _ENABLED:
        return fn(x, sigma, cond=cond)
    bypass_reason = _graph_denoiser_bypass_reason(denoiser)
    if bypass_reason is not None:
        _record_bypass(bypass_reason)
        return fn(x, sigma, cond=cond)
    if _MAX_CACHE_SIZE <= 0:
        _record_bypass("cache_disabled")
        return fn(x, sigma, cond=cond)
    if not torch.cuda.is_available() or not torch.is_tensor(x) or x.device.type != "cuda" or torch.is_grad_enabled():
        with _LOCK:
            _STATS["fallbacks"] += 1
        return fn(x, sigma, cond=cond)

    key = _cache_key(fn, x, sigma, cond, denoiser)
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
            _evict_if_needed_locked()
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
