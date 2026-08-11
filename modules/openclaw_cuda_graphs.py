from __future__ import annotations

import contextlib
import os
import threading
import traceback
from typing import Any

import torch

from modules import openclaw_cache_epochs

_ENABLED = False
_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_KEY_LOCKS: dict[tuple[Any, ...], threading.RLock] = {}
_LOCK = threading.RLock()
_RUNTIME_LOCK = threading.RLock()
_STATS = {
    "captures": 0,
    "replays": 0,
    "fallbacks": 0,
    "bypasses": 0,
    "bypass_reasons": {},
    "last_bypass_reason": None,
    "invalidations": 0,
    "invalidation_reasons": {},
    "last_invalidation_reason": None,
    "last_invalidation_details": None,
    "failures": 0,
    "last_error": None,
    "last_key": None,
}
_FAILED_KEYS: set[tuple[Any, ...]] = set()
_SEEN_KEYS: set[tuple[Any, ...]] = set()
_LIFECYCLE_STATE: dict[str, Any] = {}
_MISSING = object()


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
        "invalidations": 0,
        "invalidation_reasons": {},
        "last_invalidation_reason": None,
        "last_invalidation_details": None,
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
    # SEG graphing remains opt-in. SEG mutates Python attention hook state and
    # CUDA graph replay skips Python, so live should keep this false until a
    # full static-equivalence validation exists for the active SEG pipeline.
    return _env_flag("OPENCLAW_CUDA_GRAPH_ALLOW_SEG", False)


def _min_key_hits_before_capture() -> int:
    try:
        return max(1, int(os.environ.get("OPENCLAW_CUDA_GRAPH_MIN_KEY_HITS", "2") or 2))
    except ValueError:
        return 2


def status() -> dict[str, Any]:
    with _LOCK:
        return {
            "enabled": _ENABLED,
            "cache_size": len(_CACHE),
            "max_cache_size": _MAX_CACHE_SIZE,
            "allow_seg": _allow_seg_graphs(),
            "min_key_hits_before_capture": _min_key_hits_before_capture(),
            **_STATS,
            "lifecycle_state_keys": sorted(_LIFECYCLE_STATE),
        }


def set_enabled(enabled: bool, clear: bool = False) -> dict[str, Any]:
    global _ENABLED
    with _LOCK:
        _ENABLED = bool(enabled)
        if clear or not _ENABLED:
            cleared = len(_CACHE)
            _CACHE.clear()
            _KEY_LOCKS.clear()
            _FAILED_KEYS.clear()
            _SEEN_KEYS.clear()
            _LIFECYCLE_STATE.clear()
            _reset_stats()
            if cleared:
                openclaw_cache_epochs.observe("E11", "invalidate", reason="cache_disabled" if not _ENABLED else "cache_cleared", count=cleared)
            openclaw_cache_epochs.set_size("E11", current_size=0, capacity=_MAX_CACHE_SIZE)
        return status()


def _clear_cache_locked() -> bool:
    had_state = bool(_CACHE or _KEY_LOCKS or _FAILED_KEYS or _SEEN_KEYS)
    _CACHE.clear()
    _KEY_LOCKS.clear()
    _FAILED_KEYS.clear()
    _SEEN_KEYS.clear()
    return had_state


def invalidate(reason: str, details: Any | None = None) -> dict[str, Any]:
    """Clear captured CUDA graphs after a mutable runtime boundary changes."""
    reason = str(reason or "unknown")
    # Invalidation can run from model CPU/device/trash movement while API workers
    # are concurrently copying static graph inputs or capturing a new graph. Hold
    # the runtime lock so invalidation cannot clear per-key ownership underneath
    # an in-flight replay/capture, and so a capture cannot publish a stale entry
    # after the boundary has changed.
    with _RUNTIME_LOCK:
        with _LOCK:
            had_state = _clear_cache_locked()
            if had_state:
                openclaw_cache_epochs.observe("E11", "invalidate", reason="dependency_changed")
                openclaw_cache_epochs.set_size("E11", current_size=0, capacity=_MAX_CACHE_SIZE)
                _STATS["invalidations"] += 1
                _STATS["invalidation_reasons"][reason] = _STATS["invalidation_reasons"].get(reason, 0) + 1
                _STATS["last_invalidation_reason"] = reason
                _STATS["last_invalidation_details"] = repr(details)[:1000] if details is not None else None
            return status()


@contextlib.contextmanager
def mutable_runtime_boundary(reason: str, details: Any | None = None):
    """Serialize graph invalidation with a model/UNet mutation boundary.

    CUDA graph replay skips Python and owns static tensors captured against the
    current CUDA module storage. Model movement or quantized parameter rewrites
    must not overlap replay/capture, and replay/capture must not start again
    until the movement is complete.
    """
    with _RUNTIME_LOCK:
        invalidate(reason, details)
        yield


def invalidate_if_changed(boundary: str, state: Any, reason: str | None = None) -> dict[str, Any]:
    """Invalidate once when a named lifecycle boundary changes state.

    First observation is registered without thrashing an empty cache; if cache
    state already exists, an unobserved boundary is treated as unsafe and cleared.
    Repeated identical observations are no-ops.
    """
    reason = reason or boundary
    with _RUNTIME_LOCK:
        with _LOCK:
            previous = _LIFECYCLE_STATE.get(boundary, _MISSING)
            if previous == state:
                return status()
            _LIFECYCLE_STATE[boundary] = state
            had_state = _clear_cache_locked()
            if had_state:
                openclaw_cache_epochs.observe("E11", "invalidate", reason="dependency_changed")
                openclaw_cache_epochs.set_size("E11", current_size=0, capacity=_MAX_CACHE_SIZE)
                _STATS["invalidations"] += 1
                _STATS["invalidation_reasons"][reason] = _STATS["invalidation_reasons"].get(reason, 0) + 1
                _STATS["last_invalidation_reason"] = reason
                _STATS["last_invalidation_details"] = repr({"boundary": boundary, "previous": previous, "current": state})[:1000]
            return status()

def clear() -> dict[str, Any]:
    with _LOCK:
        had_state = _clear_cache_locked()
        _LIFECYCLE_STATE.clear()
        _reset_stats()
        if had_state:
            openclaw_cache_epochs.observe("E11", "invalidate", reason="cache_cleared")
        openclaw_cache_epochs.set_size("E11", current_size=0, capacity=_MAX_CACHE_SIZE)
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


def _runtime_boundary_state() -> tuple[Any, ...]:
    try:
        from modules import devices, shared, sd_hijack_optimizations
    except Exception:
        devices = shared = sd_hijack_optimizations = None

    model = getattr(shared, "sd_model", None) if shared is not None else None
    checkpoint_info = getattr(model, "sd_checkpoint_info", None)
    checkpoint_key = (
        id(model) if model is not None else None,
        getattr(checkpoint_info, "filename", None),
        getattr(checkpoint_info, "hash", None),
        getattr(checkpoint_info, "sha256", None),
        repr(getattr(model, "used_config", None)),
    )
    vae_key = (
        getattr(model, "loaded_vae_file", None),
        id(getattr(model, "first_stage_model", None)) if model is not None else None,
    )
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
    try:
        attention_key = sd_hijack_optimizations.sdpa_backend_status() if sd_hijack_optimizations is not None else None
    except Exception:
        attention_key = None
    precision_key = (
        getattr(devices, "dtype", None),
        getattr(devices, "dtype_unet", None),
        getattr(devices, "dtype_vae", None),
        getattr(devices, "unet_needs_upcast", None),
        getattr(devices, "fp8", None),
        getattr(devices, "mxfp8", None),
        getattr(devices, "nvfp4", None),
        os.environ.get("OPENCLAW_SDPA_BACKEND"),
        os.environ.get("OPENCLAW_CUDA_GRAPHS"),
        os.environ.get("OPENCLAW_CUDA_GRAPH_ALLOW_SEG"),
    )
    dependency_epochs = openclaw_cache_epochs.epoch_subset((
        "checkpoint_object_epoch", "model_movement_epoch", "vae_object_epoch",
        "lora_applied_epoch", "forward_hook_epoch", "precision_epoch",
        "device_epoch", "attention_epoch", "compile_epoch",
    ))
    return (checkpoint_key, vae_key, lora_key, repr(attention_key), tuple(map(str, precision_key)), dependency_epochs)


def refresh_runtime_state() -> dict[str, Any]:
    return invalidate_if_changed("runtime", _runtime_boundary_state(), "runtime_changed")


def note_model_loaded(model: Any | None = None, reason: str = "model_changed") -> dict[str, Any]:
    checkpoint_info = getattr(model, "sd_checkpoint_info", None)
    state = (
        id(model) if model is not None else None,
        getattr(checkpoint_info, "filename", None),
        getattr(checkpoint_info, "hash", None),
        getattr(checkpoint_info, "sha256", None),
        repr(getattr(model, "used_config", None)),
    )
    return invalidate_if_changed("model", state, reason)


def note_vae_loaded(model: Any | None = None, reason: str = "vae_changed") -> dict[str, Any]:
    state = (
        id(model) if model is not None else None,
        getattr(model, "loaded_vae_file", None),
        id(getattr(model, "first_stage_model", None)) if model is not None else None,
    )
    return invalidate_if_changed("vae", state, reason)


def note_lora_loaded(reason: str = "lora_changed") -> dict[str, Any]:
    try:
        import networks

        state = tuple(
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
        state = None
    return invalidate_if_changed("lora", state, reason)


def _evict_if_needed_locked() -> None:
    if _MAX_CACHE_SIZE <= 0:
        _CACHE.clear()
        _KEY_LOCKS.clear()
        return
    while len(_CACHE) >= _MAX_CACHE_SIZE:
        evicted_key = next(iter(_CACHE))
        _CACHE.pop(evicted_key)
        _KEY_LOCKS.pop(evicted_key, None)
        openclaw_cache_epochs.observe("E11", "eviction", reason="capacity", semantic_key=evicted_key)


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


def _seg_window(denoiser: Any, seg_params: Any) -> tuple[int | None, int | None, int | None]:
    total_steps = getattr(denoiser, "total_steps", None) or getattr(denoiser, "steps", None)
    if total_steps is None:
        try:
            from modules.shared import state

            total_steps = state.sampling_steps
        except Exception:
            total_steps = None

    try:
        total = int(total_steps) if total_steps else None
    except (TypeError, ValueError):
        total = None
    try:
        start_step = int(getattr(seg_params, "seg_start_step", 0) or 0)
    except (TypeError, ValueError):
        start_step = None
    try:
        end_step = int(getattr(seg_params, "seg_end_step", -1) or -1)
    except (TypeError, ValueError):
        end_step = None
    return total, start_step, end_step


def _seg_active_for_all_graph_steps(denoiser: Any, seg_params: Any) -> bool:
    # SEG toggles Python attention hooks per step. CUDA graph replay bypasses
    # Python, so replay is safe only when the SEG hook flag is enabled for every
    # denoiser call in the sampling window. Partial/intermittent SEG stays eager
    # to preserve image quality over speed.
    total_steps, start_step, end_step = _seg_window(denoiser, seg_params)
    if total_steps is None or start_step is None or end_step is None or total_steps <= 0:
        return False
    return start_step <= 0 and end_step >= total_steps - 1


def _seg_module_signature(seg_params: Any) -> tuple[Any, ...] | None:
    modules = getattr(seg_params, "crossattn_modules", None)
    if not modules:
        return None
    signature = []
    for module in modules:
        to_q = getattr(module, "to_q", None)
        if to_q is None or not hasattr(to_q, "seg_enable"):
            return None
        signature.append((
            getattr(module, "network_layer_name", None),
            type(module).__module__,
            type(module).__qualname__,
            int(getattr(module, "heads", 0) or 0),
            id(to_q),
        ))
    return tuple(signature)


def _seg_graph_state_key(denoiser: Any | None) -> Any:
    seg_params = _seg_params(denoiser)
    if denoiser is None or seg_params is None or not bool(getattr(seg_params, "seg_active", False)):
        return None

    p = getattr(denoiser, "p", None)
    try:
        import modules.shared as shared

        batch_cond_uncond = bool(getattr(shared.opts, "batch_cond_uncond", False))
    except Exception:
        batch_cond_uncond = None

    total_steps, start_step, end_step = _seg_window(denoiser, seg_params)
    module_signature = _seg_module_signature(seg_params)
    return (
        "seg",
        _allow_seg_graphs(),
        bool(getattr(seg_params, "seg_active", False)),
        float(getattr(seg_params, "seg_blur_sigma", 0.0) or 0.0),
        float(getattr(seg_params, "seg_blur_threshold", 0.0) or 0.0),
        start_step,
        end_step,
        total_steps,
        _seg_active_for_all_graph_steps(denoiser, seg_params),
        int(getattr(p, "height", 0) or 0),
        int(getattr(p, "width", 0) or 0),
        batch_cond_uncond,
        module_signature,
    )

def _img2img_graph_state_key(denoiser: Any | None) -> Any:
    if denoiser is None:
        return None

    p = getattr(denoiser, "p", None)
    return (
        "img2img",
        getattr(denoiser, "init_latent", None) is not None,
        bool(getattr(denoiser, "mask_before_denoising", False)),
        getattr(p, "init_latent", None) is not None if p is not None else False,
        getattr(p, "image_conditioning", None) is not None if p is not None else False,
        bool(getattr(p, "init_images", None)) if p is not None else False,
    )


def _denoiser_graph_key(denoiser: Any | None) -> Any:
    p = getattr(denoiser, "p", None) if denoiser is not None else None
    sd_model = getattr(p, "sd_model", None) if p is not None else None
    unet = getattr(getattr(sd_model, "model", None), "diffusion_model", None)
    return (
        _seg_graph_state_key(denoiser),
        _img2img_graph_state_key(denoiser),
        # TeaCache is implemented as a per-request Python UNet.forward patch. The
        # CUDA graph cache key must include this active hook state; otherwise a graph
        # captured by a disabled request can replay for a later TeaCache-enabled
        # request and bypass TeaCache entirely while infotext still records accepted
        # TeaCache args.
        bool(getattr(unet, "_teacache_patched", False)),
        getattr(unet, "_openclaw_teacache_original_forward", None) is not None,
    )


def _graph_denoiser_bypass_reason(denoiser: Any | None) -> str | None:
    if denoiser is None:
        return None

    # Masked/inpaint blending mutates the latent around the wrapped UNet call and
    # can invoke arbitrary mask-blend scripts. Keep every mask-bearing path eager
    # until mask tensors/script effects are modeled as explicit graph inputs.
    if getattr(denoiser, "mask", None) is not None or getattr(denoiser, "nmask", None) is not None:
        return "denoiser_mask"

    p = getattr(denoiser, "p", None)
    if p is not None:
        if getattr(p, "mask", None) is not None or getattr(p, "nmask", None) is not None:
            return "processing_mask"

        unet = getattr(getattr(getattr(p, "sd_model", None), "model", None), "diffusion_model", None)
        if getattr(unet, "_teacache_patched", False) or getattr(unet, "_openclaw_teacache_original_forward", None) is not None:
            # TeaCache is implemented as a per-request Python UNet.forward patch.
            # CUDA graph replay bypasses that Python hook after capture, making
            # TeaCache-enabled requests report accepted args while recording zero
            # TeaCache cache hits/full refreshes. Keep active TeaCache requests eager
            # so the TeaCache gate can choose cached vs full UNet branches itself.
            return "teacache_unet_forward_hook"
        if getattr(unet, "_original_forward", None) is not None:
            # Extensions such as ControlNet install a Python UNet forward hook that
            # mutates conditioning state around each call. Capturing beneath that
            # hook can fail or replay stale hook state; keep these paths eager.
            return "external_unet_forward_hook"

        # Unmasked img2img/hires state is graphable here. By the time
        # CFGDenoiser.run_inner_model calls us, init_latent/init_images have been
        # consumed by sampler setup and any image conditioning used by the UNet is
        # present in the cond argument copied into static graph inputs. The init
        # latent itself is only used for pre/post mask blending, and mask-bearing
        # variants remain bypassed above.

        seg_params = _seg_params(denoiser)
        if bool(getattr(seg_params, "seg_active", False)):
            # SEG mutates Python attention hooks and module fields during sampling.
            # Graph only when the effective hook state and affected module set are
            # static for the entire sampling window and match A1111's paired CFG
            # attention batch behavior. Otherwise keep SEG eager.
            if not _allow_seg_graphs():
                return "seg_disabled"
            if not _seg_active_for_all_graph_steps(denoiser, seg_params):
                return "seg_active"
            try:
                import modules.shared as shared

                if not bool(getattr(shared.opts, "batch_cond_uncond", False)):
                    return "seg_unpaired_cfg"
            except Exception:
                return "seg_unpaired_cfg"
            if _seg_module_signature(seg_params) is None:
                return "seg_hooks_unready"

    return None


def _record_bypass(reason: str) -> None:
    with _LOCK:
        _STATS["bypasses"] += 1
        reasons = dict(_STATS.get("bypass_reasons") or {})
        reasons[reason] = reasons.get(reason, 0) + 1
        _STATS["bypass_reasons"] = reasons
        _STATS["last_bypass_reason"] = reason
    openclaw_cache_epochs.observe("E11", "bypass", reason="cache_disabled" if reason == "cache_disabled" else "unsafe_input")


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
        key_lock = _KEY_LOCKS.setdefault(key, threading.RLock())
    if failed_before:
        with _LOCK:
            _STATS["fallbacks"] += 1
            _STATS["last_key"] = repr(key)
        return fn(x, sigma, cond=cond)
    if entry is None and _min_key_hits_before_capture() > 1:
        with _LOCK:
            seen_before = key in _SEEN_KEYS
            if not seen_before:
                _SEEN_KEYS.add(key)
                _STATS["last_key"] = repr(key)
        if not seen_before:
            _record_bypass("cache_warmup")
            return fn(x, sigma, cond=cond)

    # A CUDA graph entry owns mutable static input/output tensors and a graph
    # replay object. Copy/replay/capture must be serialized per key; otherwise
    # concurrent API workers with the same shape/model key can interleave static
    # input copies and replay stale or mixed conditioning. Keep the lock narrow
    # to preserve concurrency across distinct graph keys.
    with _RUNTIME_LOCK:
        with key_lock:
            with _LOCK:
                entry = _CACHE.get(key)
                failed_before = key in _FAILED_KEYS
            if failed_before:
                with _LOCK:
                    _STATS["fallbacks"] += 1
                    _STATS["last_key"] = repr(key)
                return fn(x, sigma, cond=cond)
            if entry is not None:
                openclaw_cache_epochs.observe("E11", "hit", reason="cache_hit", semantic_key=key)
                _copy_into_static(entry["x"], x)
                _copy_into_static(entry["sigma"], sigma)
                _copy_into_static(entry["cond"], cond)
                entry["graph"].replay()
                with _LOCK:
                    _STATS["replays"] += 1
                    _STATS["last_key"] = repr(key)
                return entry["out"].clone()

            openclaw_cache_epochs.observe("E11", "miss", reason="cache_miss", semantic_key=key)
            try:
                static_x = _clone_static(x)
                static_sigma = _clone_static(sigma)
                static_cond = _clone_static(cond)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    warmup_out = fn(static_x, static_sigma, cond=static_cond)
                torch.cuda.current_stream().wait_stream(stream)
                # The capture pass invokes the UNet a second time for the same denoise
                # step. Some active attention/guidance stacks are call-sensitive even
                # when graph replay is exact, so return the first eager result for the
                # current step and keep the captured output only as the graph-owned
                # static replay buffer for subsequent steps.
                capture_return = _clone_static(warmup_out)

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    static_out = fn(static_x, static_sigma, cond=static_cond)
                entry = {"graph": graph, "x": static_x, "sigma": static_sigma, "cond": static_cond, "out": static_out}
                with _LOCK:
                    _evict_if_needed_locked()
                    _CACHE[key] = entry
                    _STATS["captures"] += 1
                    openclaw_cache_epochs.observe("E11", "publish", reason="published", semantic_key=key)
                    openclaw_cache_epochs.set_size("E11", current_size=len(_CACHE), capacity=_MAX_CACHE_SIZE)
                    _STATS["last_error"] = None
                    _STATS["last_key"] = repr(key)
                return capture_return
            except Exception as exc:
                with _LOCK:
                    _STATS["failures"] += 1
                    _FAILED_KEYS.add(key)
                    openclaw_cache_epochs.observe("E11", "reject", reason="capture_failed", semantic_key=key)
                    _STATS["last_error"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-8000:]
                    _STATS["last_key"] = repr(key)
                return fn(x, sigma, cond=cond)
