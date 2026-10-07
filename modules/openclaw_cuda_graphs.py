from __future__ import annotations

import contextlib
import itertools
import sys
import threading
import traceback
import weakref
from collections import OrderedDict
from typing import Any

import torch

from modules import openclaw_cache_epochs, openclaw_env

_ENABLED = False
# Insertion order is recency order: hits move_to_end, eviction pops the least recently used entry.
_CACHE: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
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
_LIFECYCLE_STATE: dict[str, Any] = {}
_MISSING = object()
# One memory pool shared by every denoiser graph capture (allocated lazily, dropped with the cache).
# Sharing is safe only because of invariants run() enforces, see _graph_pool().
_GRAPH_POOL: Any = None
# denoiser wrapper -> (schedule tensors, their versions, value signature); see _schedule_signature().
_SCHEDULE_SIGNATURES: weakref.WeakKeyDictionary[Any, tuple[Any, ...]] = weakref.WeakKeyDictionary()
# Full schedule value signature -> the compact token graph keys carry instead (see _schedule_signature). Tokens come
# from a counter that never repeats, so clearing this table cannot make a stale per-wrapper token equal a new one.
_SCHEDULE_TOKENS: dict[tuple[Any, ...], tuple[str, int]] = {}
_SCHEDULE_TOKEN_IDS = itertools.count()


_MAX_CACHE_SIZE = openclaw_env.env_int("OPENCLAW_CUDA_GRAPH_CACHE_MAX", 8, minimum=0)


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


def status() -> dict[str, Any]:
    with _LOCK:
        return {
            "enabled": _ENABLED,
            "cache_size": len(_CACHE),
            "max_cache_size": _MAX_CACHE_SIZE,
            **_STATS,
            "lifecycle_state_keys": sorted(_LIFECYCLE_STATE),
        }


def set_enabled(enabled: bool, clear: bool = False) -> dict[str, Any]:
    global _ENABLED
    with _LOCK:
        _ENABLED = bool(enabled)
        if clear or not _ENABLED:
            cleared = len(_CACHE)
            _clear_cache_locked()
            _LIFECYCLE_STATE.clear()
            _reset_stats()
            if cleared:
                openclaw_cache_epochs.observe("E11", "invalidate", reason="cache_disabled" if not _ENABLED else "cache_cleared", count=cleared)
            openclaw_cache_epochs.set_size("E11", current_size=0, capacity=_MAX_CACHE_SIZE)
        return status()


def _clear_cache_locked() -> bool:
    global _GRAPH_POOL
    had_state = bool(_CACHE or _KEY_LOCKS or _FAILED_KEYS)
    _CACHE.clear()
    _KEY_LOCKS.clear()
    _FAILED_KEYS.clear()
    _SCHEDULE_TOKENS.clear()
    # Every graph that captured into the shared pool is gone; the next capture starts a fresh pool.
    _GRAPH_POOL = None
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


def _checkpoint_signature(model: Any | None) -> tuple[Any, ...]:
    checkpoint_info = getattr(model, "sd_checkpoint_info", None)
    return (
        getattr(checkpoint_info, "filename", None),
        getattr(checkpoint_info, "hash", None),
        getattr(checkpoint_info, "sha256", None),
    )


def _lora_signature() -> tuple[Any, ...] | None:
    """Identity of the loaded LoRA set; None when the Lora extension is not loaded.

    Runs on every graph-eligible denoiser call, so it must stay O(loaded networks) attribute reads. `source_key`
    is the immutable identity (canonical path + SHA-256 of the bytes actually parsed + parser revision) that
    networks.load_network/load_networks stamp on every network; never re-hash LoRA files here.
    """
    networks = sys.modules.get("networks")  # the builtin Lora extension's module
    if networks is None:
        return None

    return tuple(
        (
            getattr(net, "name", None),
            getattr(net, "mentioned_name", None),
            getattr(net, "te_multiplier", None),
            getattr(net, "unet_multiplier", None),
            getattr(net, "dyn_dim", None),
            getattr(net, "source_key", None),
        )
        for net in networks.loaded_networks
    )


def note_model_loaded(model: Any | None = None, reason: str = "model_changed") -> dict[str, Any]:
    state = (
        id(model) if model is not None else None,
        *_checkpoint_signature(model),
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
    return invalidate_if_changed("lora", _lora_signature(), reason)


def _evict_if_needed_locked() -> None:
    if _MAX_CACHE_SIZE <= 0:
        _CACHE.clear()
        _KEY_LOCKS.clear()
        return
    while len(_CACHE) >= _MAX_CACHE_SIZE:
        evicted_key, _ = _CACHE.popitem(last=False)
        _KEY_LOCKS.pop(evicted_key, None)
        openclaw_cache_epochs.observe("E11", "eviction", reason="capacity", semantic_key=evicted_key)


def _graph_pool() -> Any:
    """Return the memory pool shared by all denoiser graph captures.

    A private pool per capture retains every graph's whole activation peak (several GB per SDXL key on unified
    memory). Sharing one pool lets a capture reuse memory that earlier captures freed, which is safe because:
    - every replay and capture runs under _RUNTIME_LOCK on the device's default stream (run() bypasses graphs
      on any other stream), so no two graphs ever execute concurrently;
    - static inputs are cloned outside the capture, so they never live in the pool, and each replay's static
      output is cloned on the same stream before the lock is released; a later replay of another graph may
      overwrite that pool memory, but nothing reads a static output except the clone right after its replay;
    - torch.cuda.graph synchronizes the device before each capture.
    """
    global _GRAPH_POOL
    if _GRAPH_POOL is None:
        _GRAPH_POOL = torch.cuda.graph_pool_handle()
    return _GRAPH_POOL


def _schedule_tensors(fn: Any) -> tuple[tuple[str, torch.Tensor], ...]:
    """Tensors outside the UNet that the captured denoiser wrapper reads during replay.

    k-diffusion wrappers (CompVisDenoiser/CompVisVDenoiser) read their own `sigmas`/`log_sigmas` buffers in
    sigma_to_t; the timesteps v-wrapper reads the model's `alphas_cumprod`. Both derive from the model's alpha
    schedule, which per-request options replace (alpha-bar downcast, zero-terminal SNR).
    """
    tensors = list(fn.named_buffers(recurse=False)) if isinstance(fn, torch.nn.Module) else []
    alphas = getattr(getattr(fn, "inner_model", None), "alphas_cumprod", None)
    if torch.is_tensor(alphas):
        tensors.append(("inner_model.alphas_cumprod", alphas))
    return tuple(tensors)


def _schedule_signature(fn: Any) -> tuple[str, int]:
    """Value identity of _schedule_tensors(fn), read to the host once per wrapper and tensor version.

    A replay reads the schedule tensors at the addresses captured from the capturing wrapper (the cache entry
    keeps them alive), so a graph may replay for another wrapper only when the values are equal. One wrapper
    exists per sampling run (update_inner_model builds a new one on a refiner switch), so the device-to-host
    read happens once per run; replacing a tensor or mutating it in place changes the identity/version check.

    The result is a token interned per distinct value signature (equal values, equal token), not the ~3000 floats
    themselves: the key is hashed, compared, repr'd and digested on every denoiser call, ~1.5 ms of host time per
    call with the floats inline (CPU-measured on the GB10 host).
    """
    tensors = _schedule_tensors(fn)
    versions = tuple(tensor._version for _name, tensor in tensors)
    try:
        cached = _SCHEDULE_SIGNATURES.get(fn)
    except TypeError:  # not weak-referenceable (plain callables in tests); nothing to memoize per run
        cached = None
    if (
        cached is not None
        and len(cached[0]) == len(tensors)
        and all(held is tensor for held, (_name, tensor) in zip(cached[0], tensors))
        and cached[1] == versions
    ):
        return cached[2]
    values = tuple(
        (name, tuple(tensor.shape), str(tensor.dtype), str(tensor.device), tuple(tensor.detach().cpu().reshape(-1).tolist()))
        for name, tensor in tensors
    )
    with _LOCK:
        signature = _SCHEDULE_TOKENS.get(values)
        if signature is None:
            signature = _SCHEDULE_TOKENS[values] = ("schedule", next(_SCHEDULE_TOKEN_IDS))
    try:
        # Holding the tensors keeps their ids from being reused by a different tensor while memoized.
        _SCHEDULE_SIGNATURES[fn] = (tuple(tensor for _name, tensor in tensors), versions, signature)
    except TypeError:
        pass
    return signature


def _wrapper_scalars(fn: Any) -> tuple[Any, ...]:
    """Plain attributes of the denoiser wrapper that steer its Python control flow (e.g. k-diffusion `quantize`)."""
    attributes = getattr(fn, "__dict__", {})
    return tuple(
        (name, value) for name, value in sorted(attributes.items())
        if not name.startswith("_") and isinstance(value, (bool, int, float, str))
    )


def _model_signature(fn: Any) -> tuple[Any, ...]:
    try:
        from modules import shared

        checkpoint_key = _checkpoint_signature(getattr(shared, "sd_model", None))
    except ImportError:
        checkpoint_key = None

    return (
        type(fn).__module__,
        type(fn).__qualname__,
        checkpoint_key,
        _lora_signature(),
        _wrapper_scalars(fn),
        _schedule_signature(fn),
    )


def _attention_key() -> str | None:
    # sd_hijack_optimizations installs the SDPA attention forwards; until it is imported no backend selection exists.
    sd_hijack_optimizations = sys.modules.get("modules.sd_hijack_optimizations")
    return sd_hijack_optimizations.active_sdpa_backend() if sd_hijack_optimizations is not None else None


def _runtime_branch_key() -> tuple[Any, ...]:
    """Process-wide Python state the captured UNet call branches on, which a replay never re-reads.

    - the CrossAttention forward installed by the cross attention optimization (changing the setting swaps the
      function through sd_hijack.redo_hijack without touching graphs);
    - upcast_attn: the attention forwards run float32 attention with autocast off;
    - lora_functional: the Lora extension's per-layer functional path instead of merged weights, and the UNet
      norms' bf16-native eligibility (sd_hijack_unet.bf16_native_norm_eligible).
    Request override_settings set these without callbacks, so they must be part of the key.
    """
    shared = sys.modules.get("modules.shared")
    opts = getattr(shared, "opts", None)
    cross_attention = getattr(sys.modules.get("sgm.modules.attention"), "CrossAttention", None)
    return (
        getattr(cross_attention, "forward", None),
        bool(getattr(opts, "upcast_attn", False)),
        bool(getattr(opts, "lora_functional", False)),
    )


def _cache_key(fn: Any, x: torch.Tensor, sigma: torch.Tensor, cond: Any, denoiser: Any | None = None) -> tuple[Any, ...]:
    return (_model_signature(fn), _tensor_signature(x), _tensor_signature(sigma), _structure_signature(cond), _attention_key(), _runtime_branch_key(), _denoiser_graph_key(denoiser))


def _seg_params(denoiser: Any | None) -> Any | None:
    p = getattr(denoiser, "p", None) if denoiser is not None else None
    incant_cfg = getattr(p, "incant_cfg_params", None)
    return incant_cfg.get("seg_params") if isinstance(incant_cfg, dict) else None


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
        _img2img_graph_state_key(denoiser),
        # TeaCache is implemented as a per-request Python UNet.forward patch. The
        # CUDA graph cache key must include this active hook state; otherwise a graph
        # captured by a disabled request can replay for a later TeaCache-enabled
        # request and bypass TeaCache entirely while infotext still records accepted
        # TeaCache args.
        bool(getattr(unet, "_teacache_patched", False)),
        getattr(unet, "_openclaw_teacache_original_forward", None) is not None,
    )


def instance_overrides(obj: Any, name: str) -> bool:
    """True when an instance attribute `name` shadows the class method with different code.

    Python-level monkeypatches (Tiled Diffusion, Tiled VAE, ...) run only when the call goes through Python, so
    graph capture would freeze them. Their restore paths assign the saved bound class method back onto the
    instance; that leftover attribute calls exactly the class method and is not an override.
    """
    value = getattr(obj, "__dict__", {}).get(name)
    if value is None:
        return False
    return not (getattr(value, "__self__", None) is obj and getattr(value, "__func__", None) is getattr(type(obj), name, None))


def _denoiser_models(fn: Any | None, p: Any | None) -> tuple[Any, ...]:
    """The model objects whose apply_model/UNet the captured call runs: the wrapper's model and p.sd_model."""
    models = []
    for model in (getattr(fn, "inner_model", None), getattr(p, "sd_model", None)):
        if model is not None and all(model is not seen for seen in models):
            models.append(model)
    return tuple(models)


def _graph_denoiser_bypass_reason(denoiser: Any | None, fn: Any | None = None) -> str | None:
    if denoiser is None and fn is None:
        return None

    # Masked/inpaint blending mutates the latent around the wrapped UNet call and
    # can invoke arbitrary mask-blend scripts. Keep every mask-bearing path eager
    # until mask tensors/script effects are modeled as explicit graph inputs.
    if getattr(denoiser, "mask", None) is not None or getattr(denoiser, "nmask", None) is not None:
        return "denoiser_mask"

    p = getattr(denoiser, "p", None) if denoiser is not None else None
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
        if (
            getattr(unet, "_original_forward", None) is not None
            or getattr(unet, "_controlnet_forward_hook_owner", None) is not None
        ):
            # Extensions such as ControlNet install a Python UNet forward hook that
            # mutates conditioning state around each call. The owner-scoped
            # ControlNet lifecycle no longer uses the legacy _original_forward
            # marker, so include its explicit live-owner marker as well. Capturing
            # beneath either form can fail or replay stale hook state.
            return "external_unet_forward_hook"

        # Unmasked img2img/hires state is graphable here. By the time
        # CFGDenoiser.run_inner_model calls us, init_latent/init_images have been
        # consumed by sampler setup and any image conditioning used by the UNet is
        # present in the cond argument copied into static graph inputs. The init
        # latent itself is only used for pre/post mask blending, and mask-bearing
        # variants remain bypassed above.

    # Hypernetworks run as Python inside every attention forward (modules/hypernetworks/hypernetwork.py), reading the
    # loaded set and each network's multiplier per call; a replay would apply the set that was loaded at capture.
    if getattr(sys.modules.get("modules.shared"), "loaded_hypernetworks", None):
        return "hypernetworks"

    models = _denoiser_models(fn, p)
    # A per-request instance override (Tiled Diffusion: MultiDiffusion/DemoFusion replace inner_model.forward or
    # the CFG denoiser's forward, MixtureOfDiffusers replaces sd_model.apply_model) is Python that replay skips:
    # a graph captured under it would replay the tile loop for later plain requests (and the reverse), freeze its
    # per-step host state, and read request-owned buffers freed after the request. Keep every such call eager.
    if instance_overrides(fn, "forward") or instance_overrides(denoiser, "forward"):
        return "python_forward_override"
    for model in models:
        wrapper = getattr(model, "model", None)
        if (
            instance_overrides(model, "apply_model")
            or instance_overrides(wrapper, "forward")
            or instance_overrides(getattr(wrapper, "diffusion_model", None), "forward")
        ):
            return "python_forward_override"
        # Hypertile draws a random tiling per attention call; a capture would freeze one draw (and its RNG advance)
        # and replay it for every later request with the same key, including requests with hypertile off.
        if getattr(wrapper, "__webui_hypertile_enabled", False):
            return "hypertile_unet"
        # tomesd swaps transformer block classes per request; the graph key does not model the merge ratio.
        if getattr(model, "applied_token_merged_ratio", 0):
            return "token_merging"

    if p is not None:
        seg_params = _seg_params(denoiser)
        if bool(getattr(seg_params, "seg_active", False)):
            # SEG owns mutable Python attention-hook state. Even when its effective
            # parameters span the full sampling window, replay bypasses the Python
            # hook lifecycle and has produced process-dependent repeated-SEG output.
            # Keep SEG eager until the hook state is represented as explicit graph
            # input/state; never trade fixed-seed quality for graph speed.
            return "seg_attention_hooks"

    return None


def _record_bypass(reason: str) -> None:
    with _LOCK:
        _STATS["bypasses"] += 1
        reasons = dict(_STATS.get("bypass_reasons") or {})
        reasons[reason] = reasons.get(reason, 0) + 1
        _STATS["bypass_reasons"] = reasons
        _STATS["last_bypass_reason"] = reason
    openclaw_cache_epochs.observe("E11", "bypass", reason="cache_disabled" if reason == "cache_disabled" else "unsafe_input")


def on_default_stream(device: torch.device) -> bool:
    return torch.cuda.current_stream(device) == torch.cuda.default_stream(device)


def run(fn: Any, x: torch.Tensor, sigma: torch.Tensor, cond: Any, *, denoiser: Any | None = None):
    if not _ENABLED:
        return fn(x, sigma, cond=cond)
    bypass_reason = _graph_denoiser_bypass_reason(denoiser, fn)
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
    if not on_default_stream(x.device):
        # Static buffers and the shared graph pool are only safe while every replay is ordered on one stream.
        _record_bypass("non_default_stream")
        return fn(x, sigma, cond=cond)

    key = _cache_key(fn, x, sigma, cond, denoiser)
    with _LOCK:
        failed_before = key in _FAILED_KEYS
        key_lock = _KEY_LOCKS.setdefault(key, threading.RLock())
    if failed_before:
        with _LOCK:
            _STATS["fallbacks"] += 1
            _STATS["last_key"] = repr(key)
        return fn(x, sigma, cond=cond)

    # A CUDA graph entry owns mutable static input/output tensors and a graph
    # replay object, and all entries share one memory pool (see _graph_pool).
    # _RUNTIME_LOCK serializes every copy/replay/clone and capture across all
    # keys; otherwise concurrent API workers could interleave static input copies
    # or overwrite another graph's pool memory before its output is cloned.
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
                    if key in _CACHE:
                        _CACHE.move_to_end(key)
                    _STATS["replays"] += 1
                    _STATS["last_key"] = repr(key)
                return entry["out"].clone()

            openclaw_cache_epochs.observe("E11", "miss", reason="cache_miss", semantic_key=key)
            try:
                static_x = _clone_static(x)
                static_sigma = _clone_static(sigma)
                static_cond = _clone_static(cond)
                # The one warm-up a capture needs: run on a side stream so lazy CUDA state (library handles and
                # workspaces, kernel loading, allocator growth) is initialized outside the capture. A first-sighting
                # key used to get an extra eager run on the caller's stream as well; its result was discarded (the
                # caller receives the replay below), so it only added a UNet forward per new key.
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    fn(static_x, static_sigma, cond=static_cond)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=_graph_pool()):
                    static_out = fn(static_x, static_sigma, cond=static_cond)
                entry = {
                    "graph": graph,
                    # The replay reads the wrapper's schedule tensors at their captured addresses; keep the wrapper
                    # and those tensors alive for the entry's lifetime (the request that created them may end).
                    "fn": fn,
                    "schedule": tuple(tensor for _name, tensor in _schedule_tensors(fn)),
                    "x": static_x,
                    "sigma": static_sigma,
                    "cond": static_cond,
                    "out": static_out,
                }
                with _LOCK:
                    _evict_if_needed_locked()
                    _CACHE[key] = entry
                    _STATS["captures"] += 1
                    openclaw_cache_epochs.observe("E11", "publish", reason="published", semantic_key=key)
                    openclaw_cache_epochs.set_size("E11", current_size=len(_CACHE), capacity=_MAX_CACHE_SIZE)
                    _STATS["last_error"] = None
                    _STATS["last_key"] = repr(key)
                # The capture execution can include one-time backend/autotune
                # transitions. Replay once with the same static inputs and return that
                # output so the first request has the same graph-replay semantics as
                # every cache hit.
                graph.replay()
                return _clone_static(static_out)
            except Exception as exc:
                with _LOCK:
                    _STATS["failures"] += 1
                    _FAILED_KEYS.add(key)
                    openclaw_cache_epochs.observe("E11", "reject", reason="capture_failed", semantic_key=key)
                    _STATS["last_error"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-8000:]
                    _STATS["last_key"] = repr(key)
                return fn(x, sigma, cond=cond)
