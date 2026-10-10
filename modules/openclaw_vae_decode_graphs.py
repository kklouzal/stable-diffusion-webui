from __future__ import annotations

import sys
import threading
import traceback
from collections import OrderedDict
from typing import Any

import torch

from modules import openclaw_cache_epochs, openclaw_env
from modules.openclaw_cuda_graphs import instance_overrides, on_default_stream, publish_graph_cache_size
from modules.sd_unet_row_memo import tensor_version

# Read once at import (T2): an invalid value fails startup instead of being logged and ignored.
_ENABLED = openclaw_env.env_bool("OPENCLAW_VAE_DECODE_GRAPHS", False)
_GRAPH_CONTRACT_VERSION = 3
# lora_applied_epoch is added only while LoRA can reach the VAE; see _lora_reaches_vae(). Hooks, precision and the
# attention backend are keyed directly (_module_revision, _runtime_identity, _attention_identity).
_EPOCH_DIMENSIONS = (
    "checkpoint_object_epoch",
    "model_movement_epoch",
    "vae_object_epoch",
    "device_epoch",
)


_CACHE_MAX = openclaw_env.env_int("OPENCLAW_VAE_DECODE_GRAPH_CACHE_MAX", 4, minimum=0)
_LOCK = threading.RLock()
# CUDA graph replay and lifecycle invalidation share one execution lock. VAE graph
# entries retain mutable input/output storage and CUDA graph pools; serializing the
# whole family prevents teardown/replacement racing an in-flight replay or capture.
_EXECUTION_LOCK = threading.RLock()
_CACHE: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
_FAILED_KEYS: OrderedDict[tuple[Any, ...], None] = OrderedDict()
_COUNTERS = {"captures": 0, "replays": 0, "bypasses": 0, "failures": 0, "invalidations": 0, "evictions": 0}
_BYPASS_REASONS: dict[str, int] = {}
_INVALIDATION_REASONS: dict[str, int] = {}
_LAST_ERROR: str | None = None
_LAST_KEY: tuple[Any, ...] | None = None
# One memory pool shared by every decode graph capture (allocated lazily, dropped with the cache): replays and
# captures are serialized by _EXECUTION_LOCK on the device's default stream, static inputs are cloned outside the
# capture, and each replay's output is cloned before the lock is released, so captures may reuse each other's
# freed memory. Private pools retained one full decode peak per entry.
_GRAPH_POOL: Any = None


def _observe_bypass(reason: str) -> None:
    _COUNTERS["bypasses"] += 1
    _BYPASS_REASONS[reason] = _BYPASS_REASONS.get(reason, 0) + 1
    telemetry_reason = "cache_disabled" if reason in {"disabled", "cache_disabled"} else "unsafe_input"
    openclaw_cache_epochs.observe("E11", "bypass", reason=telemetry_reason)


def _clear_cache_locked() -> bool:
    global _GRAPH_POOL
    had_state = bool(_CACHE or _FAILED_KEYS)
    _CACHE.clear()
    _FAILED_KEYS.clear()
    _GRAPH_POOL = None
    return had_state


def _graph_pool() -> Any:
    global _GRAPH_POOL
    if _GRAPH_POOL is None:
        _GRAPH_POOL = torch.cuda.graph_pool_handle()
    return _GRAPH_POOL


def status() -> dict[str, Any]:
    with _LOCK:
        return {
            "enabled": _ENABLED,
            "cache_size": len(_CACHE),
            "failed_key_count": len(_FAILED_KEYS),
            "max_cache_size": _CACHE_MAX,
            **_COUNTERS,
            "bypass_reasons": dict(_BYPASS_REASONS),
            "invalidation_reasons": dict(_INVALIDATION_REASONS),
            "last_error": _LAST_ERROR,
            "last_key": repr(_LAST_KEY) if _LAST_KEY is not None else None,
            "contract_version": _GRAPH_CONTRACT_VERSION,
        }


def set_enabled(enabled: bool | None = None, clear_cache: bool = False) -> dict[str, Any]:
    """Update enablement and/or atomically reset retained graph state.

    A reset preserves user enablement unless explicitly supplied, clears
    failed-key latches and buffers, and keeps cumulative telemetry observable.
    """
    global _ENABLED, _LAST_ERROR, _LAST_KEY
    with _EXECUTION_LOCK, _LOCK:
        if enabled is not None:
            _ENABLED = bool(enabled)
        if clear_cache:
            _clear_cache_locked()
            _LAST_ERROR = None
            _LAST_KEY = None
            _COUNTERS["invalidations"] += 1
            _INVALIDATION_REASONS["manual_reset"] = _INVALIDATION_REASONS.get("manual_reset", 0) + 1
            openclaw_cache_epochs.observe("E11", "invalidate", reason="manual")
            publish_graph_cache_size()
        return status()

def invalidate(reason: str) -> dict[str, Any]:
    with _EXECUTION_LOCK, _LOCK:
        if _clear_cache_locked():
            _COUNTERS["invalidations"] += 1
            _INVALIDATION_REASONS[reason] = _INVALIDATION_REASONS.get(reason, 0) + 1
            openclaw_cache_epochs.observe("E11", "invalidate", reason="dependency_changed")
            publish_graph_cache_size()
    return status()


def _callable_identity(value: Any) -> tuple[Any, ...] | None:
    if value is None:
        return None
    function = getattr(value, "__func__", value)
    return (id(function), getattr(function, "__module__", None), getattr(function, "__qualname__", None))


def _module_revision(module: Any) -> tuple[Any, ...]:
    """Identity of the VAE module's parameters, buffers, hooks and convolution padding modes.

    In-place weight loads bump the version counters. Tiling (sd_hijack.model_hijack.apply_circular) switches every
    Conv2d to circular padding without touching a tensor; a capture freezes the padding it ran.
    """
    named_tensors = list(module.named_parameters(recurse=True)) + list(module.named_buffers(recurse=True))
    tensors = tuple(
        (
            name,
            id(tensor),
            tensor_version(tensor),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            str(tensor.dtype),
            str(tensor.device),
            tensor.data_ptr() if tensor.device.type != "meta" else None,
        )
        for name, tensor in named_tensors
    )
    hooks = []
    for attribute in ("_forward_pre_hooks", "_forward_hooks", "_backward_hooks"):
        values = getattr(module, attribute, None)
        if values:
            hooks.append((attribute, tuple((key, _callable_identity(value)) for key, value in values.items())))
    padding_modes = tuple(
        (name, submodule.padding_mode)
        for name, submodule in module.named_modules()
        if getattr(submodule, "padding_mode", "zeros") != "zeros"
    )
    return (id(module), type(module).__module__, type(module).__qualname__, bool(getattr(module, "training", False)), tensors, tuple(hooks), padding_modes)


def _tensor_key(x: torch.Tensor) -> tuple[Any, ...]:
    # The capture reads a contiguous clone and replay copies into it, so the input's own layout never matters.
    return (tuple(x.shape), str(x.dtype), str(x.device))


def _option_identity() -> tuple[Any, ...]:
    """Options a capture freezes. Decode method, approximation and VAE hypertile are bypassed in _bypass_reason."""
    from modules import shared

    return (
        # sdp_attnblock_forward reads it per call; a capture freezes the branch taken.
        bool(getattr(shared.opts, "upcast_attn", False)),
        # The NHWC GroupNorm switch picks the VAE GroupNorm kernel (and fused swish) a capture freezes.
        _nhwc_group_norm_state(),
    )


def _nhwc_group_norm_state() -> tuple[str, ...] | None:
    nhwc_group_norm = sys.modules.get("modules.openclaw_nhwc_groupnorm")
    return nhwc_group_norm.state_key() if nhwc_group_norm is not None else None


def _attention_identity(vae: Any) -> tuple[Any, ...]:
    """The attention implementation a capture freezes into the graph.

    sd_hijack installs the VAE AttnBlock.forward on the class (cross-attention optimization), and the SDPA backend can
    be switched at runtime (/sdapi/v1/openclaw/sdpa-backend); neither is visible in the module's parameters or hooks.
    """
    attention_block = getattr(getattr(getattr(vae, "decoder", None), "mid", None), "attn_1", None)
    forward = _callable_identity(getattr(type(attention_block), "forward", None)) if attention_block is not None else None
    # Same lookup as the UNet graph key: no backend selection exists until sd_hijack_optimizations is imported.
    optimizations = sys.modules.get("modules.sd_hijack_optimizations")
    return (forward, optimizations.active_sdpa_backend() if optimizations is not None else None)


def execution_identity(vae: Any) -> tuple[Any, ...]:
    """What a VAE pass (decode or encode) computes with beyond its weights and hooks: the options it reads per call
    (_option_identity) and the installed attention implementation (_attention_identity; the encoder uses the same
    AttnBlock class). The img2img init latent cache (modules/processing.py) keys its encodes on it too."""
    return (_option_identity(), _attention_identity(vae))


def _runtime_identity(model: Any) -> tuple[Any, ...]:
    info = getattr(model, "sd_checkpoint_info", None)
    model_identity = (
        id(model),
        getattr(info, "filename", None),
        getattr(info, "shorthash", None),
        getattr(info, "sha256", None),
        getattr(model, "sd_model_hash", None),
    )
    vae_identity = (
        _module_revision(model.first_stage_model),
        _callable_identity(getattr(model, "decode_first_stage", None)),
    )
    # modules.devices is loaded long before any decode; the lookup only keeps this module importable without it.
    devices = sys.modules.get("modules.devices")
    device_identity = (
        str(getattr(devices, "dtype_vae", None)),
        str(getattr(devices, "dtype", None)),
        str(getattr(devices, "device", None)),
    )
    return (model_identity, vae_identity, device_identity)


def _lora_reaches_vae() -> bool:
    """Whether the loaded LoRA set can change a VAE decode.

    LoRA never edits VAE weights: networks.assign_network_names_to_compvis_modules maps only the text encoders and
    sd_model.model (the UNet), and network_apply_weights returns at once for modules without a network_layer_name.
    The legacy lora_functional path (networks.network_forward) is the exception: while any LoRA is loaded it routes
    every Linear/Conv/norm input, VAE ones included, through devices.cond_cast_unet.
    """
    from modules import shared

    return bool(getattr(shared.opts, "lora_functional", False))


def _mutation_epochs() -> tuple[tuple[str, int], ...]:
    dimensions = _EPOCH_DIMENSIONS + (("lora_applied_epoch",) if _lora_reaches_vae() else ())
    return openclaw_cache_epochs.epoch_subset(dimensions)


def _bypass_reason(model: Any, x: Any, approximation: int) -> str | None:
    # Decode only: encoding samples a posterior, so a capture-only invocation would consume an
    # extra RNG result on the cold path. Encode stays eager until its RNG state is a tested graph input.
    if not _ENABLED:
        return "disabled"
    if approximation != 0:
        return "vae_approximation"
    if not torch.is_tensor(x):
        return "not_tensor"
    if not x.is_cuda:
        return "not_cuda"
    if x.ndim != 4 or x.shape[1] != 4:
        return "unsupported_shape"
    try:
        from modules import lowvram, shared

        opts = getattr(shared, "opts", None)
        if getattr(opts, "sd_vae_decode_method", "Full") != "Full":
            return "vae_method"
        if getattr(opts, "hypertile_enable_vae", False):
            return "hypertile_vae"
        if lowvram.is_enabled(model):
            return "lowvram"
        # samples_to_images_tensor then decodes with the live-preview method instead of the full VAE.
        if getattr(opts, "live_preview_fast_interrupt", False) and getattr(getattr(shared, "state", None), "interrupted", False):
            return "fast_interrupt"
    except Exception:
        return "state_probe_failed"
    vae = getattr(model, "first_stage_model", None)
    if vae is None or not hasattr(model, "decode_first_stage"):
        return "missing_vae"
    if bool(getattr(vae, "training", False)):
        return "vae_training"
    # Tiled VAE (VAEHook) and similar extensions replace decoder.forward per request. The key does not see that
    # Python override, and capture would either freeze it or fail (tiling syncs to the host) and latch the key.
    if instance_overrides(vae, "decode") or instance_overrides(getattr(vae, "decoder", None), "forward"):
        return "vae_forward_override"
    if not on_default_stream(x.device):
        # The shared pool and static buffers rely on every replay being ordered on one stream.
        return "non_default_stream"
    return None


def _key(model: Any, x: torch.Tensor) -> tuple[Any, ...]:
    return (
        "vae_cuda_graph",
        _runtime_identity(model),
        _tensor_key(x),
        execution_identity(model.first_stage_model),
        _mutation_epochs(),
    )


def _execute(model: Any, x: torch.Tensor) -> torch.Tensor:
    from modules import devices, sd_samplers_common

    with torch.no_grad(), devices.without_autocast():
        return model.decode_first_stage(sd_samplers_common.vae_decode_input(model, x))


def _remember_failed_key_locked(key: tuple[Any, ...]) -> None:
    _FAILED_KEYS[key] = None
    _FAILED_KEYS.move_to_end(key)
    failure_capacity = max(1, _CACHE_MAX)
    while len(_FAILED_KEYS) > failure_capacity:
        _FAILED_KEYS.popitem(last=False)


def _evict_locked() -> None:
    while len(_CACHE) > _CACHE_MAX:
        evicted_key, _ = _CACHE.popitem(last=False)
        _COUNTERS["evictions"] += 1
        openclaw_cache_epochs.observe("E11", "eviction", reason="capacity", semantic_key=evicted_key)


def run(model: Any, x: Any, approximation: int = 0) -> torch.Tensor | None:
    global _LAST_ERROR, _LAST_KEY
    with _EXECUTION_LOCK:
        reason = _bypass_reason(model, x, approximation)
        if reason is not None:
            with _LOCK:
                _observe_bypass(reason)
            return None
        if _CACHE_MAX <= 0:
            with _LOCK:
                _observe_bypass("cache_disabled")
            return None

        try:
            key = _key(model, x)
        except Exception as exc:
            # Fail closed: an identity that cannot be read must never fall back to one shared by other models or
            # VAEs (a replay would run against another graph's captured parameter addresses). Decode eagerly.
            with _LOCK:
                _LAST_ERROR = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                _observe_bypass("key_probe_failed")
            return None
        with _LOCK:
            _LAST_KEY = key
            entry = _CACHE.get(key)
            failed_before = key in _FAILED_KEYS
        if entry is not None:
            openclaw_cache_epochs.observe("E11", "hit", reason="cache_hit", semantic_key=key)
            entry["input"].copy_(x, non_blocking=True)
            entry["graph"].replay()
            output = entry["output"].clone()
            with _LOCK:
                if key in _CACHE:
                    _CACHE.move_to_end(key)
                _COUNTERS["replays"] += 1
            return output
        if failed_before:
            with _LOCK:
                _observe_bypass("failed_key")
            return None

        openclaw_cache_epochs.observe("E11", "miss", reason="cache_miss", semantic_key=key)
        try:
            static_input = x.detach().contiguous().clone()
            stream = torch.cuda.Stream(device=x.device)
            stream.wait_stream(torch.cuda.current_stream(x.device))
            with torch.cuda.stream(stream):
                _execute(model, static_input)
            torch.cuda.current_stream(x.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=_graph_pool()):
                static_output = _execute(model, static_input)
            # Defensive device barrier before the first-use entry is published and replayed (the UNet graph cache
            # has none); kept until a GPU eager/capture/replay check shows it is unnecessary.
            torch.cuda.synchronize()
            entry = {"graph": graph, "input": static_input, "output": static_output}
            with _LOCK:
                # Publish only the fully captured entry. Invalidation cannot
                # interleave because it shares _EXECUTION_LOCK.
                _CACHE[key] = entry
                _CACHE.move_to_end(key)
                _FAILED_KEYS.pop(key, None)
                _evict_locked()
                _COUNTERS["captures"] += 1
                _LAST_ERROR = None
                openclaw_cache_epochs.observe("E11", "publish", reason="published", semantic_key=key)
                publish_graph_cache_size()
            # The capture execution can include one-time backend/autotune
            # transitions. Replay once with the same static input and return that
            # output so misses and hits have identical graph-replay semantics.
            graph.replay()
            # Replay and clone are enqueued on the caller's current stream; the
            # returned tensor carries the normal CUDA stream dependency without
            # a device-wide barrier that can perturb later request scheduling.
            return static_output.clone()
        except Exception as exc:
            # Locals are deliberately not published; dropping all references
            # releases partial graph/static allocations after this frame exits.
            graph = static_input = static_output = None
            with _LOCK:
                _remember_failed_key_locked(key)
                _COUNTERS["failures"] += 1
                _LAST_ERROR = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                openclaw_cache_epochs.observe("E11", "reject", reason="capture_failed", semantic_key=key)
                _observe_bypass("capture_failed")
                publish_graph_cache_size()
            return None


publish_graph_cache_size()
