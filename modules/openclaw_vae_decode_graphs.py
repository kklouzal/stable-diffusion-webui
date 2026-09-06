from __future__ import annotations

import os
import threading
import traceback
import weakref
from collections import OrderedDict
from typing import Any

import torch

from modules import openclaw_cache_epochs

_ENABLED = False
_OPERATION_DECODE = "decode"
_OPERATION_ENCODE = "encode"
_GRAPH_CONTRACT_VERSION = 2
_EPOCH_DIMENSIONS = (
    "checkpoint_object_epoch",
    "model_movement_epoch",
    "vae_object_epoch",
    "lora_applied_epoch",
    "forward_hook_epoch",
    "precision_epoch",
    "device_epoch",
    "attention_epoch",
    "compile_epoch",
)


def _read_cache_max() -> int:
    try:
        return max(0, int(os.environ.get("OPENCLAW_VAE_DECODE_GRAPH_CACHE_MAX", "4") or 0))
    except ValueError:
        return 4


_CACHE_MAX = _read_cache_max()
_LOCK = threading.RLock()
# CUDA graph replay and lifecycle invalidation share one execution lock. VAE graph
# entries retain mutable input/output storage and CUDA graph pools; serializing the
# whole family prevents teardown/replacement racing an in-flight replay or capture.
_EXECUTION_LOCK = threading.RLock()
_CACHE: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
_KEY_LOCKS: weakref.WeakValueDictionary[tuple[Any, ...], threading.RLock] = weakref.WeakValueDictionary()
_FAILED_KEYS: OrderedDict[tuple[Any, ...], None] = OrderedDict()
_COUNTERS = {"captures": 0, "replays": 0, "bypasses": 0, "failures": 0, "invalidations": 0, "evictions": 0}
_BYPASS_REASONS: dict[str, int] = {}
_INVALIDATION_REASONS: dict[str, int] = {}
_LAST_ERROR: str | None = None
_LAST_KEY: tuple[Any, ...] | None = None
_LIFECYCLE_STATE: dict[str, str] = {}


def _flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def _observe_bypass(reason: str) -> None:
    _COUNTERS["bypasses"] += 1
    _BYPASS_REASONS[reason] = _BYPASS_REASONS.get(reason, 0) + 1
    telemetry_reason = "cache_disabled" if reason in {"disabled", "cache_disabled"} else "unsafe_input"
    openclaw_cache_epochs.observe("E11", "bypass", reason=telemetry_reason)


def _clear_cache_locked() -> bool:
    had_state = bool(_CACHE or _KEY_LOCKS or _FAILED_KEYS)
    _CACHE.clear()
    _KEY_LOCKS.clear()
    _FAILED_KEYS.clear()
    return had_state


def _key_lock(key: tuple[Any, ...]) -> threading.RLock:
    with _LOCK:
        key_lock = _KEY_LOCKS.get(key)
        if key_lock is None:
            key_lock = threading.RLock()
            _KEY_LOCKS[key] = key_lock
        return key_lock


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
            "lifecycle_state_keys": sorted(_LIFECYCLE_STATE),
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
            _LIFECYCLE_STATE.clear()
            _COUNTERS["invalidations"] += 1
            _INVALIDATION_REASONS["manual_reset"] = _INVALIDATION_REASONS.get("manual_reset", 0) + 1
            openclaw_cache_epochs.observe("E10", "invalidate", reason="manual_reset")
        return status()

def invalidate(reason: str, details: Any | None = None) -> dict[str, Any]:
    del details  # Details may contain paths or object reprs and never enter telemetry.
    with _EXECUTION_LOCK, _LOCK:
        if _clear_cache_locked():
            _COUNTERS["invalidations"] += 1
            _INVALIDATION_REASONS[reason] = _INVALIDATION_REASONS.get(reason, 0) + 1
            openclaw_cache_epochs.observe("E11", "invalidate", reason="dependency_changed")
            openclaw_cache_epochs.set_size("E11", current_size=0, capacity=_CACHE_MAX)
    return status()


def invalidate_if_changed(boundary: str, state: Any, reason: str | None = None) -> dict[str, Any]:
    marker = repr(state)
    with _EXECUTION_LOCK, _LOCK:
        old = _LIFECYCLE_STATE.get(boundary)
        _LIFECYCLE_STATE[boundary] = marker
        if old is None or old == marker:
            return status()
        if _clear_cache_locked():
            why = reason or f"{boundary}_changed"
            _COUNTERS["invalidations"] += 1
            _INVALIDATION_REASONS[why] = _INVALIDATION_REASONS.get(why, 0) + 1
            openclaw_cache_epochs.observe("E11", "invalidate", reason="dependency_changed")
            openclaw_cache_epochs.set_size("E11", current_size=0, capacity=_CACHE_MAX)
    return status()


def note_model_loaded(state: Any = None) -> dict[str, Any]:
    return invalidate_if_changed("model", state if state is not None else _runtime_identity(None)[0], "model_changed")


def note_vae_loaded(state: Any = None) -> dict[str, Any]:
    return invalidate_if_changed("vae", state if state is not None else _runtime_identity(None)[1], "vae_changed")


def _callable_identity(value: Any) -> tuple[Any, ...] | None:
    if value is None:
        return None
    function = getattr(value, "__func__", value)
    return (id(function), getattr(function, "__module__", None), getattr(function, "__qualname__", None))


def _file_revision(path: Any) -> tuple[Any, ...] | None:
    if not isinstance(path, (str, os.PathLike)) or not path:
        return None
    canonical = os.path.realpath(os.fspath(path))
    try:
        stat = os.stat(canonical)
        return (canonical, stat.st_mtime_ns, stat.st_size)
    except OSError:
        return (canonical, None, None)


def _module_revision(module: Any) -> tuple[Any, ...] | None:
    if module is None:
        return None
    tensors = []
    try:
        named_tensors = list(module.named_parameters(recurse=True)) + list(module.named_buffers(recurse=True))
    except Exception:
        named_tensors = []
    for name, tensor in named_tensors:
        if torch.is_tensor(tensor):
            tensors.append(
                (
                    name,
                    id(tensor),
                    getattr(tensor, "_version", None),
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    str(tensor.dtype),
                    str(tensor.device),
                    tensor.data_ptr() if tensor.device.type != "meta" else None,
                )
            )
    hooks = []
    for attribute in ("_forward_pre_hooks", "_forward_hooks", "_backward_hooks"):
        values = getattr(module, attribute, None)
        if values:
            hooks.append((attribute, tuple((key, _callable_identity(value)) for key, value in values.items())))
    return (id(module), type(module).__module__, type(module).__qualname__, bool(getattr(module, "training", False)), tuple(tensors), tuple(hooks))


def _tensor_key(x: torch.Tensor) -> tuple[Any, ...]:
    return (
        tuple(x.shape),
        tuple(x.stride()),
        x.storage_offset(),
        str(x.dtype),
        str(x.device),
        bool(x.is_contiguous()),
        bool(x.is_contiguous(memory_format=torch.channels_last)) if x.ndim == 4 else False,
    )


def _option_identity(operation: str, approximation: int) -> tuple[Any, ...]:
    try:
        from modules import shared

        opts = getattr(shared, "opts", None)
        cmd_opts = getattr(shared, "cmd_opts", None)
        method_name = "sd_vae_decode_method" if operation == _OPERATION_DECODE else "sd_vae_encode_method"
        return (
            approximation,
            getattr(opts, method_name, None),
            bool(getattr(opts, "hypertile_enable_vae", False)),
            bool(getattr(cmd_opts, "no_half_vae", False)),
            bool(getattr(cmd_opts, "upcast_sampling", False)),
            bool(getattr(cmd_opts, "precision", None) == "full"),
        )
    except Exception:
        return (approximation, None, None, None, None, None)


def _runtime_identity(model: Any) -> tuple[Any, ...]:
    try:
        from modules import shared

        try:
            from modules import devices
        except Exception:
            devices = None

        active_model = model if model is not None else getattr(shared, "sd_model", None)
        vae = getattr(active_model, "first_stage_model", None)
        info = getattr(active_model, "sd_checkpoint_info", None)
        model_identity = (
            id(active_model),
            getattr(info, "filename", None),
            getattr(info, "shorthash", None),
            getattr(info, "sha256", None),
            getattr(active_model, "sd_model_hash", None),
        )
        vae_identity = (
            _module_revision(vae),
            _file_revision(getattr(active_model, "loaded_vae_file", None)),
            _callable_identity(getattr(active_model, "decode_first_stage", None)),
            _callable_identity(getattr(active_model, "encode_first_stage", None)),
            _callable_identity(getattr(active_model, "get_first_stage_encoding", None)),
        )
        device_identity = (
            str(getattr(devices, "dtype_vae", None)),
            str(getattr(devices, "dtype", None)),
            str(getattr(devices, "device", None)),
        )
        return (model_identity, vae_identity, device_identity)
    except Exception:
        return (None, None, None)


def _graph_runtime_identity() -> tuple[Any, ...]:
    return (
        _GRAPH_CONTRACT_VERSION,
        torch.__version__,
        getattr(torch.version, "cuda", None),
        id(torch.cuda.CUDAGraph),
        os.environ.get("OPENCLAW_VAE_DECODE_GRAPHS"),
    )


def _mutation_epochs() -> tuple[tuple[str, int], ...]:
    try:
        return openclaw_cache_epochs.epoch_subset(_EPOCH_DIMENSIONS)
    except Exception:
        return tuple()


def _bypass_reason(model: Any, x: Any, approximation: int, operation: str) -> str | None:
    if not _ENABLED:
        return "disabled"
    if operation not in {_OPERATION_DECODE, _OPERATION_ENCODE}:
        return "unsupported_operation"
    # Encoding commonly samples a posterior. A capture-only invocation would
    # consume an extra RNG result on the cold path, so encoding remains eager
    # until its RNG state is an explicit, equivalence-tested graph dependency.
    if operation == _OPERATION_ENCODE:
        return "encode_rng_semantics"
    if approximation != 0:
        return "vae_approximation"
    if not torch.is_tensor(x):
        return "not_tensor"
    if not x.is_cuda:
        return "not_cuda"
    expected_channels = 4 if operation == _OPERATION_DECODE else 3
    if x.ndim != 4 or x.shape[1] != expected_channels:
        return "unsupported_shape"
    try:
        from modules import lowvram, shared

        opts = getattr(shared, "opts", None)
        method_name = "sd_vae_decode_method" if operation == _OPERATION_DECODE else "sd_vae_encode_method"
        if getattr(opts, method_name, "Full") != "Full":
            return "vae_method"
        if getattr(opts, "hypertile_enable_vae", False):
            return "hypertile_vae"
        if lowvram.is_enabled(model):
            return "lowvram"
    except Exception:
        return "state_probe_failed"
    vae = getattr(model, "first_stage_model", None)
    method = "decode_first_stage" if operation == _OPERATION_DECODE else "encode_first_stage"
    if vae is None or not hasattr(model, method):
        return "missing_vae"
    if bool(getattr(vae, "training", False)):
        return "vae_training"
    return None


def _key(model: Any, x: torch.Tensor, approximation: int, operation: str = _OPERATION_DECODE) -> tuple[Any, ...]:
    return (
        "vae_cuda_graph",
        operation,
        _runtime_identity(model),
        _tensor_key(x),
        _option_identity(operation, approximation),
        _mutation_epochs(),
        _graph_runtime_identity(),
    )


def _execute(model: Any, x: torch.Tensor, operation: str) -> torch.Tensor:
    from modules import devices

    with torch.no_grad(), devices.without_autocast():
        vae_input = x.to(model.first_stage_model.dtype)
        if operation == _OPERATION_DECODE:
            return model.decode_first_stage(vae_input)
        encoded = model.encode_first_stage(vae_input)
        return model.get_first_stage_encoding(encoded)


def _remember_failed_key_locked(key: tuple[Any, ...]) -> None:
    _FAILED_KEYS[key] = None
    _FAILED_KEYS.move_to_end(key)
    failure_capacity = max(1, _CACHE_MAX)
    while len(_FAILED_KEYS) > failure_capacity:
        _FAILED_KEYS.popitem(last=False)


def _evict_locked() -> None:
    while len(_CACHE) > _CACHE_MAX:
        evicted_key, _ = _CACHE.popitem(last=False)
        _KEY_LOCKS.pop(evicted_key, None)
        _COUNTERS["evictions"] += 1
        openclaw_cache_epochs.observe("E11", "eviction", reason="capacity", semantic_key=evicted_key)


def run(model: Any, x: Any, approximation: int = 0, *, operation: str = _OPERATION_DECODE) -> torch.Tensor | None:
    global _LAST_ERROR, _LAST_KEY
    with _EXECUTION_LOCK:
        reason = _bypass_reason(model, x, approximation, operation)
        if reason is not None:
            with _LOCK:
                _observe_bypass(reason)
            return None
        if _CACHE_MAX <= 0:
            with _LOCK:
                _observe_bypass("cache_disabled")
            return None

        key = _key(model, x, approximation, operation)
        with _LOCK:
            _LAST_KEY = key
        key_lock = _key_lock(key)
        with key_lock:
            with _LOCK:
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
            graph = None
            static_input = None
            static_output = None
            try:
                static_input = x.detach().contiguous().clone()
                stream = torch.cuda.Stream(device=x.device)
                stream.wait_stream(torch.cuda.current_stream(x.device))
                with torch.cuda.stream(stream):
                    warmup_output = _execute(model, static_input, operation)
                torch.cuda.current_stream(x.device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    static_output = _execute(model, static_input, operation)
                # CUDA graph capture completes asynchronously. Synchronize before
                # publishing or replaying a first-use entry so capture work cannot
                # race the request stream and seed a process-local output basin.
                torch.cuda.synchronize()
                entry = {"graph": graph, "input": static_input, "output": static_output, "operation": operation}
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
                    openclaw_cache_epochs.set_size("E11", current_size=len(_CACHE), capacity=_CACHE_MAX)
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
                    openclaw_cache_epochs.set_size("E11", current_size=len(_CACHE), capacity=_CACHE_MAX)
                return None


def run_encode(model: Any, x: Any, approximation: int = 0) -> torch.Tensor | None:
    """Reserved encode direction with explicit safe eager fallback semantics."""
    return run(model, x, approximation, operation=_OPERATION_ENCODE)


set_enabled(_flag("OPENCLAW_VAE_DECODE_GRAPHS", False), clear_cache=True)
