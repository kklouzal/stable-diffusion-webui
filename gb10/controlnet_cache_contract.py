from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import platform
import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

CACHE_SCHEMA = 1


def _file_digest(path: str) -> tuple:
    path = os.path.realpath(path)
    stat = os.stat(path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return (path, stat.st_size, stat.st_mtime_ns, digest.hexdigest())


def callable_identity(value: Any) -> tuple:
    target = value.__func__ if inspect.ismethod(value) else value
    module = inspect.getmodule(target)
    source = inspect.getsourcefile(target) or (getattr(module, "__file__", None) if module else None)
    source_identity = _file_digest(source) if source and os.path.isfile(source) else None
    code = getattr(target, "__code__", None)
    code_digest = hashlib.sha256(code.co_code).hexdigest() if code is not None else None
    return (
        getattr(target, "__module__", None),
        getattr(target, "__qualname__", type(target).__qualname__),
        source_identity,
        code_digest,
    )


def runtime_identity(torch_module=None) -> tuple:
    torch_version = getattr(torch_module, "__version__", None)
    cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
    return (platform.python_implementation(), platform.python_version(), np.__version__, torch_version, cuda_version)


def ndarray_identity(value: np.ndarray) -> tuple:
    array = np.asarray(value)
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()
    return (array.shape, array.strides, array.dtype.str, digest)


def freeze(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return ("ndarray", ndarray_identity(value))
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), freeze(v)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(freeze(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(freeze(v) for v in value))
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return value
    if hasattr(value, "device") and hasattr(value, "dtype") and hasattr(value, "shape"):
        return (type(value).__module__, type(value).__qualname__, tuple(value.shape), str(value.dtype), str(value.device))
    return (type(value).__module__, type(value).__qualname__, repr(value))


def clone_result(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, tuple):
        return type(value)(*(clone_result(v) for v in value)) if hasattr(value, "_fields") else tuple(clone_result(v) for v in value)
    if isinstance(value, list):
        return [clone_result(v) for v in value]
    if isinstance(value, dict):
        return {k: clone_result(v) for k, v in value.items()}
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


@dataclass(frozen=True)
class CacheLookup:
    hit: bool
    reason: str


class AtomicLRU:
    """Bounded LRU with per-key serialization and failure-atomic publication."""

    def __init__(self, max_size: int, name: str):
        self.max_size = max(0, int(max_size))
        self.name = name
        self._values = OrderedDict()
        self._locks = {}
        self._lock_refs = Counter()
        self._lock = threading.RLock()
        self._stats = Counter()
        self._last_lookup = CacheLookup(False, "cold")

    def get_or_compute(self, key: Any, compute: Callable[[], Any], *, clone: bool = False) -> Any:
        if self.max_size == 0:
            with self._lock:
                self._stats["bypass:disabled"] += 1
                self._last_lookup = CacheLookup(False, "disabled")
            return compute()
        with self._lock:
            if key in self._values:
                value = self._values.pop(key)
                self._values[key] = value
                self._stats["hit"] += 1
                self._last_lookup = CacheLookup(True, "hit")
                return clone_result(value) if clone else value
            key_lock = self._locks.setdefault(key, threading.Lock())
            self._lock_refs[key] += 1
        try:
            with key_lock:
                with self._lock:
                    if key in self._values:
                        value = self._values.pop(key)
                        self._values[key] = value
                        self._stats["hit:after-wait"] += 1
                        self._last_lookup = CacheLookup(True, "hit-after-wait")
                        return clone_result(value) if clone else value
                    self._stats["miss"] += 1
                    self._last_lookup = CacheLookup(False, "key-change" if self._values else "cold")
                try:
                    value = compute()
                except BaseException:
                    with self._lock:
                        self._stats["failure:not-published"] += 1
                    raise
                published = clone_result(value) if clone else value
                with self._lock:
                    self._values[key] = published
                    while len(self._values) > self.max_size:
                        self._values.popitem(last=False)
                        self._stats["evict:lru"] += 1
                return clone_result(published) if clone else published
        finally:
            with self._lock:
                self._lock_refs[key] -= 1
                if self._lock_refs[key] == 0:
                    del self._lock_refs[key]
                    if self._locks.get(key) is key_lock:
                        del self._locks[key]

    def discard(self, key: Any, reason: str = "volatile") -> None:
        with self._lock:
            if self._values.pop(key, None) is not None:
                self._stats[f"invalidate:{reason}"] += 1
                self._last_lookup = CacheLookup(False, reason)

    def clear(self, reason: str = "explicit") -> None:
        with self._lock:
            count = len(self._values)
            self._values.clear()
            self._stats[f"invalidate:{reason}"] += count or 1
            self._last_lookup = CacheLookup(False, reason)

    def set_max_size(self, max_size: int, reason: str = "capacity-change") -> None:
        max_size = max(0, int(max_size))
        with self._lock:
            if max_size != self.max_size:
                self.max_size = max_size
                while len(self._values) > self.max_size:
                    self._values.popitem(last=False)
                    self._stats["evict:capacity-change"] += 1
                self._stats[f"invalidate:{reason}"] += 1
                self._last_lookup = CacheLookup(False, reason)

    def info(self) -> dict:
        with self._lock:
            return {"name": self.name, "size": len(self._values), "max_size": self.max_size, "inflight": len(self._locks), "stats": dict(self._stats), "last_lookup": self._last_lookup.__dict__}
