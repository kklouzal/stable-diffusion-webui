import importlib.util
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).parents[1] / "gb10/controlnet_cache_contract.py"
spec = importlib.util.spec_from_file_location("controlnet_cache_contract", MODULE_PATH)
cache_contract = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cache_contract
spec.loader.exec_module(cache_contract)
AtomicLRU = cache_contract.AtomicLRU
callable_identity = cache_contract.callable_identity
freeze = cache_contract.freeze
ndarray_identity = cache_contract.ndarray_identity
runtime_identity = cache_contract.runtime_identity


def test_cold_hit_input_layout_dtype_options_source_device_stochastic_separation(tmp_path):
    cache = AtomicLRU(16, "test")
    calls = 0
    source = tmp_path / "annotator.py"
    source.write_text("revision-one")

    def run(key):
        nonlocal calls
        calls += 1
        return np.array([calls], dtype=np.int64)

    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    source_revision = cache_contract._file_digest(source)
    base = (source_revision, ndarray_identity(image), (2, 2, 3), "rgb", "crop", "cpu", "float32", 7, freeze({"threshold": 1}), runtime_identity())
    first = cache.get_or_compute(base, lambda: run(base), clone=True)
    second = cache.get_or_compute(base, lambda: run(base), clone=True)
    assert calls == 1
    assert np.array_equal(first, second)
    second[0] = 99
    assert cache.get_or_compute(base, lambda: run(base), clone=True)[0] == 1

    variants = [
        base[:1] + (ndarray_identity(image.copy() + 1),) + base[2:],
        base[:2] + ((1, 4, 3),) + base[3:],
        base[:5] + ("cuda:0",) + base[6:],
        base[:6] + ("float16",) + base[7:],
        base[:7] + (8,) + base[8:],
        base[:8] + (freeze({"threshold": 2}),) + base[9:],
    ]
    source.write_text("revision-two")
    os.utime(source, ns=(source.stat().st_atime_ns, source.stat().st_mtime_ns + 1))
    variants.append((cache_contract._file_digest(source),) + base[1:])
    for key in variants:
        cache.get_or_compute(key, lambda key=key: run(key), clone=True)
    assert calls == 1 + len(variants)


def test_failure_is_not_published_and_retry_succeeds():
    cache = AtomicLRU(2, "test")
    calls = 0

    def compute():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("partial")
        return "complete"

    try:
        cache.get_or_compute("key", compute)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")
    assert cache.info()["size"] == 0
    assert cache.get_or_compute("key", compute) == "complete"
    assert calls == 2
    assert cache.info()["stats"]["failure:not-published"] == 1


def test_same_key_concurrency_computes_once_and_publishes_complete_value():
    cache = AtomicLRU(2, "test")
    calls = 0
    barrier = threading.Barrier(8)
    results = []

    def compute():
        nonlocal calls
        calls += 1
        time.sleep(0.03)
        return {"complete": True}

    def worker():
        barrier.wait()
        results.append(cache.get_or_compute("key", compute))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert calls == 1
    assert results == [{"complete": True}] * 8
    assert cache.info()["inflight"] == 0


def test_lru_bound_capacity_change_clear_and_observability():
    cache = AtomicLRU(2, "test")
    for key in ("a", "b", "c"):
        cache.get_or_compute(key, lambda key=key: key)
    assert cache.info()["size"] == 2
    assert cache.info()["stats"]["evict:lru"] == 1
    cache.set_max_size(1)
    assert cache.info()["size"] == 1
    cache.clear("extension-reload")
    info = cache.info()
    assert info["size"] == 0
    assert info["last_lookup"]["reason"] == "extension-reload"
    assert info["stats"]["invalidate:extension-reload"] == 1


def test_callable_and_runtime_identity_change_with_implementation(tmp_path):
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("def f(): return 1\n")
    second.write_text("def f(): return 2\n")

    def load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.f

    assert callable_identity(load(first, "first")) != callable_identity(load(second, "second"))
    assert len(runtime_identity()) == 5


def test_model_contract_source_base_runtime_options_and_loader_separation(tmp_path):
    source = tmp_path / "model.safetensors"
    source.write_bytes(b"model-one")
    stat = source.stat()
    source_revision = (str(source.resolve()), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    base = ("unet", "checkpoint-a")
    options = freeze({"control_net_lowvram": False})
    loader = ("loader", 1)
    key = (source_revision, base, "float16", "cuda:0", runtime_identity(), loader, options)
    cache = AtomicLRU(8, "model")
    calls = 0
    def build():
        nonlocal calls
        calls += 1
        return object()
    assert cache.get_or_compute(key, build) is cache.get_or_compute(key, build)
    variants = [
        key[:1] + (("unet", "checkpoint-b"),) + key[2:],
        key[:2] + ("float32",) + key[3:],
        key[:3] + ("cpu",) + key[4:],
        key[:5] + (("loader", 2),) + key[6:],
        key[:6] + (freeze({"control_net_lowvram": True}),),
    ]
    for variant in variants: cache.get_or_compute(variant, build)
    assert calls == 1 + len(variants)


def test_forced_clean_equivalence_for_deterministic_array_result():
    image = np.arange(9, dtype=np.float32).reshape(3, 3)
    options = {"threshold": 3}
    def preprocess(): return np.where(image > options["threshold"], 255, 0).astype(np.uint8)
    key = (ndarray_identity(image), freeze(options), "cpu", "float32")
    cache = AtomicLRU(2, "preprocessor")
    cached = cache.get_or_compute(key, preprocess, clone=True)
    forced_clean = preprocess()
    assert np.array_equal(cached, forced_clean)
    cache.clear("forced-clean")
    assert np.array_equal(cache.get_or_compute(key, preprocess, clone=True), forced_clean)
