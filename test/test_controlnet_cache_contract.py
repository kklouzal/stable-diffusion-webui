import importlib.util
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

CONTROLNET = Path(__file__).parents[1] / "extensions/sd-webui-controlnet"
MODULE_PATH = CONTROLNET / "internal_controlnet/cache_contract.py"
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


def _parse(rel):
    import ast

    return ast.parse((CONTROLNET / rel).read_text(encoding="utf-8"))


def _exec_definitions(rel, wanted, namespace, class_name=None):
    """Execute the named top-level functions of a ControlNet source file (or methods of its class_name) in
    namespace; methods come back on a class of the same name holding only them."""
    import ast

    tree = _parse(rel)
    body = tree.body if class_name is None else next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name).body
    nodes = [node for node in body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert {node.name for node in nodes} == set(wanted)
    if class_name is not None:
        nodes = [ast.ClassDef(name=class_name, bases=[], keywords=[], body=nodes, decorator_list=[])]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(CONTROLNET / rel), "exec"), namespace)
    return namespace


def test_cache_helper_lives_outside_scanned_scripts_root():
    import ast

    # A1111 executes every module under an extension's scripts/ as a script.
    assert not (CONTROLNET / "scripts/cache_contract.py").exists()
    for rel in ("scripts/controlnet.py", "scripts/supported_preprocessor.py"):
        imported = {node.module for node in ast.walk(_parse(rel)) if isinstance(node, ast.ImportFrom)}
        assert "internal_controlnet.cache_contract" in imported
        assert "scripts.cache_contract" not in imported


def test_controlnet_correctness_fixes_live_in_tracked_source():
    # Source checks (normalized by ast.unparse): the inpaint hijack runs only inside a full ControlNet UNet forward
    # of a 9-channel inpaint model, and the openpose average differs only with several scales (scale_search has one).
    import ast

    def statements(rel):
        return {ast.unparse(node) for node in ast.walk(_parse(rel)) if isinstance(node, (ast.Assign, ast.AugAssign))}

    # The cached inpaint latent follows x's device/dtype (the .to() result was discarded before).
    assert "param.used_hint_inpaint_hijack = param.used_hint_inpaint_hijack.to(device=x.device, dtype=x.dtype)" in statements("scripts/hook.py")
    body = statements("annotator/openpose/body.py")
    assert "heatmap_avg += heatmap / len(multiplier)" in body
    assert "heatmap_avg += heatmap_avg + heatmap / len(multiplier)" not in body


def test_both_preprocessor_path_settings_are_registered_once():
    # annotator_path.py reads both options (test_controlnet_preprocessor_path_contract.py).
    from types import SimpleNamespace

    registered = []

    class OptionInfo:
        def __init__(self, default=None, label="", component=None, component_args=None, section=None):
            self.default, self.section, self.reload_ui = default, section, False

        def needs_reload_ui(self):
            self.reload_ui = True
            return self

    shared = SimpleNamespace(OptionInfo=OptionInfo, opts=SimpleNamespace(add_option=lambda key, info: registered.append((key, info))))
    namespace = {"shared": shared, "global_state": SimpleNamespace(default_detectedmap_dir="detected_maps"),
                 "gr": SimpleNamespace(Slider="Slider", Checkbox="Checkbox")}
    _exec_definitions("scripts/controlnet.py", ["on_ui_settings"], namespace)["on_ui_settings"]()

    keys = [key for key, _info in registered]
    infos = dict(registered)
    for key in ("control_net_modules_path", "control_net_preprocessor_models_path"):
        assert keys.count(key) == 1
        assert infos[key].default == "" and infos[key].reload_ui
        assert infos[key].section == ("control_net", "ControlNet")


def test_option_snapshot_covers_exactly_the_controlnet_options(tmp_path):
    # One snapshot for both cache keys (Script._model_cache_key, Preprocessor result keys): every
    # "control_net*"/"controlnet*" option and nothing else, independent of insertion order.
    from types import SimpleNamespace

    import torch

    data = {"control_net_unit_count": 3, "sd_model_checkpoint": "x", "controlnet_clip_detector_on_cpu": False,
            "CN_other": 1}
    snapshot = cache_contract.controlnet_option_snapshot(data)
    assert snapshot == freeze({"control_net_unit_count": 3, "controlnet_clip_detector_on_cpu": False})
    assert snapshot == cache_contract.controlnet_option_snapshot(dict(reversed(list(data.items()))))
    assert snapshot != cache_contract.controlnet_option_snapshot({**data, "control_net_unit_count": 4})

    # Both keys read the live options through it.
    opts = SimpleNamespace(data=dict(data))
    common = {"shared": SimpleNamespace(opts=opts), "torch": torch,
              "devices": SimpleNamespace(device="cpu", dtype=torch.float32, dtype_unet=torch.float32),
              "runtime_identity": runtime_identity, "callable_identity": callable_identity,
              "controlnet_option_snapshot": cache_contract.controlnet_option_snapshot, "freeze": freeze}
    script = _exec_definitions("scripts/controlnet.py", ["_model_cache_key"],
                               {**common, "os": os, "build_model_by_guess": lambda *args: None}, "Script")["Script"]
    model_file = tmp_path / "control.safetensors"
    model_file.write_bytes(b"weights")
    script._resolve_model_path = staticmethod(lambda model: (model, str(model_file)))
    preprocessor_class = _exec_definitions("scripts/supported_preprocessor.py", ["_cache_identity"], dict(common), "Preprocessor")["Preprocessor"]
    preprocessor_class.__call__ = lambda self, *args, **kwargs: None
    preprocessor = preprocessor_class()
    preprocessor.__dict__.update(model=None, name="canny", label="Canny", device="cpu", cache_ignored_kwargs=set())
    p = SimpleNamespace(sd_model=SimpleNamespace(dtype=torch.float16))
    unet = object()

    def keys():
        return script._model_cache_key(p, unet, "control"), preprocessor._cache_identity((1,), {"low": 100})

    before = keys()
    opts.data["sd_model_checkpoint"] = "y"
    assert keys() == before
    opts.data["control_net_unit_count"] = 4
    after = keys()
    assert after[0] != before[0] and after[1] != before[1]
