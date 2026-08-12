import ast
import importlib.util
import os
import sys
import threading
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class MemoryCache(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()


def load_cache(monkeypatch, tmp_path):
    diskcache = types.ModuleType("diskcache")
    diskcache.Cache = MemoryCache
    monkeypatch.setitem(sys.modules, "diskcache", diskcache)
    tqdm = types.ModuleType("tqdm"); tqdm.tqdm = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "tqdm", tqdm)
    paths = types.ModuleType("modules.paths")
    paths.data_path = str(tmp_path)
    paths.script_path = str(tmp_path)
    modules = types.ModuleType("modules")
    modules.__path__ = []
    monkeypatch.setitem(sys.modules, "modules", modules)
    monkeypatch.setitem(sys.modules, "modules.paths", paths)
    spec = importlib.util.spec_from_file_location("modules.cache", ROOT / "modules/cache.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "modules.cache", module)
    spec.loader.exec_module(module)
    return module


def test_file_cache_detects_same_size_same_mtime_replacement_and_schema(monkeypatch, tmp_path):
    cache = load_cache(monkeypatch, tmp_path)
    source = tmp_path / "source"
    source.write_text("one")
    original_mtime = source.stat().st_mtime_ns
    calls = []

    def parse():
        calls.append(1)
        return source.read_text()

    assert cache.cached_data_for_file("s", "t", source, parse, schema_revision="v1") == "one"
    assert cache.cached_data_for_file("s", "t", source, parse, schema_revision="v1") == "one"
    replacement = tmp_path / "replacement"
    replacement.write_text("two")
    os.utime(replacement, ns=(original_mtime, original_mtime))
    os.replace(replacement, source)
    os.utime(source, ns=(original_mtime, original_mtime))
    assert cache.cached_data_for_file("s", "t", source, parse, schema_revision="v1") == "two"
    assert cache.cached_data_for_file("s", "t", source, parse, schema_revision="v2") == "two"
    assert len(calls) == 3


def test_file_cache_single_publication_under_concurrency(monkeypatch, tmp_path):
    cache = load_cache(monkeypatch, tmp_path)
    source = tmp_path / "source"
    source.write_text("data")
    calls = 0
    barrier = threading.Barrier(8)
    lock = threading.Lock()

    def parse():
        nonlocal calls
        with lock:
            calls += 1
        time.sleep(0.02)
        return source.read_text()

    results = []
    def worker():
        barrier.wait()
        results.append(cache.cached_data_for_file("s", "t", source, parse, schema_revision="v1"))
    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert results == ["data"] * 8
    assert calls == 1


def test_git_revision_tracks_head_ref_content(monkeypatch, tmp_path):
    modules = types.ModuleType("modules")
    modules.__path__ = []
    shared = types.SimpleNamespace(cmd_opts=types.SimpleNamespace(), opts=types.SimpleNamespace())
    for name, value in {"shared": shared, "errors": types.SimpleNamespace(report=lambda *a, **k: None), "cache": object(), "scripts": types.SimpleNamespace(ScriptFile=object)}.items():
        setattr(modules, name, value)
        monkeypatch.setitem(sys.modules, f"modules.{name}", value if isinstance(value, types.ModuleType) else types.ModuleType(f"modules.{name}"))
    monkeypatch.setitem(sys.modules, "modules", modules)
    git = types.ModuleType("modules.gitpython_hack"); git.Repo = object
    paths = types.ModuleType("modules.paths_internal"); paths.extensions_dir=str(tmp_path/"exts"); paths.extensions_builtin_dir=str(tmp_path/"builtin"); paths.script_path=str(tmp_path)
    monkeypatch.setitem(sys.modules, "modules.gitpython_hack", git); monkeypatch.setitem(sys.modules, "modules.paths_internal", paths)
    spec = importlib.util.spec_from_file_location("derived_extensions", ROOT / "modules/extensions.py")
    extension = importlib.util.module_from_spec(spec); monkeypatch.setitem(sys.modules, "derived_extensions", extension); spec.loader.exec_module(extension)
    repo = tmp_path / "repo"; gitdir = repo / ".git"; (gitdir / "refs/heads").mkdir(parents=True)
    (gitdir / "HEAD").write_text("ref: refs/heads/main\n"); (gitdir / "refs/heads/main").write_text("a" * 40 + "\n")
    first = extension.git_repository_revision(repo)
    (gitdir / "refs/heads/main").write_text("b" * 40 + "\n")
    assert extension.git_repository_revision(repo) != first


def test_static_cache_reload_and_upscale_contracts():
    sampler = (ROOT / "modules/sd_samplers.py").read_text()
    multi = (ROOT / "extensions/openclaw-multi-sampler/scripts/openclaw_multi_sampler.py").read_text()
    upscale = (ROOT / "scripts/postprocessing_upscale.py").read_text()
    imgalt = (ROOT / "scripts/img2imgalt.py").read_text()
    hypertile = (ROOT / "extensions-builtin/hypertile/hypertile.py").read_text()
    ast.parse(sampler); ast.parse(multi); ast.parse(upscale); ast.parse(imgalt); ast.parse(hypertile)
    assert "get_sampler_and_scheduler.cache_clear()" in sampler
    register = multi[multi.index("def _register_definitions"):multi.index("def _upsert_custom")]
    assert "_SAMPLER_FUNC_CACHE.clear()" in register and "_SIGNATURE_PARAM_CACHE.clear()" in register
    assert "hashlib.sha256(image.tobytes()).digest()" in upscale
    assert "id(scaler)" in upscale
    assert "cached_image.copy()" in upscale and "image.copy()" in upscale
    assert "upscale_cache_lock" in upscale and "popitem(last=False)" in upscale
    assert "noise_cache = None" in imgalt and "torch.equal(noise_cache.latent, lat)" in imgalt
    assert "self.cache.noise" not in imgalt
    assert hypertile.count("@lru_cache(maxsize=256)") == 3 and "from functools import wraps, lru_cache" in hypertile
