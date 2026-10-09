"""Contract of modelloader.load_cached_spandrel_model (shared upscaler model cache)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from test.helpers import load_source, module

torch = pytest.importorskip("torch")
pytest.importorskip("spandrel")


@pytest.fixture()
def modelloader(tmp_path):
    stubs = {
        "modules": module("modules", package=True),
        "modules.paths": module("modules.paths", data_path=str(tmp_path), script_path=str(tmp_path)),
        "modules.shared": module("modules.shared"),
        "modules.upscaler": module("modules.upscaler", **{name: type(name, (), {}) for name in ("Upscaler", "UpscalerLanczos", "UpscalerNearest", "UpscalerNone")}),
        "modules.util": module("modules.util", load_file_from_url=lambda *args, **kwargs: None),
    }
    stubs["modules.cache"] = load_source("modules.cache", "modules/cache.py", stubs)  # the real file identity helpers
    return load_source("modules.modelloader", "modules/modelloader.py", stubs)


def _write_tiny_esrgan(path: Path, seed: int) -> None:
    # safetensors, not .pth: a .pth goes through torch.load, which modules.safe replaces once another test imported
    # the real webui modules.
    from safetensors.torch import save_file
    from spandrel.architectures.ESRGAN import ESRGAN

    torch.manual_seed(seed)
    save_file(ESRGAN(in_nc=3, out_nc=3, num_filters=8, num_blocks=1, scale=2).state_dict(), str(path))


def _count_loads(monkeypatch, modelloader):
    calls = []
    original = modelloader.load_spandrel_model

    def counting(path, **kwargs):
        calls.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(modelloader, "load_spandrel_model", counting)
    return calls


def test_hit_returns_same_eval_model_with_exact_weights(tmp_path, monkeypatch, modelloader):
    path = tmp_path / "tiny.safetensors"
    _write_tiny_esrgan(path, seed=1)
    calls = _count_loads(monkeypatch, modelloader)

    first = modelloader.load_cached_spandrel_model(path, load_device="cpu", device="cpu", expected_architecture="ESRGAN")
    second = modelloader.load_cached_spandrel_model(str(path), device=torch.device("cpu"), expected_architecture="ESRGAN")

    assert second is first
    assert len(calls) == 1
    assert not first.model.training
    assert first.device == torch.device("cpu")
    reference = modelloader.load_spandrel_model(path, device="cpu", expected_architecture="ESRGAN")
    cached_state, reference_state = first.model.state_dict(), reference.model.state_dict()
    assert cached_state.keys() == reference_state.keys()
    assert all(torch.equal(cached_state[k], reference_state[k]) for k in reference_state)


def test_replaced_file_reloads_and_drops_stale_entry(tmp_path, monkeypatch, modelloader):
    path = tmp_path / "tiny.safetensors"
    _write_tiny_esrgan(path, seed=1)
    calls = _count_loads(monkeypatch, modelloader)
    first = modelloader.load_cached_spandrel_model(path, device="cpu")

    _write_tiny_esrgan(path, seed=2)
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    second = modelloader.load_cached_spandrel_model(path, device="cpu")

    assert second is not first
    assert len(calls) == 2
    assert len(modelloader._spandrel_model_cache) == 1
    assert not torch.equal(next(first.model.parameters()), next(second.model.parameters()))


def test_same_size_replacement_with_preserved_mtime_reloads(tmp_path, monkeypatch, modelloader):
    # cp -p / rsync -t / tar x: equal size and mtime, but a new inode (and ctime).
    path = tmp_path / "tiny.safetensors"
    _write_tiny_esrgan(path, seed=1)
    calls = _count_loads(monkeypatch, modelloader)
    first = modelloader.load_cached_spandrel_model(path, device="cpu")

    original = os.stat(path)
    replacement = tmp_path / "replacement.safetensors"
    _write_tiny_esrgan(replacement, seed=2)
    os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
    os.replace(replacement, path)
    replaced = os.stat(path)
    assert (replaced.st_size, replaced.st_mtime_ns) == (original.st_size, original.st_mtime_ns)

    second = modelloader.load_cached_spandrel_model(path, device="cpu")

    assert second is not first
    assert len(calls) == 2
    assert len(modelloader._spandrel_model_cache) == 1
    assert not torch.equal(next(first.model.parameters()), next(second.model.parameters()))


def test_load_arguments_are_part_of_the_key(tmp_path, monkeypatch, modelloader):
    path = tmp_path / "tiny.safetensors"
    _write_tiny_esrgan(path, seed=1)
    calls = _count_loads(monkeypatch, modelloader)

    full = modelloader.load_cached_spandrel_model(path, device="cpu")
    half = modelloader.load_cached_spandrel_model(path, device="cpu", prefer_half=True)

    assert half is not full
    assert len(calls) == 2
    assert full.dtype == torch.float32
    assert half.dtype == torch.float16


def test_lru_is_bounded_and_refreshed_on_hit(tmp_path, monkeypatch, modelloader):
    paths = [tmp_path / f"tiny{i}.safetensors" for i in range(3)]
    for seed, path in enumerate(paths):
        _write_tiny_esrgan(path, seed=seed)
    calls = _count_loads(monkeypatch, modelloader)

    a = modelloader.load_cached_spandrel_model(paths[0], device="cpu")
    modelloader.load_cached_spandrel_model(paths[1], device="cpu")
    assert modelloader.load_cached_spandrel_model(paths[0], device="cpu") is a  # refresh a
    modelloader.load_cached_spandrel_model(paths[2], device="cpu")  # evicts paths[1]

    assert len(modelloader._spandrel_model_cache) == modelloader._SPANDREL_MODEL_CACHE_SIZE == 2
    assert modelloader.load_cached_spandrel_model(paths[0], device="cpu") is a
    modelloader.load_cached_spandrel_model(paths[1], device="cpu")
    assert calls == [os.path.realpath(p) for p in (paths[0], paths[1], paths[2], paths[1])]


def test_failed_load_is_not_cached(tmp_path, monkeypatch, modelloader):
    path = tmp_path / "tiny.safetensors"
    _write_tiny_esrgan(path, seed=1)
    original = modelloader.load_spandrel_model

    def failing(*args, **kwargs):
        raise RuntimeError("corrupt model")

    monkeypatch.setattr(modelloader, "load_spandrel_model", failing)
    with pytest.raises(RuntimeError, match="corrupt model"):
        modelloader.load_cached_spandrel_model(path, device="cpu")
    assert not modelloader._spandrel_model_cache

    monkeypatch.setattr(modelloader, "load_spandrel_model", original)
    assert modelloader.load_cached_spandrel_model(path, device="cpu").scale == 2


def test_missing_file_raises(tmp_path, modelloader):
    with pytest.raises(FileNotFoundError):
        modelloader.load_cached_spandrel_model(tmp_path / "missing.pth", device="cpu")
    assert not modelloader._spandrel_model_cache


def test_compiled_model_is_its_own_entry(tmp_path, monkeypatch, modelloader):
    # SwinIR's SWIN_torch_compile: compiled once per file revision, never compiling the plain entry other callers share.
    path = tmp_path / "tiny.safetensors"
    _write_tiny_esrgan(path, seed=1)
    calls = _count_loads(monkeypatch, modelloader)

    plain = modelloader.load_cached_spandrel_model(path, device="cpu")
    compiled = modelloader.load_cached_spandrel_model(path, device="cpu", compile_model=True)

    assert compiled is not plain
    assert modelloader.load_cached_spandrel_model(path, device="cpu", compile_model=True) is compiled
    assert len(calls) == 2
    assert compiled.model._compiled_call_impl is not None  # torch.nn.Module.compile's in-place wrapper
    assert plain.model._compiled_call_impl is None
