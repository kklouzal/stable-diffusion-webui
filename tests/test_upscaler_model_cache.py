"""Contract of modelloader.load_cached_spandrel_model (shared upscaler model cache)."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("spandrel")

MODULE_PATH = Path(__file__).resolve().parents[1] / "modules" / "modelloader.py"
_STUBBED = ("modules", "modules.shared", "modules.upscaler", "modules.util", "modules.modelloader")


@pytest.fixture()
def modelloader():
    previous = {name: sys.modules.get(name) for name in _STUBBED}
    modules_pkg = types.ModuleType("modules")
    modules_pkg.__path__ = []
    upscaler = types.ModuleType("modules.upscaler")
    for name in ("Upscaler", "UpscalerLanczos", "UpscalerNearest", "UpscalerNone"):
        setattr(upscaler, name, type(name, (), {}))
    util = types.ModuleType("modules.util")
    util.load_file_from_url = lambda *args, **kwargs: None
    sys.modules.update({
        "modules": modules_pkg,
        "modules.shared": types.ModuleType("modules.shared"),
        "modules.upscaler": upscaler,
        "modules.util": util,
    })
    try:
        spec = importlib.util.spec_from_file_location("modules.modelloader", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules["modules.modelloader"] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def _write_tiny_esrgan(path: Path, seed: int) -> None:
    from spandrel.architectures.ESRGAN import ESRGAN

    torch.manual_seed(seed)
    torch.save(ESRGAN(in_nc=3, out_nc=3, num_filters=8, num_blocks=1, scale=2).state_dict(), path)


def _count_loads(monkeypatch, modelloader):
    calls = []
    original = modelloader.load_spandrel_model

    def counting(path, **kwargs):
        calls.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(modelloader, "load_spandrel_model", counting)
    return calls


def test_hit_returns_same_eval_model_with_exact_weights(tmp_path, monkeypatch, modelloader):
    path = tmp_path / "tiny.pth"
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
    path = tmp_path / "tiny.pth"
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


def test_load_arguments_are_part_of_the_key(tmp_path, monkeypatch, modelloader):
    path = tmp_path / "tiny.pth"
    _write_tiny_esrgan(path, seed=1)
    calls = _count_loads(monkeypatch, modelloader)

    full = modelloader.load_cached_spandrel_model(path, device="cpu")
    half = modelloader.load_cached_spandrel_model(path, device="cpu", prefer_half=True)

    assert half is not full
    assert len(calls) == 2
    assert full.dtype == torch.float32
    assert half.dtype == torch.float16


def test_lru_is_bounded_and_refreshed_on_hit(tmp_path, monkeypatch, modelloader):
    paths = [tmp_path / f"tiny{i}.pth" for i in range(3)]
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
    path = tmp_path / "tiny.pth"
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
