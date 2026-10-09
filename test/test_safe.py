import importlib.util
import io
import pickle
import sys
import types
import zipfile
from pathlib import Path

import numpy
import pytest


def module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def load_safe_module(monkeypatch):
    storage = module("torch.storage", TypedStorage=type("TypedStorage", (), {}))
    torch = module("torch", storage=storage, load=lambda *args, **kwargs: None)
    torch._utils = module("torch._utils")
    torch.nn = module("torch.nn", modules=module("torch.nn.modules", container=module("torch.nn.modules.container")))

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "modules.errors", module("modules.errors", report=lambda *a, **k: None))

    spec = importlib.util.spec_from_file_location("test_loaded_safe", Path("modules/safe.py"))
    safe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(safe)
    return safe


def test_check_zip_filenames_requires_literal_metadata_dot_prefixes(monkeypatch):
    safe = load_safe_module(monkeypatch)

    safe.check_zip_filenames("model.pt", ["archive/data.pkl", "archive/.data/serialization_id"])

    with pytest.raises(Exception, match="bad file inside"):
        safe.check_zip_filenames("model.pt", ["archive/data.pkl", "archive/xdata/serialization_id"])

    with pytest.raises(Exception, match="bad file inside"):
        safe.check_zip_filenames("model.pt", ["archive/data.pkl", "archive/xformat_version"])


def test_restricted_unpickler_accepts_numpy_scalars_and_arrays_from_numpy_1_and_2(monkeypatch, tmp_path):
    safe = load_safe_module(monkeypatch)
    # torch.save writes pickle protocol 2 into <archive>/data.pkl; NumPy 2 names numpy._core.multiarray there.
    data = pickle.dumps({"step": numpy.int64(7), "values": numpy.arange(3, dtype=numpy.float32)}, protocol=2)
    assert b"numpy._core.multiarray" in data

    for label, payload in (("numpy2", data), ("numpy1", data.replace(b"numpy._core.multiarray", b"numpy.core.multiarray"))):
        path = tmp_path / f"{label}.pt"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("archive/data.pkl", payload)
            archive.writestr("archive/version", "3\n")
        safe.check_pt(str(path), None)

    unpickler = safe.RestrictedUnpickler(io.BytesIO())
    assert unpickler.find_class("numpy.core.multiarray", "scalar") is numpy._core.multiarray.scalar
    with pytest.raises(Exception, match="is forbidden"):
        unpickler.find_class("numpy._core.multiarray", "frombuffer")


def test_check_pt_reads_real_torch_storages(monkeypatch, tmp_path):
    import torch

    monkeypatch.setattr(torch, "load", torch.load)  # executing safe.py replaces torch.load; put it back afterwards
    monkeypatch.setitem(sys.modules, "modules.errors", module("modules.errors", report=lambda *a, **k: None))
    spec = importlib.util.spec_from_file_location("test_loaded_safe_real_torch", Path("modules/safe.py"))
    safe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(safe)

    path = tmp_path / "tensor.pt"
    torch.save({"weight": torch.ones(2)}, path)
    safe.check_pt(str(path), None)  # every storage reaches persistent_load, which returns a TypedStorage placeholder

    storage = safe.RestrictedUnpickler(io.BytesIO()).persistent_load(("storage",))
    assert type(storage) is torch.storage.TypedStorage
