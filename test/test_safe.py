import io
import pickle
import zipfile

import numpy
import pytest

from test.helpers import load_source, module


def _errors():
    return module("modules.errors", report=lambda *a, **k: None)


def load_safe_module():
    torch = module(
        "torch",
        storage=module("torch.storage", TypedStorage=type("TypedStorage", (), {})),
        load=lambda *args, **kwargs: None,
        _utils=module("torch._utils"),
        nn=module("torch.nn", modules=module("torch.nn.modules", container=module("torch.nn.modules.container"))),
    )
    return load_source("test_loaded_safe", "modules/safe.py", {"torch": torch, "modules.errors": _errors()})


def test_check_zip_filenames_requires_literal_metadata_dot_prefixes():
    safe = load_safe_module()

    safe.check_zip_filenames("model.pt", ["archive/data.pkl", "archive/.data/serialization_id"])

    with pytest.raises(Exception, match="bad file inside"):
        safe.check_zip_filenames("model.pt", ["archive/data.pkl", "archive/xdata/serialization_id"])

    with pytest.raises(Exception, match="bad file inside"):
        safe.check_zip_filenames("model.pt", ["archive/data.pkl", "archive/xformat_version"])


def test_restricted_unpickler_accepts_numpy_scalars_and_arrays_from_numpy_1_and_2(tmp_path):
    safe = load_safe_module()
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
    safe = load_source("test_loaded_safe_real_torch", "modules/safe.py", {"modules.errors": _errors()})

    path = tmp_path / "tensor.pt"
    torch.save({"weight": torch.ones(2)}, path)
    safe.check_pt(str(path), None)  # every storage reaches persistent_load, which returns a TypedStorage placeholder

    storage = safe.RestrictedUnpickler(io.BytesIO()).persistent_load(("storage",))
    assert type(storage) is torch.storage.TypedStorage
