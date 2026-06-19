import importlib.util
import sys
import types
from pathlib import Path

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
    numpy = module("numpy", core=module("numpy.core", multiarray=module("numpy.core.multiarray")))

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "numpy", numpy)
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
