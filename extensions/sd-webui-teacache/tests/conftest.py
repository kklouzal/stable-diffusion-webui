"""Shared harness: loads scripts/teacache.py against minimal A1111/sgm stubs.

Every stub goes through monkeypatch, so sys.modules is restored after each test, and every test gets a fresh
module object (its own session global, lock and caches).
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

TEACACHE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "teacache.py"


def _install_stub_modules(monkeypatch):
    def module(name, **attrs):
        mod = ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    class InputAccordion:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return False

        def __exit__(self, *args):
            return False

    module("sgm")
    module("sgm.modules")
    module("sgm.modules.diffusionmodules")
    # Timestep-dependent stand-in for the sinusoidal embedding.
    module(
        "sgm.modules.diffusionmodules.openaimodel",
        timestep_embedding=lambda timesteps, dim, repeat_only=False: timesteps[:, None].float().expand(-1, dim) * 0.01,
    )
    module("modules")
    module("modules.headless_ui", Row=lambda *a, **k: None, Slider=lambda *a, **k: None, Number=lambda *a, **k: None)
    module("modules.processing", StableDiffusionProcessing=object)
    module("modules.script_callbacks", on_cfg_after_cfg=lambda callback: callback)
    module("modules.scripts", Script=object, AlwaysVisible=object())
    module("modules.sd_samplers_common", setup_img2img_steps=lambda p, steps=None: (steps or p.steps, steps or p.steps))
    module("modules.sd_hijack_unet", th=torch)
    module("modules.ui_components", InputAccordion=InputAccordion)


@pytest.fixture
def load_teacache(monkeypatch):
    """Install the stubs, then return a loader; each call executes teacache.py as a new module named `name`."""
    _install_stub_modules(monkeypatch)

    def load(name="teacache_under_test"):
        spec = importlib.util.spec_from_file_location(name, TEACACHE_PATH)
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, mod)
        spec.loader.exec_module(mod)
        return mod

    return load


@pytest.fixture
def teacache(load_teacache):
    return load_teacache()
