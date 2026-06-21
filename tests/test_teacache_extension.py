import importlib.util
import math
import sys
import types
from pathlib import Path

import torch


def load_teacache_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "extensions" / "sd-webui-teacache" / "scripts" / "teacache.py"

    modules_pkg = types.ModuleType("modules")
    modules_pkg.__path__ = []
    sys.modules["modules"] = modules_pkg

    headless_ui = types.ModuleType("modules.headless_ui")
    headless_ui.Row = object
    headless_ui.Slider = lambda *args, **kwargs: None
    headless_ui.Number = lambda *args, **kwargs: None
    sys.modules["modules.headless_ui"] = headless_ui

    processing = types.ModuleType("modules.processing")
    processing.StableDiffusionProcessing = object
    sys.modules["modules.processing"] = processing

    script_callbacks = types.ModuleType("modules.script_callbacks")
    script_callbacks.on_cfg_after_cfg = lambda *args, **kwargs: None
    sys.modules["modules.script_callbacks"] = script_callbacks

    scripts = types.ModuleType("modules.scripts")
    scripts.Script = object
    scripts.AlwaysVisible = object()
    sys.modules["modules.scripts"] = scripts

    sd_hijack_unet = types.ModuleType("modules.sd_hijack_unet")
    sd_hijack_unet.th = torch
    sys.modules["modules.sd_hijack_unet"] = sd_hijack_unet

    sd_samplers_common = types.ModuleType("modules.sd_samplers_common")
    sd_samplers_common.setup_img2img_steps = lambda p, total_steps=None: (total_steps or p.steps, p.steps)
    sys.modules["modules.sd_samplers_common"] = sd_samplers_common

    ui_components = types.ModuleType("modules.ui_components")
    ui_components.InputAccordion = object
    sys.modules["modules.ui_components"] = ui_components

    sgm = types.ModuleType("sgm")
    sgm_modules = types.ModuleType("sgm.modules")
    diffusionmodules = types.ModuleType("sgm.modules.diffusionmodules")
    openaimodel = types.ModuleType("sgm.modules.diffusionmodules.openaimodel")
    openaimodel.timestep_embedding = lambda *args, **kwargs: None
    sys.modules["sgm"] = sgm
    sys.modules["sgm.modules"] = sgm_modules
    sys.modules["sgm.modules.diffusionmodules"] = diffusionmodules
    sys.modules["sgm.modules.diffusionmodules.openaimodel"] = openaimodel

    spec = importlib.util.spec_from_file_location("teacache_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_args_clamps_and_orders_range():
    teacache = load_teacache_module()

    assert teacache.normalize_args([1, "2.5", "12.8", 0.9, -0.2]) == (True, 1.0, 12, 0.0, 0.9)
    assert teacache.normalize_args([]) == (False, 0.3, 0, 0.3, 1.0)
    assert teacache.normalize_args([True]) == (True, 0.3, 0, 0.3, 1.0)
    assert teacache.normalize_args(["false"]) == (False, 0.3, 0, 0.3, 1.0)
    assert teacache.normalize_args([True, "bad", None, "bad", "bad"]) == (True, 0.3, 0, 0.3, 1.0)


def test_relative_l1_distance_handles_zero_baseline():
    teacache = load_teacache_module()
    prev = torch.zeros((2, 2), dtype=torch.float32)
    curr = torch.ones((2, 2), dtype=torch.float32)

    distance = teacache.relative_l1_distance(prev, curr)

    assert math.isfinite(distance)
    assert distance > 0


def test_session_requires_residual_for_current_call_index():
    teacache = load_teacache_module()
    session = teacache.TeaCacheSession(threshold=1.0, max_consecutive=0, start=0.0, end=1.0, steps=10)
    session.use_cache = True
    session.residuals[0] = torch.ones((1, 1), dtype=torch.float32)

    assert session.can_use_current_residual()
    session.call_index = 1
    assert not session.can_use_current_residual()


if __name__ == "__main__":
    test_normalize_args_clamps_and_orders_range()
    test_relative_l1_distance_handles_zero_baseline()
    test_session_requires_residual_for_current_call_index()
