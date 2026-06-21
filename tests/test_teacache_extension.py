import importlib.util
import ast
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
    assert teacache.normalize_args([]) == (False, 0.25, 4, 0.35, 0.9)
    assert teacache.normalize_args([True]) == (True, 0.25, 4, 0.35, 0.9)
    assert teacache.normalize_args(["false"]) == (False, 0.25, 4, 0.35, 0.9)
    assert teacache.normalize_args([True, "bad", None, "bad", "bad"]) == (True, 0.25, 4, 0.35, 0.9)


def test_relative_l1_distance_handles_zero_baseline():
    teacache = load_teacache_module()
    prev = torch.zeros((2, 2), dtype=torch.float32)
    curr = torch.ones((2, 2), dtype=torch.float32)

    distance = teacache.relative_l1_distance(prev, curr)

    assert isinstance(distance, torch.Tensor)
    assert distance.shape == torch.Size([])
    assert distance.device == prev.device
    assert math.isfinite(distance.item())
    assert distance.item() > 0



def test_sdxl_polynomial_distance_matches_coefficients_on_tensor_device():
    teacache = load_teacache_module()
    relative = torch.tensor(0.125, dtype=torch.float32)
    coeffs = relative.new_tensor(teacache.SDXL_POLYNOMIAL_COEFFICIENTS)

    distance = teacache.sdxl_polynomial_distance(relative, coeffs)
    expected = sum(float(coeff) * (float(relative) ** power) for power, coeff in enumerate(teacache.SDXL_POLYNOMIAL_COEFFICIENTS))

    assert isinstance(distance, torch.Tensor)
    assert distance.device == relative.device
    assert math.isclose(distance.item(), expected, rel_tol=1e-6, abs_tol=1e-6)


def test_session_caches_device_tensors_for_hot_path_constants():
    teacache = load_teacache_module()
    session = teacache.TeaCacheSession(threshold=1.0, max_consecutive=0, start=0.0, end=1.0, steps=10)
    signature = ((1, 2), "torch.float32", "cpu")
    old = torch.ones((1, 2), dtype=torch.float32)
    session.previous_fb[0] = old
    session.residuals[0] = (signature, torch.zeros((1, 2), dtype=torch.float32))

    session.update_condition(old * 1.001, signature)

    assert session.use_cache
    assert session.distances[0].device == old.device
    assert session._threshold_tensors[(str(old.device), torch.float32)].device == old.device
    assert session._coefficient_tensors[(str(old.device), torch.float32)].device == old.device


def test_hot_path_sync_constructs_are_explicitly_allowlisted():
    root = Path(__file__).resolve().parents[1]
    source = (root / "extensions" / "sd-webui-teacache" / "scripts" / "teacache.py").read_text()
    hot_source = source[source.index("def relative_l1_distance"):]
    tree = ast.parse(hot_source)
    banned_attrs = {"cpu", "numpy", "tolist", "item"}
    banned_calls = []
    bool_calls = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in banned_attrs:
                banned_calls.append((func.attr, node.lineno))
            elif isinstance(func, ast.Name) and func.id in {"float", "int", "bool"}:
                segment = ast.get_source_segment(hot_source, node) or ""
                if "torch.ge" in segment:
                    bool_calls.append(node.lineno)
                else:
                    banned_calls.append((func.id, node.lineno))

    assert banned_calls == []
    assert len(bool_calls) == 1
    assert "Intentional sync point" in source

def test_session_requires_residual_for_current_call_index():
    teacache = load_teacache_module()
    session = teacache.TeaCacheSession(threshold=1.0, max_consecutive=0, start=0.0, end=1.0, steps=10)
    session.use_cache = True
    signature = ((1, 1), "torch.float32", "cpu")
    session.residuals[0] = (signature, torch.ones((1, 1), dtype=torch.float32))

    assert session.can_use_current_residual()
    assert session.current_residual(signature) is not None
    assert session.current_residual(((2, 1), "torch.float32", "cpu")) is None
    session.call_index = 1
    assert not session.can_use_current_residual()


def test_session_isolates_cache_by_call_signature():
    teacache = load_teacache_module()
    session = teacache.TeaCacheSession(threshold=1.0, max_consecutive=0, start=0.0, end=1.0, steps=10)
    old = torch.ones((1, 2), dtype=torch.float32)
    signature = ((1, 2), "torch.float32", "cpu")
    other_signature = ((2, 2), "torch.float32", "cpu")

    session.previous_fb[0] = old
    session.residuals[0] = (other_signature, torch.ones((1, 2), dtype=torch.float32))
    session.update_condition(old * 1.01, signature)

    assert not session.use_cache
    session.store_current_residual(signature, torch.full((1, 2), 3.0))
    session.use_cache = True
    assert torch.equal(session.current_residual(signature), torch.full((1, 2), 3.0))
    assert session.current_residual(other_signature) is None


def test_session_window_and_max_consecutive_are_quality_guards():
    teacache = load_teacache_module()
    signature = ((1, 2), "torch.float32", "cpu")
    session = teacache.TeaCacheSession(threshold=1.0, max_consecutive=1, start=0.25, end=0.75, steps=4, initial_step=1)
    session.previous_fb[0] = torch.ones((1, 2), dtype=torch.float32)
    session.residuals[0] = (signature, torch.zeros((1, 2), dtype=torch.float32))

    session.update_condition(torch.ones((1, 2), dtype=torch.float32), signature)
    assert not session.use_cache

    session.next_step()
    session.update_condition(torch.ones((1, 2), dtype=torch.float32), signature)
    assert session.use_cache
    assert session.consecutive_hits == 1

    session.next_step()
    session.update_condition(torch.ones((1, 2), dtype=torch.float32), signature)
    assert not session.use_cache


def test_masked_denoising_disables_cache():
    teacache = load_teacache_module()
    p = types.SimpleNamespace(mask=torch.zeros((1, 1)), nmask=None, image_mask=None)
    assert teacache._has_masked_denoising(p)

    p = types.SimpleNamespace(mask=None, nmask=None, image_mask=None)
    assert not teacache._has_masked_denoising(p)


if __name__ == "__main__":
    test_normalize_args_clamps_and_orders_range()
    test_relative_l1_distance_handles_zero_baseline()
    test_sdxl_polynomial_distance_matches_coefficients_on_tensor_device()
    test_session_caches_device_tensors_for_hot_path_constants()
    test_hot_path_sync_constructs_are_explicitly_allowlisted()
    test_session_requires_residual_for_current_call_index()
    test_session_isolates_cache_by_call_signature()
    test_session_window_and_max_consecutive_are_quality_guards()
    test_masked_denoising_disables_cache()
