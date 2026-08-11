import importlib.util
import ast
import math
import sys
import types
from pathlib import Path

import pytest
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
                if segment == "bool(should_refresh)":
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


@pytest.mark.parametrize("marker", ["_original_forward", "_controlnet_forward_hook_owner"])
def test_external_unet_forward_hook_is_detected(marker):
    teacache = load_teacache_module()
    unet = types.SimpleNamespace()
    setattr(unet, marker, object())
    p = types.SimpleNamespace(
        sd_model=types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=unet))
    )

    assert teacache._has_external_unet_forward_hook(p)


def test_process_does_not_wrap_controlnet_owned_unet():
    teacache = load_teacache_module()
    original_forward = lambda x, **kwargs: x
    owner = object()
    unet = types.SimpleNamespace(
        forward=original_forward,
        _controlnet_forward_hook_owner=owner,
    )
    p = types.SimpleNamespace(
        sd_model=types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=unet))
    )

    teacache.TeaCacheScript().process(p, True)

    assert unet.forward is original_forward
    assert unet._controlnet_forward_hook_owner is owner
    assert not getattr(unet, "_teacache_patched", False)
    assert not teacache._has_external_unet_forward_hook(types.SimpleNamespace())


class _GuardLock:
    def __init__(self):
        self.depth = 0
        self.entries = 0

    def __enter__(self):
        self.depth += 1
        self.entries += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        self.depth -= 1


def test_global_session_accessors_are_lock_guarded():
    teacache = load_teacache_module()
    guard = _GuardLock()
    teacache._cache_lock = guard
    session = teacache.TeaCacheSession(threshold=1.0, max_consecutive=0, start=0.0, end=1.0, steps=10)

    teacache._set_cache(session)
    assert teacache._get_cache() is session
    teacache._set_cache(None)

    assert guard.entries == 3
    assert guard.depth == 0


def test_api_generation_paths_hold_queue_lock_for_teacache_global_state():
    source = Path(__file__).resolve().parents[1].joinpath("modules/api/api.py").read_text()
    text2img = source[source.index("    def text2imgapi"):source.index("    def img2imgapi")]
    img2img = source[source.index("    def img2imgapi"):source.index("    def _run_extras")]

    assert "with self.queue_lock:" in text2img
    assert "processed = self._run_generation_with_scripts" in text2img
    assert text2img.index("with self.queue_lock:") < text2img.index("processed = self._run_generation_with_scripts")
    assert "with self.queue_lock:" in img2img
    assert "processed = self._run_generation_with_scripts" in img2img
    assert img2img.index("with self.queue_lock:") < img2img.index("processed = self._run_generation_with_scripts")



def test_session_refreshes_on_nonfinite_distance_and_does_not_cache_nan():
    teacache = load_teacache_module()
    signature = ((1, 2), "torch.float32", "cpu")
    session = teacache.TeaCacheSession(threshold=1.0, max_consecutive=0, start=0.0, end=1.0, steps=10)
    session.previous_fb[0] = torch.ones((1, 2), dtype=torch.float32)
    session.residuals[0] = (signature, torch.zeros((1, 2), dtype=torch.float32))

    session.update_condition(torch.full((1, 2), float("nan"), dtype=torch.float32), signature)

    assert not session.use_cache
    assert torch.equal(session.distances[0], torch.zeros((), dtype=torch.float32))


def test_session_detaches_first_block_residual_before_distance_math():
    teacache = load_teacache_module()
    signature = ((1, 2), "torch.float32", "cpu")
    session = teacache.TeaCacheSession(threshold=1.0, max_consecutive=0, start=0.0, end=1.0, steps=10)
    session.previous_fb[0] = torch.ones((1, 2), dtype=torch.float32)
    session.residuals[0] = (signature, torch.zeros((1, 2), dtype=torch.float32))

    session.update_condition(torch.ones((1, 2), dtype=torch.float32, requires_grad=True), signature)

    assert session.use_cache
    assert not session.previous_fb[0].requires_grad
    assert not session.distances[0].requires_grad


def test_postprocess_clears_session_even_when_current_unet_is_unpatched():
    teacache = load_teacache_module()
    script = teacache.TeaCacheScript()
    current_unet = types.SimpleNamespace(_teacache_patched=False)
    p = types.SimpleNamespace(sd_model=types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=current_unet)))
    teacache._set_cache(teacache.TeaCacheSession(threshold=1.0, max_consecutive=0, start=0.0, end=1.0, steps=10))

    script.postprocess(p)

    assert teacache._get_cache() is None
    assert script.original_forward is None
    assert script.patched_unet is None



def test_patched_forward_exception_restores_unet_and_clears_cache():
    teacache = load_teacache_module()

    def original_forward(x, timesteps=None, context=None, y=None, **kwargs):
        raise RuntimeError("synthetic model failure")

    unet = types.SimpleNamespace(num_classes=None)
    unet.forward = teacache.patched_forward.__get__(unet)
    unet._teacache_patched = True
    unet._openclaw_teacache_original_forward = original_forward
    session = teacache.TeaCacheSession(threshold=1.0, max_consecutive=0, start=0.0, end=1.0, steps=10)
    session.disabled_reason = "synthetic exception path"
    teacache._set_cache(session)

    try:
        unet.forward(torch.zeros((1, 1)))
    except RuntimeError as exc:
        assert "synthetic model failure" in str(exc)
    else:
        raise AssertionError("expected synthetic model failure")

    assert unet.forward is original_forward
    assert unet._teacache_patched is False
    assert not hasattr(unet, "_openclaw_teacache_original_forward")
    assert teacache._get_cache() is None

def test_process_restores_previous_patched_unet_when_model_object_changes_before_disable():
    teacache = load_teacache_module()
    script = teacache.TeaCacheScript()

    def original_forward(x, timesteps=None, context=None, y=None, **kwargs):
        return x + 1

    first_unet = types.SimpleNamespace(forward=original_forward)
    first_p = types.SimpleNamespace(sd_model=types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=first_unet)))
    script.process(first_p, True)
    patched_forward = first_unet.forward
    assert getattr(first_unet, "_teacache_patched", False)
    assert patched_forward is not original_forward

    second_unet = types.SimpleNamespace(forward=lambda x, **kwargs: x)
    second_p = types.SimpleNamespace(sd_model=types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=second_unet)))
    script.process(second_p, False)

    assert first_unet.forward is original_forward
    assert not first_unet._teacache_patched
    assert not hasattr(first_unet, "_openclaw_teacache_original_forward")
    assert teacache._get_cache() is None


def test_storing_fresh_residual_resets_accumulated_distance():
    teacache = load_teacache_module()
    signature = ((1, 2), "torch.float32", "cpu")
    session = teacache.TeaCacheSession(threshold=1.0, max_consecutive=0, start=0.0, end=1.0, steps=10)
    session.distances[0] = torch.tensor(0.75)

    session.store_current_residual(signature, torch.ones((1, 2), dtype=torch.float32))

    torch.testing.assert_close(session.distances[0], torch.zeros(()))


def test_process_restores_old_unet_before_patching_new_model_object():
    teacache = load_teacache_module()
    script = teacache.TeaCacheScript()

    def first_forward(x, timesteps=None, context=None, y=None, **kwargs):
        return x + 1

    def second_forward(x, timesteps=None, context=None, y=None, **kwargs):
        return x + 2

    first_unet = types.SimpleNamespace(forward=first_forward)
    second_unet = types.SimpleNamespace(forward=second_forward)
    script.process(types.SimpleNamespace(sd_model=types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=first_unet))), True)
    script.process(types.SimpleNamespace(sd_model=types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=second_unet))), True)

    assert first_unet.forward is first_forward
    assert not first_unet._teacache_patched
    assert not hasattr(first_unet, "_openclaw_teacache_original_forward")
    assert getattr(second_unet, "_teacache_patched", False)
    assert second_unet._openclaw_teacache_original_forward is second_forward

def test_patched_forward_falls_back_to_original_when_session_disabled():
    teacache = load_teacache_module()
    x = torch.zeros((1, 1), dtype=torch.float32)
    timesteps = torch.zeros((1,), dtype=torch.float32)
    calls = []

    def original_forward(x_arg, timesteps=None, context=None, y=None, **kwargs):
        calls.append((x_arg, timesteps, context, y, kwargs))
        return x_arg + 2

    unet = types.SimpleNamespace(
        num_classes=None,
        _openclaw_teacache_original_forward=original_forward,
    )
    teacache._cache = teacache.TeaCacheSession(
        threshold=1.0,
        max_consecutive=0,
        start=0.0,
        end=1.0,
        steps=10,
        disabled_reason="external UNet forward hook",
    )
    try:
        result = teacache.patched_forward(unet, x, timesteps=timesteps)
    finally:
        teacache._cache = None

    torch.testing.assert_close(result, x + 2)
    assert calls == [(x, timesteps, None, None, {})]


def test_patched_forward_falls_back_to_original_for_conditioning_kwargs():
    teacache = load_teacache_module()
    x = torch.zeros((1, 1), dtype=torch.float32)
    timesteps = torch.zeros((1,), dtype=torch.float32)
    calls = []

    def original_forward(x_arg, timesteps=None, context=None, y=None, **kwargs):
        calls.append(kwargs)
        return x_arg + 3

    unet = types.SimpleNamespace(
        num_classes=None,
        _openclaw_teacache_original_forward=original_forward,
    )
    teacache._cache = teacache.TeaCacheSession(threshold=1.0, max_consecutive=0, start=0.0, end=1.0, steps=10)
    try:
        result = teacache.patched_forward(unet, x, timesteps=timesteps, transformer_options={"control": True})
    finally:
        teacache._cache = None

    torch.testing.assert_close(result, x + 3)
    assert calls == [{"transformer_options": {"control": True}}]


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
    test_external_unet_forward_hook_is_detected()
    test_global_session_accessors_are_lock_guarded()
    test_api_generation_paths_hold_queue_lock_for_teacache_global_state()
    test_session_refreshes_on_nonfinite_distance_and_does_not_cache_nan()
    test_session_detaches_first_block_residual_before_distance_math()
    test_postprocess_clears_session_even_when_current_unet_is_unpatched()
    test_process_restores_previous_patched_unet_when_model_object_changes_before_disable()
    test_storing_fresh_residual_resets_accumulated_distance()
    test_process_restores_old_unet_before_patching_new_model_object()
    test_patched_forward_falls_back_to_original_when_session_disabled()
    test_patched_forward_falls_back_to_original_for_conditioning_kwargs()
