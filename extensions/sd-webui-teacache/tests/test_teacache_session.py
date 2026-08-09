from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
import unittest

import torch

EXT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT_ROOT))


def install_a1111_stubs() -> None:
    modules_pkg = types.ModuleType("modules")
    headless_ui_mod = types.ModuleType("modules.headless_ui")
    processing_mod = types.ModuleType("modules.processing")
    processing_mod.StableDiffusionProcessing = object
    script_callbacks_mod = types.ModuleType("modules.script_callbacks")
    script_callbacks_mod.on_cfg_after_cfg = lambda _fn: None
    scripts_mod = types.ModuleType("modules.scripts")
    scripts_mod.Script = object
    sd_hijack_unet_mod = types.ModuleType("modules.sd_hijack_unet")
    sd_hijack_unet_mod.th = torch
    sd_samplers_common_mod = types.ModuleType("modules.sd_samplers_common")
    sd_samplers_common_mod.setup_img2img_steps = lambda _p, steps=None: (steps or getattr(_p, "steps", 1), steps or getattr(_p, "steps", 1))
    ui_components_mod = types.ModuleType("modules.ui_components")

    class InputAccordion:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return False
        def __exit__(self, *args):
            return False

    ui_components_mod.InputAccordion = InputAccordion

    sgm_pkg = types.ModuleType("sgm")
    sgm_modules_pkg = types.ModuleType("sgm.modules")
    sgm_diff_pkg = types.ModuleType("sgm.modules.diffusionmodules")
    openaimodel_mod = types.ModuleType("sgm.modules.diffusionmodules.openaimodel")
    openaimodel_mod.timestep_embedding = lambda timesteps, channels, repeat_only=False: torch.zeros((timesteps.shape[0], channels), device=timesteps.device)

    sys.modules.update({
        "modules": modules_pkg,
        "modules.headless_ui": headless_ui_mod,
        "modules.processing": processing_mod,
        "modules.script_callbacks": script_callbacks_mod,
        "modules.scripts": scripts_mod,
        "modules.sd_hijack_unet": sd_hijack_unet_mod,
        "modules.sd_samplers_common": sd_samplers_common_mod,
        "modules.ui_components": ui_components_mod,
        "sgm": sgm_pkg,
        "sgm.modules": sgm_modules_pkg,
        "sgm.modules.diffusionmodules": sgm_diff_pkg,
        "sgm.modules.diffusionmodules.openaimodel": openaimodel_mod,
    })


class TeaCacheSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_a1111_stubs()
        cls.teacache = importlib.import_module("scripts.teacache")

    def test_normalize_args_clamps_and_orders_bounds(self):
        assert self.teacache.normalize_args(["true", "2", "999", "0.9", "0.1"]) == (True, 1.0, 150, 0.1, 0.9)

    def test_progress_start_is_exclusive_and_end_is_inclusive(self):
        session = self.teacache.TeaCacheSession(threshold=1.0, max_consecutive=4, start=0.2, end=0.8, steps=10, initial_step=2)
        signature = ((1,),)
        session.previous_fb[0] = torch.zeros(1)
        session.residuals[0] = (signature, torch.ones(1))
        session.update_condition(torch.zeros(1), signature)
        self.assertFalse(session.use_cache)

        session.current_step = 8
        session.update_condition(torch.zeros(1), signature)
        self.assertTrue(session.use_cache)

    def test_signature_separates_conditioning_batch_shape_and_kwargs(self):
        h = torch.zeros(2, 4, 8, 8)
        timesteps = torch.zeros(2)
        context = torch.zeros(2, 77, 2048)
        y = torch.zeros(2, 2816)
        sig_a = self.teacache._call_signature(h, timesteps, context, y, {"control": torch.zeros(2, 1)})
        sig_b = self.teacache._call_signature(h, timesteps, context[:1], y, {"control": torch.zeros(2, 1)})
        sig_c = self.teacache._call_signature(h, timesteps, context, y, {"other": torch.zeros(2, 1)})
        self.assertNotEqual(sig_a, sig_b)
        self.assertNotEqual(sig_a, sig_c)


    def test_nonfinite_distance_forces_refresh_and_resets_accumulator(self):
        session = self.teacache.TeaCacheSession(threshold=1.0, max_consecutive=0, start=0.0, end=1.0, steps=10)
        signature = ((1,),)
        session.previous_fb[0] = torch.ones(1)
        session.residuals[0] = (signature, torch.ones(1))
        session.update_condition(torch.full((1,), float("inf")), signature)
        self.assertFalse(session.use_cache)
        torch.testing.assert_close(session.distances[0], torch.zeros(()))

    def test_first_block_residual_state_is_detached(self):
        session = self.teacache.TeaCacheSession(threshold=1.0, max_consecutive=0, start=0.0, end=1.0, steps=10)
        signature = ((1,),)
        session.previous_fb[0] = torch.zeros(1)
        session.residuals[0] = (signature, torch.ones(1))
        session.update_condition(torch.zeros(1, requires_grad=True), signature)
        self.assertFalse(session.previous_fb[0].requires_grad)
        self.assertFalse(session.distances[0].requires_grad)

    def test_cached_residual_is_cloned_not_mutable_alias(self):
        session = self.teacache.TeaCacheSession(threshold=1.0, max_consecutive=4, start=0.0, end=1.0, steps=10)
        signature = ((1,),)
        residual = torch.ones(2)
        session.store_current_residual(signature, residual)
        residual.add_(10)
        torch.testing.assert_close(session.current_residual(signature), torch.ones(2))


if __name__ == "__main__":
    unittest.main()
