import importlib
import sys
import types
from pathlib import Path
import unittest

import torch

EXT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT_ROOT))


def install_a1111_stubs():
    modules_pkg = types.ModuleType("modules")
    scripts_mod = types.ModuleType("modules.scripts")
    scripts_mod.AlwaysVisible = object()
    script_callbacks_mod = types.ModuleType("modules.script_callbacks")

    class CFGDenoiserParams:
        pass

    callback_registry = []
    script_callbacks_mod.CFGDenoiserParams = CFGDenoiserParams
    script_callbacks_mod.callback_registry = callback_registry
    def on_cfg_denoiser(callback, *args, **kwargs):
        callback_registry.append(callback)

    def remove_callbacks_for_function(callback):
        callback_registry[:] = [c for c in callback_registry if c is not callback]

    script_callbacks_mod.on_cfg_denoiser = on_cfg_denoiser
    script_callbacks_mod.remove_callbacks_for_function = remove_callbacks_for_function
    processing_mod = types.ModuleType("modules.processing")
    processing_mod.StableDiffusionProcessing = object
    shared_mod = types.ModuleType("modules.shared")
    shared_mod.device = torch.device("cpu")

    modules_pkg.scripts = scripts_mod
    modules_pkg.script_callbacks = script_callbacks_mod
    modules_pkg.processing = processing_mod
    modules_pkg.shared = shared_mod
    sys.modules.update({
        "modules": modules_pkg,
        "modules.scripts": scripts_mod,
        "modules.script_callbacks": script_callbacks_mod,
        "modules.processing": processing_mod,
        "modules.shared": shared_mod,
    })


class DynamicThresholdingTests(unittest.TestCase):
    def test_relative_path_preserves_dtype_and_finiteness(self):
        from dynthres_core import DynThresh

        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            dt = DynThresh(7.0, 1.0, "Constant", 0.0, "Constant", 0.0, 4.0, 0, 10, True, "MEAN", "AD", 1.0)
            dt.step = 1
            uncond = torch.randn(2, 4, 8, 8, dtype=dtype)
            relative = torch.randn_like(uncond) * 0.1
            out = dt.dynthresh_from_relative(relative, uncond, 12.0)
            self.assertEqual(out.dtype, dtype)
            self.assertTrue(torch.isfinite(out.float()).all())

    def test_ragged_multicond_equivalence_with_manual_relative(self):
        from dynthres_core import DynThresh

        dt = DynThresh(7.0, 1.0, "Constant", 0.0, "Constant", 0.0, 4.0, 0, 10, True, "MEAN", "AD", 1.0)
        dt.step = 1
        uncond = torch.randn(2, 4, 4, 4)
        x_out = torch.randn(5, 4, 4, 4)
        conds_list = [[(0, 0.5), (2, 1.25)], [(1, 0.75), (3, -0.1), (4, 0.25)]]
        relative = torch.zeros_like(uncond)
        for i, conds in enumerate(conds_list):
            for cond_index, weight in conds:
                relative[i] += (x_out[cond_index] - uncond[i]) * weight
        out = dt.dynthresh_from_relative(relative, uncond, 9.0)
        self.assertEqual(out.shape, uncond.shape)
        self.assertTrue(torch.isfinite(out).all())


class CFGCombinerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_a1111_stubs()
        cls.cfg_combiner = importlib.import_module("scripts.cfg_combiner")

    def test_no_pag_delegates_to_original(self):
        called = {"value": False}

        def original(x_out, conds_list, uncond, cond_scale):
            called["value"] = True
            return torch.full_like(x_out[-uncond.shape[0]:], cond_scale)

        x_out = torch.randn(3, 4, 4, 4)
        uncond = torch.randn(1, 77, 32)
        out = self.cfg_combiner.combine_denoised_pass_conds_list(
            x_out, [[(0, 1.0)]], uncond, 7.5, original_func=original, cfg_dict={"pag_params": None}
        )
        self.assertTrue(called["value"])
        self.assertTrue(torch.equal(out, torch.full_like(out, 7.5)))

    def test_sdxl_uncond_dict_uses_crossattn_shape_and_missing_pag_falls_back(self):
        class Pag:
            pag_active = True
            pag_x_out = None
            pag_scale = 3.0
            pag_start_step = 0
            pag_end_step = 10
            step = 1
            pag_sanf = False
            cfg_interval_enable = False
            cfg_interval_scheduled_value = 7.0

        def original(x_out, conds_list, uncond, cond_scale):
            return torch.full_like(x_out[-uncond["crossattn"].shape[0]:], cond_scale)

        x_out = torch.randn(4, 4, 4, 4)
        uncond = {"crossattn": torch.randn(2, 77, 32), "vector": torch.randn(2, 1280)}
        out = self.cfg_combiner.combine_denoised_pass_conds_list(
            x_out, [[(0, 1.0)], [(1, 1.0)]], uncond, 6.0, original_func=original, cfg_dict={"pag_params": Pag()}
        )
        self.assertEqual(tuple(out.shape), (2, 4, 4, 4))
        self.assertTrue(torch.equal(out, torch.full_like(out, 6.0)))

    def test_cfg_combiner_process_batch_skips_inactive_callback(self):
        callbacks = sys.modules["modules.script_callbacks"].callback_registry
        callbacks.clear()
        script = self.cfg_combiner.CFGCombinerScript()

        class Processing:
            incant_cfg_params = {"pag_params": None}

        script.process_batch(Processing())
        self.assertEqual(callbacks, [])

    def test_cfg_combiner_process_batch_registers_when_pag_active(self):
        callbacks = sys.modules["modules.script_callbacks"].callback_registry
        callbacks.clear()
        script = self.cfg_combiner.CFGCombinerScript()

        class Processing:
            incant_cfg_params = {"pag_params": object()}

        script.process_batch(Processing())
        self.assertEqual(len(callbacks), 1)
        script.remove_callbacks()
        self.assertEqual(callbacks, [])

    def test_cfg_combiner_wrapper_restores_only_own_wrapper(self):
        script = self.cfg_combiner.CFGCombinerScript()

        def original(x_out, conds_list, uncond, cond_scale):
            return torch.full_like(x_out[-uncond.shape[0]:], cond_scale)

        class Denoiser:
            combine_denoised = staticmethod(original)

        denoiser = Denoiser()
        cfg_dict = {"denoiser": None, "original_combine_denoised": None, "wrapped_combine_denoised": None, "pag_params": None}
        script.patch_cfg_denoiser(denoiser, cfg_dict)
        wrapped = denoiser.combine_denoised
        self.assertIs(cfg_dict["wrapped_combine_denoised"], wrapped)
        self.assertIsNot(wrapped, original)

        script.restore_cfg_denoiser(cfg_dict)
        self.assertIs(denoiser.combine_denoised, original)
        self.assertIsNone(cfg_dict["denoiser"])

        script.patch_cfg_denoiser(denoiser, cfg_dict)
        wrapped = denoiser.combine_denoised

        def external_wrapper(*args, **kwargs):
            return wrapped(*args, **kwargs)

        denoiser.combine_denoised = external_wrapper
        script.restore_cfg_denoiser(cfg_dict)
        self.assertIs(denoiser.combine_denoised, external_wrapper)
        self.assertIsNone(cfg_dict["denoiser"])


class ModuleHookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_a1111_stubs()
        cls.module_hooks = importlib.import_module("scripts.incant_utils.module_hooks")

    def test_forward_hook_handle_removal_is_local(self):
        layer = torch.nn.Linear(2, 2, bias=False)
        calls = {"count": 0}

        def hook(module, args, output):
            calls["count"] += 1
            return output

        handle = self.module_hooks.module_add_forward_hook(layer, hook)
        layer(torch.ones(1, 2))
        self.assertEqual(calls["count"], 1)
        handle.remove()
        layer(torch.ones(1, 2))
        self.assertEqual(calls["count"], 1)


if __name__ == "__main__":
    unittest.main()
