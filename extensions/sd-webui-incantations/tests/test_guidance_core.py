import importlib
import sys
import types
from pathlib import Path
import unittest

import torch

EXT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT_ROOT))

DynThresh = importlib.import_module("dynthres_core").DynThresh


def install_a1111_stubs():
    modules_pkg = types.ModuleType("modules")
    headless_ui_mod = types.ModuleType("modules.headless_ui")
    scripts_mod = types.ModuleType("modules.scripts")
    scripts_mod.AlwaysVisible = object()
    script_callbacks_mod = types.ModuleType("modules.script_callbacks")

    class CFGDenoiserParams:
        pass

    class CFGDenoisedParams:
        pass

    callback_registry = []
    script_callbacks_mod.CFGDenoiserParams = CFGDenoiserParams
    script_callbacks_mod.CFGDenoisedParams = CFGDenoisedParams
    script_callbacks_mod.callback_registry = callback_registry

    def on_cfg_denoiser(callback, *args, **kwargs):
        callback_registry.append(callback)

    def on_cfg_denoised(callback, *args, **kwargs):
        callback_registry.append(callback)

    def remove_callbacks_for_function(callback):
        callback_registry[:] = [c for c in callback_registry if c is not callback]

    script_callbacks_mod.on_cfg_denoiser = on_cfg_denoiser
    script_callbacks_mod.on_cfg_denoised = on_cfg_denoised
    script_callbacks_mod.remove_callbacks_for_function = remove_callbacks_for_function
    processing_mod = types.ModuleType("modules.processing")
    processing_mod.StableDiffusionProcessing = object
    shared_mod = types.ModuleType("modules.shared")
    shared_mod.device = torch.device("cpu")
    shared_mod.opts = types.SimpleNamespace(
        batch_cond_uncond=False,
        uni_pc_variant="bh1",
        uni_pc_skip_type="time_uniform",
        uni_pc_order=3,
        uni_pc_lower_order_final=True,
    )
    models_mod = types.ModuleType("modules.models")
    diffusion_mod = types.ModuleType("modules.models.diffusion")
    uni_pc_mod = types.ModuleType("modules.models.diffusion.uni_pc")

    class UniPCSampler:
        def __init__(self, model, **kwargs):
            self.model = model

        def before_sample(self, x, t, cond, uncond):
            return x, t, cond, uncond

        def after_sample(self, *args, **kwargs):
            return None

        def after_update(self, *args, **kwargs):
            return None

    uni_pc_mod.sampler = types.SimpleNamespace(UniPCSampler=UniPCSampler)
    uni_pc_mod.uni_pc = types.SimpleNamespace()
    samplers_mod = types.ModuleType("modules.sd_samplers_cfg_denoiser")

    def catenate_conds(conds):
        if not isinstance(conds[0], dict):
            return torch.cat(conds)
        return {key: torch.cat([x[key] for x in conds]) for key in conds[0]}

    def subscript_cond(cond, a, b):
        if not isinstance(cond, dict):
            return cond[a:b]
        return {key: value[a:b] for key, value in cond.items()}

    samplers_mod.catenate_conds = catenate_conds
    samplers_mod.subscript_cond = subscript_cond

    modules_pkg.scripts = scripts_mod
    modules_pkg.headless_ui = headless_ui_mod
    modules_pkg.script_callbacks = script_callbacks_mod
    modules_pkg.processing = processing_mod
    modules_pkg.shared = shared_mod
    modules_pkg.models = models_mod
    models_mod.diffusion = diffusion_mod
    diffusion_mod.uni_pc = uni_pc_mod
    sys.modules.update(
        {
            "modules": modules_pkg,
            "modules.models": models_mod,
            "modules.models.diffusion": diffusion_mod,
            "modules.models.diffusion.uni_pc": uni_pc_mod,
            "modules.headless_ui": headless_ui_mod,
            "modules.scripts": scripts_mod,
            "modules.script_callbacks": script_callbacks_mod,
            "modules.processing": processing_mod,
            "modules.shared": shared_mod,
            "modules.sd_samplers_cfg_denoiser": samplers_mod,
        }
    )


class DynThreshUniPCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_a1111_stubs()
        cls.dynthres_unipc = importlib.import_module("dynthres_unipc")

    def test_unipc_before_sample_starts_schedule_at_zero(self):
        class Model:
            betas = torch.zeros(1)
            alphas_cumprod = torch.ones(1)
            parameterization = "eps"

            @staticmethod
            def apply_model(x, t, cond, **kwargs):
                return x

        dt = types.SimpleNamespace(step=None)
        sampler = self.dynthres_unipc.CustomUniPCSampler(Model())
        sampler.main_class = dt
        sampler.alphas_cumprod = Model.alphas_cumprod

        captured_before_sample = None
        test_case = self

        class FakeNoiseScheduleVP:
            def __init__(self, *args, **kwargs):
                self.total_N = 1000

        class FakeUniPC:
            def __init__(self, *args, before_sample=None, **kwargs):
                nonlocal captured_before_sample
                captured_before_sample = before_sample

            def sample(self, img, **kwargs):
                for expected_step in (0, 1, 2):
                    captured_before_sample(img, torch.zeros(img.shape[0]), None, None)
                    test_case.assertEqual(dt.step, expected_step)
                return img

        original_noise_schedule = getattr(self.dynthres_unipc.uni_pc.uni_pc, "NoiseScheduleVP", None)
        original_unipc = getattr(self.dynthres_unipc.uni_pc.uni_pc, "UniPC", None)
        self.dynthres_unipc.uni_pc.uni_pc.NoiseScheduleVP = FakeNoiseScheduleVP
        self.dynthres_unipc.uni_pc.uni_pc.UniPC = FakeUniPC
        try:
            sampler.sample(3, 1, (4, 2, 2), conditioning=torch.zeros(1, 1, 1))
        finally:
            if original_noise_schedule is None:
                del self.dynthres_unipc.uni_pc.uni_pc.NoiseScheduleVP
            else:
                self.dynthres_unipc.uni_pc.uni_pc.NoiseScheduleVP = original_noise_schedule
            if original_unipc is None:
                del self.dynthres_unipc.uni_pc.uni_pc.UniPC
            else:
                self.dynthres_unipc.uni_pc.uni_pc.UniPC = original_unipc


class DynamicThresholdingTests(unittest.TestCase):
    @staticmethod
    def _reference_experiment_mode3(actual_res, step, max_steps):
        coefs = torch.tensor(
            [
                [0.298, 0.207, 0.208, 0.0],
                [0.187, 0.286, 0.173, 0.0],
                [-0.158, 0.189, 0.264, 0.0],
                [-0.184, -0.271, -0.473, 1.0],
            ],
            device=actual_res.device,
            dtype=actual_res.dtype,
        )
        res_rgb = torch.einsum("laxy,ab -> lbxy", actual_res, coefs)
        max_r, max_g, max_b = (
            res_rgb[0][0].max(),
            res_rgb[0][1].max(),
            res_rgb[0][2].max(),
        )
        max_w = res_rgb[0][3].max()
        max_rgb = torch.maximum(max_r, torch.maximum(max_g, max_b))
        if step / max(max_steps - 1, 1) > 0.2:
            if bool((max_rgb < 2.0) & (max_w < 3.0)):
                res_rgb /= max_rgb.div(2.4).clamp_min(torch.finfo(actual_res.dtype).eps)
        else:
            if bool((max_rgb > 2.4) & (max_w > 3.0)):
                res_rgb /= max_rgb.div(2.4).clamp_min(torch.finfo(actual_res.dtype).eps)
        return torch.einsum("laxy,ab -> lbxy", res_rgb, coefs.inverse())

    def test_relative_path_preserves_dtype_and_finiteness(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            dt = DynThresh(
                7.0,
                1.0,
                "Constant",
                0.0,
                "Constant",
                0.0,
                4.0,
                0,
                10,
                True,
                "MEAN",
                "AD",
                1.0,
            )
            dt.step = 1
            uncond = torch.randn(2, 4, 8, 8, dtype=dtype)
            relative = torch.randn_like(uncond) * 0.1
            out = dt.dynthresh_from_relative(relative, uncond, 12.0)
            self.assertEqual(out.dtype, dtype)
            self.assertTrue(torch.isfinite(out.float()).all())

    def test_ragged_multicond_equivalence_with_manual_relative(self):
        dt = DynThresh(
            7.0,
            1.0,
            "Constant",
            0.0,
            "Constant",
            0.0,
            4.0,
            0,
            10,
            True,
            "MEAN",
            "AD",
            1.0,
        )
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

    def test_experiment_mode3_matches_reference_float32(self):
        torch.manual_seed(1234)
        base = DynThresh(
            7.0,
            1.0,
            "Constant",
            0.0,
            "Constant",
            0.0,
            4.0,
            0,
            10,
            True,
            "MEAN",
            "AD",
            1.0,
        )
        exp3 = DynThresh(
            7.0,
            1.0,
            "Constant",
            0.0,
            "Constant",
            0.0,
            4.0,
            3,
            10,
            True,
            "MEAN",
            "AD",
            1.0,
        )
        base.step = exp3.step = 3
        uncond = torch.randn(2, 4, 8, 8)
        relative = torch.randn_like(uncond) * 0.1

        expected = self._reference_experiment_mode3(
            base.dynthresh_from_relative(relative, uncond, 12.0), 3, 10
        )
        actual = exp3.dynthresh_from_relative(relative, uncond, 12.0)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_experiment_mode3_preserves_dtype_and_handles_single_step(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            dt = DynThresh(
                7.0,
                1.0,
                "Constant",
                0.0,
                "Constant",
                0.0,
                4.0,
                3,
                1,
                True,
                "MEAN",
                "AD",
                1.0,
            )
            dt.step = 0
            uncond = torch.randn(2, 4, 8, 8, dtype=dtype)
            relative = torch.randn_like(uncond) * 0.1
            out = dt.dynthresh_from_relative(relative, uncond, 12.0)
            self.assertEqual(out.dtype, dtype)
            self.assertEqual(out.shape, uncond.shape)
            self.assertTrue(torch.isfinite(out.float()).all())


class CFGCombinerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_a1111_stubs()
        cls.cfg_combiner = importlib.import_module("scripts.cfg_combiner")

    def test_no_pag_delegates_to_original(self):
        called = {"value": False}

        def original(x_out, conds_list, uncond, cond_scale):
            called["value"] = True
            return torch.full_like(x_out[-uncond.shape[0] :], cond_scale)

        x_out = torch.randn(3, 4, 4, 4)
        uncond = torch.randn(1, 77, 32)
        out = self.cfg_combiner.combine_denoised_pass_conds_list(
            x_out,
            [[(0, 1.0)]],
            uncond,
            7.5,
            original_func=original,
            cfg_dict={"pag_params": None},
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
            return torch.full_like(x_out[-uncond["crossattn"].shape[0] :], cond_scale)

        x_out = torch.randn(4, 4, 4, 4)
        uncond = {"crossattn": torch.randn(2, 77, 32), "vector": torch.randn(2, 1280)}
        out = self.cfg_combiner.combine_denoised_pass_conds_list(
            x_out,
            [[(0, 1.0)], [(1, 1.0)]],
            uncond,
            6.0,
            original_func=original,
            cfg_dict={"pag_params": Pag()},
        )
        self.assertEqual(tuple(out.shape), (2, 4, 4, 4))
        self.assertTrue(torch.equal(out, torch.full_like(out, 6.0)))

    def test_cfg_combiner_process_batch_skips_inactive_callback(self):
        callbacks = sys.modules["modules.script_callbacks"].callback_registry
        callbacks.clear()
        script = self.cfg_combiner.CFGCombinerScript()

        class Processing:
            def __init__(self):
                self.incant_cfg_params = {"pag_params": None}

        script.process_batch(Processing())
        self.assertEqual(callbacks, [])

    def test_cfg_combiner_process_batch_registers_when_pag_active(self):
        callbacks = sys.modules["modules.script_callbacks"].callback_registry
        callbacks.clear()
        script = self.cfg_combiner.CFGCombinerScript()

        class Processing:
            def __init__(self):
                self.incant_cfg_params = {"pag_params": object()}

        script.process_batch(Processing())
        self.assertEqual(len(callbacks), 1)
        script.remove_callbacks()
        self.assertEqual(callbacks, [])

    def test_cfg_combiner_wrapper_restores_only_own_wrapper(self):
        script = self.cfg_combiner.CFGCombinerScript()

        def original(x_out, conds_list, uncond, cond_scale):
            return torch.full_like(x_out[-uncond.shape[0] :], cond_scale)

        class Denoiser:
            combine_denoised = staticmethod(original)

        denoiser = Denoiser()
        cfg_dict = {
            "denoiser": None,
            "original_combine_denoised": None,
            "wrapped_combine_denoised": None,
            "pag_params": None,
        }
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


class PAGBatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_a1111_stubs()
        cls.pag = importlib.import_module("scripts.pag")

    def test_pag_extra_pass_splits_sdxl_cond_uncond_when_token_counts_differ(self):
        calls = []

        def inner_model(x_in, sigma_in, cond):
            calls.append(
                (
                    tuple(x_in.shape),
                    cond["crossattn"].shape[1],
                    tuple(cond["c_concat"][0].shape),
                )
            )
            return torch.ones_like(x_in) * len(calls)

        def make_condition_dict(c_crossattn, c_concat):
            return {**c_crossattn, "c_concat": [c_concat]}

        x_in = torch.zeros(3, 4, 2, 2)
        sigma_in = torch.zeros(3)
        image_cond = torch.zeros(3, 1, 2, 2)
        tensor = {"crossattn": torch.randn(2, 539, 8), "vector": torch.randn(2, 4)}
        uncond = {"crossattn": torch.randn(1, 462, 8), "vector": torch.randn(1, 4)}

        out = self.pag.pag_inner_model_x_out(
            inner_model,
            x_in,
            sigma_in,
            tensor,
            uncond,
            image_cond,
            make_condition_dict,
            1,
        )

        self.assertEqual(tuple(out.shape), tuple(x_in.shape))
        self.assertEqual(len(calls), 3)
        self.assertEqual([call[1] for call in calls], [539, 539, 462])

    def test_pag_extra_pass_concatenates_when_token_counts_match(self):
        calls = []

        def inner_model(x_in, sigma_in, cond):
            calls.append(cond["crossattn"].shape)
            return torch.ones_like(x_in)

        def make_condition_dict(c_crossattn, c_concat):
            return {**c_crossattn, "c_concat": [c_concat]}

        x_in = torch.zeros(3, 4, 2, 2)
        sigma_in = torch.zeros(3)
        image_cond = torch.zeros(3, 1, 2, 2)
        tensor = {"crossattn": torch.randn(2, 77, 8), "vector": torch.randn(2, 4)}
        uncond = {"crossattn": torch.randn(1, 77, 8), "vector": torch.randn(1, 4)}

        out = self.pag.pag_inner_model_x_out(
            inner_model,
            x_in,
            sigma_in,
            tensor,
            uncond,
            image_cond,
            make_condition_dict,
            1,
        )

        self.assertEqual(tuple(out.shape), tuple(x_in.shape))
        self.assertEqual(calls, [torch.Size([3, 77, 8])])

    def test_pag_extra_pass_split_allows_missing_image_conditioning(self):
        calls = []

        def inner_model(x_in, sigma_in, cond):
            calls.append((cond["crossattn"].shape[1], cond["c_concat"][0]))
            return torch.ones_like(x_in)

        def make_condition_dict(c_crossattn, c_concat):
            return {**c_crossattn, "c_concat": [c_concat]}

        x_in = torch.zeros(3, 4, 2, 2)
        sigma_in = torch.zeros(3)
        tensor = {"crossattn": torch.randn(2, 847, 8), "vector": torch.randn(2, 4)}
        uncond = {"crossattn": torch.randn(1, 770, 8), "vector": torch.randn(1, 4)}

        out = self.pag.pag_inner_model_x_out(
            inner_model, x_in, sigma_in, tensor, uncond, None, make_condition_dict, 1
        )

        self.assertEqual(tuple(out.shape), tuple(x_in.shape))
        self.assertEqual([call[0] for call in calls], [847, 847, 770])
        self.assertTrue(all(call[1] is None for call in calls))


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
