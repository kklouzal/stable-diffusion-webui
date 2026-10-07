import functools
import importlib
import importlib.util
import math
import sys
import types
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F

EXT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXT_ROOT.parents[1]
sys.path.insert(0, str(EXT_ROOT))

DynThresh = importlib.import_module("dynthres_core").DynThresh


def install_a1111_stubs():
    modules_pkg = types.ModuleType("modules")
    headless_ui_mod = types.ModuleType("modules.headless_ui")
    scripts_mod = types.ModuleType("modules.scripts")
    scripts_mod.Script = object
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

    def on_before_ui(callback, *args, **kwargs):
        callback_registry.append(callback)

    script_callbacks_mod.on_cfg_denoiser = on_cfg_denoiser
    script_callbacks_mod.on_cfg_denoised = on_cfg_denoised
    script_callbacks_mod.on_before_ui = on_before_ui
    script_callbacks_mod.remove_callbacks_for_function = remove_callbacks_for_function
    processing_mod = types.ModuleType("modules.processing")
    processing_mod.StableDiffusionProcessing = object
    shared_mod = types.ModuleType("modules.shared")
    shared_mod.device = torch.device("cpu")
    shared_mod.opts = types.SimpleNamespace(batch_cond_uncond=False)
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
    sd_samplers_mod = types.ModuleType("modules.sd_samplers")
    sd_samplers_mod.all_samplers_map = {}

    def create_sampler(name, sd_model):
        return types.SimpleNamespace(name=name, sd_model=sd_model)

    sd_samplers_mod.create_sampler = create_sampler
    sd_samplers_common_mod = types.ModuleType("modules.sd_samplers_common")

    class SamplerData:
        def __init__(self, name, constructor, aliases=None, options=None):
            self.name = name
            self.constructor = constructor
            self.aliases = aliases or []
            self.options = options or {}

    sd_samplers_common_mod.SamplerData = SamplerData
    sd_samplers_kdiffusion_mod = types.ModuleType("modules.sd_samplers_kdiffusion")

    class CFGDenoiserKDiffusion:
        def __init__(self, model):
            self.model = model

    sd_samplers_kdiffusion_mod.CFGDenoiserKDiffusion = CFGDenoiserKDiffusion

    # The main-pass row memo is real core code with no WebUI dependencies.
    row_memo_spec = importlib.util.spec_from_file_location("modules.sd_unet_row_memo", REPO_ROOT / "modules" / "sd_unet_row_memo.py")
    row_memo_mod = importlib.util.module_from_spec(row_memo_spec)
    row_memo_spec.loader.exec_module(row_memo_mod)

    modules_pkg.sd_unet_row_memo = row_memo_mod
    modules_pkg.scripts = scripts_mod
    modules_pkg.headless_ui = headless_ui_mod
    modules_pkg.script_callbacks = script_callbacks_mod
    modules_pkg.processing = processing_mod
    modules_pkg.shared = shared_mod
    sys.modules.update(
        {
            "modules": modules_pkg,
            "modules.headless_ui": headless_ui_mod,
            "modules.scripts": scripts_mod,
            "modules.script_callbacks": script_callbacks_mod,
            "modules.processing": processing_mod,
            "modules.shared": shared_mod,
            "modules.sd_samplers_cfg_denoiser": samplers_mod,
            "modules.sd_samplers": sd_samplers_mod,
            "modules.sd_samplers_common": sd_samplers_common_mod,
            "modules.sd_samplers_kdiffusion": sd_samplers_kdiffusion_mod,
            "modules.sd_unet_row_memo": row_memo_mod,
        }
    )


class DynamicThresholdingTests(unittest.TestCase):
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



    def test_std_variability_handles_single_spatial_sample_without_nan(self):
        dt = DynThresh(
            7.0, 1.0, "Constant", 0.0, "Constant", 0.0, 1.0, 1, False, "MEAN", "STD", 1.0
        )
        dt.step = 0
        uncond = torch.zeros(1, 4, 1, 1, dtype=torch.float32)
        relative = torch.ones_like(uncond) * 0.25

        out = dt.dynthresh_from_relative(relative, uncond, 9.0)

        self.assertEqual(out.shape, uncond.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_ragged_multicond_equivalence_with_manual_relative(self):
        dt = DynThresh(
            7.0,
            1.0,
            "Constant",
            0.0,
            "Constant",
            0.0,
            4.0,
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

    def test_dynthresh_rejects_ragged_batch_ratio_without_assert(self):
        dt = DynThresh(
            7.0,
            1.0,
            "Constant",
            0.0,
            "Constant",
            0.0,
            4.0,
            10,
            True,
            "MEAN",
            "AD",
            1.0,
        )
        dt.step = 1
        cond = torch.randn(3, 4, 4, 4)
        uncond = torch.randn(2, 4, 4, 4)

        with self.assertRaisesRegex(ValueError, "constant across batches"):
            dt.dynthresh(cond, uncond, 9.0, None)


class DynamicThresholdingLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_a1111_stubs()
        cls.dynamic_thresholding = importlib.import_module("scripts.dynamic_thresholding")

    def setUp(self):
        self.sd_samplers = sys.modules["modules.sd_samplers"]
        self.sd_samplers.all_samplers_map.clear()

        def constructor(model):
            return types.SimpleNamespace(model_wrap_cfg=types.SimpleNamespace(inner_model=model))

        sampler_data = sys.modules["modules.sd_samplers_common"].SamplerData(
            "Euler",
            constructor,
            [],
            {},
        )
        self.sd_samplers.all_samplers_map["Euler"] = sampler_data

    def test_disabled_next_batch_restores_sampler_object_after_failed_cleanup(self):
        script = self.dynamic_thresholding.Script()
        p = types.SimpleNamespace(
            sampler_name="Euler",
            sampler=types.SimpleNamespace(name="Euler"),
            sd_model=object(),
            steps=4,
            enable_hr=False,
            extra_generation_params={},
        )

        script.process_batch(
            p,
            True,
            7.0,
            100.0,
            "Constant",
            0.0,
            "Constant",
            0.0,
            4.0,
            True,
            "MEAN",
            "AD",
            1.0,
            0,
            [],
            [],
            [],
        )
        self.assertNotEqual(p.sampler_name, "Euler")
        self.assertTrue(p.sampler_name.startswith("Euler_dynthres"))
        self.assertIn(p.sampler_name, self.sd_samplers.all_samplers_map)

        p.dynthres_enabled = False
        script.process_batch(
            p,
            False,
            7.0,
            100.0,
            "Constant",
            0.0,
            "Constant",
            0.0,
            4.0,
            True,
            "MEAN",
            "AD",
            1.0,
            1,
            [],
            [],
            [],
        )

        self.assertEqual(p.sampler_name, "Euler")
        self.assertEqual(p.sampler.name, "Euler")
        self.assertFalse(any(name.startswith("Euler_dynthres") for name in self.sd_samplers.all_samplers_map))
        self.assertFalse(hasattr(p, "orig_sampler_name"))


class CFGCombinerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_a1111_stubs()
        # Rebind to the stubs installed above even if another test class imported the module first.
        sys.modules.pop("scripts.cfg_combiner", None)
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

    def pag_params(self, pag_x_out):
        return types.SimpleNamespace(
            pag_active=True,
            pag_x_out=pag_x_out,
            pag_scale=2.0,
            pag_start_step=0,
            pag_end_step=10,
            step=1,
            pag_sanf=False,
            cfg_interval_enable=False,
            cfg_interval_scheduled_value=7.0,
        )

    def test_pag_guidance_reads_cond_rows_of_pag_output(self):
        def original(x_out, conds_list, uncond, cond_scale):
            return torch.zeros_like(x_out[-uncond.shape[0] :])

        # AND prompt: image 0 has cond rows 0 and 1, image 1 has cond row 2; rows 3-4 are uncond.
        x_out = torch.randn(5, 4, 2, 2)
        pag_x_out = torch.randn(3, 4, 2, 2)
        conds_list = [[(0, 0.5), (1, 0.5)], [(2, 1.0)]]
        out = self.cfg_combiner.combine_denoised_pass_conds_list(
            x_out, conds_list, torch.zeros(2, 77, 8), 6.0,
            original_func=original, cfg_dict={"pag_params": self.pag_params(pag_x_out)},
        )
        expected0 = (x_out[0] - pag_x_out[0]) * 1.0 + (x_out[1] - pag_x_out[1]) * 1.0
        torch.testing.assert_close(out[0], expected0)
        torch.testing.assert_close(out[1], (x_out[2] - pag_x_out[2]) * 2.0)

    def test_pag_guidance_fails_fast_on_wrong_pag_row_count(self):
        def original(x_out, conds_list, uncond, cond_scale):
            return torch.zeros_like(x_out[-uncond.shape[0] :])

        x_out = torch.randn(4, 4, 2, 2)
        with self.assertRaisesRegex(RuntimeError, "expected the 2 cond rows"):
            self.cfg_combiner.combine_denoised_pass_conds_list(
                x_out, [[(0, 1.0)], [(1, 1.0)]], torch.zeros(2, 77, 8), 6.0,
                original_func=original, cfg_dict={"pag_params": self.pag_params(torch.randn(4, 4, 2, 2))},
            )

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

    def test_sanf_blur_handles_single_pixel_spatial_side(self):
        img = torch.randn(2, 4, 1, 9)
        out = self.cfg_combiner.sanf_gaussian_blur3(img)
        self.assertEqual(tuple(out.shape), tuple(img.shape))
        self.assertTrue(torch.isfinite(out).all())

    def test_sanf_guidance_blend_matches_stack_argmax_and_prefers_cfg_on_tie(self):
        cfg_x = torch.tensor(
            [
                [[1.0, -2.0], [3.0, -4.0]],
                [[-0.5, 0.25], [-0.75, 1.5]],
            ]
        )
        pag_x = torch.tensor(
            [
                [[0.25, -4.0], [1.0, -8.0]],
                [[-1.0, 0.5], [-2.0, 0.25]],
            ]
        )
        omega_rt = self.cfg_combiner.sanf_gaussian_blur3(cfg_x.abs()).float()
        omega_rs = self.cfg_combiner.sanf_gaussian_blur3(pag_x.abs()).float()
        soft_rt = torch.softmax(omega_rt, dim=0)
        soft_rs = torch.softmax(omega_rs, dim=0)
        _, argmax_indices = torch.max(torch.stack([soft_rt, soft_rs], dim=0), dim=0)
        expected = cfg_x * (argmax_indices == 0) + pag_x * (argmax_indices == 1)

        torch.testing.assert_close(self.cfg_combiner._sanf_guidance_blend(cfg_x, pag_x), expected)
        torch.testing.assert_close(self.cfg_combiner._sanf_guidance_blend(cfg_x, -cfg_x), cfg_x)

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
        sys.modules.pop("scripts.pag", None)
        cls.pag = importlib.import_module("scripts.pag")
        cls.row_memo = sys.modules["modules.sd_unet_row_memo"]

    def setUp(self):
        shared = sys.modules["modules.shared"]
        shared.opts.batch_cond_uncond = False
        shared.opts.pad_cond_uncond = False
        shared.opts.pad_cond_uncond_v0 = False
        shared.sd_model = types.SimpleNamespace(
            cond_stage_model_empty_prompt=torch.zeros(1, 77, 8))

    def record_main_pass(self, n_cond, chunks, rows=None, row_subset_ok=None):
        """Record main-pass calls the way CFGDenoiser makes them: ordered row slices of one x_in."""
        rows = rows if rows is not None else sum(chunks)
        x_in = torch.randn(rows, 4, 2, 2)
        sigma_in = torch.rand(rows)
        cond_in = {"crossattn": torch.randn(rows, 77, 8), "vector": torch.randn(rows, 4), "c_concat": [torch.randn(rows, 1, 2, 2)]}
        denoiser = types.SimpleNamespace(run_inner_model=lambda x, sigma, cond: x)
        memo = self.row_memo.arm(denoiser, n_cond)
        start = 0
        for index, chunk in enumerate(chunks):
            end = start + chunk
            call_cond = {"crossattn": cond_in["crossattn"][start:end], "vector": cond_in["vector"][start:end], "c_concat": [cond_in["c_concat"][0][start:end]]}
            denoiser.run_inner_model(x_in[start:end], sigma_in[start:end], call_cond)
            if row_subset_ok is not None:
                memo.calls[index].row_subset_ok = row_subset_ok[index]
            start = end
        self.assertIs(self.row_memo.disarm(denoiser), memo)
        return memo, x_in, sigma_in, cond_in

    def replay(self, memo, preserve_call_sequence=False):
        calls = []

        def inner_model(x, sigma, cond):
            calls.append((x, sigma, cond))
            return x + 1

        out = self.pag.pag_cond_rows_x_out(inner_model, memo, preserve_call_sequence)
        return out, calls

    def assert_call_rows(self, call, x_in, sigma_in, cond_in, start, end):
        x, sigma, cond = call
        self.assertTrue(torch.equal(x, x_in[start:end]))
        self.assertEqual(x.data_ptr(), x_in[start:end].data_ptr())
        self.assertTrue(torch.equal(sigma, sigma_in[start:end]))
        self.assertTrue(torch.equal(cond["crossattn"], cond_in["crossattn"][start:end]))
        self.assertTrue(torch.equal(cond["vector"], cond_in["vector"][start:end]))
        self.assertTrue(torch.equal(cond["c_concat"][0], cond_in["c_concat"][0][start:end]))

    def test_pag_replays_only_cond_rows_of_batched_main_call(self):
        # batch_cond_uncond with matching (or padded) tokens: one [cond, uncond] call.
        memo, x_in, sigma_in, cond_in = self.record_main_pass(n_cond=2, chunks=[3])
        out, calls = self.replay(memo)
        self.assertEqual(len(calls), 1)
        self.assert_call_rows(calls[0], x_in, sigma_in, cond_in, 0, 2)
        torch.testing.assert_close(out, x_in[:2] + 1)

    def test_pag_mirrors_split_main_pass_and_skips_uncond_calls(self):
        # Token mismatch without padding (AND prompt, batch_size 1): cond chunk of 2, then 1, then uncond.
        memo, x_in, sigma_in, cond_in = self.record_main_pass(n_cond=3, chunks=[2, 1, 1])
        out, calls = self.replay(memo)
        self.assertEqual(len(calls), 2)
        self.assert_call_rows(calls[0], x_in, sigma_in, cond_in, 0, 2)
        self.assert_call_rows(calls[1], x_in, sigma_in, cond_in, 2, 3)
        torch.testing.assert_close(out, x_in[:3] + 1)

    def test_pag_mirrors_batching_disabled_chunks(self):
        # batch_cond_uncond off, batch_size 2: chunks [c0 c1] [u0 u1].
        memo, x_in, sigma_in, cond_in = self.record_main_pass(n_cond=2, chunks=[2, 2])
        out, calls = self.replay(memo)
        self.assertEqual(len(calls), 1)
        self.assert_call_rows(calls[0], x_in, sigma_in, cond_in, 0, 2)
        torch.testing.assert_close(out, x_in[:2] + 1)

    def test_pag_replays_chunks_that_straddle_the_cond_boundary(self):
        # batch_cond_uncond off, batch_size 2, AND prompt: chunks [c0 c1] [c2 u0] [u1].
        memo, x_in, sigma_in, cond_in = self.record_main_pass(n_cond=3, chunks=[2, 2, 1])
        out, calls = self.replay(memo)
        self.assertEqual(len(calls), 2)
        self.assert_call_rows(calls[0], x_in, sigma_in, cond_in, 0, 2)
        self.assert_call_rows(calls[1], x_in, sigma_in, cond_in, 2, 3)
        torch.testing.assert_close(out, x_in[:3] + 1)

    def test_pag_handles_skip_uncond_main_pass(self):
        memo, x_in, sigma_in, cond_in = self.record_main_pass(n_cond=2, chunks=[2])
        out, calls = self.replay(memo)
        self.assertEqual(len(calls), 1)
        self.assert_call_rows(calls[0], x_in, sigma_in, cond_in, 0, 2)
        torch.testing.assert_close(out, x_in[:2] + 1)

    def test_pag_replays_row_coupled_calls_whole(self):
        memo, x_in, sigma_in, cond_in = self.record_main_pass(n_cond=1, chunks=[2, 2], row_subset_ok=[False, False])
        out, calls = self.replay(memo)
        self.assertEqual(len(calls), 2)
        self.assert_call_rows(calls[0], x_in, sigma_in, cond_in, 0, 2)
        self.assert_call_rows(calls[1], x_in, sigma_in, cond_in, 2, 4)
        torch.testing.assert_close(out, x_in[:1] + 1)

    def test_pag_preserves_call_sequence_when_requested(self):
        memo, x_in, sigma_in, cond_in = self.record_main_pass(n_cond=1, chunks=[1, 1])
        out, calls = self.replay(memo, preserve_call_sequence=True)
        self.assertEqual(len(calls), 2)
        self.assert_call_rows(calls[0], x_in, sigma_in, cond_in, 0, 1)
        self.assert_call_rows(calls[1], x_in, sigma_in, cond_in, 1, 2)
        torch.testing.assert_close(out, x_in[:1] + 1)

    def test_pag_replays_whole_calls_when_requested(self):
        memo, x_in, sigma_in, cond_in = self.record_main_pass(n_cond=1, chunks=[2, 1])
        calls = []

        def inner_model(x, sigma, cond):
            calls.append((x, sigma, cond))
            return x + 1

        out = self.pag.pag_cond_rows_x_out(inner_model, memo, False, whole_calls=True)
        self.assertEqual(len(calls), 2)
        self.assert_call_rows(calls[0], x_in, sigma_in, cond_in, 0, 2)
        self.assert_call_rows(calls[1], x_in, sigma_in, cond_in, 2, 3)
        torch.testing.assert_close(out, x_in[:1] + 1)

    def test_pag_fails_fast_when_main_pass_does_not_cover_cond_rows(self):
        memo, *_ = self.record_main_pass(n_cond=3, chunks=[2], rows=2)
        with self.assertRaisesRegex(RuntimeError, "covers 2 of 3 cond rows"):
            self.replay(memo)

    def test_pag_callbacks_record_main_pass_and_restore_hook_state(self):
        script = self.pag.PAGExtensionScript()
        pag_params = self.pag.PAGStateParams()
        pag_params.pag_active = True
        pag_params.pag_scale = 3.0
        pag_params.pag_start_step = 0
        pag_params.pag_end_step = 10
        pag_params.noise_levels = [1.0] * 4
        attn = types.SimpleNamespace(pag_enable=False)
        to_q = types.SimpleNamespace(seg_enable=True)
        pag_params.crossattn_modules = [attn]
        pag_params.seg_q_modules = [to_q]

        class Denoiser(torch.nn.Module):
            def run_inner_model(self, x, sigma, cond):
                return x * 0

        denoiser = Denoiser()
        x_in = torch.randn(2, 4, 2, 2)
        cond = {"crossattn": torch.randn(1, 77, 8), "vector": torch.randn(1, 4)}
        params = types.SimpleNamespace(sampling_step=1, denoiser=denoiser, text_cond=cond)
        script.on_cfg_denoiser_callback(params, pag_params)
        denoiser.run_inner_model(x_in, torch.ones(2), {"crossattn": torch.randn(2, 77, 8)})

        seen = {}

        def inner_model(x, sigma, cond):
            seen["state"] = (attn.pag_enable, to_q.seg_enable, x.shape[0])
            return x - 1

        script.on_cfg_denoised_callback(types.SimpleNamespace(sampling_step=1, inner_model=inner_model), pag_params)
        self.assertEqual(seen["state"], (True, False, 1))
        torch.testing.assert_close(pag_params.pag_x_out, x_in[:1] - 1)
        self.assertFalse(attn.pag_enable)
        self.assertTrue(to_q.seg_enable)
        self.assertIsNone(self.row_memo.disarm(denoiser))

        # A failing PAG pass still restores hook state and releases the memo.
        script.on_cfg_denoiser_callback(params, pag_params)
        denoiser.run_inner_model(x_in, torch.ones(2), {"crossattn": torch.randn(2, 77, 8)})

        def failing_inner_model(x, sigma, cond):
            raise ValueError("unet failed")

        with self.assertRaisesRegex(ValueError, "unet failed"):
            script.on_cfg_denoised_callback(types.SimpleNamespace(sampling_step=1, inner_model=failing_inner_model), pag_params)
        self.assertFalse(attn.pag_enable)
        self.assertTrue(to_q.seg_enable)
        self.assertIsNone(self.row_memo.disarm(denoiser))

        # Outside the PAG interval nothing is recorded and the recorder passes through.
        script.on_cfg_denoiser_callback(types.SimpleNamespace(sampling_step=20, denoiser=denoiser, text_cond=cond), pag_params)
        self.assertIsNone(self.row_memo.disarm(denoiser))

        script.postprocess_batch(types.SimpleNamespace(incant_cfg_params={"pag_params": None}))
        self.assertNotIn("run_inner_model", denoiser.__dict__)

    def test_pag_denoised_callback_fails_fast_without_recorded_main_pass(self):
        script = self.pag.PAGExtensionScript()
        pag_params = self.pag.PAGStateParams()
        pag_params.pag_active = True
        pag_params.pag_scale = 3.0
        with self.assertRaisesRegex(RuntimeError, "not recorded"):
            script.on_cfg_denoised_callback(types.SimpleNamespace(sampling_step=1, inner_model=None), pag_params)

    def test_pag_noise_helpers_handle_degenerate_step_counts(self):
        self.assertEqual(self.pag.calculate_noise_level(0, 0), 0.0)
        self.assertEqual(self.pag.calculate_noise_level(-1, 4), 80.0)
        self.assertEqual(self.pag.calculate_noise_level(4, 4), 0.0)
        self.assertEqual(self.pag.find_closest_index(1.0, 0), 0)

    def test_pag_find_closest_index_honors_custom_noise_curve(self):
        target = self.pag.calculate_noise_level(2, 5, sigma_min=0.01, sigma_max=10.0, rho=5)

        self.assertEqual(
            self.pag.find_closest_index(target, 5, sigma_min=0.01, sigma_max=10.0, rho=5),
            2,
        )

    def test_cfg_schedulers_handle_zero_steps_and_clamp_progress(self):
        for schedule in self.pag._CFG_SCHEDULE_DISPATCH:
            self.assertTrue(math.isfinite(self.pag.cfg_scheduler(schedule, 0, 0, 8.0)))
        self.assertEqual(self.pag.cfg_scheduler("Linear", 0, 0, 8.0), 8.0)
        self.assertEqual(self.pag.linear_schedule(-1, 4, 8.0), 16.0)
        self.assertEqual(self.pag.invlinear_schedule(5, 4, 8.0), 16.0)

    def test_cfg_interval_registers_without_pag_attention_modules(self):
        callbacks = sys.modules["modules.script_callbacks"].callback_registry
        callbacks.clear()
        script = self.pag.PAGExtensionScript()

        def fail_if_called():
            raise AssertionError("CFG interval should not need PAG attention modules")

        script.get_cross_attn_modules = fail_if_called

        class Processing:
            def __init__(self):
                self.incant_cfg_params = {"pag_params": None}
                self.steps = 4
                self.cfg_scale = 8.0
                self.batch_size = 1
                self.extra_generation_params = {}

        p = Processing()
        script.pag_process_batch(
            p,
            False,
            0.0,
            0,
            150,
            True,
            "Linear",
            0.0,
            100.0,
            False,
        )

        self.assertEqual(len(callbacks), 1)
        pag_params = p.incant_cfg_params["pag_params"]
        params = types.SimpleNamespace(sampling_step=1)
        callbacks[0](params)
        self.assertEqual(
            pag_params.cfg_interval_scheduled_value,
            self.pag.cfg_scheduler("Linear", 1, 4, 8.0),
        )
        script.remove_callbacks()
        self.assertEqual(callbacks, [])

    def test_inactive_pag_batch_clears_previous_callbacks(self):
        callbacks = sys.modules["modules.script_callbacks"].callback_registry
        callbacks.clear()
        script = self.pag.PAGExtensionScript()

        class Processing:
            def __init__(self):
                self.incant_cfg_params = {"pag_params": None}
                self.steps = 4
                self.cfg_scale = 8.0
                self.batch_size = 1
                self.extra_generation_params = {}

        p = Processing()
        script.pag_process_batch(
            p,
            False,
            0.0,
            0,
            150,
            True,
            "Linear",
            0.0,
            100.0,
            False,
        )

        self.assertEqual(len(callbacks), 1)
        self.assertIsNotNone(script._cfg_denoiser_callback)

        script.pag_process_batch(
            p,
            False,
            0.0,
            0,
            150,
            False,
            "Linear",
            0.0,
            100.0,
            False,
        )

        self.assertEqual(callbacks, [])
        self.assertIsNone(script._cfg_denoiser_callback)

    def test_pag_without_attention_modules_does_not_leave_inactive_combiner_state(self):
        callbacks = sys.modules["modules.script_callbacks"].callback_registry
        callbacks.clear()
        script = self.pag.PAGExtensionScript()
        script.get_cross_attn_modules = lambda: []

        class Processing:
            def __init__(self):
                self.incant_cfg_params = {"pag_params": None}
                self.steps = 4
                self.cfg_scale = 8.0
                self.batch_size = 1
                self.extra_generation_params = {}

        p = Processing()
        script.pag_process_batch(
            p,
            True,
            3.0,
            0,
            150,
            False,
            "Constant",
            0.0,
            100.0,
            False,
        )

        self.assertEqual(callbacks, [])
        self.assertIsNone(p.incant_cfg_params["pag_params"])


def _legacy_gaussian_blur_2d(img, kernel_size, sigma):
    # Pre-separable production SEG blur (reflect pad + k x k depthwise conv of
    # query-dtype outer-product taps), verbatim minus its kernel cache. Kept as
    # an independent oracle for gaussian_blur_queries.
    min_spatial = min(img.shape[-2:])
    max_reflect_kernel = min_spatial - (min_spatial % 2 - 1)
    kernel_size = min(kernel_size, max_reflect_kernel)
    channels = img.shape[-3]
    ksize_half = (kernel_size - 1) * 0.5
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size, device=img.device, dtype=torch.float32)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    x_kernel = (pdf / pdf.sum()).to(dtype=img.dtype)
    base_kernel = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = base_kernel.expand(channels, 1, base_kernel.shape[0], base_kernel.shape[1]).contiguous()

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]
    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=channels)

    return img


def _legacy_blur_seg_cond_queries(output, *, heads, head_dim, downscale_h, downscale_w, blur_fn):
    # Pre-separable query path: blur_fn sees the (half*heads, head_dim, H, W) layout.
    output_batch = output.shape[0]
    half_batch = output_batch // 2
    seq_len = downscale_h * downscale_w
    q_passthrough, q_blur = output.split(half_batch, dim=0)
    q_blur = q_blur.view(half_batch, -1, heads, head_dim).transpose(1, 2)
    q_blur = q_blur.permute(0, 1, 3, 2).reshape(
        half_batch * heads, head_dim, downscale_h, downscale_w
    )
    q_blur = blur_fn(q_blur)
    q_blur = q_blur.reshape(half_batch, heads, head_dim, seq_len)
    q_blur = q_blur.view(half_batch, heads * head_dim, seq_len).transpose(1, 2)
    return torch.cat((q_passthrough, q_blur), dim=0)


class SEGBlurTests(unittest.TestCase):
    # (H, W, kernel_size, sigma). The kernel clamps to the shorter side as in
    # production: 49 -> 41 at 40x40, 25 at 40x24, 7 at 9x7, 3 at 2x9 and 3x3.
    BLUR_CASES = (
        (40, 40, 49, 8.0),
        (32, 32, 33, 5.5),
        (40, 24, 49, 8.0),
        (24, 40, 49, 8.0),
        (9, 7, 49, 8.0),
        (2, 9, 13, 2.0),
        (3, 3, 49, 8.0),
        (5, 6, 1, 2.0),
    )
    # (half_batch, heads, head_dim)
    QUERY_SHAPES = ((1, 2, 3), (2, 1, 4), (2, 5, 2))

    @classmethod
    def setUpClass(cls):
        install_a1111_stubs()
        cls.seg = importlib.import_module("scripts.smoothed_energy_guidance")

    def _check_against_legacy(self, dtype, check):
        gen = torch.Generator().manual_seed(1234)
        for h, w, kernel_size, sigma in self.BLUR_CASES:
            for half, heads, head_dim in self.QUERY_SHAPES:
                output = torch.randn(2 * half, h * w, heads * head_dim, generator=gen).to(dtype)
                geometry = dict(heads=heads, head_dim=head_dim, downscale_h=h, downscale_w=w)
                new = self.seg._blur_seg_cond_queries(output, kernel_size=kernel_size, sigma=sigma, is_inf_blur=False, **geometry)
                blur_fn = functools.partial(_legacy_gaussian_blur_2d, kernel_size=kernel_size, sigma=sigma)
                old = _legacy_blur_seg_cond_queries(output, blur_fn=blur_fn, **geometry)
                with self.subTest(h=h, w=w, kernel_size=kernel_size, half=half, heads=heads, head_dim=head_dim):
                    self.assertEqual(new.shape, output.shape)
                    self.assertEqual(new.dtype, dtype)
                    self.assertTrue(torch.equal(new[:half], output[:half]))
                    check(output, new, old, geometry, blur_fn)

    def test_separable_blur_matches_legacy_conv_fp32(self):
        def check(output, new, old, geometry, blur_fn):
            torch.testing.assert_close(new, old, rtol=1e-5, atol=1e-5)

        self._check_against_legacy(torch.float32, check)

    def test_separable_blur_matches_legacy_conv_bf16(self):
        def check(output, new, old, geometry, blur_fn):
            torch.testing.assert_close(new.float(), old.float(), rtol=2e-2, atol=2e-2)
            # fp32 taps and accumulation: one bf16 rounding of the exact blur.
            exact = _legacy_blur_seg_cond_queries(output.float(), blur_fn=blur_fn, **geometry)
            torch.testing.assert_close(new.float(), exact, rtol=2**-8, atol=1e-5)

        self._check_against_legacy(torch.bfloat16, check)

    def test_separable_blur_keeps_fp32_math_under_autocast(self):
        # The hook runs under bf16 autocast, which would otherwise round the
        # fp32 operators and the intermediate of the matmuls to bf16.
        q = torch.randn(2, 40 * 24, 6).to(torch.bfloat16)
        expected = self.seg.gaussian_blur_queries(q, 40, 24, kernel_size=49, sigma=8.0)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = self.seg.gaussian_blur_queries(q, 40, 24, kernel_size=49, sigma=8.0)
        self.assertTrue(torch.equal(out, expected))

    def test_gaussian_blur_clamps_kernel_to_shorter_spatial_side(self):
        q = torch.randn(2, 2 * 9, 3)
        out = self.seg.gaussian_blur_queries(q, 2, 9, kernel_size=13, sigma=2.0)
        self.assertEqual(tuple(out.shape), tuple(q.shape))
        self.assertTrue(torch.isfinite(out).all())
        # A 2-pixel side admits at most a 3-tap reflect kernel, on both axes.
        self.assertTrue(torch.equal(out, self.seg.gaussian_blur_queries(q, 2, 9, kernel_size=3, sigma=2.0)))

    def test_blur_operator_rows_sum_to_one(self):
        for n, kernel_size in ((40, 41), (24, 25), (7, 7), (2, 3), (1, 1)):
            operator = self.seg._gaussian_blur_operator(n, kernel_size, 8.0, torch.device("cpu"))
            with self.subTest(n=n, kernel_size=kernel_size):
                self.assertEqual(operator.dtype, torch.float32)
                self.assertEqual(tuple(operator.shape), (n, n))
                self.assertTrue((operator >= 0).all())
                torch.testing.assert_close(operator.sum(dim=1), torch.ones(n), rtol=0, atol=1e-6)

    def test_seg_query_blur_mutates_legacy_tail_half_only(self):
        output = torch.randn(4, 6 * 5, 2 * 3, dtype=torch.bfloat16)
        geometry = dict(heads=2, head_dim=3, downscale_h=6, downscale_w=5)

        for is_inf_blur in (False, True):
            out = self.seg._blur_seg_cond_queries(output, kernel_size=49, sigma=8.0, is_inf_blur=is_inf_blur, **geometry)
            with self.subTest(is_inf_blur=is_inf_blur):
                self.assertEqual(out.dtype, output.dtype)
                self.assertTrue(torch.equal(out[:2], output[:2]))
                self.assertFalse(torch.equal(out[2:], output[2:]))

    def test_infinite_blur_matches_legacy_path_bitwise(self):
        for dtype in (torch.float32, torch.bfloat16):
            output = torch.randn(4, 6 * 5, 2 * 3).to(dtype)
            geometry = dict(heads=2, head_dim=3, downscale_h=6, downscale_w=5)
            new = self.seg._blur_seg_cond_queries(output, kernel_size=49, sigma=2.0**11, is_inf_blur=True, **geometry)
            old = _legacy_blur_seg_cond_queries(
                output,
                blur_fn=lambda q: self.seg.gaussian_blur_inf(q, 1.0, 2.0**11),
                **geometry,
            )
            with self.subTest(dtype=dtype):
                self.assertTrue(torch.equal(new, old))


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


class SharedHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_a1111_stubs()
        cls.timing = importlib.import_module("scripts.incant_utils.timing")
        cls.ui_wrapper = importlib.import_module("scripts.ui_wrapper")

    def test_hook_timings_fold_into_processing_record(self):
        p = types.SimpleNamespace()
        hooks = {}
        self.timing.record(hooks, "cfg_denoiser_callback", 0.25)
        self.timing.record(hooks, "cfg_denoiser_callback", 0.5)
        self.timing.record(hooks.setdefault("details", {}), "pag_hidden_denoise", 0.125)
        self.timing.merge_into_processing(p, "Incantations.PAGExtensionScript", {"process": {"total_seconds": 1.0, "calls": 1}})
        self.timing.merge_into_processing(p, "Incantations.PAGExtensionScript", hooks)

        self.assertEqual(p.openclaw_extension_timings, {
            "total_seconds": 1.75,
            "extensions": {
                "Incantations.PAGExtensionScript": {
                    "total_seconds": 1.75,
                    "calls": 3,
                    "hooks": {"process": 1.0, "cfg_denoiser_callback": 0.75},
                    "details": {"pag_hidden_denoise": {"total_seconds": 0.125, "calls": 1}},
                },
            },
        })

    def test_xyz_field_setter_enables_feature_only_when_unset(self):
        setter = self.ui_wrapper.xyz_field_setter
        p = types.SimpleNamespace()
        setter("cfg_interval_schedule", "pag_active", also_enable="cfg_interval_enable")(p, "Linear", [])
        self.assertEqual(vars(p), {"cfg_interval_schedule": "Linear", "pag_active": True, "cfg_interval_enable": True})

        p = types.SimpleNamespace(pag_active=False)
        setter("pag_sanf", "pag_active", boolean=True)(p, "True", [])
        self.assertEqual(vars(p), {"pag_active": False, "pag_sanf": True})


if __name__ == "__main__":
    unittest.main()
