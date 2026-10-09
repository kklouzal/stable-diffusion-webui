import collections
import functools
import importlib
import importlib.util
import itertools
import math
import sys
import types
from pathlib import Path
import unittest
import unittest.mock

import torch
from torch.nn import functional as F

EXT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXT_ROOT.parents[1]
sys.path.insert(0, str(EXT_ROOT))

dynthres_core = importlib.import_module("dynthres_core")
DynThresh = dynthres_core.DynThresh

# Package roots install_a1111_stubs() replaces; restored when this file finishes so later files see their own.
_STUBBED_PACKAGES = ("modules", "scripts")
_saved_modules = {}


def setUpModule():
    _saved_modules.update({key: value for key, value in sys.modules.items() if key.split(".")[0] in _STUBBED_PACKAGES})


def tearDownModule():
    for key in [key for key in sys.modules if key.split(".")[0] in _STUBBED_PACKAGES]:
        del sys.modules[key]
    sys.modules.update(_saved_modules)
    _saved_modules.clear()


def install_a1111_stubs():
    # The extension's modules bind these stubs when imported: drop the ones imported against an earlier install.
    for name in [name for name in sys.modules if name == "scripts" or name.startswith("scripts.")]:
        del sys.modules[name]
    modules_pkg = types.ModuleType("modules")
    # The real inert component surface (no WebUI dependencies): ui() defaults are what the API's script args use.
    headless_ui_spec = importlib.util.spec_from_file_location("modules.headless_ui", REPO_ROOT / "modules" / "headless_ui.py")
    headless_ui_mod = importlib.util.module_from_spec(headless_ui_spec)
    sys.modules["modules.headless_ui"] = headless_ui_mod  # it registers itself as "gradio" while loading
    headless_ui_spec.loader.exec_module(headless_ui_mod)
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
    shared_mod.opts = types.SimpleNamespace(batch_cond_uncond=False)
    sd_samplers_mod = types.ModuleType("modules.sd_samplers")
    sd_samplers_mod.all_samplers_map = {}

    def create_sampler(name, sd_model):
        return types.SimpleNamespace(name=name, sd_model=sd_model)

    sd_samplers_mod.create_sampler = create_sampler
    sd_samplers_common_mod = types.ModuleType("modules.sd_samplers_common")

    # modules/sd_samplers_common.py's SamplerData, verbatim.
    SamplerDataTuple = collections.namedtuple('SamplerData', ['name', 'constructor', 'aliases', 'options'])

    class SamplerData(SamplerDataTuple):
        def total_steps(self, steps):
            if self.options.get("second_order", False):
                steps = steps * 2

            return steps

    sd_samplers_common_mod.SamplerData = SamplerData
    # The timestep sampler registry, as modules/sd_samplers_timesteps.py declares it.
    sd_samplers_timesteps_mod = types.ModuleType("modules.sd_samplers_timesteps")
    sd_samplers_timesteps_mod.samplers_data_timesteps = [
        SamplerData(name, None, [], {}) for name in ("DDIM", "DDIM CFG++", "PLMS", "UniPC")
    ]
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
            "modules.sd_samplers": sd_samplers_mod,
            "modules.sd_samplers_common": sd_samplers_common_mod,
            "modules.sd_samplers_timesteps": sd_samplers_timesteps_mod,
            "modules.sd_samplers_kdiffusion": sd_samplers_kdiffusion_mod,
            "modules.sd_unet_row_memo": row_memo_mod,
        }
    )
    # The CUDA-graph module's instance_overrides is the shared monkeypatch test; it and its imports are real core
    # code with no WebUI dependencies.
    for name in ("openclaw_env", "openclaw_cache_epochs", "openclaw_cuda_graphs"):
        spec = importlib.util.spec_from_file_location(f"modules.{name}", REPO_ROOT / "modules" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"modules.{name}"] = mod
        setattr(modules_pkg, name, mod)
        spec.loader.exec_module(mod)


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

    @staticmethod
    def _bits(tensor):
        return tensor.contiguous().view({2: torch.int16, 4: torch.int32, 8: torch.int64}[tensor.element_size()])

    def test_cfg_relative_fast_path_is_bitwise_the_accumulation(self):
        # Signed zeros and equal cond/uncond values are where 0 + d and d could differ.
        for dtype in (torch.float32, torch.bfloat16):
            x_out = torch.randn(4, 4, 6, 6).to(dtype)
            x_out.view(-1)[:20] = 0.0
            x_out[0].view(-1)[:6] = -0.0
            x_out[2:].view(-1)[30:40] = x_out[:2].reshape(-1)[30:40]
            uncond = x_out[2:]
            for conds_list in ([[(0, 1.0)], [(1, 1.0)]], [[(0, 0.5)], [(1, 1.0)]], [[(1, 1.0)], [(0, 1.0)]], [[(0, 1.0), (1, 0.25)], [(1, 1.0)]]):
                fast = dynthres_core.cfg_relative(x_out, conds_list, uncond)
                relative = torch.zeros_like(uncond)
                for i, conds in enumerate(conds_list):
                    for cond_index, weight in conds:
                        relative[i] += (x_out[cond_index] - uncond[i]) * weight
                dt = DynThresh(6.5, 1.0, "Constant", 0.0, "Constant", 0.0, 4.0, 15, True, "MEAN", "AD", 1.0)
                dt.step = 2
                with self.subTest(dtype=dtype, conds_list=conds_list):
                    self.assertTrue(torch.equal(fast, relative))
                    self.assertTrue(torch.equal(self._bits(dt.dynthresh_from_relative(fast, uncond, 9.0)), self._bits(dt.dynthresh_from_relative(relative, uncond, 9.0))))

    def test_dynthresh_matches_legacy_formulation_bitwise(self):
        gen = torch.Generator().manual_seed(7)
        for dtype in (torch.float32, torch.bfloat16, torch.float64):
            uncond = (torch.randn(2, 4, 9, 7, generator=gen) * 2).to(dtype)
            relative = (torch.randn(2, 4, 9, 7, generator=gen) * 0.3).to(dtype)
            relative.view(-1)[:5] = 0.0
            for percentile, sep, start, measure, phi in itertools.product((1.0, 0.95), (True, False), ("MEAN", "ZERO"), ("AD", "STD"), (1.0, 0.6)):
                dt = DynThresh(6.5, percentile, "Linear Down", 1.0, "Cosine Down", 2.0, 4.0, 15, sep, start, measure, phi)
                dt.step = 4
                with self.subTest(dtype=dtype, percentile=percentile, sep=sep, start=start, measure=measure, phi=phi):
                    new = dt.dynthresh_from_relative(relative, uncond, 9.0)
                    old = _legacy_dynthresh_from_relative(dt, relative, uncond, 9.0)
                    self.assertTrue(torch.equal(self._bits(new), self._bits(old)))

    def test_dynthresh_propagates_non_finite_values_like_legacy(self):
        for bad in (float("nan"), float("inf")):
            uncond = torch.randn(1, 4, 8, 8)
            relative = torch.randn(1, 4, 8, 8)
            relative[0, 1, 2, 3] = bad
            for sep in (True, False):
                dt = DynThresh(6.5, 1.0, "Constant", 0.0, "Constant", 0.0, 4.0, 15, sep, "MEAN", "AD", 1.0)
                dt.step = 1
                with self.subTest(bad=bad, sep=sep):
                    new = dt.dynthresh_from_relative(relative, uncond, 9.0)
                    old = _legacy_dynthresh_from_relative(dt, relative, uncond, 9.0)
                    self.assertTrue(torch.equal(self._bits(new), self._bits(old)))


def _legacy_dynthresh_from_relative(dt, relative, uncond, cfg_scale):
    # Pre-kernel-economy DynThresh.dynthresh_from_relative, verbatim: independent oracle for the
    # fused |x| max, the skipped percentile-100 clamp and the dropped clamp of max_scaleref.
    orig_dtype = uncond.dtype
    stats_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
    uncond_f = uncond.to(dtype=stats_dtype)
    relative_f = relative.to(dtype=stats_dtype)
    mimic_scale = dt.interpret_scale(dt.mimic_scale, dt.mimic_mode, dt.mimic_scale_min)
    cfg_scale = dt.interpret_scale(cfg_scale, dt.cfg_mode, dt.cfg_scale_min)
    mim_target = uncond_f + relative_f * mimic_scale
    cfg_target = uncond_f + relative_f * cfg_scale
    mim_flattened = mim_target.flatten(2)
    cfg_flattened = cfg_target.flatten(2)
    mim_means = mim_flattened.mean(dim=2).unsqueeze(2)
    cfg_means = cfg_flattened.mean(dim=2).unsqueeze(2)
    mim_centered = mim_flattened - mim_means
    cfg_centered = cfg_flattened - cfg_means
    if dt.sep_feat_channels:
        if dt.variability_measure == 'STD':
            mim_scaleref = mim_centered.std(dim=2, correction=0).unsqueeze(2)
            cfg_scaleref = cfg_centered.std(dim=2, correction=0).unsqueeze(2)
        else:
            mim_scaleref = mim_centered.abs().amax(dim=2).unsqueeze(2)
            cfg_abs = cfg_centered.abs()
            if dt.threshold_percentile >= 1.0:
                cfg_scaleref = cfg_abs.amax(dim=2).unsqueeze(2)
            else:
                cfg_scaleref = torch.quantile(cfg_abs, dt.threshold_percentile, dim=2).unsqueeze(2)
    else:
        if dt.variability_measure == 'STD':
            mim_scaleref = mim_centered.std(correction=0)
            cfg_scaleref = cfg_centered.std(correction=0)
        else:
            mim_scaleref = mim_centered.abs().amax()
            cfg_abs = cfg_centered.abs()
            if dt.threshold_percentile >= 1.0:
                cfg_scaleref = cfg_abs.amax()
            else:
                cfg_scaleref = torch.quantile(cfg_abs, dt.threshold_percentile)
    eps = torch.finfo(stats_dtype).eps
    cfg_scaleref = cfg_scaleref.clamp_min(eps)
    mim_scaleref = mim_scaleref.clamp_min(eps)
    if dt.scaling_startpoint == 'ZERO':
        result = cfg_flattened * (mim_scaleref / cfg_scaleref)
    else:
        if dt.variability_measure == 'STD':
            cfg_renormalized = (cfg_centered / cfg_scaleref) * mim_scaleref
        else:
            max_scaleref = torch.maximum(mim_scaleref, cfg_scaleref).clamp_min(eps)
            cfg_clamped = cfg_centered.clamp(-max_scaleref, max_scaleref)
            cfg_renormalized = (cfg_clamped / max_scaleref) * mim_scaleref
        result = cfg_renormalized + cfg_means
    actual_res = result.unflatten(2, mim_target.shape[2:])
    if dt.interpolate_phi != 1.0:
        actual_res = actual_res * dt.interpolate_phi + cfg_target * (1.0 - dt.interpolate_phi)
    return actual_res.to(dtype=orig_dtype)


class DynamicThresholdingLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_a1111_stubs()
        cls.dynamic_thresholding = importlib.import_module("scripts.dynamic_thresholding")

    def setUp(self):
        self.sd_samplers = sys.modules["modules.sd_samplers"]
        self.sd_samplers.all_samplers_map.clear()
        self.dynamic_thresholding.Script.registered_samplers.clear()

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

    def test_next_request_unregisters_sampler_left_by_failed_generation(self):
        script = self.dynamic_thresholding.Script()
        args = (7.0, 100.0, "Constant", 0.0, "Constant", 0.0, 4.0, True, "MEAN", "AD", 1.0, 0, [], [], [])

        def request():
            return types.SimpleNamespace(sampler_name="Euler", sampler=None, sd_model=object(), steps=4, extra_generation_params={})

        failed = request()
        script.process_batch(failed, True, *args)  # its generation raises: postprocess_batch never runs
        leaked = failed.sampler_name
        self.assertIn(leaked, self.sd_samplers.all_samplers_map)

        for enabled in (False, True):
            p = request()
            script.process_batch(p, enabled, *args)
            with self.subTest(enabled=enabled):
                self.assertNotIn(leaked, self.sd_samplers.all_samplers_map)
                expected = {"Euler", p.sampler_name}
                self.assertEqual(set(self.sd_samplers.all_samplers_map), expected)
            script.postprocess_batch(p, enabled, *args[:-3], [])
        self.assertEqual(set(self.sd_samplers.all_samplers_map), {"Euler"})
        self.assertEqual(self.dynamic_thresholding.Script.registered_samplers, set())

    def test_every_timestep_sampler_is_rejected(self):
        # They run CFGDenoiserTimesteps; DDIM CFG++ used to slip through a hard-coded name list and get a
        # k-diffusion CFG denoiser, i.e. silently wrong math.
        script = self.dynamic_thresholding.Script()
        args = (7.0, 100.0, "Constant", 0.0, "Constant", 0.0, 4.0, True, "MEAN", "AD", 1.0, 0, [], [], [])
        for name in ("DDIM", "DDIM CFG++", "PLMS", "UniPC"):
            p = types.SimpleNamespace(sampler_name=name, sampler=None, sd_model=object(), steps=4, extra_generation_params={})
            with self.subTest(sampler=name), self.assertRaisesRegex(RuntimeError, "Cannot use sampler"):
                script.process_batch(p, True, *args)
            self.assertEqual(set(self.sd_samplers.all_samplers_map), {"Euler"})

    ARGS = (7.0, 100.0, "Constant", 0.0, "Constant", 0.0, 4.0, True, "MEAN", "AD", 1.0, 0, [], [], [])

    def test_renamed_sampler_keeps_the_sampler_data_subclass_and_fields(self):
        # A "Multi" chain is a MultiSamplerData whose total_steps sums its stages; a plain SamplerData copy counted
        # steps once per step (and second_order chains twice), so the hires/conds step counts were wrong.
        SamplerData = sys.modules["modules.sd_samplers_common"].SamplerData

        class MultiSamplerData(SamplerData):
            def total_steps(self, steps):
                return steps + 3

        chain = MultiSamplerData("Multi: oi2", self.sd_samplers.all_samplers_map["Euler"].constructor, ["oi2"], {"scheduler": "exponential", "openclaw_chain": "{}"})
        self.sd_samplers.all_samplers_map[chain.name] = chain
        script = self.dynamic_thresholding.Script()
        p = types.SimpleNamespace(sampler_name=chain.name, sampler=None, sd_model=object(), steps=4, extra_generation_params={})
        script.process_batch(p, True, *self.ARGS)
        renamed = self.sd_samplers.all_samplers_map[p.sampler_name]
        self.assertIsInstance(renamed, MultiSamplerData)
        self.assertEqual(renamed.total_steps(10), 13)
        self.assertEqual((renamed.name, renamed.aliases, renamed.options), (p.sampler_name, chain.aliases, chain.options))
        self.assertIsInstance(renamed.constructor(object()).model_wrap_cfg, self.dynamic_thresholding.CustomCFGDenoiser)
        script.postprocess_batch(p)
        self.assertEqual(set(self.sd_samplers.all_samplers_map), {"Euler", chain.name})

    def test_hires_sampler_gets_dynamic_thresholding_too(self):
        SamplerData = sys.modules["modules.sd_samplers_common"].SamplerData
        euler = self.sd_samplers.all_samplers_map["Euler"]
        self.sd_samplers.all_samplers_map["Heun"] = SamplerData("Heun", euler.constructor, [], {"second_order": True})
        script = self.dynamic_thresholding.Script()

        def request(hr_sampler_name):
            return types.SimpleNamespace(sampler_name="Euler", hr_sampler_name=hr_sampler_name, sampler=None, sd_model=object(), steps=4, extra_generation_params={})

        # A different hires sampler gets its own renamed sampler.
        p = request("Heun")
        script.process_batch(p, True, *self.ARGS)
        self.assertTrue(p.hr_sampler_name.startswith("Heun_dynthres"))
        hires = self.sd_samplers.all_samplers_map[p.hr_sampler_name]
        self.assertEqual(hires.options, {"second_order": True})
        self.assertIsInstance(hires.constructor(object()).model_wrap_cfg, self.dynamic_thresholding.CustomCFGDenoiser)
        self.assertEqual(set(self.sd_samplers.all_samplers_map), {"Euler", "Heun", p.sampler_name, p.hr_sampler_name})
        script.postprocess_batch(p)
        self.assertEqual((p.sampler_name, p.hr_sampler_name), ("Euler", "Heun"))
        self.assertEqual(set(self.sd_samplers.all_samplers_map), {"Euler", "Heun"})
        self.assertEqual(self.dynamic_thresholding.Script.registered_samplers, set())

        # The first pass's sampler named explicitly shares its renamed sampler; None keeps following sampler_name.
        for hr_sampler_name in ("Euler", None):
            p = request(hr_sampler_name)
            script.process_batch(p, True, *self.ARGS)
            with self.subTest(hr_sampler_name=hr_sampler_name):
                self.assertEqual(p.hr_sampler_name, p.sampler_name if hr_sampler_name else None)
                self.assertEqual(len(self.sd_samplers.all_samplers_map), 3)
            script.postprocess_batch(p)
            self.assertEqual((p.sampler_name, p.hr_sampler_name), ("Euler", hr_sampler_name))

        # A timestep hires sampler is rejected like a first-pass one.
        p = request("DDIM")
        with self.assertRaisesRegex(RuntimeError, "Cannot use sampler DDIM"):
            script.process_batch(p, True, *self.ARGS)
        self.assertEqual(set(self.sd_samplers.all_samplers_map), {"Euler", "Heun"})

        # A failed generation (no postprocess_batch) is cleaned up by the next batch, hires name included.
        p = request("Heun")
        script.process_batch(p, True, *self.ARGS)
        script.process_batch(p, False, *self.ARGS)
        self.assertEqual((p.sampler_name, p.hr_sampler_name), ("Euler", "Heun"))
        self.assertEqual(set(self.sd_samplers.all_samplers_map), {"Euler", "Heun"})

    def test_ui_defaults_enable_dynthres_without_explicit_minimums(self):
        # Requests that enable DynThres but leave the scheduler minimums to their defaults (partial args, an
        # infotext re-run of a constant-mode image, the X/Y/Z mimic scale axis) get the ui() values.
        script = self.dynamic_thresholding.Script()
        controls = script.ui(False)
        defaults = [control.value for control in controls]
        self.assertEqual([c.label for c in controls if c.label.startswith("Minimum value")], [
            "Minimum value of the Mimic Scale Scheduler", "Minimum value of the CFG Scale Scheduler"])
        self.assertEqual((defaults[4], defaults[6]), (0.0, 0.0))

        p = types.SimpleNamespace(sampler_name="Euler", sampler=None, sd_model=object(), steps=4, extra_generation_params={})
        script.process_batch(p, True, *defaults[1:], 0, [], [], [])
        denoiser = self.sd_samplers.all_samplers_map[p.sampler_name].constructor(object()).model_wrap_cfg
        denoiser.step, denoiser.total_steps = 1, 4
        x_out = torch.randn(2, 4, 4, 4)
        out = denoiser.combine_denoised(x_out, [[(0, 1.0)]], {"crossattn": torch.zeros(1, 77, 8)}, 7.0)
        self.assertEqual(tuple(out.shape), (1, 4, 4, 4))
        self.assertTrue(torch.isfinite(out).all())
        script.postprocess_batch(p, True, *defaults[1:], batch_number=0, images=[])
        self.assertEqual(set(self.sd_samplers.all_samplers_map), {"Euler"})


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
        attn = types.SimpleNamespace(pag_enable=False)
        to_q = types.SimpleNamespace(seg_enable=True)
        pag_params.crossattn_modules = [attn]
        pag_params.seg_q_modules = [to_q]

        class Denoiser(torch.nn.Module):
            def run_inner_model(self, x, sigma, cond):
                return x * 0

        denoiser = Denoiser()
        denoiser.step, denoiser.steps, denoiser.total_steps = 1, 20, 20
        x_in = torch.randn(2, 4, 2, 2)
        cond = {"crossattn": torch.randn(1, 77, 8), "vector": torch.randn(1, 4)}
        params = types.SimpleNamespace(denoiser=denoiser, text_cond=cond)
        script.on_cfg_denoiser_callback(params, pag_params)
        denoiser.run_inner_model(x_in, torch.ones(2), {"crossattn": torch.randn(2, 77, 8)})

        seen = {}

        def inner_model(x, sigma, cond):
            seen["state"] = (attn.pag_enable, to_q.seg_enable, x.shape[0])
            return x - 1

        script.on_cfg_denoised_callback(types.SimpleNamespace(inner_model=inner_model), pag_params)
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
            script.on_cfg_denoised_callback(types.SimpleNamespace(inner_model=failing_inner_model), pag_params)
        self.assertFalse(attn.pag_enable)
        self.assertTrue(to_q.seg_enable)
        self.assertIsNone(self.row_memo.disarm(denoiser))

        # Outside the PAG interval nothing is recorded and the recorder passes through.
        denoiser.step = 20
        script.on_cfg_denoiser_callback(params, pag_params)
        self.assertIsNone(self.row_memo.disarm(denoiser))

        script.postprocess_batch(types.SimpleNamespace(incant_cfg_params={"pag_params": None}))
        self.assertNotIn("run_inner_model", denoiser.__dict__)

    def test_pag_interval_uses_the_step_of_the_denoiser_call(self):
        # state.sampling_step reads 0 during both step 0 and step 1; a [1, 1] interval must run PAG on step 1 only.
        script = self.pag.PAGExtensionScript()
        pag_params = self.pag.PAGStateParams()
        pag_params.pag_active, pag_params.pag_scale = True, 3.0
        pag_params.pag_start_step = pag_params.pag_end_step = 1
        pag_params.crossattn_modules = [types.SimpleNamespace(pag_enable=False)]
        pag_params.seg_q_modules = []

        class Denoiser(torch.nn.Module):
            def run_inner_model(self, x, sigma, cond):
                return x

        denoiser = Denoiser()
        denoiser.steps = denoiser.total_steps = 3
        cond = {"crossattn": torch.randn(1, 77, 8)}
        ran = []
        for call in range(3):
            denoiser.step = call
            script.on_cfg_denoiser_callback(types.SimpleNamespace(sampling_step=max(0, call - 1), denoiser=denoiser, text_cond=cond), pag_params)
            denoiser.run_inner_model(torch.randn(2, 4, 2, 2), torch.ones(2), {"crossattn": torch.randn(2, 77, 8)})
            script.on_cfg_denoised_callback(types.SimpleNamespace(inner_model=lambda x, sigma, cond: x), pag_params)
            ran.append(pag_params.pag_x_out is not None)
        self.assertEqual(ran, [False, True, False])
        script.remove_main_pass_recorder()

    def test_pag_denoised_callback_fails_fast_without_recorded_main_pass(self):
        script = self.pag.PAGExtensionScript()
        pag_params = self.pag.PAGStateParams()
        pag_params.pag_active = True
        pag_params.pag_scale = 3.0
        with self.assertRaisesRegex(RuntimeError, "not recorded"):
            script.on_cfg_denoised_callback(types.SimpleNamespace(inner_model=None), pag_params)

    @staticmethod
    def attention_module(attn_mask=None):
        # The shape of the SDPA CrossAttention forward (modules/sd_hijack_optimizations.py): q/k/v projections,
        # attention, then to_out = [Linear, Dropout]. attn_mask stands in for the attention map in tests.
        class CrossAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = 2
                self.to_q = torch.nn.Linear(8, 8, bias=False)
                self.to_k = torch.nn.Linear(8, 8, bias=False)
                self.to_v = torch.nn.Linear(8, 8, bias=False)
                self.to_out = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Dropout(0.0))

            def forward(self, x, context=None):
                context = x if context is None else context
                b, n, inner = x.shape
                q, k, v = (t.view(b, -1, self.heads, inner // self.heads).transpose(1, 2) for t in (self.to_q(x), self.to_k(context), self.to_v(context)))
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
                return self.to_out(out.transpose(1, 2).reshape(b, n, inner))

        torch.manual_seed(0)
        return CrossAttention()

    @staticmethod
    def processing():
        return types.SimpleNamespace(incant_cfg_params={"pag_params": None}, extra_generation_params={})

    def test_script_args_keep_the_positions_api_callers_send(self):
        # The controller sends the 13 Incantations args positionally (README "A1111 API argument order"): PAG's
        # slice is 9 of them, with the removed CFG Scheduler's 4 placeholders before PAG SANF.
        controls = self.pag.PAGExtensionScript().setup_ui(False)
        self.assertEqual([c.label for c in controls], [
            "PAG Active", "PAG Scale", "PAG Start Step", "PAG End Step",
            "Enable CFG Scheduler", "CFG Schedule Type", "CFG Noise Interval Low", "CFG Noise Interval High",
            "Use Saliency-Adaptive Noise Fusion",
        ])
        self.assertEqual([c.value for c in controls[4:8]], [False, "Constant", 0, 100])

    def test_removed_cfg_scheduler_fails_the_request(self):
        callbacks = sys.modules["modules.script_callbacks"].callback_registry
        callbacks.clear()
        script = self.pag.PAGExtensionScript()
        script.get_cross_attn_modules = lambda: [self.attention_module()]
        for active in (False, True):
            p = self.processing()
            with self.subTest(pag_active=active), self.assertRaisesRegex(ValueError, "CFG Scheduler"):
                script.pag_process_batch(p, active, 3.0, 0, 150, True, "Constant", 0.0, 100.0, False)
            self.assertEqual(callbacks, [])
            self.assertEqual(p.extra_generation_params, {})
            self.assertIsNone(p.incant_cfg_params["pag_params"])

        # Its other placeholders are ignored, as they were while it was off.
        p = self.processing()
        script.pag_process_batch(p, True, 3.0, 0, 150, False, "Linear", 10.0, 20.0, False)
        self.assertEqual(len(callbacks), 2)
        self.assertNotIn("CFG Interval Enable", p.extra_generation_params)
        script.postprocess_batch(p)
        self.assertEqual(callbacks, [])

    def test_inactive_pag_batch_clears_previous_callbacks_and_hooks(self):
        # A failed generation skips postprocess_batch; the next (PAG-off) batch must not keep its state.
        callbacks = sys.modules["modules.script_callbacks"].callback_registry
        callbacks.clear()
        attn = self.attention_module()
        script = self.pag.PAGExtensionScript()
        script.get_cross_attn_modules = lambda: [attn]

        p = self.processing()
        script.pag_process_batch(p, True, 3.0, 0, 150, False, "Constant", 0.0, 100.0, False)
        self.assertEqual(len(callbacks), 2)
        self.assertIs(p.incant_cfg_params["pag_params"].crossattn_modules[0], attn)
        self.assertEqual(p.extra_generation_params["PAG Active"], True)
        x = torch.randn(1, 3, 8)
        with torch.no_grad():
            plain = attn(x)
            attn.pag_enable = True
            # The perturbed pass is the attention with an identity map: the output projection of the values.
            self.assertTrue(torch.equal(attn(x), attn.to_out(attn.to_v(x))))
            attn.pag_enable = False

        script.pag_process_batch(self.processing(), False, 3.0, 0, 150, False, "Constant", 0.0, 100.0, False)
        self.assertEqual(callbacks, [])
        self.assertEqual(script._callbacks, [])
        self.assertFalse(hasattr(attn, "pag_enable"))
        self.assertFalse(hasattr(attn.to_v, "pag_parent_module"))
        self.assertEqual(len(attn._forward_hooks), 0)
        self.assertEqual(len(attn.to_v._forward_hooks), 0)
        with torch.no_grad():
            self.assertTrue(torch.equal(attn(x), plain))

    def test_perturbed_attention_is_the_identity_attention_map_through_to_out(self):
        attn = self.attention_module()
        script = self.pag.PAGExtensionScript()
        script.get_cross_attn_modules = lambda: [attn]
        p = self.processing()
        script.pag_process_batch(p, True, 3.0, 0, 150, False, "Constant", 0.0, 100.0, False)
        # A LoRA on the output projection (a forward hook on to_out[0], like the functional LoRA path) applies too.
        lora = attn.to_out[0].register_forward_hook(lambda module, args, output: output + 0.25)
        x = torch.randn(2, 5, 8)
        # Independent oracle: the same attention module run with an identity attention map.
        identity = torch.full((5, 5), float("-inf")).fill_diagonal_(0.0)
        oracle = self.attention_module(attn_mask=identity)
        oracle.load_state_dict(attn.state_dict())
        oracle.to_out[0].register_forward_hook(lambda module, args, output: output + 0.25)
        with torch.no_grad():
            attn.pag_enable = True
            perturbed = attn(x)
            attn.pag_enable = False
            plain = attn(x)
            expected = oracle(x)
        torch.testing.assert_close(perturbed, expected, rtol=0, atol=1e-6)
        self.assertFalse(torch.allclose(perturbed, plain))
        lora.remove()
        script.postprocess_batch(p)

    def test_perturbed_attention_fails_on_split_attention_input(self):
        # Hypertile tiling the middle block runs to_v on (rows * tiles) tile rows; their values are not the layer's.
        attn = self.attention_module()
        script = self.pag.PAGExtensionScript()
        script.get_cross_attn_modules = lambda: [attn]
        p = self.processing()
        script.pag_process_batch(p, True, 3.0, 0, 150, False, "Constant", 0.0, 100.0, False)
        tiled = attn.to_v.register_forward_hook(lambda module, args, output: output.reshape(4, 2, 8), prepend=True)
        attn.pag_enable = True
        with torch.no_grad(), self.assertRaisesRegex(RuntimeError, "PAG: the attention input was split"):
            attn(torch.randn(2, 4, 8))
        attn.pag_enable = False
        tiled.remove()
        script.postprocess_batch(p)

    def test_pag_fails_when_model_has_no_middle_attention(self):
        # Requested PAG must not silently render without PAG under an infotext that says "PAG Active".
        callbacks = sys.modules["modules.script_callbacks"].callback_registry
        callbacks.clear()
        script = self.pag.PAGExtensionScript()
        script.get_cross_attn_modules = lambda: []
        p = self.processing()
        with self.assertRaisesRegex(RuntimeError, "PAG: no middle-block"):
            script.pag_process_batch(p, True, 3.0, 0, 150, False, "Constant", 0.0, 100.0, False)
        self.assertEqual(callbacks, [])
        self.assertEqual(p.extra_generation_params, {})
        self.assertIsNone(p.incant_cfg_params["pag_params"])


def _legacy_gaussian_blur_2d(img, kernel_size, sigma):
    # Pre-separable production SEG blur (reflect pad + k x k depthwise conv of
    # query-dtype outer-product taps), verbatim minus its kernel cache. Kept as
    # an independent oracle for the production query blur.
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


def _blur_queries(seg, q, height, width, kernel_size, sigma):
    # The production query blur as _blur_seg_uncond_queries applies it: fp32 result rounded to the query dtype.
    return seg._gaussian_blur_queries_fp32(q, height, width, kernel_size, sigma).to(q.dtype)


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
                new = self.seg._blur_seg_uncond_queries(output.clone(), half, kernel_size=kernel_size, sigma=sigma, is_inf_blur=False, **geometry)
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
        expected = _blur_queries(self.seg, q, 40, 24, kernel_size=49, sigma=8.0)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = _blur_queries(self.seg, q, 40, 24, kernel_size=49, sigma=8.0)
        self.assertTrue(torch.equal(out, expected))

    def test_gaussian_blur_clamps_kernel_to_shorter_spatial_side(self):
        q = torch.randn(2, 2 * 9, 3)
        out = _blur_queries(self.seg, q, 2, 9, kernel_size=13, sigma=2.0)
        self.assertEqual(tuple(out.shape), tuple(q.shape))
        self.assertTrue(torch.isfinite(out).all())
        # A 2-pixel side admits at most a 3-tap reflect kernel, on both axes.
        self.assertTrue(torch.equal(out, _blur_queries(self.seg, q, 2, 9, kernel_size=3, sigma=2.0)))

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
            out = self.seg._blur_seg_uncond_queries(output.clone(), 2, kernel_size=49, sigma=8.0, is_inf_blur=is_inf_blur, **geometry)
            with self.subTest(is_inf_blur=is_inf_blur):
                self.assertEqual(out.dtype, output.dtype)
                self.assertTrue(torch.equal(out[:2], output[:2]))
                self.assertFalse(torch.equal(out[2:], output[2:]))

    def test_infinite_blur_matches_legacy_path_bitwise(self):
        for dtype in (torch.float32, torch.bfloat16):
            output = torch.randn(4, 6 * 5, 2 * 3).to(dtype)
            geometry = dict(heads=2, head_dim=3, downscale_h=6, downscale_w=5)
            new = self.seg._blur_seg_uncond_queries(output.clone(), 2, kernel_size=49, sigma=2.0**11, is_inf_blur=True, **geometry)
            old = _legacy_blur_seg_cond_queries(
                output,
                blur_fn=lambda q: self.seg.gaussian_blur_inf(q),
                **geometry,
            )
            with self.subTest(dtype=dtype):
                self.assertTrue(torch.equal(new, old))

    def test_query_blur_writes_the_to_q_output_in_place_like_the_concatenating_path(self):
        # The hook's to_q output is a fresh tensor: the blurred uncond rows are written into it (one
        # rounding copy) instead of concatenating a new batch. Oracle: the previous cat formulation.
        for dtype, is_inf_blur, (n_cond, n_rows) in itertools.product(
            (torch.float32, torch.bfloat16, torch.float16), (False, True), ((1, 2), (2, 4), (3, 4))
        ):
            output = torch.randn(n_rows, 7 * 6, 4 * 3).to(dtype)
            geometry = dict(heads=4, head_dim=3, downscale_h=7, downscale_w=6, kernel_size=49, sigma=8.0, is_inf_blur=is_inf_blur)
            q_passthrough, q_blur = output.split((n_cond, n_rows - n_cond), dim=0)
            if is_inf_blur:
                q_blur = q_blur.view(q_blur.shape[0], -1, 4, 3).transpose(1, 2).permute(0, 1, 3, 2).reshape(-1, 3, 7, 6)
                q_blur = self.seg.gaussian_blur_inf(q_blur).reshape(-1, 4, 3, 42).view(-1, 12, 42).transpose(1, 2)
            else:
                q_blur = _blur_queries(self.seg, q_blur, 7, 6, 49, 8.0)
            expected = torch.cat((q_passthrough, q_blur), dim=0)
            target = output.clone()
            out = self.seg._blur_seg_uncond_queries(target, n_cond, **geometry)
            with self.subTest(dtype=dtype, is_inf_blur=is_inf_blur, n_cond=n_cond):
                self.assertIs(out, target)
                self.assertTrue(torch.equal(out, expected))

    def test_attention_grid_follows_the_unet_downsample_chain(self):
        # The old sqrt(seq * H / W) floor + divisor walk gave 30x68 for 1080x1920 (true 34x60) and 25x76
        # for 1200x1600 (true 38x50), blurring along a wrapped grid. Arguments are the latent's size.
        cases = {
            (2040, 135, 240): (34, 60),
            (1900, 150, 200): (38, 50),
            (1600, 160, 160): (40, 40),
            (988, 104, 152): (26, 38),
            (135 * 240, 135, 240): (135, 240),
            (19 * 13, 152, 104): (19, 13),
            (48 * 32, 192, 128): (48, 32),  # 1024x1024 hires-cropped to a 1536x1024 latent
        }
        for (seq_len, height, width), expected in cases.items():
            with self.subTest(seq_len=seq_len, height=height, width=width):
                self.assertEqual(self.seg.seg_attention_grid(seq_len, height, width), expected)
        for height in range(1, 260, 3):
            for width in range(1, 260, 5):
                rows, cols = height, width
                for _level in range(4):
                    with self.subTest(height=height, width=width, grid=(rows, cols)):
                        self.assertEqual(self.seg.seg_attention_grid(rows * cols, height, width), (rows, cols))
                    rows, cols = math.ceil(rows / 2), math.ceil(cols / 2)

    def test_attention_grid_matches_the_pixel_size_grid_wherever_that_was_right(self):
        # The grid used to come from p.height x p.width; from the latent's own size it is bit-identical wherever
        # that pixel size was the latent's (first passes), and right where it was not (a cropped hires pass).
        def legacy_grid(seq_len, height, width):
            rows, cols = max(1, height // 8), max(1, width // 8)
            while rows * cols > seq_len:
                rows, cols = (rows + 1) // 2, (cols + 1) // 2
                if rows * cols == 1:
                    break
            if rows * cols == seq_len:
                return rows, cols
            aspect = math.log(height / width)
            rows = min((r for r in range(1, seq_len + 1) if seq_len % r == 0), key=lambda r: abs(math.log(r * r / seq_len) - aspect))
            return rows, seq_len // rows

        for height in range(64, 2049, 8):
            for width in range(64, 2049, 24):
                rows, cols = height // 8, width // 8
                for _level in range(3):
                    with self.subTest(height=height, width=width, grid=(rows, cols)):
                        self.assertEqual(self.seg.seg_attention_grid(rows * cols, height // 8, width // 8), legacy_grid(rows * cols, height, width))
                    rows, cols = (rows + 1) // 2, (cols + 1) // 2
        # 1024x1024 hires-fixed to 1536x1024 with cropping (p.width/p.height stay 1024): the middle level of the
        # 128x192 latent is 32x48; the pixel size gave its transpose.
        self.assertEqual(legacy_grid(32 * 48, 1024, 1024), (48, 32))
        self.assertEqual(self.seg.seg_attention_grid(32 * 48, 128, 192), (32, 48))

    def test_attention_grid_rejects_split_tokens(self):
        # Hypertile tiles of a 32x32 level: 512 tokens are no level of the 128x128 latent.
        self.assertIsNone(self.seg.seg_attention_grid(512, 128, 128))

    def test_cfg_call_rows_locates_row_slices_of_the_cfg_batch(self):
        x_in = torch.randn(5, 4, 3, 2)
        n_cond = 3
        cases = {
            (0, 5): (3, 5),  # one call, the whole batch
            (0, 2): (2, 2),  # chunks [c0 c1] [c2 u0] [u1]
            (2, 4): (1, 2),
            (4, 5): (0, 1),
            (3, 5): (0, 2),  # the uncond call of a split batch
            (0, 3): (3, 3),  # skip-uncond: cond rows only
        }
        for (start, end), expected in cases.items():
            with self.subTest(rows=(start, end)):
                self.assertEqual(self.seg.cfg_call_rows(x_in, x_in[start:end], n_cond), expected)
        for x in (x_in.clone(), x_in[:, :2], x_in[1:3].clone(), torch.randn(2, 4, 3, 2), None):
            with self.subTest(shape=None if x is None else tuple(x.shape)):
                self.assertIsNone(self.seg.cfg_call_rows(x_in, x, n_cond))
        self.assertIsNone(self.seg.cfg_call_rows(None, x_in, n_cond))

    def test_blur_operator_built_under_cpu_autocast_stays_fp32(self):
        # The operator is first built inside the UNet forward; an active CPU autocast must not
        # turn its conv into bf16 and cache a bf16 operator (the fp32 matmul then fails).
        self.seg._gaussian_blur_operator.cache_clear()
        q = torch.randn(2, 9 * 7, 4).to(torch.bfloat16)
        try:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                out = _blur_queries(self.seg, q, 9, 7, kernel_size=49, sigma=8.0)
                operator = self.seg._gaussian_blur_operator(9, 7, 8.0, torch.device("cpu"))
        finally:
            self.seg._gaussian_blur_operator.cache_clear()
        self.assertEqual(operator.dtype, torch.float32)
        self.assertTrue(torch.equal(out, _blur_queries(self.seg, q, 9, 7, kernel_size=49, sigma=8.0)))

    def test_uncond_blur_follows_cond_row_count_for_and_prompts(self):
        # AND prompts: 3 cond rows + 1 uncond row. Only the uncond row is blurred; a half split
        # would blur cond row 2 as well (guidance toward the smoothed prediction for that prompt).
        output = torch.randn(4, 6 * 5, 2 * 3)
        geometry = dict(heads=2, head_dim=3, downscale_h=6, downscale_w=5)
        for is_inf_blur in (False, True):
            out = self.seg._blur_seg_uncond_queries(output.clone(), 3, kernel_size=49, sigma=8.0, is_inf_blur=is_inf_blur, **geometry)
            paired = self.seg._blur_seg_uncond_queries(output[2:].clone(), 1, kernel_size=49, sigma=8.0, is_inf_blur=is_inf_blur, **geometry)
            with self.subTest(is_inf_blur=is_inf_blur):
                self.assertTrue(torch.equal(out[:3], output[:3]))
                self.assertTrue(torch.equal(out[3], paired[1]))
                self.assertFalse(torch.equal(out[3], output[3]))

    # The SEG harness: a latent of LATENT_H x LATENT_W whose middle level (two ceil-halvings) is 6x5.
    LATENT_H, LATENT_W = 24, 20
    SEQ = 6 * 5

    def _hooked_seg(self):
        class CrossAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = 2
                self.to_q = torch.nn.Linear(6, 6, bias=False)

        class InnerModel(torch.nn.Module):
            """The denoiser's inner model: the UNet call. Row r of x carries index r in x[r, 0, 0, 0]; the
            middle-block attention sees that row's tokens."""

            def __init__(self, attn, tokens):
                super().__init__()
                self.attn = attn
                self.tokens = tokens

            def forward(self, x, sigma, cond=None):
                return self.attn.to_q(self.tokens[x[:, 0, 0, 0].long()])

        attn = CrossAttention()
        script = self.seg.SEGExtensionScript()
        params = self.seg.SEGStateParams()
        params.seg_active = True
        params.seg_blur_threshold = 10.5
        params.crossattn_modules = [attn]
        script.ready_hijack_forward(params, 3.0)
        self.addCleanup(script.remove_all_hooks)
        shared = sys.modules["modules.shared"]
        saved_model = shared.__dict__.get("sd_model")
        shared.sd_model = types.SimpleNamespace()
        self.addCleanup(setattr, shared, "sd_model", saved_model)
        tokens = torch.randn(8, self.SEQ, 6, generator=torch.Generator().manual_seed(7))
        denoiser = types.SimpleNamespace(step=0, steps=20, total_steps=20, inner_model=InnerModel(attn, tokens))
        return script, params, attn, denoiser

    def _step(self, script, params, denoiser, n_cond, n_uncond, rows=None):
        """Run the cfg_denoiser callback for a CFG batch of n_cond + n_uncond rows; returns x_in."""
        rows = n_cond + n_uncond if rows is None else rows
        x_in = torch.zeros(rows, 4, self.LATENT_H, self.LATENT_W)
        x_in[:, 0, 0, 0] = torch.arange(rows, dtype=x_in.dtype)
        step = types.SimpleNamespace(
            x=x_in,
            denoiser=denoiser,
            text_cond={"crossattn": torch.zeros(n_cond, 77, 8), "vector": torch.zeros(n_cond, 4)},
            text_uncond={"crossattn": torch.zeros(n_uncond, 77, 8), "vector": torch.zeros(n_uncond, 4)},
        )
        script.on_cfg_denoiser_callback(step, params)
        return x_in

    def _blur_oracle(self, denoiser, n_cond, rows):
        """The to_q output of a CFG batch with its uncond rows [n_cond:rows) blurred, via the blur helper."""
        attn = denoiser.inner_model.attn
        plain = torch.nn.functional.linear(denoiser.inner_model.tokens[:rows], attn.to_q.weight)
        return self.seg._blur_seg_uncond_queries(plain.clone(), n_cond, heads=2, head_dim=3, downscale_h=6, downscale_w=5, kernel_size=49, sigma=8.0, is_inf_blur=False), plain

    def test_seg_blurs_exactly_the_uncond_rows_in_every_cfg_call_layout(self):
        script, params, attn, denoiser = self._hooked_seg()
        # (n_cond, n_uncond, the UNet calls as row ranges): one call; batch 2; AND prompts (3 cond rows for 1 uncond);
        # batch_cond_uncond off ([c0 c1] [u0 u1], and AND prompts straddling a chunk); prompt and negative prompt
        # of different token lengths without padding (cond chunks, then one uncond call); skip-uncond (cond only).
        layouts = (
            (1, 1, ((0, 2),)),
            (2, 2, ((0, 4),)),
            (3, 1, ((0, 4),)),
            (2, 2, ((0, 2), (2, 4))),
            (3, 2, ((0, 2), (2, 4), (4, 5))),
            (3, 2, ((0, 2), (2, 3), (3, 5))),
            (2, 2, ((0, 2),)),
        )
        with torch.no_grad():
            for n_cond, n_uncond, calls in layouts:
                with self.subTest(n_cond=n_cond, n_uncond=n_uncond, calls=calls):
                    x_in = self._step(script, params, denoiser, n_cond, n_uncond)
                    out = torch.cat([denoiser.inner_model(x_in[a:b], torch.ones(b - a)) for a, b in calls])
                    expected, plain = self._blur_oracle(denoiser, n_cond, calls[-1][1])
                    self.assertTrue(torch.equal(out, expected))
                    self.assertTrue(torch.equal(out[:n_cond], plain[:n_cond]))
                    for row in range(n_cond, out.shape[0]):
                        self.assertFalse(torch.equal(out[row], plain[row]))
                    self.assertIsNone(params.call_rows)

    def test_seg_blurs_with_batch_cond_uncond_off(self):
        # It used to switch itself off for the whole request when batch_cond_uncond was off.
        shared = sys.modules["modules.shared"]
        saved = shared.opts.batch_cond_uncond
        shared.opts.batch_cond_uncond = False
        try:
            script, params, attn, denoiser = self._hooked_seg()
            with torch.no_grad():
                x_in = self._step(script, params, denoiser, 1, 1)
                self.assertTrue(attn.to_q.seg_enable)
                out = torch.cat([denoiser.inner_model(x_in[:1], torch.ones(1)), denoiser.inner_model(x_in[1:], torch.ones(1))])
            self.assertTrue(torch.equal(out, self._blur_oracle(denoiser, 1, 2)[0]))
        finally:
            shared.opts.batch_cond_uncond = saved

    def test_seg_fails_on_a_unet_call_it_cannot_place(self):
        script, params, attn, denoiser = self._hooked_seg()
        with torch.no_grad():
            x_in = self._step(script, params, denoiser, 1, 1)
            # An extension calling the UNet on rows of its own: SEG cannot tell which of them are uncond.
            with self.assertRaisesRegex(RuntimeError, "not a row slice of the step's CFG batch"):
                denoiser.inner_model(x_in.clone(), torch.ones(2))
            # Hypertile tiling the middle block: the attention sees (rows x tiles) rows of a tile's tokens.
            tiled = attn.to_q.register_forward_pre_hook(lambda module, args: (args[0].reshape(4, self.SEQ // 2, 6),))
            try:
                with self.assertRaisesRegex(RuntimeError, "split into tiles"):
                    denoiser.inner_model(x_in, torch.ones(2))
            finally:
                tiled.remove()
            # Outside the SEG interval nothing is placed and any call runs.
            params.seg_end_step = -1
            self._step(script, params, denoiser, 1, 1)
            out = denoiser.inner_model(x_in.clone(), torch.ones(2))
        self.assertTrue(torch.equal(out, self._blur_oracle(denoiser, 1, 2)[1]))

    def test_seg_rejects_a_cfg_batch_that_is_not_cond_then_uncond(self):
        # InstructPix2Pix image CFG: [cond, uncond (image), uncond] rows.
        script, params, attn, denoiser = self._hooked_seg()
        with self.assertRaisesRegex(RuntimeError, r"SEG supports the \[cond, uncond\] batch only"):
            self._step(script, params, denoiser, 1, 1, rows=3)

    def test_seg_fails_under_tiled_diffusion_overrides(self):
        shared = sys.modules["modules.shared"]

        class Model:
            def apply_model(self, x, t, cond):
                return x

        class Wrapper(torch.nn.Module):
            def forward(self, x, sigma, cond=None):
                return x

        class Denoiser:
            def forward(self, x):
                return x

        cases = {
            # MultiDiffusion and DemoFusion replace the inner model's forward, DemoFusion the CFG denoiser's,
            # Mixture of Diffusers and DemoFusion sd_model.apply_model.
            "inner model forward": lambda den, model: setattr(den.inner_model, "forward", lambda x, sigma, cond=None: x),
            "CFG denoiser forward": lambda den, model: setattr(den, "forward", lambda x: x),
            "sd_model.apply_model": lambda den, model: setattr(model, "apply_model", lambda x, t, cond: x),
        }
        for label, override in cases.items():
            script, params, attn, _ = self._hooked_seg()
            denoiser = Denoiser()
            denoiser.step, denoiser.steps, denoiser.total_steps, denoiser.inner_model = 0, 20, 20, Wrapper()
            shared.sd_model = Model()
            # Restores that assign the bound class method back are not overrides.
            denoiser.forward = denoiser.forward
            shared.sd_model.apply_model = shared.sd_model.apply_model
            self._step(script, params, denoiser, 1, 1)
            override(denoiser, shared.sd_model)
            with self.subTest(override=label), self.assertRaisesRegex(RuntimeError, f"replaces the {label} \\(Tiled Diffusion\\)"):
                self._step(script, params, denoiser, 1, 1)

    def test_inactive_seg_batch_clears_callback_left_by_failed_batch(self):
        # A failed generation skips postprocess_batch; the next (SEG-off) batch must not keep its callback.
        callbacks = sys.modules["modules.script_callbacks"].callback_registry
        callbacks.clear()
        script, params, attn, denoiser = self._hooked_seg()
        self._step(script, params, denoiser, 1, 1)
        callbacks.append(script.track_callback(lambda step: script.on_cfg_denoiser_callback(step, params)))
        p = types.SimpleNamespace(extra_generation_params={}, incant_cfg_params={})
        script.seg_process_batch(p, False, 3.0, 0, 150)
        self.assertEqual(callbacks, [])
        self.assertEqual(script._callbacks, [])
        self.assertFalse(hasattr(attn.to_q, "seg_enable"))
        self.assertEqual(len(attn.to_q._forward_hooks), 0)
        self.assertEqual(len(denoiser.inner_model._forward_pre_hooks), 0)
        self.assertEqual(len(denoiser.inner_model._forward_hooks), 0)

    def test_active_seg_fails_when_model_has_no_middle_attention(self):
        # Requested SEG must not silently render without SEG (its infotext already says "SEG Active").
        callbacks = sys.modules["modules.script_callbacks"].callback_registry
        callbacks.clear()
        script = self.seg.SEGExtensionScript()
        p = types.SimpleNamespace(extra_generation_params={}, incant_cfg_params={}, height=64, width=64)
        with unittest.mock.patch.object(script, "get_cross_attn_modules", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "SEG"):
                script.seg_process_batch(p, True, 3.0, 0, 150)
        self.assertEqual(callbacks, [])

    def test_seg_steps_outside_the_interval_run_plain_attention(self):
        script, params, attn, denoiser = self._hooked_seg()
        params.seg_start_step, params.seg_end_step = 2, 3
        with torch.no_grad():
            for call, blurred in ((1, False), (2, True), (3, True), (4, False)):
                denoiser.step = call
                x_in = self._step(script, params, denoiser, 1, 1)
                out = denoiser.inner_model(x_in, torch.ones(2))
                expected, plain = self._blur_oracle(denoiser, 1, 2)
                with self.subTest(step=call):
                    self.assertTrue(torch.equal(out, expected if blurred else plain))


class SamplerStepTests(unittest.TestCase):
    """sampler_step against the real k-diffusion samplers: every model call maps to the step that makes it."""

    @classmethod
    def setUpClass(cls):
        install_a1111_stubs()
        cls.ui_wrapper = importlib.import_module("scripts.ui_wrapper")
        sys.path.insert(0, str(REPO_ROOT / "repositories" / "k-diffusion"))
        try:
            cls.k_sampling = importlib.import_module("k_diffusion.sampling")
        finally:
            sys.path.remove(str(REPO_ROOT / "repositories" / "k-diffusion"))

    def run_sampler(self, func, steps, calls_per_step):
        """Run ``func`` with a CFG-denoiser-like model; returns (sampler_step, state.sampling_step, sigma) per call."""
        denoiser = types.SimpleNamespace(step=0, steps=steps, total_steps=steps * calls_per_step)
        state = {"sampling_step": 0}
        calls = []

        def model(x, sigma, **kwargs):
            calls.append((self.ui_wrapper.sampler_step(denoiser), state["sampling_step"], float(sigma[0])))
            denoiser.step += 1
            return x * 0.5

        def callback(d):
            state["sampling_step"] = d["i"]

        sigmas = torch.cat([torch.linspace(10.0, 0.5, steps), torch.zeros(1)])
        torch.manual_seed(0)
        func(model, torch.ones(1, 1, 2, 2), sigmas, callback=callback, disable=True)
        return calls, sigmas

    def test_first_order_samplers_map_each_call_to_its_step(self):
        for name in ("sample_euler", "sample_euler_ancestral", "sample_dpmpp_2m", "sample_dpmpp_2m_sde", "sample_lms"):
            calls, sigmas = self.run_sampler(getattr(self.k_sampling, name), 7, 1)
            with self.subTest(sampler=name):
                self.assertEqual([step for step, _, _ in calls], list(range(7)))
                self.assertEqual([sigma for _, _, sigma in calls], sigmas[:7].tolist())
                # The lag this replaces: state.sampling_step reads 0 during steps 0 and 1, then i - 1.
                self.assertEqual([lagged for _, lagged, _ in calls], [0] + list(range(6)))

    def test_second_order_samplers_map_both_calls_of_a_step_to_it(self):
        for name in ("sample_heun", "sample_dpm_2", "sample_dpmpp_2s_ancestral", "sample_dpmpp_sde"):
            calls, sigmas = self.run_sampler(getattr(self.k_sampling, name), 6, 2)
            with self.subTest(sampler=name):
                # Two calls per step; the final step to sigma 0 makes one.
                self.assertEqual([step for step, _, _ in calls], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5])
                # Each step's first call evaluates at that step's sigma.
                self.assertEqual([sigma for _, _, sigma in calls[::2]], sigmas[:6].tolist())

    def test_missing_step_counts_fail(self):
        with self.assertRaisesRegex(RuntimeError, "no step count"):
            self.ui_wrapper.sampler_step(types.SimpleNamespace(step=0, steps=None, total_steps=None))


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

    def test_timed_records_the_block_also_when_it_raises(self):
        timings = {}
        with self.timing.timed(timings, "hook"):
            pass
        with self.assertRaisesRegex(ValueError, "boom"):
            with self.timing.timed(timings, "hook"):
                raise ValueError("boom")
        self.assertEqual(timings["hook"]["calls"], 2)
        self.assertGreaterEqual(timings["hook"]["total_seconds"], 0.0)

    def test_xyz_field_setter_enables_feature_only_when_unset(self):
        setter = self.ui_wrapper.xyz_field_setter
        p = types.SimpleNamespace()
        setter("pag_scale", "pag_active")(p, 2.5, [])
        self.assertEqual(vars(p), {"pag_scale": 2.5, "pag_active": True})

        p = types.SimpleNamespace(pag_active=False)
        setter("pag_sanf", "pag_active", boolean=True)(p, "True", [])
        self.assertEqual(vars(p), {"pag_active": False, "pag_sanf": True})

    def test_xyz_axis_options_are_appended_once_in_order(self):
        def option(label):
            return types.SimpleNamespace(label=label)

        xyz_grid = types.SimpleNamespace(axis_options=[option("Seed"), option("[SEG] Active")])
        self.ui_wrapper.add_xyz_axis_options(xyz_grid, [option("[SEG] Active"), option("[PAG] Active"), option("[PAG] SANF")])
        self.ui_wrapper.add_xyz_axis_options(xyz_grid, [option("[PAG] Active")])
        self.assertEqual([o.label for o in xyz_grid.axis_options], ["Seed", "[SEG] Active", "[PAG] Active", "[PAG] SANF"])


if __name__ == "__main__":
    unittest.main()
