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
    scripts_mod = types.ModuleType("modules.scripts")
    scripts_mod.Script = object
    scripts_mod.AlwaysVisible = object()
    script_callbacks_mod = types.ModuleType("modules.script_callbacks")
    script_callbacks_mod.ExtraNoiseParams = lambda noise, x, xi: types.SimpleNamespace(noise=noise, x=x, xi=xi)
    script_callbacks_mod.extra_noise_callback = lambda _params: None
    script_callbacks_mod.on_app_started = lambda _callback: None
    script_loading_mod = types.ModuleType("modules.script_loading")
    script_loading_mod.loaded_scripts = {}
    sd_samplers_common_mod = types.ModuleType("modules.sd_samplers_common")

    class SamplerData:
        def __init__(self, name, constructor, aliases=None, options=None):
            self.name = name
            self.constructor = constructor
            self.aliases = aliases or []
            self.options = options or {}

    class InterruptedException(Exception):
        pass

    sd_samplers_common_mod.SamplerData = SamplerData
    sd_samplers_common_mod.InterruptedException = InterruptedException
    sd_samplers_common_mod.setup_img2img_steps = lambda p, steps=None: (steps or p.steps, getattr(p, "t_enc", steps or p.steps))
    sd_samplers_common_mod.TorchHijack = lambda _p: torch
    sd_samplers_common_mod.samples_to_images_tensor = lambda latent, approximation=2: latent
    sd_samplers_mod = types.ModuleType("modules.sd_samplers")
    sd_samplers_mod.all_samplers = []
    sd_samplers_mod.all_samplers_map = {}
    sd_samplers_mod.set_samplers = lambda: None
    sd_schedulers_mod = types.ModuleType("modules.sd_schedulers")
    sd_schedulers_mod.schedulers_map = {
        "Automatic": object(),
        "Karras": object(),
        "Exponential": object(),
        "Normal": object(),
    }
    sd_schedulers_mod.schedulers = [
        types.SimpleNamespace(label="Automatic"),
        types.SimpleNamespace(label="Karras"),
        types.SimpleNamespace(label="Exponential"),
        types.SimpleNamespace(label="Normal"),
    ]
    shared_mod = types.ModuleType("modules.shared")
    shared_mod.opts = types.SimpleNamespace(
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        sgm_noise_multiplier=False,
        img2img_extra_noise=0.0,
    )
    shared_mod.state = types.SimpleNamespace(sampling_step=0, sampling_steps=0)
    shared_mod.total_tqdm = types.SimpleNamespace(update=lambda: None)
    shared_mod.sd_model = types.SimpleNamespace()
    sd_samplers_kdiffusion_mod = types.ModuleType("modules.sd_samplers_kdiffusion")

    class FakeSamplerConfig:
        def __init__(self, name, options=None):
            self.name = name
            self.options = options or {}

        @staticmethod
        def total_steps(steps):
            return steps

    sampler_configs = {
        "Euler": FakeSamplerConfig("Euler"),
        "DPM++ 2M SDE": FakeSamplerConfig("DPM++ 2M SDE", {"brownian_noise": True}),
        "Heun": FakeSamplerConfig("Heun", {"solver_type": "heun"}),
    }
    sd_samplers_kdiffusion_mod.samplers_k_diffusion = [
        ("Euler", "sample_euler"),
        ("DPM++ 2M SDE", "sample_dpmpp_2m_sde"),
        ("Heun", "sample_heun"),
    ]
    sd_samplers_kdiffusion_mod.k_diffusion_samplers_map = sampler_configs
    sd_samplers_kdiffusion_mod.sampler_extra_params = {
        "sample_euler": ["s_churn", "s_tmin", "s_tmax", "s_noise"],
        "sample_dpmpp_2m_sde": [],
        "sample_heun": [],
    }

    class KDiffusionSampler:
        def __init__(self, _funcname, _sd_model):
            self.config = sampler_configs["Euler"]
            self.model_wrap_cfg = types.SimpleNamespace()
            self.eta_option_field = "eta"
            self.eta = 0.0
            self.stop_at = None

        def get_sigmas(self, _p, steps):
            return torch.linspace(float(steps), 0.0, steps + 1)

        def sample_img2img(self, *args, **kwargs):
            return args, kwargs

        def create_noise_sampler(self, *_args, **_kwargs):
            return None

        def add_infotext(self, _p):
            return None

    sd_samplers_kdiffusion_mod.KDiffusionSampler = KDiffusionSampler
    k_diffusion_pkg = types.ModuleType("k_diffusion")
    sampling_mod = types.ModuleType("k_diffusion.sampling")
    sampling_mod.sample_euler = lambda _model, x, **_kwargs: x
    sampling_mod.sample_dpmpp_2m_sde = lambda _model, x, **_kwargs: x
    sampling_mod.sample_heun = lambda _model, x, **_kwargs: x
    k_diffusion_pkg.sampling = sampling_mod

    modules_pkg.headless_ui = headless_ui_mod
    modules_pkg.script_callbacks = script_callbacks_mod
    modules_pkg.script_loading = script_loading_mod
    modules_pkg.scripts = scripts_mod
    modules_pkg.sd_samplers = sd_samplers_mod
    modules_pkg.sd_samplers_common = sd_samplers_common_mod
    modules_pkg.sd_samplers_kdiffusion = sd_samplers_kdiffusion_mod
    modules_pkg.sd_schedulers = sd_schedulers_mod
    modules_pkg.shared = shared_mod

    sys.modules.update(
        {
            "k_diffusion": k_diffusion_pkg,
            "k_diffusion.sampling": sampling_mod,
            "modules": modules_pkg,
            "modules.headless_ui": headless_ui_mod,
            "modules.script_callbacks": script_callbacks_mod,
            "modules.script_loading": script_loading_mod,
            "modules.scripts": scripts_mod,
            "modules.sd_samplers": sd_samplers_mod,
            "modules.sd_samplers_common": sd_samplers_common_mod,
            "modules.sd_samplers_kdiffusion": sd_samplers_kdiffusion_mod,
            "modules.sd_schedulers": sd_schedulers_mod,
            "modules.shared": shared_mod,
        }
    )


class MultiSamplerCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_a1111_stubs()
        cls.multi = importlib.import_module("scripts.openclaw_multi_sampler")

    def setUp(self):
        self.multi._DENOISE_RAMP_FUNC = None
        self.multi._DENOISE_RAMP_FUNC_LOADED = False
        sys.modules["modules.script_loading"].loaded_scripts = {}

    def test_chain_boundaries_reject_empty_stage_after_clamping(self):
        definition = {"samplers": ["Euler", "Heun"], "switch_ats": [20]}

        with self.assertRaisesRegex(ValueError, "no sampling steps"):
            self.multi._chain_boundaries(definition, steps=20)

    def test_normalize_definition_preserves_arbitrary_length_scheduler_chain(self):
        definition = self.multi._normalize_definition(
            {
                "name": "three stage",
                "samplers": ["Euler", "Heun", "DPM++ 2M SDE"],
                "switch_ats": [4, 8],
                "schedulers": ["Karras", "Exponential", "Normal"],
                "created_at": 123.0,
            }
        )

        self.assertEqual(definition["name"], "Multi: three stage")
        self.assertEqual(definition["samplers"], ["Euler", "Heun", "DPM++ 2M SDE"])
        self.assertEqual(definition["switch_ats"], [4, 8])
        self.assertEqual(definition["schedulers"], ["Karras", "Exponential", "Normal"])
        self.assertEqual(definition["created_at"], 123.0)
        self.assertEqual(definition["sampler_3"], "DPM++ 2M SDE")
        self.assertEqual(definition["scheduler_3"], "Normal")

    def test_scheduler_stage_split_preserves_continuous_boundary_sigma(self):
        sampler = object.__new__(self.multi.MultiKDiffusionSampler)
        sampler.definition = {
            "samplers": ["Euler", "Heun"],
            "switch_ats": [2],
            "schedulers": ["Karras", "Exponential"],
        }
        sources = {
            "Karras": torch.tensor([10.0, 9.0, 8.0, 7.0, 0.0]),
            "Exponential": torch.tensor([100.0, 90.0, 80.0, 70.0, 0.0]),
        }
        sampler._sigmas_for_scheduler = lambda _p, _source_steps, _sampler_name, scheduler_name: sources[scheduler_name]

        stages = sampler._build_stages(types.SimpleNamespace(), torch.zeros(5), steps=4)

        torch.testing.assert_close(stages[0][2], torch.tensor([10.0, 9.0, 8.0]))
        torch.testing.assert_close(stages[1][2], torch.tensor([8.0, 70.0, 0.0]))

    def test_denoise_ramp_helper_is_not_imported_as_fallback(self):
        original_sample_img2img = sys.modules["modules.sd_samplers_kdiffusion"].KDiffusionSampler.sample_img2img

        self.assertIsNone(self.multi._load_denoise_ramp_func())

        self.assertIs(sys.modules["modules.sd_samplers_kdiffusion"].KDiffusionSampler.sample_img2img, original_sample_img2img)
        self.assertTrue(self.multi._DENOISE_RAMP_FUNC_LOADED)

    def test_denoise_ramp_helper_uses_loaded_script_module(self):
        target = EXT_ROOT.parent / "openclaw-denoise-ramp" / "scripts" / "openclaw_denoise_ramp.py"

        def ramp(*_args, **_kwargs):
            return None

        sys.modules["modules.script_loading"].loaded_scripts = {str(target): types.SimpleNamespace(ramp_sigmas_for_img2img=ramp)}

        self.assertIs(self.multi._load_denoise_ramp_func(), ramp)


if __name__ == "__main__":
    unittest.main()
