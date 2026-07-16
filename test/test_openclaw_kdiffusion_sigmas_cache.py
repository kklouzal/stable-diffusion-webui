import importlib.util
import os
import sys
import types
import unittest

import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def load_kdiffusion_module():
    originals = {}

    def put(name, module):
        originals[name] = sys.modules.get(name)
        sys.modules[name] = module

    profile_spec = importlib.util.spec_from_file_location(
        "test_openclaw_generation_profile_module",
        "modules/openclaw_generation_profile.py",
    )
    profile = importlib.util.module_from_spec(profile_spec)
    sys.modules[profile_spec.name] = profile
    try:
        profile_spec.loader.exec_module(profile)
    finally:
        sys.modules.pop(profile_spec.name, None)
    profile.clear()

    sampling_module = _module("k_diffusion.sampling")
    k_diffusion_module = _module("k_diffusion", sampling=sampling_module)
    sd_samplers_common = _module(
        "modules.sd_samplers_common",
        SamplerData=lambda label, constructor, aliases, options: types.SimpleNamespace(
            name=label,
            label=label,
            constructor=constructor,
            aliases=aliases,
            options=options,
        ),
        Sampler=type("Sampler", (), {"__init__": lambda self, funcname: setattr(self, "funcname", funcname)}),
    )
    sd_samplers_cfg_denoiser = _module(
        "modules.sd_samplers_cfg_denoiser",
        CFGDenoiser=type("CFGDenoiser", (torch.nn.Module,), {"__init__": lambda self, sampler: torch.nn.Module.__init__(self)}),
    )
    devices = _module("modules.devices", cpu=torch.device("cpu"))
    script_callbacks = _module(
        "modules.script_callbacks",
        ExtraNoiseParams=object,
        extra_noise_callback=lambda *args, **kwargs: None,
    )
    opts = types.SimpleNamespace(
        always_discard_next_to_last_sigma=False,
        use_old_karras_scheduler_sigmas=False,
        sigma_min=0,
        sigma_max=0,
        rho=0,
        beta_dist_alpha=0.6,
        beta_dist_beta=0.6,
    )
    shared = _module(
        "modules.shared",
        opts=opts,
        sd_model=types.SimpleNamespace(
            is_sdxl=False,
            is_sd2=False,
            parameterization="eps",
            sd_checkpoint_info=types.SimpleNamespace(filename="sd15.safetensors", shorthash="sd15"),
        ),
    )

    def schedule_function(n, sigma_min, sigma_max, device):
        value = 2.0 if shared.sd_model.is_sdxl else 1.0
        return torch.tensor([value, 0.0], device=device)

    scheduler = types.SimpleNamespace(
        label="Align Your Steps",
        function=schedule_function,
        default_rho=-1,
        need_inner_model=False,
    )
    sd_schedulers = _module(
        "modules.sd_schedulers",
        schedulers=[],
        schedulers_map={"Align Your Steps": scheduler},
    )

    modules_pkg = _module("modules")
    modules_pkg.__path__ = ["modules"]

    for name, module in (
        ("k_diffusion", k_diffusion_module),
        ("k_diffusion.sampling", sampling_module),
        ("modules", modules_pkg),
        ("modules.sd_samplers_common", sd_samplers_common),
        ("modules.sd_samplers_extra", _module("modules.sd_samplers_extra", restart_sampler=lambda *args, **kwargs: None)),
        ("modules.sd_samplers_cfg_denoiser", sd_samplers_cfg_denoiser),
        ("modules.sd_schedulers", sd_schedulers),
        ("modules.devices", devices),
        ("modules.openclaw_generation_profile", profile),
        ("modules.script_callbacks", script_callbacks),
        ("modules.shared", shared),
    ):
        put(name, module)

    try:
        spec = importlib.util.spec_from_file_location("test_kdiffusion_module", "modules/sd_samplers_kdiffusion.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["test_kdiffusion_module"] = module
        spec.loader.exec_module(module)
        return module, shared, profile
    finally:
        sys.modules.pop("test_kdiffusion_module", None)
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class KDiffusionSigmasCacheTests(unittest.TestCase):
    def test_sigmas_cache_is_invalidated_by_model_schedule_signature(self):
        module, shared, profile = load_kdiffusion_module()
        sampler = object.__new__(module.KDiffusionSampler)
        sampler.config = types.SimpleNamespace(name="Euler", options={})
        sampler.funcname = "sample_euler"
        sampler.model_wrap = types.SimpleNamespace(sigmas=torch.tensor([0.1, 10.0]))
        p = types.SimpleNamespace(
            hr_scheduler=None,
            is_hr_pass=False,
            scheduler="Align Your Steps",
            extra_generation_params={},
            sampler_noise_scheduler_override=None,
        )

        sd15_sigmas = sampler.get_sigmas(p, 20)
        shared.sd_model = types.SimpleNamespace(
            is_sdxl=True,
            is_sd2=False,
            parameterization="eps",
            sd_checkpoint_info=types.SimpleNamespace(filename="sdxl.safetensors", shorthash="sdxl"),
        )
        sdxl_sigmas = sampler.get_sigmas(p, 20)

        torch.testing.assert_close(sd15_sigmas, torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(sdxl_sigmas, torch.tensor([2.0, 0.0]))
        self.assertEqual(profile.status()["hits"], 0)
        self.assertEqual(profile.status()["misses"], 2)


if __name__ == "__main__":
    unittest.main()
