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
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, child_name, module)

    os.environ["OPENCLAW_GENERATION_PROFILE_CACHE_MAX"] = "16"

    sampling_module = _module(
        "k_diffusion.sampling",
        sample_euler=lambda *args, **kwargs: None,
    )
    k_diffusion_module = _module("k_diffusion", sampling=sampling_module)
    brownian_interval = _module("torchsde._brownian.brownian_interval", _randn=None)
    torchsde_brownian = _module("torchsde._brownian", brownian_interval=brownian_interval)
    torchsde_module = _module("torchsde", _brownian=torchsde_brownian)

    devices = _module(
        "modules.devices",
        cpu=torch.device("cpu"),
        device=torch.device("cpu"),
        dtype=torch.float32,
        dtype_vae=torch.float32,
        randn_local=lambda seed, size: torch.randn(size),
        without_autocast=lambda: torch.no_grad(),
        torch_gc=lambda: None,
    )
    opts = types.SimpleNamespace(
        always_discard_next_to_last_sigma=False,
        use_old_karras_scheduler_sigmas=False,
        sigma_min=0,
        sigma_max=0,
        rho=0,
        beta_dist_alpha=0.6,
        beta_dist_beta=0.6,
        img2img_fix_steps=False,
        show_progress_type="Full",
        live_preview_fast_interrupt=False,
        live_preview_allow_lowvram_full=False,
        sd_vae_decode_method="Full",
        sd_vae_encode_method="Full",
        live_previews_enable=False,
        show_progress_every_n_steps=0,
        no_dpmpp_sde_batch_determinism=False,
    )
    shared = _module(
        "modules.shared",
        opts=opts,
        state=types.SimpleNamespace(interrupted=False, sampling_step=0, assign_current_image=lambda image: None),
        sd_model=types.SimpleNamespace(
            is_sdxl=False,
            is_sd2=False,
            parameterization="eps",
            sd_checkpoint_info=types.SimpleNamespace(filename="sd15.safetensors", shorthash="sd15"),
            model=types.SimpleNamespace(conditioning_key="crossattn"),
        ),
        device=torch.device("cpu"),
        parallel_processing_allowed=True,
        total_tqdm=types.SimpleNamespace(update=lambda *args, **kwargs: None),
    )

    def schedule_function(n, sigma_min, sigma_max, device, inner_model=None):
        value = 2.0 if shared.sd_model.is_sdxl else 1.0
        middle = float(inner_model.sigmas[1]) if inner_model is not None and inner_model.sigmas.numel() > 2 else 0.0
        return torch.tensor([value, middle, float(sigma_min), float(sigma_max), 0.0], device=device)

    scheduler = types.SimpleNamespace(
        label="Align Your Steps",
        function=schedule_function,
        default_rho=-1,
        need_inner_model=True,
    )
    sd_schedulers = _module(
        "modules.sd_schedulers",
        schedulers=[],
        schedulers_map={"Align Your Steps": scheduler},
    )
    sd_samplers_cfg_denoiser = _module(
        "modules.sd_samplers_cfg_denoiser",
        CFGDenoiser=type("CFGDenoiser", (torch.nn.Module,), {"__init__": lambda self, sampler: torch.nn.Module.__init__(self)}),
    )
    script_callbacks = _module(
        "modules.script_callbacks",
        ExtraNoiseParams=object,
        extra_noise_callback=lambda *args, **kwargs: None,
    )
    modules_pkg = _module("modules")
    modules_pkg.__path__ = ["modules"]

    for name, module in (
        ("modules", modules_pkg),
        ("k_diffusion", k_diffusion_module),
        ("k_diffusion.sampling", sampling_module),
        ("torchsde", torchsde_module),
        ("torchsde._brownian", torchsde_brownian),
        ("torchsde._brownian.brownian_interval", brownian_interval),
        ("modules.devices", devices),
        ("modules.images", _module("modules.images", image_grid=lambda images: images)),
        ("modules.sd_vae_approx", _module("modules.sd_vae_approx")),
        ("modules.sd_vae_taesd", _module("modules.sd_vae_taesd")),
        ("modules.sd_models", _module("modules.sd_models", SkipWritingToConfig=lambda: torch.no_grad(), reload_model_weights=lambda *args, **kwargs: None)),
        ("modules.sd_samplers", _module("modules.sd_samplers", find_sampler_config=lambda name: None)),
        ("modules.shared", shared),
        ("modules.sd_samplers_extra", _module("modules.sd_samplers_extra", restart_sampler=lambda *args, **kwargs: None)),
        ("modules.sd_samplers_cfg_denoiser", sd_samplers_cfg_denoiser),
        ("modules.sd_schedulers", sd_schedulers),
        ("modules.script_callbacks", script_callbacks),
    ):
        put(name, module)

    try:
        common_spec = importlib.util.spec_from_file_location("modules.sd_samplers_common", "modules/sd_samplers_common.py")
        common = importlib.util.module_from_spec(common_spec)
        put("modules.sd_samplers_common", common)
        common_spec.loader.exec_module(common)

        profile_spec = importlib.util.spec_from_file_location("modules.openclaw_generation_profile", "modules/openclaw_generation_profile.py")
        profile = importlib.util.module_from_spec(profile_spec)
        put("modules.openclaw_generation_profile", profile)
        profile_spec.loader.exec_module(profile)
        profile.clear()

        spec = importlib.util.spec_from_file_location("test_kdiffusion_module", "modules/sd_samplers_kdiffusion.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["test_kdiffusion_module"] = module
        spec.loader.exec_module(module)
        return module, shared, profile
    finally:
        sys.modules.pop("test_kdiffusion_module", None)
        for name, original in reversed(list(originals.items())):
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def make_sampler(module, model_sigmas=None):
    sampler = object.__new__(module.KDiffusionSampler)
    sampler.config = types.SimpleNamespace(name="DPM++ SDE", options={})
    sampler.funcname = "sample_euler"
    sampler.model_wrap = types.SimpleNamespace(sigmas=model_sigmas if model_sigmas is not None else torch.tensor([0.1, 1.0, 10.0]))
    return sampler


def make_processing():
    return types.SimpleNamespace(
        hr_scheduler=None,
        is_hr_pass=False,
        scheduler="Align Your Steps",
        extra_generation_params={},
        sampler_noise_scheduler_override=None,
    )


class KDiffusionSigmasCacheTests(unittest.TestCase):
    def test_sigmas_cache_hits_return_equal_fresh_tensors(self):
        module, _shared, profile = load_kdiffusion_module()
        sampler = make_sampler(module)

        first = sampler.get_sigmas(make_processing(), 20)
        second = sampler.get_sigmas(make_processing(), 20)

        torch.testing.assert_close(second, first)
        self.assertIsNot(second, first)
        status = profile.status()
        self.assertEqual(status["hits"], 1)
        self.assertEqual(status["misses"], 1)
        self.assertEqual(status["stores"], 1)

    def test_in_place_sigma_mutation_does_not_poison_cached_schedule(self):
        module, _shared, profile = load_kdiffusion_module()
        sampler = make_sampler(module)

        first = sampler.get_sigmas(make_processing(), 20)
        expected = first.clone()
        first.add_(999.0)
        second = sampler.get_sigmas(make_processing(), 20)

        torch.testing.assert_close(second, expected)
        self.assertEqual(profile.status()["hits"], 1)

    def test_model_schedule_in_place_mutation_changes_semantic_cache_key(self):
        module, _shared, profile = load_kdiffusion_module()
        sampler = make_sampler(module)

        first = sampler.get_sigmas(make_processing(), 20)
        sampler.model_wrap.sigmas[1] = 2.0
        second = sampler.get_sigmas(make_processing(), 20)

        self.assertEqual(float(first[1]), 1.0)
        self.assertEqual(float(second[1]), 2.0)
        status = profile.status()
        self.assertEqual(status["misses"], 2)
        self.assertEqual(status["hits"], 0)

    def test_sigmas_cache_is_invalidated_by_option_and_model_schedule_signature(self):
        module, shared, profile = load_kdiffusion_module()
        sampler = make_sampler(module)

        sd15_sigmas = sampler.get_sigmas(make_processing(), 20)
        shared.opts.sigma_min = 0.2
        option_sigmas = sampler.get_sigmas(make_processing(), 20)
        shared.sd_model = types.SimpleNamespace(
            is_sdxl=True,
            is_sd2=False,
            parameterization="eps",
            sd_checkpoint_info=types.SimpleNamespace(filename="sdxl.safetensors", shorthash="sdxl"),
            model=types.SimpleNamespace(conditioning_key="crossattn"),
        )
        sdxl_sigmas = sampler.get_sigmas(make_processing(), 20)

        self.assertEqual(float(sd15_sigmas[0]), 1.0)
        self.assertAlmostEqual(float(option_sigmas[2]), 0.2, places=6)
        self.assertEqual(float(sdxl_sigmas[0]), 2.0)
        self.assertEqual(profile.status()["misses"], 3)

    def test_generation_profile_cache_preserves_dtype_device_values_and_rng_state(self):
        _module, _shared, profile = load_kdiffusion_module()
        key = ("direct", "dtype-device")
        expected = torch.tensor([0.125, 0.5, 2.0], dtype=torch.float64, device=torch.device("cpu"))

        torch.manual_seed(12345)
        rng_before = torch.random.get_rng_state().clone()
        first = profile.cached_tensor("direct", "dtype-device", None, 3, torch.device("cpu"), torch.float64, lambda: expected.clone(), params=key)
        first.mul_(1000.0)
        second = profile.cached_tensor("direct", "dtype-device", None, 3, torch.device("cpu"), torch.float64, lambda: torch.zeros_like(expected), params=key)
        rng_after = torch.random.get_rng_state()

        torch.testing.assert_close(second, expected)
        torch.testing.assert_close(rng_after, rng_before)
        self.assertEqual(second.dtype, torch.float64)
        self.assertEqual(second.device, torch.device("cpu"))

    def test_cache_can_be_bounded_by_environment(self):
        _module, _shared, profile = load_kdiffusion_module()
        os.environ["OPENCLAW_GENERATION_PROFILE_CACHE_MAX"] = "1"
        profile.clear()

        profile.cached_tensor("direct", "a", None, 1, torch.device("cpu"), torch.float32, lambda: torch.ones(1))
        profile.cached_tensor("direct", "b", None, 1, torch.device("cpu"), torch.float32, lambda: torch.ones(1) * 2)

        status = profile.status()
        self.assertEqual(status["cache_size"], 1)
        self.assertEqual(status["evictions"], 1)


if __name__ == "__main__":
    unittest.main()
