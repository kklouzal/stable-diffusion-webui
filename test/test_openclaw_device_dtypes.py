import sys
import types
import unittest
from unittest import mock

import torch

from test.helpers import available_test_device, load_source, module, stub_modules


def load_unipc_module():
    return load_source("test_unipc_module", "modules/models/diffusion/uni_pc/uni_pc.py")


def load_timesteps_impl_module():
    unipc_module = load_unipc_module()
    shared_module = module("modules.shared", opts=types.SimpleNamespace())
    torch_utils_module = module("modules.torch_utils", float64=lambda tensor: torch.float32 if tensor.device.type == "xpu" else torch.float64)
    k_diffusion_sampling = module("k_diffusion.sampling", torch=torch)
    uni_pc_pkg = module("modules.models.diffusion.uni_pc", uni_pc=unipc_module)
    diffusion_pkg = module("modules.models.diffusion", uni_pc=uni_pc_pkg)
    models_pkg = module("modules.models", diffusion=diffusion_pkg)

    return load_source("test_timesteps_impl_module", "modules/sd_samplers_timesteps_impl.py", {
        "modules": module("modules", shared=shared_module, models=models_pkg, torch_utils=torch_utils_module),
        "modules.shared": shared_module,
        "modules.models": models_pkg,
        "modules.models.diffusion": diffusion_pkg,
        "modules.models.diffusion.uni_pc": uni_pc_pkg,
        "modules.models.diffusion.uni_pc.uni_pc": unipc_module,
        "modules.torch_utils": torch_utils_module,
        "k_diffusion": module("k_diffusion", package=True, sampling=k_diffusion_sampling),
        "k_diffusion.sampling": k_diffusion_sampling,
    })


def load_timesteps_sampler_module(device=None):
    class CFGDenoiser:
        def __init__(self, sampler):
            self.sampler = sampler
            self.model_wrap = None
            self.p = None

        def update_inner_model(self):
            self.model_wrap = None
            cond, uncond = self.p.get_conds()
            self.sampler.sampler_extra_args["cond"] = cond
            self.sampler.sampler_extra_args["uncond"] = uncond

    class Sampler:
        def __init__(self, funcname):
            self.funcname = funcname
            self.config = None

    class SamplerData(tuple):
        def __new__(cls, name, constructor, aliases, options):
            return tuple.__new__(cls, (name, constructor, aliases, options))

    generation_profile_module = module("modules.openclaw_generation_profile", cached_tensor=lambda **kwargs: kwargs["factory"]())

    return load_source("test_timesteps_sampler_module", "modules/sd_samplers_timesteps.py", {
        "modules": module("modules", openclaw_generation_profile=generation_profile_module),
        "modules.devices": module("modules.devices", device=device or torch.device("cpu")),
        "modules.script_callbacks": module("modules.script_callbacks", ExtraNoiseParams=object, extra_noise_callback=lambda params: None),
        "modules.sd_samplers_cfg_denoiser": module("modules.sd_samplers_cfg_denoiser", CFGDenoiser=CFGDenoiser),
        "modules.sd_samplers_common": module("modules.sd_samplers_common", Sampler=Sampler, SamplerData=SamplerData),
        "modules.sd_samplers_timesteps_impl": module(
            "modules.sd_samplers_timesteps_impl",
            ddim=lambda *args, **kwargs: None,
            ddim_cfgpp=lambda *args, **kwargs: None,
            plms=lambda *args, **kwargs: None,
            unipc=lambda *args, **kwargs: None,
        ),
        "modules.openclaw_generation_profile": generation_profile_module,
        "modules.shared": module(
            "modules.shared",
            opts=types.SimpleNamespace(always_discard_next_to_last_sigma=False),
            sd_model=types.SimpleNamespace(parameterization="eps", alphas_cumprod=torch.linspace(0.999, 0.001, 1000)),
        ),
        # Importing sd_samplers_timesteps aliases itself as modules.sd_samplers_compvis; that alias must not outlive the load.
        "modules.sd_samplers_compvis": None,
    })


class OpenClawDeviceDtypeTests(unittest.TestCase):
    def test_module_loaders_restore_sys_modules(self):
        sentinel = module("modules.sd_samplers_compvis")
        names = ("modules", "modules.shared", "modules.devices", "modules.openclaw_generation_profile", "modules.sd_samplers_compvis", "k_diffusion", "k_diffusion.sampling")
        with stub_modules({"modules.sd_samplers_compvis": sentinel}):
            before = {name: sys.modules.get(name) for name in names}
            load_timesteps_impl_module()
            load_timesteps_sampler_module()
            after = {name: sys.modules.get(name) for name in names}
        self.assertEqual({name: id(module) for name, module in after.items()}, {name: id(module) for name, module in before.items()})


    def test_timestep_schedule_preserves_existing_stride_for_normal_counts(self):
        sampler_module = load_timesteps_sampler_module()

        timesteps = sampler_module._make_timesteps(20, torch.device("cpu"))

        self.assertEqual(timesteps.tolist(), list(range(1, 1000, 50)))

    def test_timestep_schedule_handles_more_than_training_steps(self):
        sampler_module = load_timesteps_sampler_module()

        timesteps = sampler_module._make_timesteps(1001, torch.device("cpu"))

        self.assertEqual(len(timesteps), 1000)
        self.assertEqual(timesteps[0].item(), 1)
        self.assertEqual(timesteps[-1].item(), 999)

    def test_timestep_schedule_rejects_nonpositive_steps(self):
        sampler_module = load_timesteps_sampler_module()

        with self.assertRaises(ValueError):
            sampler_module._make_timesteps(0, torch.device("cpu"))

    def test_generation_profile_cache_returns_fresh_timesteps(self):
        module = load_timesteps_sampler_module()
        real_profile = load_source("modules.openclaw_generation_profile", "modules/openclaw_generation_profile.py")
        module.openclaw_generation_profile = real_profile

        class Sampler(module.CompVisSampler):
            def __init__(self):
                self.config = types.SimpleNamespace(name="DDIM", options={})
                self.funcname = "ddim"

        real_profile.clear()
        sampler = Sampler()
        processing = types.SimpleNamespace(extra_generation_params={})

        first = sampler.get_timesteps(processing, 20)
        expected = first.clone()
        first.fill_(123)
        second = sampler.get_timesteps(processing, 20)

        torch.testing.assert_close(second, expected)
        self.assertIsNot(second, first)
        self.assertEqual(real_profile.status()["hits"], 1)

    def test_timesteps_refiner_refreshes_alphas(self):
        module = load_timesteps_sampler_module()

        class Denoiser(module.CFGDenoiserTimesteps):
            @property
            def inner_model(self):
                return object()

        sampler = types.SimpleNamespace(sampler_extra_args={})
        denoiser = Denoiser(sampler)
        original = denoiser.alphas
        module.shared.sd_model = types.SimpleNamespace(
            parameterization="eps",
            alphas_cumprod=torch.linspace(0.5, 0.25, 1000),
        )
        denoiser.p = types.SimpleNamespace(get_conds=lambda: ("cond", "uncond"))

        denoiser.update_inner_model()

        self.assertIs(denoiser.alphas, module.shared.sd_model.alphas_cumprod)
        self.assertIsNot(denoiser.alphas, original)
        self.assertEqual(sampler.sampler_extra_args["cond"], "cond")
        self.assertEqual(sampler.sampler_extra_args["uncond"], "uncond")

    def test_vae_cheap_approximation_preserves_sample_dtype(self):
        with mock.patch.object(sys, "argv", [sys.argv[0]]):
            from modules import sd_vae_approx

        sample = torch.ones((1, 4, 2, 2), dtype=torch.float16)
        fake_model = types.SimpleNamespace(is_sd3=False, is_sdxl=False)
        fake_shared = types.SimpleNamespace(sd_model=fake_model)
        with mock.patch.object(sd_vae_approx, "shared", fake_shared):
            result = sd_vae_approx.cheap_approximation(sample)

        self.assertEqual(result.device, sample.device)
        self.assertEqual(result.dtype, sample.dtype)

    def test_unipc_time_steps_stay_on_requested_device(self):
        unipc = load_unipc_module()

        device = available_test_device()
        sampler = unipc.UniPC(lambda x, t, cond=None, uncond=None: x, unipc.NoiseScheduleVP("linear"))

        timesteps = sampler.get_time_steps("logSNR", 1.0, 0.01, 4, device)

        self.assertEqual(timesteps.device, device)

    def test_unipc_final_callback_uses_current_latent_without_extra_model_eval(self):
        unipc = load_unipc_module()

        model_calls = 0
        callbacks = []

        def model_fn(x, t, cond=None, uncond=None):
            nonlocal model_calls
            model_calls += 1
            while t.dim() < x.dim():
                t = t.unsqueeze(-1)
            return x.mul(0.25).add(t.mul(0.125))

        def after_update(x, model_x):
            callbacks.append((x.detach().clone(), None if model_x is None else model_x.detach().clone()))

        sampler = unipc.UniPC(
            model_fn,
            unipc.NoiseScheduleVP("linear"),
            predict_x0=True,
            variant="bh1",
            after_update=after_update,
        )
        x = torch.ones((1, 1, 2, 2), dtype=torch.float64)

        sampler.sample(x, steps=3, order=2, skip_type="time_uniform")

        self.assertEqual(len(callbacks), 3)
        self.assertEqual(model_calls, 3)
        self.assertTrue(all(model_x is not None for _, model_x in callbacks))
        torch.testing.assert_close(callbacks[-1][1], callbacks[-1][0])

    def test_timestep_sampler_callback_counts_match_transition_contract(self):
        sd_samplers_timesteps_impl = load_timesteps_impl_module()

        class FakeInner:
            def __init__(self):
                self.alphas_cumprod = torch.linspace(0.999, 0.001, 1000, dtype=torch.float64)

        class FakeModel:
            def __init__(self):
                self.inner_model = types.SimpleNamespace(inner_model=FakeInner())
                self.last_noise_uncond = None

            def __call__(self, x, t, **kwargs):
                self.last_noise_uncond = torch.zeros_like(x)
                return torch.zeros_like(x)

        x = torch.ones((1, 1, 2, 2), dtype=torch.float64)
        timesteps = torch.tensor([1, 251, 501, 751], dtype=torch.long)

        for sampler in (sd_samplers_timesteps_impl.ddim, sd_samplers_timesteps_impl.ddim_cfgpp, sd_samplers_timesteps_impl.plms):
            callbacks = []
            sampler(FakeModel(), x.clone(), timesteps, extra_args={}, callback=callbacks.append, disable=True)

            self.assertEqual([payload["i"] for payload in callbacks], [0, 1, 2])

    def test_unipc_wrapper_callback_count_matches_solver_steps(self):
        sd_samplers_timesteps_impl = load_timesteps_impl_module()

        class FakeInner:
            def __init__(self):
                self.alphas_cumprod = torch.linspace(0.999, 0.001, 1000, dtype=torch.float64)

        class FakeModel:
            def __init__(self):
                self.inner_model = types.SimpleNamespace(inner_model=FakeInner())

            def __call__(self, x, t, **kwargs):
                return torch.zeros_like(x)

        class QuietTqdm:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def update(self, *args, **kwargs):
                pass

        x = torch.ones((1, 1, 2, 2), dtype=torch.float64)
        timesteps = torch.tensor([1, 251, 501, 751], dtype=torch.long)
        callbacks = []
        fake_opts = types.SimpleNamespace(uni_pc_variant="bh1", uni_pc_skip_type="time_uniform", uni_pc_order=2, uni_pc_lower_order_final=True)

        with (
            mock.patch.object(sd_samplers_timesteps_impl.shared, "opts", fake_opts),
            mock.patch.object(sd_samplers_timesteps_impl.uni_pc.tqdm, "tqdm", QuietTqdm),
        ):
            sd_samplers_timesteps_impl.unipc(FakeModel(), x, timesteps, extra_args={}, callback=callbacks.append)

        self.assertEqual([payload["i"] for payload in callbacks], [0, 1, 2, 3])

    def test_unipc_multistep_coefficients_match_cpu_and_cuda(self):
        if available_test_device().type != "cuda":
            self.skipTest("CUDA is not available for UniPC coefficient parity")

        unipc = load_unipc_module()

        def run_update(device, variant, order, predict_x0, batch_size):
            dtype = torch.float64
            x = torch.arange(batch_size * 24, device=device, dtype=dtype).reshape(batch_size, 2, 3, 4) / 17.0
            t_values = [0.92, 0.74, 0.58]
            t_prev_list = [torch.full((batch_size,), t_values[i], device=device, dtype=dtype) for i in range(order)]
            t = torch.full((batch_size,), 0.42, device=device, dtype=dtype)
            model_prev_list = [x.mul(0.125 * (i + 1)).add(0.03125 * i) for i in range(order)]

            def model_fn(x_in, t_in, cond=None, uncond=None):
                while t_in.dim() < x_in.dim():
                    t_in = t_in.unsqueeze(-1)
                return x_in.mul(0.375).add(t_in.mul(0.0625))

            sampler = unipc.UniPC(
                model_fn,
                unipc.NoiseScheduleVP("linear"),
                predict_x0=predict_x0,
                variant=variant,
            )
            x_t, model_t = sampler.multistep_uni_pc_update(
                x, model_prev_list, t_prev_list, t, order, use_corrector=True
            )
            return x_t.detach().cpu(), model_t.detach().cpu()

        for variant in ("vary_coeff", "bh1", "bh2"):
            for order in (1, 2, 3):
                for predict_x0 in (False, True):
                    for batch_size in (1, 2):
                        with self.subTest(variant=variant, order=order, predict_x0=predict_x0, batch_size=batch_size):
                            cpu_x, cpu_model = run_update(torch.device("cpu"), variant, order, predict_x0, batch_size)
                            cuda_x, cuda_model = run_update(torch.device("cuda:0"), variant, order, predict_x0, batch_size)

                            torch.testing.assert_close(cuda_x, cpu_x, rtol=1e-8, atol=1e-10)
                            torch.testing.assert_close(cuda_model, cpu_model, rtol=1e-8, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
