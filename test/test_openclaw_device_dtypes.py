import importlib.util
import os
import sys
import types
import unittest
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def load_unipc_module():
    spec = importlib.util.spec_from_file_location("test_unipc_module", "modules/models/diffusion/uni_pc/uni_pc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_timesteps_impl_module():
    module_name = "test_timesteps_impl_module"
    unipc_module = load_unipc_module()
    originals = {}

    def put(name, module):
        originals[name] = sys.modules.get(name)
        sys.modules[name] = module

    modules_pkg = types.ModuleType("modules")
    models_pkg = types.ModuleType("modules.models")
    diffusion_pkg = types.ModuleType("modules.models.diffusion")
    uni_pc_pkg = types.ModuleType("modules.models.diffusion.uni_pc")
    shared_module = types.ModuleType("modules.shared")
    torch_utils_module = types.ModuleType("modules.torch_utils")
    k_diffusion_pkg = types.ModuleType("k_diffusion")
    k_diffusion_pkg.__path__ = []
    k_diffusion_sampling = types.ModuleType("k_diffusion.sampling")

    shared_module.opts = types.SimpleNamespace()
    torch_utils_module.float64 = lambda tensor: torch.float32 if tensor.device.type in ("mps", "xpu") else torch.float64
    k_diffusion_sampling.torch = torch
    k_diffusion_pkg.sampling = k_diffusion_sampling
    uni_pc_pkg.uni_pc = unipc_module
    diffusion_pkg.uni_pc = uni_pc_pkg

    modules_pkg.shared = shared_module
    modules_pkg.models = models_pkg
    modules_pkg.torch_utils = torch_utils_module
    models_pkg.diffusion = diffusion_pkg

    for name, module in (
        ("modules", modules_pkg),
        ("modules.shared", shared_module),
        ("modules.models", models_pkg),
        ("modules.models.diffusion", diffusion_pkg),
        ("modules.models.diffusion.uni_pc", uni_pc_pkg),
        ("modules.models.diffusion.uni_pc.uni_pc", unipc_module),
        ("modules.torch_utils", torch_utils_module),
        ("k_diffusion", k_diffusion_pkg),
        ("k_diffusion.sampling", k_diffusion_sampling),
    ):
        put(name, module)

    try:
        spec = importlib.util.spec_from_file_location(module_name, "modules/sd_samplers_timesteps_impl.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    return module


def available_test_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        probe = torch.empty((), device="cuda:0")
        del probe
        return torch.device("cuda:0")
    except Exception:
        return torch.device("cpu")


class OpenClawDeviceDtypeTests(unittest.TestCase):
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

    def test_unipc_singlestep_indices_stay_on_requested_device(self):
        unipc = load_unipc_module()

        device = available_test_device()
        sampler = unipc.UniPC(lambda x, t, cond=None, uncond=None: x, unipc.NoiseScheduleVP("linear"))

        timesteps, orders = sampler.get_orders_and_timesteps_for_singlestep_solver(
            steps=5, order=2, skip_type="time_uniform", t_T=1.0, t_0=0.01, device=device
        )

        self.assertEqual(timesteps.device, device)
        self.assertEqual(orders, [2, 2, 1])

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

        sampler.sample(x, steps=3, order=2, skip_type="time_uniform", method="multistep")

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
