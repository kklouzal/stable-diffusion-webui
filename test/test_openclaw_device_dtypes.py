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
