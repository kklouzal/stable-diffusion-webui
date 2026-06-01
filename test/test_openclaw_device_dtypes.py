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


if __name__ == "__main__":
    unittest.main()
