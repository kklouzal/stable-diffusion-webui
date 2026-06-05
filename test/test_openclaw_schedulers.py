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


def load_scheduler_module():
    originals = {}

    def put(name, module):
        originals[name] = sys.modules.get(name)
        sys.modules[name] = module

    sampling_module = _module(
        "k_diffusion.sampling",
        get_sigmas_karras=lambda *args, **kwargs: None,
        get_sigmas_exponential=lambda *args, **kwargs: None,
        get_sigmas_polyexponential=lambda *args, **kwargs: None,
    )
    k_diffusion_module = _module("k_diffusion", sampling=sampling_module)
    modules_pkg = _module("modules")
    shared_module = _module(
        "modules.shared",
        sd_model=types.SimpleNamespace(is_sdxl=False),
        opts=types.SimpleNamespace(beta_dist_alpha=0.6, beta_dist_beta=0.6),
    )

    for name, module in (
        ("k_diffusion", k_diffusion_module),
        ("k_diffusion.sampling", sampling_module),
        ("modules", modules_pkg),
        ("modules.shared", shared_module),
    ):
        put(name, module)

    try:
        spec = importlib.util.spec_from_file_location("test_scheduler_module", "modules/sd_schedulers.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["test_scheduler_module"] = module
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop("test_scheduler_module", None)
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    return module


class VectorInnerModel:
    sigmas = torch.linspace(10.0, 0.1, 1000)

    def __init__(self):
        self.t_to_sigma_shapes = []

    def sigma_to_t(self, sigma):
        return sigma * 10.0

    def t_to_sigma(self, t):
        self.t_to_sigma_shapes.append(tuple(t.shape))
        return t / 10.0


class ScalarOnlyInnerModel(VectorInnerModel):
    def t_to_sigma(self, t):
        self.t_to_sigma_shapes.append(tuple(t.shape))
        if t.ndim > 0:
            raise TypeError("vector timesteps unsupported")
        return t / 10.0


class OpenClawSchedulerTests(unittest.TestCase):
    def test_normal_scheduler_uses_vector_timestep_conversion_when_available(self):
        schedulers = load_scheduler_module()
        inner_model = VectorInnerModel()

        sigmas = schedulers.normal_scheduler(5, 0.1, 10.0, inner_model, torch.device("cpu"))

        self.assertEqual(inner_model.t_to_sigma_shapes, [(5,)])
        self.assertTrue(torch.allclose(sigmas, torch.tensor([10.0, 7.525, 5.05, 2.575, 0.1, 0.0])))

    def test_timestep_conversion_falls_back_for_scalar_only_inner_models(self):
        schedulers = load_scheduler_module()
        inner_model = ScalarOnlyInnerModel()

        sigmas = schedulers.normal_scheduler(5, 0.1, 10.0, inner_model, torch.device("cpu"))

        self.assertEqual(inner_model.t_to_sigma_shapes, [(5,), (), (), (), (), ()])
        self.assertTrue(torch.allclose(sigmas, torch.tensor([10.0, 7.525, 5.05, 2.575, 0.1, 0.0])))

    def test_simple_and_ddim_schedulers_preserve_existing_index_sequences(self):
        schedulers = load_scheduler_module()
        inner_model = VectorInnerModel()

        simple = schedulers.simple_scheduler(5, 0.1, 10.0, inner_model, torch.device("cpu"))
        ddim = schedulers.ddim_scheduler(5, 0.1, 10.0, inner_model, torch.device("cpu"))

        self.assertTrue(torch.equal(simple, torch.cat([inner_model.sigmas[[-1, -201, -401, -601, -801]], torch.zeros(1)])))
        self.assertTrue(torch.equal(ddim, torch.cat([inner_model.sigmas[[801, 601, 401, 201, 1]], torch.zeros(1)])))

    def test_align_your_steps_preserves_legacy_loglinear_values(self):
        schedulers = load_scheduler_module()

        interpolated = schedulers.get_align_your_steps_sigmas(5, 0.1, 10.0, torch.device("cpu"))
        native = schedulers.get_align_your_steps_sigmas(11, 0.1, 10.0, torch.device("cpu"))

        self.assertEqual(interpolated.device.type, "cpu")
        torch.testing.assert_close(
            interpolated,
            torch.tensor([14.615, 3.22693616, 1.396, 0.51004706, 0.029, 0.0]),
        )
        torch.testing.assert_close(
            native,
            torch.tensor([14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029, 0.0]),
        )

    def test_internal_schedulers_reject_nonpositive_step_counts(self):
        schedulers = load_scheduler_module()
        inner_model = VectorInnerModel()

        for scheduler in (schedulers.uniform, schedulers.sgm_uniform, schedulers.simple_scheduler, schedulers.normal_scheduler, schedulers.ddim_scheduler, schedulers.beta_scheduler):
            with self.subTest(scheduler=scheduler.__name__):
                with self.assertRaisesRegex(ValueError, "step count"):
                    scheduler(0, 0.1, 10.0, inner_model, torch.device("cpu"))

        for scheduler in (schedulers.get_align_your_steps_sigmas, schedulers.kl_optimal):
            with self.subTest(scheduler=scheduler.__name__):
                with self.assertRaisesRegex(ValueError, "step count"):
                    scheduler(0, 0.1, 10.0, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
