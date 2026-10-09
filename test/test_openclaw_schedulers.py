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


class OpenClawSchedulerTests(unittest.TestCase):
    def test_normal_scheduler_uses_vector_timestep_conversion_when_available(self):
        schedulers = load_scheduler_module()
        inner_model = VectorInnerModel()

        sigmas = schedulers.normal_scheduler(5, 0.1, 10.0, inner_model, torch.device("cpu"))

        self.assertEqual(inner_model.t_to_sigma_shapes, [(5,)])
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

    def test_scheduler_map_resolves_names_labels_and_declared_aliases(self):
        schedulers = load_scheduler_module()

        sgm = schedulers.schedulers_map["sgm_uniform"]
        self.assertIs(schedulers.schedulers_map["SGM Uniform"], sgm)
        self.assertIs(schedulers.schedulers_map["SGMUniform"], sgm)


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

    @unittest.skipUnless(torch.cuda.is_available(), "needs a second device")
    def test_beta_scheduler_follows_inner_model_device_and_matches_paper_formula(self):
        # The real wrapper's sigma_to_t returns timesteps on the model device (CUDA) while sigmas are built with
        # device=cpu; the curve must be built next to the timesteps. Oracle: upstream per-element formula (2174ce5a).
        from scipy import stats
        import numpy as np

        schedulers = load_scheduler_module()

        class CudaInnerModel:
            def sigma_to_t(self, sigma):
                return sigma.to("cuda") * 10.0

            def t_to_sigma(self, t):
                return t / 10.0

        inner_model = CudaInnerModel()
        sigmas = schedulers.beta_scheduler(8, 0.1, 10.0, inner_model, torch.device("cpu"))

        curve = [stats.beta.ppf(x, 0.6, 0.6) for x in np.linspace(1, 0, 8)]
        start = inner_model.sigma_to_t(torch.tensor(10.0))
        end = inner_model.sigma_to_t(torch.tensor(0.1))
        expected = torch.tensor([float(inner_model.t_to_sigma(end + x * (start - end))) for x in curve] + [0.0])
        self.assertEqual(sigmas.device.type, "cpu")
        torch.testing.assert_close(sigmas, expected)

    def test_align_your_steps_matches_float64_reference_bit_for_bit(self):
        # Oracle: NVIDIA's AYS how-to / upstream loglinear_interp, np.interp in float64, rounded once to float32.
        import numpy as np

        schedulers = load_scheduler_module()
        base = [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029]

        for n in range(1, 61):
            with self.subTest(n=n):
                if n == len(base):
                    reference = np.asarray(base)
                else:
                    reference = np.exp(np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(base)), np.log(base[::-1])))[::-1]
                expected = torch.tensor(np.append(reference, 0.0), dtype=torch.float32)
                actual = schedulers.get_align_your_steps_sigmas(n, 0.1, 10.0, torch.device("cpu"))
                self.assertEqual(actual.dtype, torch.float32)
                self.assertTrue(torch.equal(actual, expected), (actual, expected))

    def test_kl_optimal_ends_at_zero_and_spans_sigma_max_to_sigma_min(self):
        schedulers = load_scheduler_module()

        for n in (1, 2, 5, 15, 40):
            with self.subTest(n=n):
                sigmas = schedulers.kl_optimal(n, 0.0292, 14.6146, torch.device("cpu"))
                self.assertEqual(sigmas.shape, (n + 1,))
                self.assertEqual(float(sigmas[-1]), 0.0)
                self.assertAlmostEqual(float(sigmas[0]), 14.6146, places=3)
                if n > 1:
                    self.assertAlmostEqual(float(sigmas[-2]), 0.0292, places=5)
                    # evenly spaced in arctan(sigma)
                    gaps = torch.diff(torch.atan(sigmas[:-1].double()))
                    torch.testing.assert_close(gaps, gaps.mean().expand_as(gaps), rtol=1e-4, atol=1e-6)
                self.assertTrue(bool((sigmas[1:] < sigmas[:-1]).all()))

    def test_kl_optimal_euler_run_returns_the_clean_sample(self):
        # With an exact denoiser, Euler steps along any schedule land on x0 exactly when the schedule ends at sigma 0;
        # a schedule that ends at sigma_min leaves sigma_min worth of noise in the output.
        schedulers = load_scheduler_module()
        generator = torch.Generator().manual_seed(0)
        x0 = torch.randn(2, 4, 8, 8, generator=generator, dtype=torch.float64)
        noise = torch.randn(2, 4, 8, 8, generator=generator, dtype=torch.float64)

        sigmas = schedulers.kl_optimal(10, 0.0292, 14.6146, torch.device("cpu")).double()
        x = x0 + noise * sigmas[0]
        for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:]):
            d = (x - x0) / sigma
            x = x + d * (sigma_next - sigma)

        torch.testing.assert_close(x, x0, rtol=0, atol=1e-12)

    def test_simple_scheduler_indexes_match_the_reference_float64_loop(self):
        schedulers = load_scheduler_module()
        inner_model = VectorInnerModel()
        table = inner_model.sigmas

        for n in range(1, 1001):
            ss = len(table) / n
            expected = torch.cat([table[[-(1 + int(i * ss)) for i in range(n)]], torch.zeros(1)])
            actual = schedulers.simple_scheduler(n, 0.1, 10.0, inner_model, torch.device("cpu"))
            if not torch.equal(actual, expected):
                self.fail(f"simple scheduler n={n} differs from the reference index loop")


if __name__ == "__main__":
    unittest.main()
