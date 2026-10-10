"""The reference (reference_only/adain/adain+attn) and ReVision q_sample noise comes from a generator seeded per
process.sample() call from the batch's first image seed, not from the global RNG."""
import importlib
import types
import unittest

import torch

utils = importlib.import_module("extensions.sd-webui-controlnet.tests.utils", "utils")

from modules import rng  # noqa: E402
from scripts.hook import AbstractLowScaleModel, UnetHook  # noqa: E402

hook_leak_test = importlib.import_module("extensions.sd-webui-controlnet.tests.cn_script.hook_leak_test")
FakeUNet, control_param = hook_leak_test.FakeUNet, hook_leak_test.control_param

SHAPE = (2, 4, 8, 8)


def sample_noise(seeds, draws=2, prior_global_draws=0):
    """Hook a fake UNet, run one p.sample() that draws hook noise `draws` times, after consuming the global RNG."""
    hook = UnetHook()
    drawn = []

    def body():
        drawn.extend(hook.noise_like(torch.zeros(SHAPE, dtype=torch.bfloat16)) for _ in range(draws))
        return drawn

    sd_model = types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=FakeUNet()), is_sdxl=False)
    p = types.SimpleNamespace(sd_model=sd_model, seeds=seeds, sample=lambda *args, **kwargs: body())
    hook.hook(model=sd_model.model.diffusion_model, sd_ldm=sd_model, control_params=[control_param()], process=p)
    try:
        torch.manual_seed(prior_global_draws)
        torch.randn(prior_global_draws * 7 + 1)
        p.sample(conditioning=[], unconditional_conditioning=[])
        assert hook.noise_generator is None  # dropped when the sample() call ends
        return drawn
    finally:
        hook.restore()


class TestRequestNoise(unittest.TestCase):
    def test_same_seed_same_noise_whatever_the_global_rng_did(self):
        first = sample_noise([123, 124], prior_global_draws=0)
        second = sample_noise([123, 124], prior_global_draws=5)
        for a, b in zip(first, second):
            self.assertEqual(a.dtype, torch.bfloat16)
            self.assertEqual(tuple(a.shape), SHAPE)
            self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(first[0], first[1]))  # successive calls of one sample() draw on

    def test_noise_follows_the_first_image_seed(self):
        self.assertFalse(torch.equal(sample_noise([123])[0], sample_noise([124])[0]))
        self.assertTrue(torch.equal(sample_noise([123])[0], sample_noise([123, 999])[0]))

    def test_not_the_initial_latent_noise_of_that_seed(self):
        initial = rng.randn_without_seed(SHAPE, generator=rng.create_generator(123)).to(torch.bfloat16)
        self.assertFalse(torch.equal(sample_noise([123])[0], initial))

    def test_revision_q_sample_takes_the_hook_noise(self):
        x0 = torch.rand(1, 1280)
        t = torch.tensor([500])
        noise = torch.randn(1, 1280, generator=torch.Generator().manual_seed(0))
        sampler = AbstractLowScaleModel()
        expected = sampler.sqrt_alphas_cumprod[500].float() * x0 + sampler.sqrt_one_minus_alphas_cumprod[500].float() * noise
        torch.testing.assert_close(sampler.q_sample(x0, t, noise), expected)


if __name__ == "__main__":
    unittest.main()
