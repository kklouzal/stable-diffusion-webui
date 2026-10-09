import torch

from test.helpers import init_shared

shared = init_shared()

# sd_samplers first: importing the sampler modules on their own runs into an import cycle.
from modules import sd_samplers, sd_samplers_lcm  # noqa: E402,F401


def test_sample_lcm_does_not_modify_the_denoised_tensor_it_reported():
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(1, 4, 8, 8, generator=generator)
    noises = [torch.randn(1, 4, 8, 8, generator=generator) for _ in range(2)]
    sigmas = torch.tensor([3.0, 1.5, 0.5, 0.0])

    def model(x, sigma, **kwargs):
        return x * 0.5 + sigma.reshape(-1, 1, 1, 1)

    reported = []
    noise_iter = iter(noises)
    out = sd_samplers_lcm.sample_lcm(
        model, x, sigmas, disable=True,
        callback=lambda d: reported.append((d["denoised"], d["denoised"].clone())),
        noise_sampler=lambda sigma, sigma_next: next(noise_iter),
    )

    assert len(reported) == 3
    for kept, snapshot in reported:
        assert torch.equal(kept, snapshot)

    # oracle: the LCM step x = denoised + sigma_next * noise, with no noise on the final step
    expected = x
    for i in range(3):
        denoised = model(expected, sigmas[i] * torch.ones(1))
        expected = denoised + sigmas[i + 1] * noises[i] if i < 2 else denoised
    assert torch.equal(out, expected)


def test_lcm_denoiser_uses_every_skip_steps_th_alpha_like_the_reference_loop():
    alphas_cumprod = torch.cumprod(1 - torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=torch.float64) ** 2, 0).float()
    model = type("Model", (), {"alphas_cumprod": alphas_cumprod, "device": torch.device("cpu")})()

    denoiser = sd_samplers_lcm.LCMCompVisDenoiser(model)

    # oracle: the original per-index loop
    expected = torch.zeros(50)
    for x in range(50):
        expected[49 - x] = alphas_cumprod[999 - x * 20]
    assert torch.equal(denoiser.sigmas, ((1 - expected) / expected) ** 0.5)
