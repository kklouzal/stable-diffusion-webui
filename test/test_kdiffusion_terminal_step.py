import importlib
import inspect

import pytest
import torch

from test.helpers import add_repositories_to_sys_path, load_source

SDE_SAMPLERS = ("sample_dpmpp_2m_sde", "sample_dpmpp_3m_sde")


@pytest.fixture(scope="module")
def sampling():
    """The real k_diffusion.sampling with modules/sd_samplers_extra.py's terminal-step patch applied (it patches the
    module in place, as it does at webui startup)."""
    add_repositories_to_sys_path("k-diffusion")
    sampling = importlib.import_module("k_diffusion.sampling")
    load_source("sd_samplers_extra_under_test", "modules/sd_samplers_extra.py")
    return sampling


def toy_model(x, sigma, **kwargs):
    # A deterministic denoiser that depends on sigma and an extra argument, so dropped arguments would show.
    return x / (1 + sigma.reshape(-1, 1, 1, 1) ** 2) + kwargs.get("shift", 0.0)


def seeded_noise_sampler(x, seed):
    generator = torch.Generator().manual_seed(seed)
    return lambda sigma, sigma_next: torch.randn(x.shape, generator=generator, dtype=x.dtype)


@pytest.mark.parametrize("name", SDE_SAMPLERS)
def test_original_sampler_fails_on_a_single_terminal_step(sampling, name):
    original = getattr(sampling, name).__wrapped__
    x = torch.randn(1, 4, 8, 8)

    with pytest.raises(UnboundLocalError):
        original(toy_model, x, torch.tensor([1.5, 0.0]), disable=True, noise_sampler=seeded_noise_sampler(x, 0))


@pytest.mark.parametrize("name", SDE_SAMPLERS)
def test_single_terminal_step_is_the_denoising_step(sampling, name):
    x = torch.randn(2, 4, 8, 8, generator=torch.Generator().manual_seed(1))
    sigmas = torch.tensor([1.5, 0.0])
    callbacks = []

    out = getattr(sampling, name)(toy_model, x, sigmas, extra_args={"shift": 0.25}, callback=callbacks.append, disable=True, noise_sampler=seeded_noise_sampler(x, 0))

    expected = toy_model(x, torch.full((2,), 1.5), shift=0.25)
    assert torch.equal(out, expected)
    assert [(c["i"], float(c["sigma"])) for c in callbacks] == [(0, 1.5)]
    assert torch.equal(callbacks[0]["denoised"], expected)


@pytest.mark.parametrize("name", SDE_SAMPLERS)
@pytest.mark.parametrize("sigmas", [[1.5, 0.7, 0.0], [14.6, 3.0, 0.9, 0.2, 0.0], [2.0, 1.0]])
def test_longer_schedules_run_the_original_sampler_bit_for_bit(sampling, name, sigmas):
    patched = getattr(sampling, name)
    original = patched.__wrapped__
    x = torch.randn(2, 4, 8, 8, generator=torch.Generator().manual_seed(2))
    sigmas = torch.tensor(sigmas)

    out = patched(toy_model, x, sigmas, extra_args={"shift": 0.1}, disable=True, noise_sampler=seeded_noise_sampler(x, 3))
    expected = original(toy_model, x, sigmas, extra_args={"shift": 0.1}, disable=True, noise_sampler=seeded_noise_sampler(x, 3))

    assert torch.equal(out, expected)


@pytest.mark.parametrize("name", SDE_SAMPLERS)
def test_patch_keeps_the_signature_and_is_applied_once(sampling, name):
    patched = getattr(sampling, name)
    # KDiffusionSampler routes eta/s_noise/noise_sampler/solver_type by inspecting the signature.
    assert inspect.signature(patched) == inspect.signature(patched.__wrapped__)

    load_source("sd_samplers_extra_reloaded", "modules/sd_samplers_extra.py")
    assert getattr(sampling, name) is patched
