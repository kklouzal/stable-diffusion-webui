"""The k-diffusion stochasticity parameters (s_churn, s_tmin, s_tmax, s_noise) a sampler is called with.

The request's values win over the settings' (StableDiffusionProcessing.fill_fields_from_opts resolves them into p), a
value is passed and recorded in infotext only where it differs from k-diffusion's default, and every sampler function
that takes one of them gets it, including Euler a, DPM adaptive and Restart. Runs the real KDiffusionSampler.initialize
against the vendored k-diffusion signatures.
"""

import inspect
from types import SimpleNamespace

import pytest

from test.helpers import add_repositories_to_sys_path, init_shared

shared = init_shared()
add_repositories_to_sys_path("k-diffusion")

# sd_samplers first: importing sd_samplers_common on its own runs into a circular import
from modules import sd_samplers, sd_samplers_common, sd_samplers_kdiffusion  # noqa: E402,F401

DEFAULTS = dict(s_churn=0.0, s_tmin=0.0, s_tmax=0.0, s_noise=1.0)


@pytest.fixture
def make_sampler(monkeypatch):
    monkeypatch.setattr(shared, "sd_model", SimpleNamespace(model=SimpleNamespace(conditioning_key="crossattn"), create_denoiser=lambda: SimpleNamespace()), raising=False)

    def make(label):
        config = sd_samplers_kdiffusion.k_diffusion_samplers_map[label]
        sampler = config.constructor(shared.sd_model)
        sampler.config = config
        return sampler

    return make


def processing(**values):
    """p as process_images leaves it: s_* resolved by fill_fields_from_opts (s_tmax 0 is infinity)."""
    resolved = {**DEFAULTS, **values}
    resolved["s_tmax"] = resolved["s_tmax"] or float("inf")
    return SimpleNamespace(eta=None, s_min_uncond=0.0, extra_generation_params={}, rng=None, **resolved)


def set_opts(monkeypatch, **values):
    for name, value in {**DEFAULTS, **values}.items():
        monkeypatch.setitem(shared.opts.data, name, value)


def test_request_values_win_over_settings_and_are_recorded(monkeypatch, make_sampler):
    set_opts(monkeypatch, s_churn=0.3, s_noise=0.8)
    p = processing(s_churn=0.5, s_noise=0.9, s_tmin=0.2)

    kwargs = make_sampler("Euler").initialize(p)

    assert kwargs == {"s_churn": 0.5, "s_tmin": 0.2, "s_noise": 0.9}
    assert p.extra_generation_params == {"Sigma churn": 0.5, "Sigma tmin": 0.2, "Sigma noise": 0.9}


def test_settings_apply_when_the_request_leaves_them_unset(monkeypatch, make_sampler):
    set_opts(monkeypatch, s_churn=0.3, s_tmax=5.0)
    p = processing(s_churn=None, s_tmax=None)
    p.s_churn = p.s_tmax = None  # fill_fields_from_opts not run: the settings are the fallback

    kwargs = make_sampler("Heun").initialize(p)

    assert kwargs == {"s_churn": 0.3, "s_tmax": 5.0}
    assert p.extra_generation_params == {"Sigma churn": 0.3, "Sigma tmax": 5.0}


def test_request_value_differing_from_default_is_recorded_even_when_settings_are_default(monkeypatch, make_sampler):
    set_opts(monkeypatch)
    p = processing(s_noise=0.5)

    kwargs = make_sampler("DPM++ 2M SDE").initialize(p)

    assert kwargs == {"s_noise": 0.5, "eta": 1.0}
    assert p.extra_generation_params == {"Sigma noise": 0.5}


@pytest.mark.parametrize("label", sorted(sd_samplers_kdiffusion.k_diffusion_samplers_map))
def test_defaults_pass_nothing_and_record_nothing(monkeypatch, make_sampler, label):
    """The production path: at the defaults no s_* keyword is passed, so the sampler runs on its own defaults."""
    set_opts(monkeypatch, s_tmax=0.0)
    p = processing()

    kwargs = make_sampler(label).initialize(p)

    assert not set(kwargs) & set(DEFAULTS)
    assert not set(p.extra_generation_params) & set(sd_samplers_common.sigma_params_infotext.values())


@pytest.mark.parametrize("label", sorted(sd_samplers_kdiffusion.k_diffusion_samplers_map))
def test_every_sampler_taking_s_noise_gets_it(monkeypatch, make_sampler, label):
    set_opts(monkeypatch)
    sampler = make_sampler(label)
    p = processing(s_noise=0.7)

    kwargs = sampler.initialize(p)

    takes = "s_noise" in inspect.signature(sampler.func).parameters
    assert ("s_noise" in kwargs) == takes
    if takes:
        assert kwargs["s_noise"] == 0.7 and p.extra_generation_params["Sigma noise"] == 0.7


def test_euler_a_dpm_adaptive_and_restart_are_mapped():
    for name in ("sample_euler_ancestral", "sample_dpm_adaptive", "restart_sampler"):
        assert sd_samplers_kdiffusion.sampler_extra_params[name] == ["s_noise"]
