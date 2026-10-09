"""KDiffusionSampler.get_sigmas on the real modules: the hires schedule type it records reads back as the schedule the
hires pass ran, and the model's sigma table is copied to the host once per wrapper, not on every call."""

from types import SimpleNamespace

import pytest
import torch

from test.helpers import add_repositories_to_sys_path, init_shared

shared = init_shared()
add_repositories_to_sys_path("k-diffusion")

# sd_samplers first: importing sd_samplers_common on its own runs into a circular import
from modules import sd_models, sd_samplers, sd_samplers_common, sd_samplers_kdiffusion  # noqa: E402,F401


class Wrapper:
    """A k-diffusion wrapper stand-in: its device sigma table and its own schedule (the model default)."""

    def __init__(self):
        alphas = torch.linspace(0.9991, 0.0047, 1000, dtype=torch.float64).float()
        self.sigmas = ((1 - alphas) / alphas) ** 0.5
        self.quantize = False

    def get_sigmas(self, n):
        return torch.cat([self.sigmas.flip(0)[torch.linspace(0, 999, n).round().long()], torch.zeros(1)])


@pytest.fixture
def make_sampler(monkeypatch):
    model = SimpleNamespace(model=SimpleNamespace(conditioning_key="crossattn"), create_denoiser=Wrapper, sd_checkpoint_info=None)
    model_data = sd_models.SdModelData()
    model_data.sd_model, model_data.was_loaded_at_least_once = model, True
    monkeypatch.setattr(sd_models, "model_data", model_data)
    monkeypatch.setitem(shared.opts.data, "always_discard_next_to_last_sigma", False)

    def make(label):
        config = sd_samplers_kdiffusion.k_diffusion_samplers_map[label]
        sampler = config.constructor(shared.sd_model)
        sampler.config = config
        return sampler

    return make


def processing(**fields):
    defaults = dict(scheduler="Automatic", hr_scheduler="Automatic", is_hr_pass=False, extra_generation_params={}, sampler_noise_scheduler_override=None)
    return SimpleNamespace(**{**defaults, **fields})


def test_hires_on_the_model_schedule_is_recorded_and_reads_back(make_sampler):
    p = processing(scheduler="Karras")
    make_sampler("DPM++ 2M").get_sigmas(p, 10)
    assert p.extra_generation_params["Schedule type"] == "Karras"

    p.is_hr_pass = True
    p.extra_generation_params["Hires schedule type"] = None  # processing's placeholder
    make_sampler("Euler a").get_sigmas(p, 10)  # no default scheduler: the model's own schedule

    assert p.extra_generation_params["Hires schedule type"] == "Automatic"
    infotext = {"Sampler": "DPM++ 2M", "Schedule type": "Karras", "Hires sampler": "Euler a", "Hires schedule type": "Automatic"}
    hr_sampler, hr_scheduler = sd_samplers.get_hr_sampler_and_scheduler(infotext)
    assert (hr_sampler, hr_scheduler) == ("Euler a", "Automatic")
    resolved = sd_samplers_kdiffusion.k_diffusion_samplers_map[hr_sampler].options.get("scheduler")
    assert resolved is None  # "Automatic" with Euler a is the model's schedule again


def test_hires_with_the_same_schedule_records_nothing(make_sampler):
    p = processing(scheduler="Karras", hr_scheduler="Karras")
    make_sampler("DPM++ 2M").get_sigmas(p, 10)
    p.is_hr_pass = True
    p.extra_generation_params["Hires schedule type"] = None
    make_sampler("DPM++ 2M").get_sigmas(p, 10)

    assert p.extra_generation_params["Hires schedule type"] is None


def test_model_schedule_without_a_first_pass_schedule_type_records_nothing(make_sampler):
    p = processing()
    make_sampler("Euler a").get_sigmas(p, 10)
    p.is_hr_pass = True
    p.extra_generation_params["Hires schedule type"] = None
    make_sampler("Euler a").get_sigmas(p, 10)

    assert "Schedule type" not in p.extra_generation_params and p.extra_generation_params["Hires schedule type"] is None


def test_sigma_table_is_copied_to_the_host_once_per_wrapper(make_sampler, monkeypatch):
    sampler = make_sampler("DPM++ 2M")
    table = sampler.model_wrap.sigmas
    first = sd_samplers_common.cpu_sigmas(sampler.model_wrap)

    for steps in (10, 20, 10):
        sampler.get_sigmas(processing(scheduler="Karras"), steps)

    assert sampler.model_wrap.openclaw_cpu_sigmas[0] is table
    assert sampler.model_wrap.openclaw_cpu_sigmas[2] is first
    assert torch.equal(first, table)

    sampler.model_wrap.sigmas = table * 2  # a replaced table is copied again
    assert torch.equal(sd_samplers_common.cpu_sigmas(sampler.model_wrap), table * 2)
    with torch.no_grad():
        sampler.model_wrap.sigmas.add_(1)  # tracked in-place change (outside inference mode)
    assert torch.equal(sd_samplers_common.cpu_sigmas(sampler.model_wrap), table * 2 + 1)
