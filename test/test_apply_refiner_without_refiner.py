from types import SimpleNamespace

import torch

from test.helpers import init_shared

shared = init_shared()

# sd_samplers first: importing sd_samplers_common on its own runs into an import cycle.
from modules import sd_models, sd_samplers, sd_samplers_common  # noqa: E402,F401

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def loaded_model(monkeypatch):
    """shared.sd_model: the base checkpoint, loaded."""
    model_data = sd_models.SdModelData()
    model_data.sd_model, model_data.was_loaded_at_least_once = SimpleNamespace(sd_checkpoint_info=SimpleNamespace(short_title="base")), True
    monkeypatch.setattr(sd_models, "model_data", model_data)
    return model_data.sd_model


class _NoSigmasAccess:
    @property
    def sigmas(self):
        raise AssertionError("the sigma argmin ran although no refiner is configured")


class _CountingSigmas:
    def __init__(self, sigmas):
        self._sigmas = sigmas
        self.reads = 0

    @property
    def sigmas(self):
        self.reads += 1
        return self._sigmas


def _denoiser(inner_model, refiner_checkpoint_info=None, refiner_switch_at=0.8):
    p = SimpleNamespace(refiner_checkpoint_info=refiner_checkpoint_info, refiner_switch_at=refiner_switch_at, extra_generation_params={})
    return SimpleNamespace(p=p, inner_model=inner_model, step=3, total_steps=10)


def test_no_refiner_returns_before_the_sigma_argmin(monkeypatch):
    monkeypatch.setitem(shared.opts.data, "refiner_switch_by_sample_steps", False)
    denoiser = _denoiser(_NoSigmasAccess())

    assert sd_samplers_common.apply_refiner(denoiser, torch.tensor([2.5, 2.5])) is False
    assert denoiser.p.extra_generation_params == {}


def test_switch_by_sampling_steps_still_records_its_infotext_without_a_refiner(monkeypatch):
    monkeypatch.setitem(shared.opts.data, "refiner_switch_by_sample_steps", True)
    denoiser = _denoiser(_NoSigmasAccess())

    assert sd_samplers_common.apply_refiner(denoiser, torch.tensor([2.5])) is False
    assert denoiser.p.extra_generation_params == {"Refiner switch by sampling steps": True}

    monkeypatch.setitem(shared.opts.data, "refiner_switch_by_sample_steps", False)
    denoiser = _denoiser(_NoSigmasAccess())
    assert sd_samplers_common.apply_refiner(denoiser, None) is False
    assert denoiser.p.extra_generation_params == {"Refiner switch by sampling steps": True}


def test_configured_refiner_still_uses_the_sigma_timestep_before_its_switch_point(monkeypatch):
    monkeypatch.setitem(shared.opts.data, "refiner_switch_by_sample_steps", False)
    sigmas = _CountingSigmas(torch.linspace(0.03, 14.6, 1000))
    denoiser = _denoiser(sigmas, refiner_checkpoint_info=SimpleNamespace(short_title="refiner"), refiner_switch_at=0.8)

    # sigma at timestep 900 -> completed ratio (999 - 900) / 1000 < 0.8: no switch yet
    assert sd_samplers_common.apply_refiner(denoiser, sigmas._sigmas[900:901]) is False
    assert sigmas.reads == 1
    assert denoiser.p.extra_generation_params == {}


def test_already_switched_to_the_refiner_returns_without_reading_sigma(monkeypatch, loaded_model):
    monkeypatch.setitem(shared.opts.data, "refiner_switch_by_sample_steps", False)
    denoiser = _denoiser(_NoSigmasAccess(), refiner_checkpoint_info=loaded_model.sd_checkpoint_info)

    assert sd_samplers_common.apply_refiner(denoiser, torch.tensor([2.5])) is False
    assert denoiser.p.extra_generation_params == {}


def test_switch_decision_matches_the_reference_argmin_and_copies_the_table_once(monkeypatch):
    """The nearest-timestep match runs on the wrapper's CPU table copy, made once per wrapper: the same float32
    arithmetic and first-minimum tie break as the reference expression on the device table."""
    monkeypatch.setitem(shared.opts.data, "refiner_switch_by_sample_steps", False)
    alphas = torch.linspace(0.9991, 0.0047, 1000)
    wrapper = _CountingSigmas(((1 - alphas) / alphas) ** 0.5)
    table = wrapper._sigmas
    switches = []
    monkeypatch.setattr(sd_models, "reload_model_weights", lambda info: switches.append(info))
    monkeypatch.setattr(sd_samplers_common.devices, "torch_gc", lambda: None)
    switch_at = 0.6
    queries = torch.cat([table, (table[1:] + table[:-1]) / 2, torch.rand(300, generator=torch.Generator().manual_seed(0)) * 15])

    for sigma in queries:
        switches.clear()
        denoiser = _denoiser(wrapper, refiner_checkpoint_info=SimpleNamespace(short_title="refiner"), refiner_switch_at=switch_at)
        denoiser.p.setup_conds = lambda: None
        denoiser.update_inner_model = lambda: None
        batch_sigma = torch.stack([sigma * 0.5, sigma])

        switched = sd_samplers_common.apply_refiner(denoiser, batch_sigma)

        reference = (999 - torch.argmin(torch.abs(table - torch.max(batch_sigma)))) / 1000 >= switch_at
        assert switched == bool(reference) == bool(switches)
    assert wrapper.openclaw_cpu_sigmas[0] is table and wrapper.reads == len(queries)
