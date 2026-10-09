from types import SimpleNamespace

import torch

from test.helpers import init_shared

shared = init_shared()

# sd_samplers first: importing sd_samplers_common on its own runs into an import cycle.
from modules import sd_samplers, sd_samplers_common  # noqa: E402,F401


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
