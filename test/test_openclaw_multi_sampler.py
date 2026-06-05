from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def load_multi_sampler(monkeypatch):
    class Script:
        pass

    def sample_euler(model, x, extra_args=None, disable=False, callback=None):
        if callback is not None:
            callback({"x": x, "i": 0, "sigma": 1, "sigma_hat": 1})
        return x

    def sample_heun(model, x, extra_args=None, disable=False, callback=None):
        return x

    def sample_dpmpp_2m_sde(model, x, extra_args=None, disable=False, callback=None):
        raise AssertionError("terminal one-step DPM++ 2M SDE should use the direct denoise path")

    class SamplerData(tuple):
        def __new__(cls, name, constructor, aliases, options):
            return tuple.__new__(cls, (name, constructor, aliases, options))

        @property
        def name(self):
            return self[0]

        @property
        def options(self):
            return self[3]

    modules_pkg = _module("modules")
    monkeypatch.setitem(sys.modules, "modules", modules_pkg)
    monkeypatch.setitem(sys.modules, "modules.headless_ui", _module("modules.headless_ui"))
    monkeypatch.setitem(sys.modules, "modules.script_callbacks", _module(
        "modules.script_callbacks",
        ExtraNoiseParams=object,
        extra_noise_callback=lambda params: None,
        on_app_started=lambda callback: None,
    ))
    monkeypatch.setitem(sys.modules, "modules.script_loading", _module("modules.script_loading", loaded_scripts={}))
    monkeypatch.setitem(sys.modules, "modules.scripts", _module("modules.scripts", Script=Script, AlwaysVisible=True))
    monkeypatch.setitem(sys.modules, "modules.sd_samplers", _module(
        "modules.sd_samplers",
        all_samplers=[],
        all_samplers_map={},
        set_samplers=lambda: None,
    ))
    monkeypatch.setitem(sys.modules, "modules.sd_samplers_common", _module(
        "modules.sd_samplers_common",
        SamplerData=SamplerData,
        InterruptedException=Exception,
        TorchHijack=lambda p: None,
    ))
    monkeypatch.setitem(sys.modules, "modules.sd_samplers_kdiffusion", _module(
        "modules.sd_samplers_kdiffusion",
        KDiffusionSampler=type("KDiffusionSampler", (), {}),
        samplers_k_diffusion=[("Euler", "sample_euler"), ("Heun", "sample_heun"), ("DPM++ 2M SDE", "sample_dpmpp_2m_sde")],
        k_diffusion_samplers_map={
            "Euler": types.SimpleNamespace(options={}, total_steps=lambda steps: steps),
            "Heun": types.SimpleNamespace(options={}, total_steps=lambda steps: steps),
            "DPM++ 2M SDE": types.SimpleNamespace(options={}, total_steps=lambda steps: steps),
        },
        sampler_extra_params={},
    ))
    monkeypatch.setitem(sys.modules, "modules.sd_schedulers", _module(
        "modules.sd_schedulers",
        schedulers=[],
        schedulers_map={"Automatic": object(), "Karras": object()},
    ))
    monkeypatch.setitem(sys.modules, "modules.shared", _module(
        "modules.shared",
        opts=types.SimpleNamespace(sgm_noise_multiplier=False, img2img_extra_noise=0),
        state=types.SimpleNamespace(),
        total_tqdm=types.SimpleNamespace(update=lambda: None),
        sd_model=types.SimpleNamespace(),
    ))
    monkeypatch.setitem(sys.modules, "fastapi", _module("fastapi", FastAPI=object, Request=object))
    monkeypatch.setitem(sys.modules, "PIL", _module("PIL"))
    monkeypatch.setitem(sys.modules, "PIL.Image", _module("PIL.Image"))
    monkeypatch.setitem(sys.modules, "numpy", _module("numpy", moveaxis=lambda tensor, source, dest: tensor))
    monkeypatch.setitem(sys.modules, "torch", _module("torch", Tensor=object))
    sampling_module = _module(
        "k_diffusion.sampling",
        sample_euler=sample_euler,
        sample_heun=sample_heun,
        sample_dpmpp_2m_sde=sample_dpmpp_2m_sde,
    )
    monkeypatch.setitem(sys.modules, "k_diffusion", _module("k_diffusion", sampling=sampling_module))
    monkeypatch.setitem(sys.modules, "k_diffusion.sampling", sampling_module)

    script_path = Path(__file__).resolve().parents[1] / "extensions" / "openclaw-multi-sampler" / "scripts" / "openclaw_multi_sampler.py"
    spec = importlib.util.spec_from_file_location("openclaw_multi_sampler_under_test", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_img2img_sigma_tail_transition_count_keeps_terminal_sigma(monkeypatch):
    module = load_multi_sampler(monkeypatch)
    sigma_sched = [6, 5, 4, 3, 2, 1, 0]
    sampling_steps = module._sigma_transition_count(sigma_sched)

    sampler = object.__new__(module.MultiKDiffusionSampler)
    sampler.definition = {"samplers": ["Euler", "Heun"], "switch_ats": [3]}

    stages = sampler._build_stages(None, sigma_sched, sampling_steps)

    assert sampling_steps == 6
    assert stages[0] == ("Euler", None, [6, 5, 4, 3], 0, 3)
    assert stages[1] == ("Heun", None, [3, 2, 1, 0], 3, 6)


def test_chain_boundaries_reject_zero_length_stages(monkeypatch):
    module = load_multi_sampler(monkeypatch)

    with pytest.raises(ValueError, match="stage 1"):
        module._chain_boundaries({"samplers": ["Euler", "Heun"], "switch_ats": [0]}, 6)

    with pytest.raises(ValueError, match="stage 2"):
        module._chain_boundaries({"samplers": ["Euler", "Heun"], "switch_ats": [6]}, 6)

    with pytest.raises(ValueError, match="at least one sampling step"):
        module._chain_boundaries({"samplers": ["Euler", "Heun"], "switch_ats": [0]}, 0)


def test_stage_sigmas_must_cover_expected_span(monkeypatch):
    module = load_multi_sampler(monkeypatch)
    sampler = object.__new__(module.MultiKDiffusionSampler)
    sampler.definition = {"samplers": ["Euler", "Heun"], "switch_ats": [2]}

    with pytest.raises(ValueError, match="expected 3 sigma value"):
        sampler._build_stages(None, [4, 3, 2, 1], 4)


def test_terminal_one_step_dpmpp_2m_sde_uses_direct_denoise(monkeypatch):
    module = load_multi_sampler(monkeypatch)

    class FakeLatent:
        shape = (1,)

        def new_ones(self, shape):
            return 1

    class FakeModelWrapCfg:
        def __init__(self):
            self.calls = []

        def __call__(self, x, sigma, **kwargs):
            self.calls.append((x, sigma, kwargs))
            return "denoised"

    p = types.SimpleNamespace(
        cfg_scale=7.0,
        eta=None,
        extra_generation_params={},
        mask=None,
        nmask=None,
        rng=None,
        s_min_uncond=0.0,
    )
    sampler = object.__new__(module.MultiKDiffusionSampler)
    sampler.definition = {"name": "Multi: test", "samplers": ["Euler", "DPM++ 2M SDE"], "switch_ats": [1]}
    sampler.eta_option_field = "eta_ancestral"
    sampler.last_latent = None
    sampler.model_wrap_cfg = FakeModelWrapCfg()
    sampler.stop_at = None

    result = sampler._run_chain(
        p,
        FakeLatent(),
        conditioning=None,
        unconditional_conditioning=None,
        sigmas=[2, 1, 0],
        steps=2,
        is_img2img=True,
    )

    assert result == "denoised"
    assert sampler.last_latent == "denoised"
    assert sampler.model_wrap_cfg.calls[0][1] == 1
    assert p.extra_generation_params["Sampler chain"] == "Euler@0-1 -> DPM++ 2M SDE@1-2"
