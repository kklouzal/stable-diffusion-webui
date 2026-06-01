from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def load_multi_sampler(monkeypatch):
    class Script:
        pass

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
        samplers_k_diffusion=[("Euler", "sample_euler"), ("Heun", "sample_heun")],
        k_diffusion_samplers_map={
            "Euler": types.SimpleNamespace(options={}, total_steps=lambda steps: steps),
            "Heun": types.SimpleNamespace(options={}, total_steps=lambda steps: steps),
        },
        sampler_extra_params={},
    ))
    monkeypatch.setitem(sys.modules, "modules.sd_schedulers", _module("modules.sd_schedulers", schedulers=[], schedulers_map={}))
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
    monkeypatch.setitem(sys.modules, "k_diffusion", _module("k_diffusion"))
    monkeypatch.setitem(sys.modules, "k_diffusion.sampling", _module("k_diffusion.sampling"))

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
