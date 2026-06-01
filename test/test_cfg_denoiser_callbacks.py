from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


def _module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def load_cfg_denoiser(cfg_denoised_callback):
    modules_pkg = _module("modules")
    prompt_parser = _module("modules.prompt_parser")
    prompt_parser.reconstruct_multicond_batch = lambda cond, step: ([[(0, 1.0)]], cond)
    prompt_parser.reconstruct_cond_batch = lambda uncond, step: uncond

    sd_samplers_common = _module(
        "modules.sd_samplers_common",
        InterruptedException=Exception,
        apply_refiner=lambda denoiser, sigma: False,
        store_latent=lambda latent: None,
    )

    shared = _module(
        "modules.shared",
        state=types.SimpleNamespace(
            interrupted=False,
            skipped=False,
            sampling_step=0,
            sampling_steps=1,
        ),
        opts=types.SimpleNamespace(
            skip_early_cond=0.0,
            s_min_uncond_all=False,
            pad_cond_uncond_v0=False,
            pad_cond_uncond=False,
            batch_cond_uncond=True,
            live_preview_content="Combined",
        ),
        sd_model=types.SimpleNamespace(
            cond_stage_key="txt",
            model=types.SimpleNamespace(conditioning_key="crossattn"),
        ),
    )

    class CFGDenoiserParams:
        def __init__(self, x, image_cond, sigma, sampling_step, total_sampling_steps, text_cond, text_uncond, denoiser=None):
            self.x = x
            self.image_cond = image_cond
            self.sigma = sigma
            self.sampling_step = sampling_step
            self.total_sampling_steps = total_sampling_steps
            self.text_cond = text_cond
            self.text_uncond = text_uncond
            self.denoiser = denoiser

    class CFGDenoisedParams:
        def __init__(self, x, sampling_step, total_sampling_steps, inner_model):
            self.x = x
            self.sampling_step = sampling_step
            self.total_sampling_steps = total_sampling_steps
            self.inner_model = inner_model

    class AfterCFGCallbackParams:
        def __init__(self, x, sampling_step, total_sampling_steps):
            self.x = x
            self.sampling_step = sampling_step
            self.total_sampling_steps = total_sampling_steps

    script_callbacks = _module(
        "modules.script_callbacks",
        CFGDenoiserParams=CFGDenoiserParams,
        CFGDenoisedParams=CFGDenoisedParams,
        AfterCFGCallbackParams=AfterCFGCallbackParams,
        cfg_denoiser_callback=lambda params: None,
        cfg_denoised_callback=cfg_denoised_callback,
        cfg_after_cfg_callback=lambda params: None,
    )

    replacements = {
        "modules": modules_pkg,
        "modules.prompt_parser": prompt_parser,
        "modules.sd_samplers_common": sd_samplers_common,
        "modules.shared": shared,
        "modules.script_callbacks": script_callbacks,
    }
    module_path = Path(__file__).resolve().parents[1] / "modules" / "sd_samplers_cfg_denoiser.py"
    spec = importlib.util.spec_from_file_location("cfg_denoiser_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with mock.patch.dict(sys.modules, replacements):
        spec.loader.exec_module(module)
    return module


class CFGDenoiserCallbackTests(unittest.TestCase):
    def test_cfg_denoised_callback_can_replace_inner_model_output(self):
        def cfg_denoised_callback(params):
            params.x = params.x + 10

        module = load_cfg_denoiser(cfg_denoised_callback)

        class TestDenoiser(module.CFGDenoiser):
            @property
            def inner_model(self):
                return object()

            def run_inner_model(self, x, sigma, cond):
                return torch.tensor([[[[2.0]]], [[[1.0]]]], device=x.device, dtype=x.dtype)

        sampler = types.SimpleNamespace(sampler_extra_args={}, last_latent=None)
        denoiser = TestDenoiser(sampler)
        denoiser.p = types.SimpleNamespace(extra_generation_params={}, scripts=None)
        denoiser.steps = 1
        denoiser.total_steps = 1

        x = torch.zeros(1, 1, 1, 1)
        sigma = torch.ones(1)
        cond = torch.zeros(1, 2, 3)
        uncond = torch.zeros(1, 2, 3)
        image_cond = torch.zeros(1, 1, 1, 1)

        denoised = denoiser(x, sigma, uncond, cond, 3.0, 0.0, image_cond)

        torch.testing.assert_close(denoised, torch.tensor([[[[14.0]]]]))


if __name__ == "__main__":
    unittest.main()
