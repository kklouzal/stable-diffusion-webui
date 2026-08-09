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
    def test_batch_index_tensors_are_cached_for_a_stable_layout(self):
        module = load_cfg_denoiser(lambda _params: None)
        denoiser = module.CFGDenoiser(types.SimpleNamespace())

        with mock.patch.object(module.torch, "repeat_interleave", wraps=torch.repeat_interleave) as repeat_interleave, \
             mock.patch.object(module.torch, "arange", wraps=torch.arange) as arange, \
             mock.patch.object(module.torch, "as_tensor", wraps=torch.as_tensor) as as_tensor:
            first = denoiser.batch_index_tensors([2, 1], [0, 2], torch.device("cpu"))
            second = denoiser.batch_index_tensors([2, 1], [0, 2], torch.device("cpu"))

        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])
        torch.testing.assert_close(first[0], torch.tensor([0, 0, 1]))
        torch.testing.assert_close(first[1], torch.tensor([0, 2]))
        self.assertEqual(repeat_interleave.call_count, 1)
        self.assertEqual(arange.call_count, 1)
        self.assertEqual(as_tensor.call_count, 2)

    def test_batch_index_cache_invalidates_when_prompt_layout_changes(self):
        module = load_cfg_denoiser(lambda _params: None)
        denoiser = module.CFGDenoiser(types.SimpleNamespace())

        first = denoiser.batch_index_tensors([1, 1], [0, 1], "cpu")
        second = denoiser.batch_index_tensors([2, 1], [0, 2], "cpu")

        self.assertIsNot(first[0], second[0])
        self.assertIsNot(first[1], second[1])
        torch.testing.assert_close(second[0], torch.tensor([0, 0, 1]))
        torch.testing.assert_close(second[1], torch.tensor([0, 2]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_batch_index_cache_is_device_specific(self):
        module = load_cfg_denoiser(lambda _params: None)
        denoiser = module.CFGDenoiser(types.SimpleNamespace())

        cpu_indexes = denoiser.batch_index_tensors([2], [0], "cpu")
        cuda_indexes = denoiser.batch_index_tensors([2], [0], "cuda")

        self.assertEqual(cpu_indexes[0].device.type, "cpu")
        self.assertEqual(cuda_indexes[0].device.type, "cuda")
        self.assertEqual(cpu_indexes[0].dtype, torch.long)
        self.assertEqual(cuda_indexes[0].dtype, torch.long)

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


    def test_last_noise_uncond_requirement_disables_uncond_skip(self):
        module = load_cfg_denoiser(lambda params: None)
        module.shared.opts.skip_early_cond = 1.0

        class TestDenoiser(module.CFGDenoiser):
            @property
            def inner_model(self):
                return object()

            def run_inner_model(self, x, sigma, cond):
                self.seen_batch = x.shape[0]
                return torch.tensor([[[[2.0]]], [[[1.0]]]], device=x.device, dtype=x.dtype)

        sampler = types.SimpleNamespace(sampler_extra_args={}, last_latent=None)
        denoiser = TestDenoiser(sampler)
        denoiser.p = types.SimpleNamespace(extra_generation_params={}, scripts=None)
        denoiser.steps = 1
        denoiser.total_steps = 1
        denoiser.need_last_noise_uncond = True

        x = torch.zeros(1, 1, 1, 1)
        sigma = torch.ones(1)
        cond = torch.zeros(1, 2, 3)
        uncond = torch.zeros(1, 2, 3)
        image_cond = torch.zeros(1, 1, 1, 1)

        denoised = denoiser(x, sigma, uncond, cond, 3.0, 0.0, image_cond)

        self.assertEqual(denoiser.seen_batch, 2)
        torch.testing.assert_close(denoiser.last_noise_uncond, torch.tensor([[[[1.0]]]]))
        torch.testing.assert_close(denoised, torch.tensor([[[[4.0]]]]))
        self.assertNotIn("Skip Early CFG", denoiser.p.extra_generation_params)

    def test_pad_cond_dict_does_not_mutate_cached_condition(self):
        module = load_cfg_denoiser(lambda _params: None)
        original_crossattn = torch.zeros(1, 2, 3)
        original = {"crossattn": original_crossattn, "vector": torch.ones(1, 4)}
        empty = torch.zeros(1, 1, 3)

        padded = module.pad_cond(original, 2, empty)

        self.assertIsNot(padded, original)
        self.assertIs(padded["vector"], original["vector"])
        self.assertIs(original["crossattn"], original_crossattn)
        self.assertEqual(original["crossattn"].shape, (1, 2, 3))
        self.assertEqual(padded["crossattn"].shape, (1, 4, 3))

    def test_pad_cond_uncond_v0_dict_does_not_mutate_cached_uncond(self):
        module = load_cfg_denoiser(lambda _params: None)
        denoiser = module.CFGDenoiser(types.SimpleNamespace())
        cond = torch.zeros(1, 4, 3)
        original_crossattn = torch.ones(1, 2, 3)
        uncond = {"crossattn": original_crossattn, "vector": torch.ones(1, 4)}

        _cond, padded_uncond = denoiser.pad_cond_uncond_v0(cond, uncond)

        self.assertIsNot(padded_uncond, uncond)
        self.assertIs(padded_uncond["vector"], uncond["vector"])
        self.assertIs(uncond["crossattn"], original_crossattn)
        self.assertEqual(uncond["crossattn"].shape, (1, 2, 3))
        self.assertEqual(padded_uncond["crossattn"].shape, (1, 4, 3))
        self.assertTrue(denoiser.padded_cond_uncond_v0)

    def test_edit_model_unpadded_unequal_prompt_lengths_are_denoised_in_aligned_slices(self):
        module = load_cfg_denoiser(lambda params: None)
        module.shared.sd_model.cond_stage_key = "edit"
        module.shared.opts.batch_cond_uncond = False

        class TestDenoiser(module.CFGDenoiser):
            @property
            def inner_model(self):
                return object()

            def run_inner_model(self, x, sigma, cond):
                self.calls.append((x.clone(), sigma.clone(), cond["c_crossattn"][0].clone(), cond["c_concat"][0].clone()))
                values = []
                for c_crossattn, c_concat in zip(cond["c_crossattn"][0], cond["c_concat"][0]):
                    if torch.count_nonzero(c_concat).item() == 0:
                        values.append(1.0)
                    elif torch.count_nonzero(c_crossattn).item() == 0:
                        values.append(4.0)
                    else:
                        values.append(10.0)
                return torch.tensor(values, device=x.device, dtype=x.dtype).reshape(-1, 1, 1, 1)

        sampler = types.SimpleNamespace(sampler_extra_args={}, last_latent=None)
        denoiser = TestDenoiser(sampler)
        denoiser.calls = []
        denoiser.p = types.SimpleNamespace(extra_generation_params={}, scripts=None)
        denoiser.steps = 1
        denoiser.total_steps = 1
        denoiser.image_cfg_scale = 1.5
        denoiser.init_latent = torch.zeros(1, 1, 1, 1)

        x = torch.zeros(1, 1, 1, 1)
        sigma = torch.ones(1)
        cond = torch.full((1, 3, 2), 9.0)
        uncond = torch.zeros(1, 2, 2)
        image_cond = torch.full((1, 1, 1, 1), 5.0)

        denoised = denoiser(x, sigma, uncond, cond, 2.0, 0.0, image_cond)

        torch.testing.assert_close(denoised, torch.tensor([[[[17.5]]]]))
        self.assertEqual([call[0].shape[0] for call in denoiser.calls], [1, 1, 1])
        self.assertEqual([call[2].shape[1] for call in denoiser.calls], [3, 2, 2])
        self.assertEqual([float(call[3][0, 0, 0, 0]) for call in denoiser.calls], [5.0, 5.0, 0.0])


if __name__ == "__main__":
    unittest.main()
