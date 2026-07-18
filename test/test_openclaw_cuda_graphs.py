from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

# A1111 parses sys.argv during shared import; keep unittest flags out of it.
sys.argv = [sys.argv[0]]

shared_stub = types.ModuleType("modules.shared")
shared_stub.opts = types.SimpleNamespace(batch_cond_uncond=True)
shared_stub.sd_model = None
sys.modules["modules.shared"] = shared_stub

import torch

from modules import openclaw_cuda_graphs


def make_denoiser(*, active=True, start=0, end=4, total_steps=5, hooks=True, blur_sigma=11.0):
    if hooks:
        to_q = types.SimpleNamespace(seg_enable=True)
        crossattn_modules = [types.SimpleNamespace(
            to_q=to_q,
            heads=8,
            network_layer_name="middle_block_attn1",
        )]
    else:
        crossattn_modules = []
    seg_params = types.SimpleNamespace(
        seg_active=active,
        seg_blur_sigma=blur_sigma,
        seg_blur_threshold=10.5,
        seg_start_step=start,
        seg_end_step=end,
        crossattn_modules=crossattn_modules,
    )
    p = types.SimpleNamespace(
        mask=None,
        nmask=None,
        width=512,
        height=512,
        incant_cfg_params={"seg_params": seg_params},
    )
    return types.SimpleNamespace(mask=None, nmask=None, p=p, total_steps=total_steps)


class CudaGraphSegBypassTests(unittest.TestCase):
    def setUp(self):
        self.previous_allow_seg = os.environ.pop("OPENCLAW_CUDA_GRAPH_ALLOW_SEG", None)
        shared = sys.modules["modules.shared"]
        self.previous_batch_cond_uncond = getattr(shared.opts, "batch_cond_uncond", None)
        shared.opts.batch_cond_uncond = True

    def tearDown(self):
        if self.previous_allow_seg is None:
            os.environ.pop("OPENCLAW_CUDA_GRAPH_ALLOW_SEG", None)
        else:
            os.environ["OPENCLAW_CUDA_GRAPH_ALLOW_SEG"] = self.previous_allow_seg
        shared = sys.modules["modules.shared"]
        shared.opts.batch_cond_uncond = self.previous_batch_cond_uncond

    def test_seg_bypasses_without_explicit_operator_opt_in(self):
        reason = openclaw_cuda_graphs._graph_denoiser_bypass_reason(make_denoiser())

        self.assertEqual(reason, "seg_disabled")

    def test_full_window_seg_uses_graph_path_when_static_paired_and_opted_in(self):
        os.environ["OPENCLAW_CUDA_GRAPH_ALLOW_SEG"] = "1"

        reason = openclaw_cuda_graphs._graph_denoiser_bypass_reason(make_denoiser())

        self.assertIsNone(reason)

    def test_partial_window_seg_still_bypasses(self):
        os.environ["OPENCLAW_CUDA_GRAPH_ALLOW_SEG"] = "1"

        reason = openclaw_cuda_graphs._graph_denoiser_bypass_reason(make_denoiser(end=3))

        self.assertEqual(reason, "seg_active")

    def test_seg_bypasses_without_paired_cfg_batch(self):
        os.environ["OPENCLAW_CUDA_GRAPH_ALLOW_SEG"] = "1"
        shared = sys.modules["modules.shared"]
        shared.opts.batch_cond_uncond = False

        reason = openclaw_cuda_graphs._graph_denoiser_bypass_reason(make_denoiser())

        self.assertEqual(reason, "seg_unpaired_cfg")

    def test_seg_bypasses_when_hooks_are_not_ready(self):
        os.environ["OPENCLAW_CUDA_GRAPH_ALLOW_SEG"] = "1"

        reason = openclaw_cuda_graphs._graph_denoiser_bypass_reason(make_denoiser(hooks=False))

        self.assertEqual(reason, "seg_hooks_unready")

    def test_seg_graph_key_includes_parameters_that_change_hook_math(self):
        os.environ["OPENCLAW_CUDA_GRAPH_ALLOW_SEG"] = "1"

        base = openclaw_cuda_graphs._denoiser_graph_key(make_denoiser())
        changed_blur = openclaw_cuda_graphs._denoiser_graph_key(make_denoiser(blur_sigma=10.0))
        changed_window = openclaw_cuda_graphs._denoiser_graph_key(make_denoiser(end=5, total_steps=6))

        self.assertNotEqual(base, changed_blur)
        self.assertNotEqual(base, changed_window)

    def test_masks_still_bypass_when_seg_is_allowed(self):
        denoiser = make_denoiser()
        denoiser.p.mask = object()

        reason = openclaw_cuda_graphs._graph_denoiser_bypass_reason(denoiser)

        self.assertEqual(reason, "processing_mask")

    def test_unmasked_img2img_init_latent_is_graphable_when_seg_is_allowed(self):
        os.environ["OPENCLAW_CUDA_GRAPH_ALLOW_SEG"] = "1"
        denoiser = make_denoiser()
        denoiser.init_latent = object()

        reason = openclaw_cuda_graphs._graph_denoiser_bypass_reason(denoiser)

        self.assertIsNone(reason)

    def test_unmasked_processing_init_images_do_not_bypass_graphs(self):
        denoiser = make_denoiser(active=False)
        denoiser.p.init_images = [object()]

        reason = openclaw_cuda_graphs._graph_denoiser_bypass_reason(denoiser)

        self.assertIsNone(reason)

    def test_processing_image_conditioning_does_not_bypass_graphs_when_unmasked(self):
        denoiser = make_denoiser(active=False)
        denoiser.p.image_conditioning = object()

        reason = openclaw_cuda_graphs._graph_denoiser_bypass_reason(denoiser)

        self.assertIsNone(reason)

    def test_img2img_graph_key_changes_without_tensor_value_explosion(self):
        txt2img = make_denoiser(active=False)
        img2img = make_denoiser(active=False)
        img2img.init_latent = object()
        img2img.p.init_images = [object()]
        first = openclaw_cuda_graphs._denoiser_graph_key(img2img)
        img2img.p.init_images = [object(), object()]
        second = openclaw_cuda_graphs._denoiser_graph_key(img2img)

        self.assertNotEqual(openclaw_cuda_graphs._denoiser_graph_key(txt2img), first)
        self.assertEqual(first, second)


class CudaGraphCacheSizeTests(unittest.TestCase):
    def setUp(self):
        self.previous_max_cache_size = openclaw_cuda_graphs._MAX_CACHE_SIZE
        self.previous_min_key_hits = os.environ.get("OPENCLAW_CUDA_GRAPH_MIN_KEY_HITS")
        os.environ.pop("OPENCLAW_CUDA_GRAPH_MIN_KEY_HITS", None)
        openclaw_cuda_graphs.set_enabled(False, clear=True)

    def tearDown(self):
        openclaw_cuda_graphs._MAX_CACHE_SIZE = self.previous_max_cache_size
        if self.previous_min_key_hits is None:
            os.environ.pop("OPENCLAW_CUDA_GRAPH_MIN_KEY_HITS", None)
        else:
            os.environ["OPENCLAW_CUDA_GRAPH_MIN_KEY_HITS"] = self.previous_min_key_hits
        openclaw_cuda_graphs.set_enabled(False, clear=True)

    def test_zero_max_cache_size_clears_existing_cache_entries(self):
        openclaw_cuda_graphs._MAX_CACHE_SIZE = 0
        openclaw_cuda_graphs._CACHE[("stale",)] = {"dummy": True}

        openclaw_cuda_graphs._evict_if_needed_locked()

        self.assertEqual(openclaw_cuda_graphs.status()["cache_size"], 0)

    def test_zero_max_cache_size_bypasses_capture(self):
        openclaw_cuda_graphs._MAX_CACHE_SIZE = 0
        openclaw_cuda_graphs.set_enabled(True, clear=True)
        calls = []
        x = torch.zeros(1)

        def fn(x_arg, sigma_arg, cond=None):
            calls.append((x_arg, sigma_arg, cond))
            return x_arg + 1

        out = openclaw_cuda_graphs.run(fn, x, x, cond={"x": x})
        status = openclaw_cuda_graphs.status()

        self.assertTrue(torch.equal(out, torch.ones(1)))
        self.assertEqual(len(calls), 1)
        self.assertEqual(status["cache_size"], 0)
        self.assertEqual(status["bypass_reasons"].get("cache_disabled"), 1)


    def test_clear_removes_per_key_locks(self):
        openclaw_cuda_graphs._KEY_LOCKS[("stale",)] = object()

        openclaw_cuda_graphs.clear()

        self.assertEqual(openclaw_cuda_graphs._KEY_LOCKS, {})

    def test_evict_removes_per_key_lock_with_cache_entry(self):
        openclaw_cuda_graphs._MAX_CACHE_SIZE = 1
        openclaw_cuda_graphs._CACHE[("old",)] = {"dummy": True}
        openclaw_cuda_graphs._KEY_LOCKS[("old",)] = object()

        openclaw_cuda_graphs._evict_if_needed_locked()

        self.assertEqual(openclaw_cuda_graphs._CACHE, {})
        self.assertEqual(openclaw_cuda_graphs._KEY_LOCKS, {})

    def test_run_reuses_per_key_lock_on_warmup_bypass(self):
        class FakeTensor:
            shape = (1,)
            dtype = "float32"
            device = types.SimpleNamespace(type="cuda")
            requires_grad = False

            def stride(self):
                return (1,)

        openclaw_cuda_graphs._MAX_CACHE_SIZE = 1
        openclaw_cuda_graphs.set_enabled(True, clear=True)
        x = FakeTensor()

        def fn(x_arg, sigma_arg, cond=None):
            return x_arg

        with mock.patch.object(openclaw_cuda_graphs.torch.cuda, "is_available", return_value=True), \
             mock.patch.object(openclaw_cuda_graphs.torch, "is_grad_enabled", return_value=False), \
             mock.patch.object(openclaw_cuda_graphs.torch, "is_tensor", side_effect=lambda value: isinstance(value, FakeTensor)):
            openclaw_cuda_graphs.run(fn, x, x, cond={"x": x})
            openclaw_cuda_graphs.run(fn, x, x, cond={"x": x})

        self.assertEqual(len(openclaw_cuda_graphs._KEY_LOCKS), 1)


    def test_capture_returns_first_eager_warmup_result(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for capture return-equivalence test")

        openclaw_cuda_graphs._MAX_CACHE_SIZE = 1
        os.environ["OPENCLAW_CUDA_GRAPH_MIN_KEY_HITS"] = "1"
        openclaw_cuda_graphs.set_enabled(True, clear=True)
        x = torch.zeros(1, device="cuda")
        calls = []

        def fn(x_arg, sigma_arg, cond=None):
            calls.append(len(calls) + 1)
            return x_arg + calls[-1]

        out = openclaw_cuda_graphs.run(fn, x, x, cond={"x": x})

        self.assertTrue(torch.equal(out.cpu(), torch.ones(1)))

    def test_min_key_hits_can_be_set_to_one_for_immediate_capture(self):
        previous = os.environ.get("OPENCLAW_CUDA_GRAPH_MIN_KEY_HITS")
        os.environ["OPENCLAW_CUDA_GRAPH_MIN_KEY_HITS"] = "1"
        try:
            self.assertEqual(openclaw_cuda_graphs._min_key_hits_before_capture(), 1)
        finally:
            if previous is None:
                os.environ.pop("OPENCLAW_CUDA_GRAPH_MIN_KEY_HITS", None)
            else:
                os.environ["OPENCLAW_CUDA_GRAPH_MIN_KEY_HITS"] = previous


if __name__ == "__main__":
    unittest.main()
