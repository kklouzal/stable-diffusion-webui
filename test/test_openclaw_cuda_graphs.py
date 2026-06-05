from __future__ import annotations

import os
import types
import unittest

import torch

from modules import openclaw_cuda_graphs


def make_denoiser(*, active=True, start=0, end=4, total_steps=5):
    seg_params = types.SimpleNamespace(
        seg_active=active,
        seg_start_step=start,
        seg_end_step=end,
    )
    p = types.SimpleNamespace(
        mask=None,
        nmask=None,
        incant_cfg_params={"seg_params": seg_params},
    )
    return types.SimpleNamespace(mask=None, nmask=None, p=p, total_steps=total_steps)


class CudaGraphSegBypassTests(unittest.TestCase):
    def setUp(self):
        self.previous_allow_seg = os.environ.pop("OPENCLAW_CUDA_GRAPH_ALLOW_SEG", None)

    def tearDown(self):
        if self.previous_allow_seg is None:
            os.environ.pop("OPENCLAW_CUDA_GRAPH_ALLOW_SEG", None)
        else:
            os.environ["OPENCLAW_CUDA_GRAPH_ALLOW_SEG"] = self.previous_allow_seg

    def test_seg_bypasses_without_explicit_opt_in(self):
        reason = openclaw_cuda_graphs._graph_denoiser_bypass_reason(make_denoiser())

        self.assertEqual(reason, "seg_active")

    def test_full_window_seg_uses_graph_path_when_opted_in(self):
        os.environ["OPENCLAW_CUDA_GRAPH_ALLOW_SEG"] = "1"

        reason = openclaw_cuda_graphs._graph_denoiser_bypass_reason(make_denoiser())

        self.assertIsNone(reason)

    def test_partial_window_seg_still_bypasses_when_opted_in(self):
        os.environ["OPENCLAW_CUDA_GRAPH_ALLOW_SEG"] = "1"

        reason = openclaw_cuda_graphs._graph_denoiser_bypass_reason(make_denoiser(end=3))

        self.assertEqual(reason, "seg_active")

    def test_masks_still_bypass_when_seg_is_allowed(self):
        os.environ["OPENCLAW_CUDA_GRAPH_ALLOW_SEG"] = "1"
        denoiser = make_denoiser()
        denoiser.p.mask = object()

        reason = openclaw_cuda_graphs._graph_denoiser_bypass_reason(denoiser)

        self.assertEqual(reason, "processing_mask")


class CudaGraphCacheSizeTests(unittest.TestCase):
    def setUp(self):
        self.previous_max_cache_size = openclaw_cuda_graphs._MAX_CACHE_SIZE
        openclaw_cuda_graphs.set_enabled(False, clear=True)

    def tearDown(self):
        openclaw_cuda_graphs._MAX_CACHE_SIZE = self.previous_max_cache_size
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


if __name__ == "__main__":
    unittest.main()
