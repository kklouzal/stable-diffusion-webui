from __future__ import annotations

import os
import types
import unittest

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


if __name__ == "__main__":
    unittest.main()
