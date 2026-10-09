from __future__ import annotations

import types
import unittest

import torch

from test.helpers import load_source


def load_row_memo():
    return load_source("sd_unet_row_memo_under_test", "modules/sd_unet_row_memo.py")


class FakeDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def run_inner_model(self, x, sigma, cond):
        self.calls.append(x.shape[0])
        return x * 2


class RowMemoTests(unittest.TestCase):
    def setUp(self):
        self.memo = load_row_memo()

    def test_recorder_records_contiguous_row_slices_and_passes_through_when_disarmed(self):
        denoiser = FakeDenoiser()
        x_in = torch.randn(5, 4, 2, 2)
        sigma = torch.rand(5)
        cond = {"crossattn": torch.randn(5, 3, 8), "vector": torch.randn(5, 6)}

        memo = self.memo.arm(denoiser, n_cond=3)
        self.assertIsInstance(denoiser.__dict__["run_inner_model"], self.memo.MainPassRecorder)
        out = denoiser.run_inner_model(x_in[0:2], sigma[0:2], self.memo.slice_cond_rows(cond, 5, 2))
        torch.testing.assert_close(out, x_in[0:2] * 2)
        denoiser.run_inner_model(x_in[2:5], sigma[2:5], {key: value[2:5] for key, value in cond.items()})

        self.assertEqual([(rec.start, rec.rows) for rec in memo.calls], [(0, 2), (2, 3)])
        self.assertTrue(all(rec.inputs_unchanged() for rec in memo.calls))
        self.assertIs(self.memo.disarm(denoiser), memo)
        self.assertIsNone(self.memo.disarm(denoiser))

        denoiser.run_inner_model(x_in, sigma, cond)
        self.assertEqual(denoiser.calls, [2, 3, 5])
        self.assertEqual(len(memo.calls), 2)

        self.assertTrue(self.memo.uninstall(denoiser))
        self.assertNotIn("run_inner_model", denoiser.__dict__)
        self.assertEqual(denoiser.run_inner_model.__func__, FakeDenoiser.run_inner_model)

    def test_recorder_rejects_calls_that_are_not_the_next_row_slice(self):
        denoiser = FakeDenoiser()
        x_in = torch.randn(4, 1)
        self.memo.arm(denoiser, n_cond=2)
        denoiser.run_inner_model(x_in[0:2], torch.zeros(2), {})
        with self.assertRaisesRegex(RuntimeError, "not the next row slice"):
            denoiser.run_inner_model(x_in[3:4], torch.zeros(1), {})
        # The failed call dropped the memo: nothing stays armed or referenced.
        self.assertIsNone(self.memo.disarm(denoiser))
        self.memo.arm(denoiser, n_cond=2)
        denoiser.run_inner_model(x_in[0:2], torch.zeros(2), {})
        with self.assertRaisesRegex(RuntimeError, "not the next row slice"):
            denoiser.run_inner_model(torch.randn(2, 1), torch.zeros(2), {})

    def test_exception_inside_recorded_call_drops_the_memo(self):
        denoiser = FakeDenoiser()

        def boom(x, sigma, cond):
            raise ValueError("unet failed")

        denoiser.run_inner_model = boom
        memo = self.memo.arm(denoiser, n_cond=1)
        with self.assertRaisesRegex(ValueError, "unet failed"):
            denoiser.run_inner_model(torch.zeros(2, 1), torch.zeros(2), {})
        self.assertEqual(memo.calls, [])
        self.assertIsNone(self.memo.disarm(denoiser))

    def test_arm_replaces_an_unconsumed_memo(self):
        denoiser = FakeDenoiser()
        first = self.memo.arm(denoiser, n_cond=1)
        denoiser.run_inner_model(torch.zeros(2, 1), torch.zeros(2), {})
        second = self.memo.arm(denoiser, n_cond=1)
        self.assertIsNot(first, second)
        self.assertIs(self.memo.disarm(denoiser), second)
        self.assertEqual(second.calls, [])

    def test_uninstall_restores_a_wrapper_that_was_there_before_arm(self):
        denoiser = FakeDenoiser()
        earlier = lambda x, sigma, cond: FakeDenoiser.run_inner_model(denoiser, x, sigma, cond)
        denoiser.run_inner_model = earlier
        self.memo.arm(denoiser, n_cond=1)
        denoiser.run_inner_model(torch.zeros(1, 1), torch.zeros(1), {})
        self.assertEqual(denoiser.calls, [1])
        self.assertTrue(self.memo.uninstall(denoiser))
        self.assertIs(denoiser.run_inner_model, earlier)

    def test_inference_tensors_are_sealed_without_version_counters(self):
        denoiser = FakeDenoiser()
        with torch.inference_mode():
            x_in = torch.randn(2, 1)
            cond = {"crossattn": torch.randn(2, 3), "c_concat": [torch.randn(2, 1)]}
            memo = self.memo.arm(denoiser, n_cond=1)
            denoiser.run_inner_model(x_in, torch.zeros(2), cond)
            self.memo.disarm(denoiser)
            rec = memo.calls[0]
            self.assertTrue(x_in.is_inference())
            self.assertTrue(rec.inputs_unchanged())
            cond["c_concat"][0] = cond["c_concat"][0].clone()
            self.assertFalse(rec.inputs_unchanged())

    def test_uninstall_leaves_a_foreign_outer_wrapper(self):
        denoiser = FakeDenoiser()
        self.memo.arm(denoiser, n_cond=1)
        recorder = denoiser.run_inner_model
        foreign = lambda x, sigma, cond: recorder(x, sigma, cond)
        denoiser.run_inner_model = foreign
        self.assertFalse(self.memo.uninstall(denoiser))
        self.assertIs(denoiser.run_inner_model, foreign)

    def test_claim_goes_to_the_outermost_forward_only(self):
        rec = self.memo.CallRecord(0, torch.zeros(2, 1), torch.zeros(2), {})
        self.assertIsNone(self.memo.claim())
        with self.memo.recording(rec):
            outer = self.memo.claim()
            self.assertIsNotNone(outer)
            self.assertFalse(outer.replay)
            self.assertEqual(outer.rows, 2)
            self.assertIsNone(self.memo.claim())
        self.assertIsNone(self.memo.claim())
        with self.memo.replaying(rec, 1):
            slot = self.memo.claim()
            self.assertTrue(slot.replay)
            self.assertEqual(slot.rows, 1)
        with self.assertRaises(ValueError):
            self.memo.replaying(rec, 3)

    def test_inputs_unchanged_detects_in_place_edits_and_replaced_entries(self):
        x = torch.zeros(2, 1)
        cond = {"crossattn": torch.zeros(2, 3), "c_concat": [torch.zeros(2, 1)]}
        rec = self.memo.CallRecord(0, x, torch.zeros(2), cond)
        self.assertFalse(rec.inputs_unchanged())
        rec.seal()
        self.assertTrue(rec.inputs_unchanged())
        cond["c_concat"][0].add_(1)
        self.assertFalse(rec.inputs_unchanged())
        rec.seal()
        cond["crossattn"] = cond["crossattn"].clone()
        self.assertFalse(rec.inputs_unchanged())

    def test_slice_cond_rows_slices_batch_aligned_tensors_and_rejects_others(self):
        cond = {"crossattn": torch.randn(3, 4, 2), "c_concat": [torch.randn(3, 1)], "tag": "keep"}
        sliced = self.memo.slice_cond_rows(cond, 3, 2)
        self.assertIsNot(sliced, cond)
        self.assertEqual(sliced["crossattn"].data_ptr(), cond["crossattn"].data_ptr())
        self.assertEqual(sliced["crossattn"].shape[0], 2)
        self.assertEqual(sliced["c_concat"][0].shape[0], 2)
        self.assertEqual(sliced["tag"], "keep")
        with self.assertRaisesRegex(ValueError, "not batch-aligned"):
            self.memo.slice_cond_rows({"vector": torch.randn(1, 4)}, 3, 2)

    def recorded_call(self, rows=3, n_cond=2):
        denoiser = FakeDenoiser()
        memo = self.memo.arm(denoiser, n_cond=n_cond)
        x = torch.randn(rows, 4, 2, 2)
        cond = {"crossattn": torch.randn(rows, 5, 8), "vector": torch.randn(rows, 6)}
        denoiser.run_inner_model(x, torch.rand(rows), cond)
        self.memo.disarm(denoiser)
        return memo.calls[0]

    def test_prefix_attaches_only_for_the_recorded_conditioning_and_replays_its_rows(self):
        rec = self.recorded_call()
        owner = object()
        unet_x = rec.x * 2
        with self.memo.recording(rec), torch.no_grad():
            slot = self.memo.claim()
            self.assertTrue(self.memo.can_record(slot, unet_x))
            self.memo.attach_prefix(slot, self.memo.Prefix(owner, unet_x, rec.cond["crossattn"].clone(), rec.cond["vector"], {}))
            self.assertIsNone(rec.prefix)
            prefix = self.memo.Prefix(owner, unet_x, rec.cond["crossattn"], rec.cond["vector"], {"h": unet_x})
            self.memo.attach_prefix(slot, prefix)
            self.assertIs(rec.prefix, prefix)
        rec.seal()

        replay_cond = self.memo.slice_cond_rows(rec.cond, rec.rows, 2)
        with self.memo.replaying(rec, 2):
            slot = self.memo.claim()
            self.assertFalse(self.memo.can_record(slot, unet_x[:2]))
            self.assertIsNone(self.memo.replay_prefix(slot, object(), unet_x[:2], replay_cond["crossattn"], replay_cond["vector"]))
            self.assertIsNone(self.memo.replay_prefix(slot, owner, unet_x[:2], replay_cond["crossattn"].clone(), replay_cond["vector"]))
            self.assertIs(self.memo.replay_prefix(slot, owner, unet_x[:2], replay_cond["crossattn"], replay_cond["vector"]), prefix)
            with self.assertRaisesRegex(RuntimeError, "expects 2"):
                self.memo.replay_prefix(slot, owner, unet_x, replay_cond["crossattn"], replay_cond["vector"])
            with self.assertRaisesRegex(RuntimeError, "does not match the recorded call"):
                self.memo.replay_prefix(slot, owner, unet_x[:2].double(), replay_cond["crossattn"], replay_cond["vector"])
        self.assertTrue(torch.equal(self.memo.rows_of(prefix.state["h"], prefix.rows, 2), unet_x[:2]))
        self.assertIs(self.memo.rows_of(0.0, 3, 2), 0.0)
        single = torch.ones(1, 4)
        self.assertIs(self.memo.rows_of(single, 3, 2), single)
        with self.assertRaisesRegex(RuntimeError, "not aligned"):
            self.memo.rows_of(torch.ones(2, 4), 3, 2)

    def test_replaying_drops_the_prefix_when_inputs_changed(self):
        rec = self.recorded_call()
        rec.prefix = self.memo.Prefix(object(), rec.x, rec.cond["crossattn"], rec.cond["vector"], {})
        with self.memo.replaying(rec, 2):
            pass
        self.assertIsNotNone(rec.prefix)
        rec.cond["vector"].mul_(1.0)
        with self.memo.replaying(rec, 2):
            pass
        self.assertIsNone(rec.prefix)

    def test_uncond_only_calls_and_grad_mode_never_record(self):
        denoiser = FakeDenoiser()
        memo = self.memo.arm(denoiser, n_cond=1)
        x_in = torch.randn(2, 1)
        denoiser.run_inner_model(x_in[0:1], torch.zeros(1), {})
        denoiser.run_inner_model(x_in[1:2], torch.zeros(1), {})
        self.assertEqual([rec.cond_rows for rec in memo.calls], [1, 0])
        with self.memo.recording(memo.calls[1]), torch.no_grad():
            self.assertFalse(self.memo.can_record(self.memo.claim(), x_in))
        with self.memo.recording(memo.calls[0]), torch.enable_grad():
            self.assertFalse(self.memo.can_record(self.memo.claim(), x_in))
        memo.calls[0].prefix = object()
        memo.clear()
        self.assertEqual(memo.calls, [])

    def test_hypertile_unet_enabled_reads_wrapped_layer_state(self):
        wrapper = torch.nn.Module()
        wrapper.diffusion_model = torch.nn.Module()
        wrapper.diffusion_model.attn = torch.nn.Linear(1, 1)
        self.assertFalse(self.memo.hypertile_unet_enabled(wrapper))
        self.assertFalse(self.memo.hypertile_unet_enabled(None))

        params = types.SimpleNamespace(enabled=False)
        setattr(wrapper.diffusion_model.attn, "__webui_hypertile_params", params)
        setattr(wrapper, "__webui_hypertile_layers", {"diffusion_model.attn": 1})
        self.assertFalse(self.memo.hypertile_unet_enabled(wrapper))
        params.enabled = True
        self.assertTrue(self.memo.hypertile_unet_enabled(wrapper))


if __name__ == "__main__":
    unittest.main()
