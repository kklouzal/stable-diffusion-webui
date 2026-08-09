from __future__ import annotations

import os
import sys
import threading
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

    def test_external_unet_forward_hook_bypasses_graphs(self):
        unet = types.SimpleNamespace(_original_forward=object())
        denoiser = make_denoiser(active=False)
        denoiser.p.sd_model = types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=unet))

        reason = openclaw_cuda_graphs._graph_denoiser_bypass_reason(denoiser)

        self.assertEqual(reason, "external_unet_forward_hook")

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


class CudaGraphInvalidationTests(unittest.TestCase):
    def setUp(self):
        self.previous_env = os.environ.get("OPENCLAW_SDPA_BACKEND")
        openclaw_cuda_graphs.clear()

    def tearDown(self):
        if self.previous_env is None:
            os.environ.pop("OPENCLAW_SDPA_BACKEND", None)
        else:
            os.environ["OPENCLAW_SDPA_BACKEND"] = self.previous_env
        openclaw_cuda_graphs.clear()

    def seed_graph_state(self):
        openclaw_cuda_graphs._CACHE[("stale",)] = {"dummy": True}
        openclaw_cuda_graphs._KEY_LOCKS[("stale",)] = object()
        openclaw_cuda_graphs._FAILED_KEYS.add(("failed",))
        openclaw_cuda_graphs._SEEN_KEYS.add(("seen",))

    def test_invalidate_records_reason_only_when_state_is_cleared(self):
        status = openclaw_cuda_graphs.invalidate("model_changed", "empty")
        self.assertEqual(status["invalidations"], 0)

        self.seed_graph_state()
        status = openclaw_cuda_graphs.invalidate("model_changed", {"checkpoint": "next"})

        self.assertEqual(status["cache_size"], 0)
        self.assertEqual(openclaw_cuda_graphs._KEY_LOCKS, {})
        self.assertEqual(openclaw_cuda_graphs._FAILED_KEYS, set())
        self.assertEqual(openclaw_cuda_graphs._SEEN_KEYS, set())
        self.assertEqual(status["invalidations"], 1)
        self.assertEqual(status["invalidation_reasons"], {"model_changed": 1})
        self.assertEqual(status["last_invalidation_reason"], "model_changed")

    def test_invalidate_if_changed_does_not_thrash_repeated_state(self):
        openclaw_cuda_graphs.invalidate_if_changed("model", ("ckpt-a",), "model_changed")
        self.seed_graph_state()

        repeated = openclaw_cuda_graphs.invalidate_if_changed("model", ("ckpt-a",), "model_changed")

        self.assertEqual(repeated["invalidations"], 0)
        self.assertEqual(repeated["cache_size"], 1)

        changed = openclaw_cuda_graphs.invalidate_if_changed("model", ("ckpt-b",), "model_changed")

        self.assertEqual(changed["invalidations"], 1)
        self.assertEqual(changed["invalidation_reasons"], {"model_changed": 1})
        self.assertEqual(changed["cache_size"], 0)

        repeated_again = openclaw_cuda_graphs.invalidate_if_changed("model", ("ckpt-b",), "model_changed")

        self.assertEqual(repeated_again["invalidations"], 1)
        self.assertEqual(repeated_again["invalidation_reasons"], {"model_changed": 1})

    def test_runtime_refresh_invalidates_on_backend_state_change_once(self):
        os.environ["OPENCLAW_SDPA_BACKEND"] = "flash"
        openclaw_cuda_graphs.refresh_runtime_state()
        self.seed_graph_state()

        unchanged = openclaw_cuda_graphs.refresh_runtime_state()

        self.assertEqual(unchanged["invalidations"], 0)
        self.assertEqual(unchanged["cache_size"], 1)

        os.environ["OPENCLAW_SDPA_BACKEND"] = "math"
        changed = openclaw_cuda_graphs.refresh_runtime_state()

        self.assertEqual(changed["invalidations"], 1)
        self.assertEqual(changed["invalidation_reasons"], {"runtime_changed": 1})
        self.assertEqual(changed["cache_size"], 0)

    def test_note_model_and_vae_loaded_are_noops_for_same_state(self):
        checkpoint = types.SimpleNamespace(filename="a.safetensors", hash="short", sha256="long")
        model = types.SimpleNamespace(sd_checkpoint_info=checkpoint, used_config="cfg", loaded_vae_file="vae.pt", first_stage_model=object())
        openclaw_cuda_graphs.note_model_loaded(model)
        openclaw_cuda_graphs.note_vae_loaded(model)
        self.seed_graph_state()

        same_model = openclaw_cuda_graphs.note_model_loaded(model)
        same_vae = openclaw_cuda_graphs.note_vae_loaded(model)

        self.assertEqual(same_model["invalidations"], 0)
        self.assertEqual(same_vae["invalidations"], 0)
        self.assertEqual(same_vae["cache_size"], 1)

        model.loaded_vae_file = "other.vae.pt"
        changed_vae = openclaw_cuda_graphs.note_vae_loaded(model)

        self.assertEqual(changed_vae["invalidations"], 1)
        self.assertEqual(changed_vae["invalidation_reasons"], {"vae_changed": 1})



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

    def test_model_invalidation_waits_for_inflight_replay_and_clone(self):
        openclaw_cuda_graphs.set_enabled(True, clear=True)
        key = (("model",), ("x",), ("sigma",), ("cond",), None, None)
        copy_started = threading.Event()
        release_copy = threading.Event()
        invalidate_done = threading.Event()
        events = []

        class FakeStatic:
            def __init__(self, name):
                self.name = name

            def copy_(self, _value, non_blocking=False):
                events.append(("copy", self.name))
                if self.name == "x":
                    copy_started.set()
                    release_copy.wait(1)

        class FakeGraph:
            def replay(self):
                events.append(("replay", None))

        class FakeOutput:
            def clone(self):
                events.append(("clone", None))
                return "replayed"

        openclaw_cuda_graphs._CACHE[key] = {
            "x": FakeStatic("x"),
            "sigma": FakeStatic("sigma"),
            "cond": {"c": FakeStatic("cond")},
            "graph": FakeGraph(),
            "out": FakeOutput(),
        }

        def replay():
            result = openclaw_cuda_graphs.run(object(), object(), object(), cond={"c": object()})
            events.append(("result", result))

        with mock.patch.object(openclaw_cuda_graphs, "_cache_key", return_value=key), \
             mock.patch.object(openclaw_cuda_graphs, "_graph_denoiser_bypass_reason", return_value=None), \
             mock.patch.object(openclaw_cuda_graphs.torch.cuda, "is_available", return_value=True), \
             mock.patch.object(openclaw_cuda_graphs.torch, "is_tensor", return_value=True), \
             mock.patch.object(openclaw_cuda_graphs.torch, "is_grad_enabled", return_value=False):
            replay_thread = threading.Thread(target=replay)
            replay_thread.start()
            self.assertTrue(copy_started.wait(1))

            invalidate_thread = threading.Thread(target=lambda: (openclaw_cuda_graphs.invalidate("model_to_cpu"), invalidate_done.set()))
            invalidate_thread.start()
            self.assertFalse(invalidate_done.wait(0.05))

            release_copy.set()
            replay_thread.join(1)
            invalidate_thread.join(1)

        self.assertFalse(replay_thread.is_alive())
        self.assertFalse(invalidate_thread.is_alive())
        self.assertEqual(events[-1], ("result", "replayed"))
        self.assertEqual(openclaw_cuda_graphs.status()["cache_size"], 0)
        self.assertEqual(openclaw_cuda_graphs.status()["invalidations"], 1)



class OpenClawImportOrderTests(unittest.TestCase):
    def test_vae_graph_helper_is_imported_lazily_inside_decode_first_stage(self):
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] / "modules/sd_samplers_common.py").read_text()
        top_imports = source.split("def samples_to_images_tensor", 1)[0]
        self.assertNotIn("openclaw_vae_decode_graphs", top_imports)
        self.assertNotIn("sd_samplers,", top_imports)
        self.assertNotIn("sd_models", top_imports)
        decode_body = source.split("def decode_first_stage", 1)[1].split("def images_tensor_to_samples", 1)[0]
        self.assertIn("from modules import openclaw_vae_decode_graphs", decode_body)
        self.assertIn("from modules import sd_samplers", source)
        self.assertIn("from modules import sd_models", source)


class OpenClawVaeDecodeGraphTests(unittest.TestCase):
    def setUp(self):
        from modules import openclaw_vae_decode_graphs
        self.graphs = openclaw_vae_decode_graphs
        self.graphs.set_enabled(False, clear_cache=True)

    def test_disabled_status_and_toggle(self):
        self.assertFalse(self.graphs.status()["enabled"])
        self.assertTrue(self.graphs.set_enabled(True, clear_cache=True)["enabled"])

    def test_lifecycle_state_repeats_are_noop(self):
        self.graphs.set_enabled(True, clear_cache=True)
        first = self.graphs.invalidate_if_changed("vae", ("a",), "vae_changed")
        second = self.graphs.invalidate_if_changed("vae", ("a",), "vae_changed")
        self.assertEqual(second["invalidations"], first["invalidations"])
        self.graphs.invalidate_if_changed("vae", ("b",), "vae_changed")
        self.assertIn("vae", self.graphs.status()["lifecycle_state_keys"])


    def test_clear_cache_removes_per_key_locks(self):
        key_lock = self.graphs._key_lock(("stale",))
        self.assertIsNotNone(key_lock)

        self.graphs.set_enabled(False, clear_cache=True)

        self.assertEqual(len(self.graphs._KEY_LOCKS), 0)

    def test_unused_per_key_lock_is_reclaimed(self):
        key_lock = self.graphs._key_lock(("stale",))
        self.assertEqual(len(self.graphs._KEY_LOCKS), 1)

        del key_lock

        self.assertEqual(len(self.graphs._KEY_LOCKS), 0)

    def test_same_key_replay_is_serialized_through_output_clone(self):
        self.graphs.set_enabled(True, clear_cache=True)
        key = (("model",), ((1,), "torch.float32", "cuda:0"), 0, "1")
        first_copy_started = threading.Event()
        release_first_copy = threading.Event()
        second_copy_started = threading.Event()
        events = []

        class FakeInput:
            def copy_(self, value, non_blocking=False):
                events.append(("copy", value.name))
                if value.name == "first":
                    first_copy_started.set()
                    release_first_copy.wait(1)
                else:
                    second_copy_started.set()

        class FakeGraph:
            def replay(self):
                events.append(("replay", None))

        class FakeOutput:
            def clone(self):
                events.append(("clone", None))
                return "output"

        entry = {"input": FakeInput(), "graph": FakeGraph(), "output": FakeOutput()}
        self.graphs._CACHE[key] = entry
        results = []

        def replay(name):
            x = types.SimpleNamespace(name=name)
            results.append(self.graphs.run(object(), x))

        with mock.patch.object(self.graphs, "_bypass_reason", return_value=None), \
             mock.patch.object(self.graphs, "_key", return_value=key):
            first = threading.Thread(target=replay, args=("first",))
            second = threading.Thread(target=replay, args=("second",))
            first.start()
            self.assertTrue(first_copy_started.wait(1))
            second.start()
            self.assertFalse(second_copy_started.wait(0.05))
            release_first_copy.set()
            first.join(1)
            second.join(1)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(results, ["output", "output"])
        self.assertEqual(
            events,
            [
                ("copy", "first"),
                ("replay", None),
                ("clone", None),
                ("copy", "second"),
                ("replay", None),
                ("clone", None),
            ],
        )
        self.assertNotIn(key, self.graphs._KEY_LOCKS)
        self.assertEqual(self.graphs.status()["replays"], 2)

    def test_cache_max_env_parse_is_clamped_and_fallback_safe(self):
        previous = os.environ.get("OPENCLAW_VAE_DECODE_GRAPH_CACHE_MAX")
        try:
            os.environ["OPENCLAW_VAE_DECODE_GRAPH_CACHE_MAX"] = "-7"
            self.assertEqual(self.graphs._read_cache_max(), 0)
            os.environ["OPENCLAW_VAE_DECODE_GRAPH_CACHE_MAX"] = "not-an-int"
            self.assertEqual(self.graphs._read_cache_max(), 4)
        finally:
            if previous is None:
                os.environ.pop("OPENCLAW_VAE_DECODE_GRAPH_CACHE_MAX", None)
            else:
                os.environ["OPENCLAW_VAE_DECODE_GRAPH_CACHE_MAX"] = previous

    def test_non_full_approximation_bypasses(self):
        self.graphs.set_enabled(True, clear_cache=True)
        self.assertIsNone(self.graphs.run(object(), object(), approximation=1))
        self.assertEqual(self.graphs.status()["bypass_reasons"].get("vae_approximation"), 1)

if __name__ == "__main__":
    unittest.main()
