from __future__ import annotations

import importlib.util
import os
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

import modules
from modules import openclaw_cuda_graphs

shared_stub = types.ModuleType("modules.shared")
shared_stub.opts = types.SimpleNamespace(batch_cond_uncond=True)
shared_stub.sd_model = None

# The graph modules resolve `shared` at call time (sys.modules / `from modules import shared`). Install the stub
# only while this file's tests run: left in sys.modules it replaced the real modules.shared for every test file
# collected after this one (tests/test_inpainting_geometry_contract.py failed on a missing cmd_opts).
# (Only this key is swapped and restored: mock.patch.dict(sys.modules) would also drop every module imported meanwhile.)
_MISSING = object()
_saved_shared_module = _MISSING
_module_patches = [
    mock.patch.object(modules, "shared", shared_stub, create=True),
    mock.patch.object(sys, "argv", [sys.argv[0]]),  # a real shared import must not parse unittest flags
]


def setUpModule():
    global _saved_shared_module
    _saved_shared_module = sys.modules.get("modules.shared", _MISSING)
    sys.modules["modules.shared"] = shared_stub
    for patch in _module_patches:
        patch.start()


def tearDownModule():
    for patch in reversed(_module_patches):
        patch.stop()
    if _saved_shared_module is _MISSING:
        sys.modules.pop("modules.shared", None)
    else:
        sys.modules["modules.shared"] = _saved_shared_module


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


class CudaGraphBypassTests(unittest.TestCase):
    def test_active_seg_always_bypasses_graphs(self):
        for denoiser in (make_denoiser(), make_denoiser(end=3), make_denoiser(hooks=False)):
            self.assertEqual(openclaw_cuda_graphs._graph_denoiser_bypass_reason(denoiser), "seg_attention_hooks")

    def test_masks_bypass_graphs(self):
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

    def test_unmasked_img2img_init_latent_is_graphable(self):
        denoiser = make_denoiser(active=False)
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
        first = openclaw_cuda_graphs._img2img_graph_state_key(img2img)
        img2img.p.init_images = [object(), object()]
        second = openclaw_cuda_graphs._img2img_graph_state_key(img2img)

        self.assertNotEqual(openclaw_cuda_graphs._img2img_graph_state_key(txt2img), first)
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
        openclaw_cuda_graphs._FAILED_KEYS.add(("failed",))

    def test_invalidate_records_reason_only_when_state_is_cleared(self):
        status = openclaw_cuda_graphs.invalidate("model_changed", "empty")
        self.assertEqual(status["invalidations"], 0)

        self.seed_graph_state()
        status = openclaw_cuda_graphs.invalidate("model_changed", {"checkpoint": "next"})

        self.assertEqual(status["cache_size"], 0)
        self.assertEqual(openclaw_cuda_graphs._FAILED_KEYS, set())
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
        openclaw_cuda_graphs.set_enabled(False, clear=True)

    def tearDown(self):
        openclaw_cuda_graphs._MAX_CACHE_SIZE = self.previous_max_cache_size
        openclaw_cuda_graphs.set_enabled(False, clear=True)

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


    def test_evict_removes_least_recent_entry_at_capacity(self):
        openclaw_cuda_graphs._MAX_CACHE_SIZE = 1
        openclaw_cuda_graphs._CACHE[("old",)] = {"dummy": True}

        openclaw_cuda_graphs._evict_if_needed_locked()

        self.assertEqual(openclaw_cuda_graphs._CACHE, {})


    def test_first_capture_returns_graph_replay_after_single_side_stream_warmup(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for capture return-equivalence test")

        openclaw_cuda_graphs._MAX_CACHE_SIZE = 1
        openclaw_cuda_graphs.set_enabled(True, clear=True)
        x = torch.zeros(1, device="cuda")
        calls = []

        def fn(x_arg, sigma_arg, cond=None):
            calls.append(len(calls) + 1)
            return x_arg + calls[-1]

        with torch.no_grad():  # graphs are inference-only: with grad enabled run() falls back to one eager call
            out = openclaw_cuda_graphs.run(fn, x, x, cond={"x": x})

        # One side-stream warm-up (+1), then the capture (+2) whose replay is returned; no extra eager warm.
        self.assertEqual(calls, [1, 2])
        self.assertTrue(torch.equal(out.cpu(), torch.full((1,), 2.0)))
        status = openclaw_cuda_graphs.status()
        self.assertEqual((status["captures"], status["failures"], status["fallbacks"]), (1, 0, 0))
        with torch.no_grad():
            hit = openclaw_cuda_graphs.run(fn, torch.full((1,), 5.0, device="cuda"), x, cond={"x": x})
        self.assertEqual(calls, [1, 2])  # replay only
        self.assertTrue(torch.equal(hit.cpu(), torch.full((1,), 7.0)))

    def test_model_invalidation_waits_for_inflight_replay_and_clone(self):
        openclaw_cuda_graphs.set_enabled(True, clear=True)
        key = (("model",), ("x",), ("sigma",), ("cond",), None, None)
        copy_started = threading.Event()
        release_copy = threading.Event()
        invalidate_done = threading.Event()
        events = []

        class FakeStatic:
            shape = (1,)
            dtype = "float32"
            device = types.SimpleNamespace(type="cuda")
            requires_grad = False

            def __init__(self, name):
                self.name = name

            def copy_(self, _value, non_blocking=False):
                events.append(("copy", self.name))
                if self.name == "x":
                    copy_started.set()
                    release_copy.wait(1)

        class FakeTensor:
            shape = (1,)
            dtype = "float32"
            device = types.SimpleNamespace(type="cuda")
            requires_grad = False

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

        thread_errors = []

        def replay():
            try:
                tensor = FakeTensor()
                result = openclaw_cuda_graphs.run(tensor, tensor, tensor, cond={"c": tensor})
                events.append(("result", result))
            except Exception as exc:  # pragma: no cover - asserted below from the parent thread
                thread_errors.append(exc)

        with mock.patch.object(openclaw_cuda_graphs, "_cache_key", return_value=key), \
             mock.patch.object(openclaw_cuda_graphs, "_graph_denoiser_bypass_reason", return_value=None), \
             mock.patch.object(openclaw_cuda_graphs, "on_default_stream", return_value=True), \
             mock.patch.object(openclaw_cuda_graphs.torch.cuda, "is_available", return_value=True), \
             mock.patch.object(openclaw_cuda_graphs.torch, "is_tensor", side_effect=lambda value: isinstance(value, (FakeStatic, FakeTensor))), \
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
        self.assertEqual(thread_errors, [])
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
        with self.graphs._LOCK:
            for counter in self.graphs._COUNTERS:
                self.graphs._COUNTERS[counter] = 0
            self.graphs._BYPASS_REASONS.clear()
            self.graphs._INVALIDATION_REASONS.clear()
            self.graphs._LAST_ERROR = None
            self.graphs._LAST_KEY = None
        self.previous_opts = shared_stub.opts
        self.previous_cmd_opts = getattr(shared_stub, "cmd_opts", None)
        shared_stub.opts = types.SimpleNamespace(
            sd_vae_decode_method="Full",
            sd_vae_encode_method="Full",
            hypertile_enable_vae=False,
        )
        shared_stub.cmd_opts = types.SimpleNamespace(no_half_vae=False, upcast_sampling=False, precision="autocast")

    def tearDown(self):
        self.graphs.set_enabled(False, clear_cache=True)
        shared_stub.opts = self.previous_opts
        if self.previous_cmd_opts is None:
            delattr(shared_stub, "cmd_opts")
        else:
            shared_stub.cmd_opts = self.previous_cmd_opts

    @staticmethod
    def fake_model(name="model"):
        vae = torch.nn.Module()
        vae.register_parameter("weight", torch.nn.Parameter(torch.ones(1)))
        vae.dtype = torch.float32
        vae.eval()
        info = types.SimpleNamespace(filename=f"/{name}.safetensors", shorthash=name, sha256=f"sha-{name}")
        return types.SimpleNamespace(
            first_stage_model=vae,
            sd_checkpoint_info=info,
            sd_model_hash=name,
            loaded_vae_file=None,
            decode_first_stage=lambda x: x + 1,
            encode_first_stage=lambda x: x,
            get_first_stage_encoding=lambda x: x,
        )

    def test_disabled_status_and_toggle(self):
        self.assertFalse(self.graphs.status()["enabled"])
        status = self.graphs.set_enabled(True, clear_cache=True)
        self.assertTrue(status["enabled"])
        self.assertEqual(status["contract_version"], 3)

    def test_key_contract_distinguishes_semantic_dependencies(self):
        model_a = self.fake_model("a")
        model_b = self.fake_model("b")
        x = torch.empty_strided((1, 4, 8, 8), (256, 64, 8, 1))
        transposed = x.transpose(2, 3)

        with mock.patch.object(self.graphs, "_mutation_epochs", return_value=(("vae_object_epoch", 1),)):
            base = self.graphs._key(model_a, x)
            self.assertNotEqual(base, self.graphs._key(model_b, x))
            # The capture reads a contiguous clone and a replay copies into it: the input layout never splits keys.
            self.assertEqual(base, self.graphs._key(model_a, transposed))
            with mock.patch.object(self.graphs, "_mutation_epochs", return_value=(("vae_object_epoch", 2),)):
                self.assertNotEqual(base, self.graphs._key(model_a, x))

        clone = x.clone()
        if torch.cuda.is_available():
            self.assertNotEqual(base, self.graphs._key(model_a, clone.cuda()))
        self.assertNotEqual(self.graphs._tensor_key(x), self.graphs._tensor_key(x.to(torch.float64)))

    def test_parameter_mutation_and_callable_replacement_change_identity(self):
        model = self.fake_model()
        before = self.graphs._runtime_identity(model)
        with torch.no_grad():
            model.first_stage_model.weight.add_(1)
        after_parameter_mutation = self.graphs._runtime_identity(model)
        self.assertNotEqual(before, after_parameter_mutation)
        model.decode_first_stage = lambda x: x + 2
        self.assertNotEqual(after_parameter_mutation, self.graphs._runtime_identity(model))

    def test_lifecycle_reload_and_teardown_clear_retained_resources(self):
        self.graphs.set_enabled(True, clear_cache=True)
        with self.graphs._LOCK:
            self.graphs._COUNTERS["invalidations"] = 0
            self.graphs._INVALIDATION_REASONS.clear()
        self.graphs._CACHE[("cached",)] = {"graph": object(), "input": object(), "output": object()}
        self.graphs._FAILED_KEYS[("failed",)] = None
        status = self.graphs.invalidate("model_to_device")
        self.assertEqual(status["cache_size"], 0)
        self.assertEqual(status["failed_key_count"], 0)
        self.assertEqual(status["invalidations"], 1)
        self.assertEqual(status["invalidation_reasons"], {"model_to_device": 1})
        self.assertEqual(self.graphs.invalidate("model_to_device")["invalidations"], 1)  # nothing retained: no-op
        self.graphs._CACHE[("cached",)] = {"graph": object(), "input": object(), "output": object()}
        status = self.graphs.set_enabled(False, clear_cache=True)
        self.assertFalse(status["enabled"])
        self.assertEqual(status["cache_size"], 0)

    def test_same_key_replay_is_serialized_through_output_clone(self):
        self.graphs.set_enabled(True, clear_cache=True)
        key = ("key",)
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

        self.graphs._CACHE[key] = {"input": FakeInput(), "graph": FakeGraph(), "output": FakeOutput()}
        results = []

        def replay(name):
            results.append(self.graphs.run(object(), types.SimpleNamespace(name=name)))

        with mock.patch.object(self.graphs, "_bypass_reason", return_value=None), mock.patch.object(self.graphs, "_key", return_value=key):
            first = threading.Thread(target=replay, args=("first",))
            second = threading.Thread(target=replay, args=("second",))
            first.start()
            self.assertTrue(first_copy_started.wait(1))
            second.start()
            self.assertFalse(second_copy_started.wait(0.05))
            release_first_copy.set()
            first.join(1)
            second.join(1)

        self.assertEqual(results, ["output", "output"])
        self.assertEqual(events, [("copy", "first"), ("replay", None), ("clone", None), ("copy", "second"), ("replay", None), ("clone", None)])
        self.assertEqual(self.graphs.status()["replays"], 2)

    def test_cold_publish_then_hit_and_forced_clean_equivalence(self):
        self.graphs.set_enabled(True, clear_cache=True)
        key = ("equivalence",)
        source = types.SimpleNamespace(value=2.0, device="cuda:0", detach=lambda: source, contiguous=lambda: source, clone=lambda: StaticInput())

        class StaticInput:
            def __init__(self):
                self.value = source.value

            def copy_(self, value, non_blocking=False):
                self.value = value.value

        class Output:
            def __init__(self, static_input):
                self.static_input = static_input

            def clone(self):
                return self.static_input.value * 3

        class Graph:
            def replay(self):
                pass

        class Stream:
            def wait_stream(self, stream):
                pass

        class CurrentStream:
            def wait_stream(self, stream):
                pass

        class GraphContext:
            def __enter__(self):
                return None

            def __exit__(self, *args):
                return False

        class StreamContext(GraphContext):
            pass

        execute_calls = []

        def execute(model, static_input):
            execute_calls.append(static_input)
            return static_input.value * 3 if len(execute_calls) % 2 == 1 else Output(static_input)

        patches = (
            mock.patch.object(self.graphs, "_bypass_reason", return_value=None),
            mock.patch.object(self.graphs, "_key", return_value=key),
            mock.patch.object(self.graphs, "_execute", side_effect=execute),
            mock.patch.object(self.graphs.torch.cuda, "Stream", return_value=Stream()),
            mock.patch.object(self.graphs.torch.cuda, "current_stream", return_value=CurrentStream()),
            mock.patch.object(self.graphs.torch.cuda, "stream", return_value=StreamContext()),
            mock.patch.object(self.graphs.torch.cuda, "CUDAGraph", return_value=Graph()),
            mock.patch.object(self.graphs.torch.cuda, "graph", return_value=GraphContext()),
            mock.patch.object(self.graphs.torch.cuda, "synchronize"),
            mock.patch.object(self.graphs.torch.cuda, "graph_pool_handle", return_value=("pool",)),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9]:
            cold = self.graphs.run(object(), source)
            source.value = 4.0
            hit = self.graphs.run(object(), source)
            self.graphs.invalidate("forced_clean")
            clean = self.graphs.run(object(), source)

        self.assertEqual(cold, 6.0)
        self.assertEqual(hit, 12.0)
        self.assertEqual(clean, 12.0)
        self.assertEqual(self.graphs.status()["captures"], 2)
        self.assertEqual(self.graphs.status()["replays"], 1)

    def test_capture_failure_is_atomic_and_failed_keys_are_bounded(self):
        self.graphs.set_enabled(True, clear_cache=True)
        old_max = self.graphs._CACHE_MAX
        self.graphs._CACHE_MAX = 2
        try:
            with mock.patch.object(self.graphs, "_bypass_reason", return_value=None), mock.patch.object(self.graphs, "_execute", side_effect=RuntimeError("capture failed")):
                for index in range(4):
                    fake = types.SimpleNamespace(
                        detach=lambda: fake,
                        contiguous=lambda: fake,
                        clone=lambda: fake,
                        device="cuda:0",
                    )
                    with mock.patch.object(self.graphs, "_key", return_value=("failed", index)), mock.patch.object(self.graphs.torch.cuda, "Stream", side_effect=RuntimeError("capture failed")):
                        self.assertIsNone(self.graphs.run(object(), fake))
            status = self.graphs.status()
            self.assertEqual(status["cache_size"], 0)
            self.assertEqual(status["failed_key_count"], 2)
            self.assertEqual(status["failures"], 4)
        finally:
            self.graphs._CACHE_MAX = old_max

    def test_capacity_eviction_bounds_retained_entries(self):
        old_max = self.graphs._CACHE_MAX
        self.graphs._CACHE_MAX = 2
        try:
            for index in range(3):
                key = (index,)
                self.graphs._CACHE[key] = {"graph": object(), "input": object(), "output": object()}
                self.graphs._evict_locked()
            self.assertEqual(list(self.graphs._CACHE), [(1,), (2,)])
            self.assertEqual(self.graphs.status()["evictions"], 1)
        finally:
            self.graphs._CACHE_MAX = old_max


    def test_non_full_approximation_bypasses(self):
        self.graphs.set_enabled(True, clear_cache=True)
        self.assertIsNone(self.graphs.run(object(), object(), approximation=1))
        self.assertEqual(self.graphs.status()["bypass_reasons"].get("vae_approximation"), 1)


class FakeScheduleWrapper(torch.nn.Module):
    """k-diffusion CompVisDenoiser stand-in: own sigma buffers, the model as `inner_model`, a `quantize` flag."""

    def __init__(self, model, alphas_cumprod, quantize=False):
        super().__init__()
        self.inner_model = model
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        self.register_buffer("sigmas", sigmas)
        self.register_buffer("log_sigmas", sigmas.log())
        self.quantize = quantize

    def forward(self, x, sigma, cond=None):
        return x + 1


class FakeModel(torch.nn.Module):
    def __init__(self, alphas_cumprod):
        super().__init__()
        self.alphas_cumprod = alphas_cumprod
        self.model = torch.nn.Module()
        self.model.diffusion_model = torch.nn.Linear(1, 1)

    def apply_model(self, x, t, cond=None):
        return x


def unmasked_denoiser(model=None):
    denoiser = make_denoiser(active=False)
    if model is not None:
        denoiser.p.sd_model = model
    return denoiser


class CudaGraphRequestOverrideTests(unittest.TestCase):
    """Per-request Python overrides that a replay would skip must keep the call eager (TD-G1, B1, B2)."""

    def setUp(self):
        self.alphas = torch.linspace(0.99, 0.01, 8)
        self.model = FakeModel(self.alphas)
        self.fn = FakeScheduleWrapper(self.model, self.alphas)

    def reason(self, denoiser=None, fn=None):
        return openclaw_cuda_graphs._graph_denoiser_bypass_reason(denoiser or unmasked_denoiser(self.model), fn or self.fn)

    def test_plain_request_is_graphable(self):
        self.assertIsNone(self.reason())

    def test_multidiffusion_inner_model_forward_override_bypasses(self):
        self.fn.forward = lambda x, sigma, cond=None: x  # MultiDiffusion: inner_model.forward = delegate.kdiff_forward
        self.assertEqual(self.reason(), "python_forward_override")

    def test_demofusion_denoiser_forward_override_bypasses(self):
        denoiser = unmasked_denoiser(self.model)
        denoiser.forward = lambda *args, **kwargs: None
        self.assertEqual(self.reason(denoiser), "python_forward_override")

    def test_mixture_of_diffusers_apply_model_override_bypasses_for_wrapper_or_processing_model(self):
        delegate = FakeModel(self.alphas)
        self.model.apply_model = delegate.apply_model  # bound to another object: real override
        self.assertEqual(self.reason(), "python_forward_override")
        # p.sd_model is checked even when the wrapper wraps a different object.
        other = FakeModel(self.alphas)
        other.apply_model = lambda *args, **kwargs: None
        self.assertEqual(
            openclaw_cuda_graphs._graph_denoiser_bypass_reason(unmasked_denoiser(other), FakeScheduleWrapper(FakeModel(self.alphas), self.alphas)),
            "python_forward_override",
        )

    def test_unet_or_wrapper_instance_forward_override_bypasses(self):
        self.model.model.diffusion_model.forward = lambda *args, **kwargs: None
        self.assertEqual(self.reason(), "python_forward_override")
        del self.model.model.diffusion_model.forward
        self.model.model.forward = lambda *args, **kwargs: None
        self.assertEqual(self.reason(), "python_forward_override")

    def test_restored_bound_class_methods_are_not_overrides(self):
        # Extension restore paths (MixtureOfDiffusers, ControlNet, TeaCache, Tiled VAE) assign the saved bound class
        # method back onto the instance; that calls exactly the class method and must not disable graphs forever.
        self.fn.forward = self.fn.forward
        self.model.apply_model = self.model.apply_model
        self.model.model.diffusion_model.forward = self.model.model.diffusion_model.forward
        self.assertIn("forward", vars(self.fn))
        self.assertIsNone(self.reason())

    def test_instance_overrides_contract(self):
        module = torch.nn.Linear(1, 1)
        other = torch.nn.Linear(1, 1)
        self.assertFalse(openclaw_cuda_graphs.instance_overrides(module, "forward"))
        self.assertFalse(openclaw_cuda_graphs.instance_overrides(None, "forward"))
        module.forward = module.forward
        self.assertFalse(openclaw_cuda_graphs.instance_overrides(module, "forward"))
        module.forward = other.forward
        self.assertTrue(openclaw_cuda_graphs.instance_overrides(module, "forward"))
        module.forward = lambda x: x
        self.assertTrue(openclaw_cuda_graphs.instance_overrides(module, "forward"))

    def test_active_unet_hypertile_bypasses(self):
        setattr(self.model.model, "__webui_hypertile_enabled", True)
        self.assertEqual(self.reason(), "hypertile_unet")
        setattr(self.model.model, "__webui_hypertile_enabled", False)
        self.assertIsNone(self.reason())

    def test_hypertile_hook_model_publishes_enabled_summary(self):
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "extensions-builtin", "hypertile")
        sys.path.insert(0, root)
        try:
            import hypertile
        finally:
            sys.path.remove(root)
        wrapper = torch.nn.Module()
        parent = wrapper
        for name in "input_blocks.4.1.transformer_blocks.0".split("."):
            child = torch.nn.Module()
            parent.add_module(name, child)
            parent = child
        parent.add_module("attn1", torch.nn.Linear(1, 1))

        hypertile.hypertile_hook_model(wrapper, 1024, 1024, enable=False)
        self.assertFalse(getattr(wrapper, "__webui_hypertile_enabled", False))  # never hooked
        hypertile.hypertile_hook_model(wrapper, 1024, 1024, enable=True, max_depth=3, is_sdxl=True)
        self.assertTrue(getattr(wrapper, "__webui_hypertile_enabled"))
        hypertile.hypertile_hook_model(wrapper, 1024, 1024, enable=True, max_depth=-1, is_sdxl=True)
        self.assertFalse(getattr(wrapper, "__webui_hypertile_enabled"))  # hooked, every layer above max depth
        hypertile.hypertile_hook_model(wrapper, 1024, 1024, enable=False)
        self.assertFalse(getattr(wrapper, "__webui_hypertile_enabled"))

    def test_token_merging_bypasses(self):
        self.model.applied_token_merged_ratio = 0.3
        self.assertEqual(self.reason(), "token_merging")
        self.model.applied_token_merged_ratio = 0
        self.assertIsNone(self.reason())

    def test_loaded_hypernetworks_bypass(self):
        # Hypernetworks run inside every attention forward from shared.loaded_hypernetworks and their multipliers;
        # a graph captured with or without them would replay that state for every later request.
        shared = types.SimpleNamespace(opts=types.SimpleNamespace(), loaded_hypernetworks=[object()])
        with mock.patch.dict(sys.modules, {"modules.shared": shared}):
            self.assertEqual(self.reason(), "hypernetworks")
            shared.loaded_hypernetworks = []
            self.assertIsNone(self.reason())

    def test_override_bypass_runs_eager_and_records_reason(self):
        openclaw_cuda_graphs.set_enabled(True, clear=True)
        try:
            self.fn.forward = lambda x, sigma, cond=None: x * 3
            x = torch.ones(1)
            out = openclaw_cuda_graphs.run(self.fn, x, x, cond={}, denoiser=unmasked_denoiser(self.model))
            status = openclaw_cuda_graphs.status()
        finally:
            openclaw_cuda_graphs.set_enabled(False, clear=True)
        self.assertTrue(torch.equal(out, torch.full((1,), 3.0)))
        self.assertEqual(status["bypass_reasons"], {"python_forward_override": 1})
        self.assertEqual(status["cache_size"], 0)


class FakeCudaTensor:
    """Just enough tensor surface for run() on a CPU-only host; values live in `value`."""

    dtype = torch.float32
    device = types.SimpleNamespace(type="cuda")
    requires_grad = False

    def __init__(self, value=0.0, shape=(1,)):
        self.value = value
        self.shape = shape

    def stride(self):
        return (1,)

    def detach(self):
        return self

    def clone(self):
        return FakeCudaTensor(self.value, self.shape)

    def copy_(self, other, non_blocking=False):
        self.value = other.value
        return self


class CudaGraphCaptureContractTests(unittest.TestCase):
    """Capture path with faked CUDA objects: one warm-up, shared pool, LRU, entry ownership (G2, G3, BUG-G1)."""

    def setUp(self):
        self.previous_max = openclaw_cuda_graphs._MAX_CACHE_SIZE
        openclaw_cuda_graphs.set_enabled(True, clear=True)
        self.pools = []
        self.graph_calls = []
        real_is_tensor = torch.is_tensor

        class Graph:
            def __init__(graph):
                graph.replays = 0

            def replay(graph):
                graph.replays += 1

        class Context:
            def __enter__(ctx):
                return None

            def __exit__(ctx, *args):
                return False

        def graph(cuda_graph, pool=None):
            self.graph_calls.append(pool)
            return Context()

        def pool_handle():
            self.pools.append(("pool", len(self.pools)))
            return self.pools[-1]

        stream = types.SimpleNamespace(wait_stream=lambda other: None)
        self.patches = [
            mock.patch.object(openclaw_cuda_graphs.torch.cuda, "is_available", return_value=True),
            mock.patch.object(openclaw_cuda_graphs, "on_default_stream", return_value=True),
            mock.patch.object(openclaw_cuda_graphs.torch, "is_grad_enabled", return_value=False),
            mock.patch.object(openclaw_cuda_graphs.torch, "is_tensor", side_effect=lambda v: isinstance(v, FakeCudaTensor) or real_is_tensor(v)),
            mock.patch.object(openclaw_cuda_graphs.torch.cuda, "Stream", return_value=stream),
            mock.patch.object(openclaw_cuda_graphs.torch.cuda, "current_stream", return_value=stream),
            mock.patch.object(openclaw_cuda_graphs.torch.cuda, "stream", return_value=Context()),
            mock.patch.object(openclaw_cuda_graphs.torch.cuda, "CUDAGraph", side_effect=Graph),
            mock.patch.object(openclaw_cuda_graphs.torch.cuda, "graph", side_effect=graph),
            mock.patch.object(openclaw_cuda_graphs.torch.cuda, "graph_pool_handle", side_effect=pool_handle),
        ]
        for patch in self.patches:
            patch.start()
        self.alphas = torch.linspace(0.99, 0.01, 8)
        self.model = FakeModel(self.alphas)

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()
        openclaw_cuda_graphs._MAX_CACHE_SIZE = self.previous_max
        openclaw_cuda_graphs.set_enabled(False, clear=True)

    def make_fn(self, calls):
        fn = FakeScheduleWrapper(self.model, self.alphas)

        def forward(x, sigma, cond=None):
            calls.append(x.shape)
            return FakeCudaTensor(x.value + 1, x.shape)

        fn.forward = forward
        return fn

    def run_shape(self, fn, shape):
        x = FakeCudaTensor(0.0, shape)
        # Instance forward stands in for the class forward here; skip the override bypass this test is not about.
        with mock.patch.object(openclaw_cuda_graphs, "_graph_denoiser_bypass_reason", return_value=None):
            return openclaw_cuda_graphs.run(fn, x, FakeCudaTensor(), cond={"c": FakeCudaTensor()}, denoiser=unmasked_denoiser(self.model))

    def test_new_key_runs_one_warmup_then_capture_and_returns_replay(self):
        calls = []
        fn = self.make_fn(calls)
        out = self.run_shape(fn, (1,))
        self.assertEqual(len(calls), 2)  # side-stream warm-up + capture; no extra eager run
        self.assertEqual(out.value, 1.0)
        entry = next(iter(openclaw_cuda_graphs._CACHE.values()))
        self.assertEqual(entry["graph"].replays, 1)
        self.run_shape(fn, (1,))
        self.assertEqual(len(calls), 2)  # hit: replay only
        self.assertEqual(entry["graph"].replays, 2)

    def test_entry_owns_wrapper_and_schedule_tensors(self):
        fn = self.make_fn([])
        self.run_shape(fn, (1,))
        entry = next(iter(openclaw_cuda_graphs._CACHE.values()))
        self.assertIs(entry["fn"], fn)
        held = entry["schedule"]
        self.assertTrue(any(tensor is fn.log_sigmas for tensor in held))
        self.assertTrue(any(tensor is fn.sigmas for tensor in held))
        self.assertTrue(any(tensor is self.model.alphas_cumprod for tensor in held))

    def test_captures_share_one_pool_until_the_cache_is_cleared(self):
        fn = self.make_fn([])
        for shape in ((1,), (2,), (3,)):
            self.run_shape(fn, shape)
        self.assertEqual(self.graph_calls, [self.pools[0]] * 3)
        openclaw_cuda_graphs.invalidate("model_changed")
        self.run_shape(fn, (1,))
        self.assertEqual(self.graph_calls[-1], self.pools[1])
        self.assertEqual(len(self.pools), 2)

    def test_eviction_is_least_recently_used(self):
        openclaw_cuda_graphs._MAX_CACHE_SIZE = 2
        fn = self.make_fn([])
        self.run_shape(fn, (1,))
        self.run_shape(fn, (2,))
        self.run_shape(fn, (1,))  # hit refreshes (1,)
        self.run_shape(fn, (3,))  # evicts (2,), the least recently used
        shapes = [key[1][1] for key in openclaw_cuda_graphs._CACHE]
        self.assertEqual(shapes, [(1,), (3,)])

    def test_equal_schedules_share_a_graph_and_changed_schedules_do_not(self):
        calls = []
        first = self.make_fn(calls)
        self.run_shape(first, (1,))
        second = self.make_fn(calls)  # the next request's wrapper, same alpha schedule
        self.run_shape(second, (1,))
        self.assertEqual(openclaw_cuda_graphs.status()["captures"], 1)
        self.assertEqual(openclaw_cuda_graphs.status()["replays"], 1)
        self.model.alphas_cumprod = self.alphas.clone()
        self.model.alphas_cumprod[-1] = 0.0  # e.g. zero-terminal-SNR rescale for this request
        third = FakeScheduleWrapper(self.model, self.model.alphas_cumprod.clamp_min(1e-4))
        third.forward = second.forward
        self.run_shape(third, (1,))
        self.assertEqual(openclaw_cuda_graphs.status()["captures"], 2)

    def test_non_default_stream_bypasses(self):
        calls = []
        fn = self.make_fn(calls)
        with mock.patch.object(openclaw_cuda_graphs, "on_default_stream", return_value=False):
            out = self.run_shape(fn, (1,))
        self.assertEqual(out.value, 1.0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(openclaw_cuda_graphs.status()["bypass_reasons"], {"non_default_stream": 1})
        self.assertEqual(openclaw_cuda_graphs.status()["cache_size"], 0)

    def test_runtime_branch_settings_select_their_own_graph(self):
        # upcast_attn/lora_functional change the Python path inside the captured UNet call (override_settings sets
        # them per request without callbacks); a replay must never serve a request with the other setting.
        shared = types.SimpleNamespace(opts=types.SimpleNamespace(upcast_attn=False, lora_functional=False))
        calls = []
        fn = self.make_fn(calls)
        with mock.patch.dict(sys.modules, {"modules.shared": shared}):
            self.run_shape(fn, (1,))
            shared.opts.upcast_attn = True
            self.run_shape(fn, (1,))
            shared.opts.upcast_attn = False
            shared.opts.lora_functional = True
            self.run_shape(fn, (1,))
            self.assertEqual(openclaw_cuda_graphs.status()["captures"], 3)
            shared.opts.lora_functional = False
            self.run_shape(fn, (1,))
        self.assertEqual(openclaw_cuda_graphs.status()["captures"], 3)
        self.assertEqual(openclaw_cuda_graphs.status()["replays"], 1)


class CudaGraphKeyTests(unittest.TestCase):
    def setUp(self):
        self.alphas = torch.linspace(0.99, 0.01, 8)
        self.model = FakeModel(self.alphas)

    def test_lora_signature_uses_loaded_source_key_without_touching_files(self):
        net = types.SimpleNamespace(name="a", mentioned_name="a-alias", te_multiplier=1.0, unet_multiplier=0.5, dyn_dim=None, source_key=("/loras/a.safetensors", ("sha256", "ab"), "lora-source-v2"))

        def no_file_reads(*args, **kwargs):
            raise AssertionError("LoRA files must not be re-read per denoiser call")

        networks_stub = types.SimpleNamespace(loaded_networks=[net], network_file_signature=no_file_reads, network_lora_source_signature=no_file_reads)
        with mock.patch.dict(sys.modules, {"networks": networks_stub}):
            signature = openclaw_cuda_graphs._lora_signature()
        self.assertEqual(signature, (("a", "a-alias", 1.0, 0.5, None, net.source_key),))
        with mock.patch.dict(sys.modules):
            sys.modules.pop("networks", None)
            self.assertIsNone(openclaw_cuda_graphs._lora_signature())

    def test_schedule_signature_is_read_once_per_wrapper_and_tracks_mutation(self):
        fn = FakeScheduleWrapper(self.model, self.alphas)
        first = openclaw_cuda_graphs._schedule_signature(fn)
        self.assertIs(openclaw_cuda_graphs._schedule_signature(fn), first)  # memoized: no new host read
        self.assertEqual(first, openclaw_cuda_graphs._schedule_signature(FakeScheduleWrapper(FakeModel(self.alphas.clone()), self.alphas.clone())))
        with torch.no_grad():
            fn.log_sigmas.add_(1)  # in-place mutation bumps the version
        mutated = openclaw_cuda_graphs._schedule_signature(fn)
        self.assertNotEqual(mutated, first)
        self.model.alphas_cumprod = self.alphas.half().float()  # replaced model schedule (alpha-bar downcast)
        self.assertNotEqual(openclaw_cuda_graphs._schedule_signature(fn), mutated)

    def test_schedule_signature_of_a_wrapper_built_in_inference_mode(self):
        # process_images samples under torch.inference_mode, so the run's wrapper owns inference buffers, which
        # have no version counter (reading tensor._version raises).
        with torch.inference_mode():
            alphas = self.alphas.clone()
            fn = FakeScheduleWrapper(FakeModel(alphas), alphas)
        first = openclaw_cuda_graphs._schedule_signature(fn)
        self.assertIs(openclaw_cuda_graphs._schedule_signature(fn), first)
        self.assertEqual(first, openclaw_cuda_graphs._schedule_signature(FakeScheduleWrapper(self.model, self.alphas)))
        with torch.inference_mode():
            fn.inner_model.alphas_cumprod = alphas.half().float()  # replaced, as per-request options do
        self.assertNotEqual(openclaw_cuda_graphs._schedule_signature(fn), first)

    def test_wrapper_scalars_enter_the_key(self):
        quantized = FakeScheduleWrapper(self.model, self.alphas, quantize=True)
        plain = FakeScheduleWrapper(self.model, self.alphas, quantize=False)
        self.assertNotEqual(openclaw_cuda_graphs._model_signature(quantized), openclaw_cuda_graphs._model_signature(plain))

    def test_key_carries_a_schedule_token_not_the_schedule_values(self):
        # The key is hashed, compared, repr'd and digested on every denoiser call; with the 1000-entry SDXL schedule
        # inline (3 tensors) that was ~1.5 ms of host time per call.
        alphas = torch.linspace(0.9999, 0.0047, 1000, dtype=torch.float64).float()
        fn = FakeScheduleWrapper(FakeModel(alphas), alphas)
        signature = openclaw_cuda_graphs._model_signature(fn)
        self.assertLess(len(repr(signature)), 400)
        equal = FakeScheduleWrapper(FakeModel(alphas.clone()), alphas.clone())
        self.assertEqual(openclaw_cuda_graphs._model_signature(equal), signature)
        changed = alphas.clone()
        changed[500] = changed[500].nextafter(torch.tensor(1.0))  # one ULP anywhere is a different schedule
        self.assertNotEqual(openclaw_cuda_graphs._model_signature(FakeScheduleWrapper(FakeModel(changed), changed)), signature)

    def test_schedule_tokens_are_never_reused_after_a_clear(self):
        # A wrapper memoizes its token across a cache clear; a different schedule seen after the clear must not get
        # the same token, or its key would match graphs captured for the old values.
        fn = FakeScheduleWrapper(self.model, self.alphas)
        first = openclaw_cuda_graphs._schedule_signature(fn)
        openclaw_cuda_graphs.clear()
        other_alphas = self.alphas * 0.5
        other = openclaw_cuda_graphs._schedule_signature(FakeScheduleWrapper(FakeModel(other_alphas), other_alphas))
        self.assertNotEqual(other, first)
        self.assertIs(openclaw_cuda_graphs._schedule_signature(fn), first)

    def test_runtime_branch_key_tracks_attention_forward_upcast_and_functional_lora(self):
        attention = types.SimpleNamespace(CrossAttention=type("CrossAttention", (), {"forward": lambda self, x: x}))
        shared = types.SimpleNamespace(opts=types.SimpleNamespace(upcast_attn=False, lora_functional=False))
        with mock.patch.dict(sys.modules, {"modules.shared": shared, "sgm.modules.attention": attention}):
            base = openclaw_cuda_graphs._runtime_branch_key()
            self.assertEqual(openclaw_cuda_graphs._runtime_branch_key(), base)
            for name in ("upcast_attn", "lora_functional"):
                setattr(shared.opts, name, True)
                self.assertNotEqual(openclaw_cuda_graphs._runtime_branch_key(), base, name)
                setattr(shared.opts, name, False)
            # Changing cross_attention_optimization swaps the class forward through sd_hijack.redo_hijack.
            attention.CrossAttention.forward = lambda self, x: x
            self.assertNotEqual(openclaw_cuda_graphs._runtime_branch_key(), base)

    def test_runtime_branch_key_tracks_the_nhwc_group_norm_switch(self):
        # A captured UNet graph freezes which GroupNorm kernels and layouts ran (modules/openclaw_nhwc_groupnorm.py).
        switch = types.SimpleNamespace(state_key=lambda: ())
        shared = types.SimpleNamespace(opts=types.SimpleNamespace(upcast_attn=False, lora_functional=False))
        with mock.patch.dict(sys.modules, {"modules.shared": shared, "modules.openclaw_nhwc_groupnorm": switch}):
            off = openclaw_cuda_graphs._runtime_branch_key()
            switch.state_key = lambda: ("silu", "unet")
            self.assertNotEqual(openclaw_cuda_graphs._runtime_branch_key(), off)

    def test_attention_key_reads_active_backend_without_status_scan(self):
        optimizations = types.SimpleNamespace(active_sdpa_backend=lambda: "flash,math", sdpa_backend_status=mock.Mock(side_effect=AssertionError("per-call status scan")))
        with mock.patch.dict(sys.modules, {"modules.sd_hijack_optimizations": optimizations}):
            self.assertEqual(openclaw_cuda_graphs._attention_key(), "flash,math")
        with mock.patch.dict(sys.modules):
            sys.modules.pop("modules.sd_hijack_optimizations", None)
            self.assertIsNone(openclaw_cuda_graphs._attention_key())


class CudaGraphSharedPoolDeviceTests(unittest.TestCase):
    """GPU-only: graphs sharing one pool replay in any order with eager-identical outputs."""

    def test_alternating_replays_of_pool_sharing_graphs_match_eager(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for shared-pool replay")
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Conv2d(4, 32, 3, padding=1), torch.nn.SiLU(), torch.nn.Conv2d(32, 4, 3, padding=1)).cuda().eval()

        def fn(x, sigma, cond=None):
            return net(x * sigma.view(-1, 1, 1, 1)) + cond["c"].view(-1, 1, 1, 1)

        previous_max = openclaw_cuda_graphs._MAX_CACHE_SIZE
        openclaw_cuda_graphs._MAX_CACHE_SIZE = 4
        openclaw_cuda_graphs.set_enabled(True, clear=True)
        try:
            with torch.no_grad():
                for step in range(4):
                    for size in (16, 24):
                        x = torch.randn(2, 4, size, size, device="cuda")
                        sigma = torch.rand(2, device="cuda") + 0.5
                        cond = {"c": torch.randn(2, device="cuda")}
                        out = openclaw_cuda_graphs.run(fn, x, sigma, cond)
                        self.assertTrue(torch.equal(out, fn(x, sigma, cond)), (step, size))
            status = openclaw_cuda_graphs.status()
            self.assertEqual(status["captures"], 2)
            self.assertEqual(status["replays"], 6)
            self.assertEqual(status["failures"], 0)
        finally:
            openclaw_cuda_graphs._MAX_CACHE_SIZE = previous_max
            openclaw_cuda_graphs.set_enabled(False, clear=True)


class VaeDecodeGraphSafetyTests(unittest.TestCase):
    def setUp(self):
        from modules import openclaw_vae_decode_graphs

        self.graphs = openclaw_vae_decode_graphs
        self.graphs.set_enabled(True, clear_cache=True)
        # The module under test imports `shared`/`lowvram` lazily; other test modules may have replaced either, so
        # patch exactly what `from modules import ...` resolves.
        self.shared = types.SimpleNamespace(opts=types.SimpleNamespace(sd_vae_decode_method="Full", hypertile_enable_vae=False, lora_functional=False))
        vae = torch.nn.Module()
        vae.decoder = torch.nn.Linear(1, 1)
        vae.eval()
        self.vae = vae
        self.model = types.SimpleNamespace(first_stage_model=vae, decode_first_stage=lambda x: x, lowvram=False)
        self.x = types.SimpleNamespace(is_cuda=True, ndim=4, shape=(1, 4, 8, 8), device="cuda:0")
        import modules

        lowvram = types.SimpleNamespace(is_enabled=lambda model: model.lowvram)
        self.patches = [
            mock.patch.dict(sys.modules, {"modules.lowvram": lowvram, "modules.shared": self.shared}),
            mock.patch.object(modules, "lowvram", lowvram, create=True),
            mock.patch.object(modules, "shared", self.shared, create=True),
            mock.patch.object(self.graphs.torch, "is_tensor", return_value=True),
            mock.patch.object(self.graphs, "on_default_stream", return_value=True),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()
        self.graphs.set_enabled(False, clear_cache=True)

    def test_plain_decode_is_graphable(self):
        self.assertIsNone(self.graphs._bypass_reason(self.model, self.x, 0))

    def test_tiled_vae_decoder_forward_override_bypasses(self):
        self.vae.decoder.forward = lambda z: z  # Tiled VAE: decoder.forward = VAEHook(...)
        self.assertEqual(self.graphs._bypass_reason(self.model, self.x, 0), "vae_forward_override")
        self.assertIsNone(self.graphs.run(self.model, self.x))
        self.assertEqual(self.graphs.status()["failed_key_count"], 0)  # no failed-key latch for later plain decodes

    def test_restored_decoder_forward_is_not_an_override(self):
        self.vae.decoder.forward = self.vae.decoder.forward  # Tiled VAE restore: original_forward written back
        self.assertIsNone(self.graphs._bypass_reason(self.model, self.x, 0))

    def test_non_default_stream_bypasses(self):
        with mock.patch.object(self.graphs, "on_default_stream", return_value=False):
            self.assertEqual(self.graphs._bypass_reason(self.model, self.x, 0), "non_default_stream")

    def test_lora_epoch_enters_vae_key_only_when_lora_can_reach_the_vae(self):
        with mock.patch.object(self.graphs.openclaw_cache_epochs, "epoch_subset", side_effect=lambda dims: tuple((dim, 0) for dim in dims)):
            plain = dict(self.graphs._mutation_epochs())
            self.shared.opts.lora_functional = True
            functional = dict(self.graphs._mutation_epochs())
        self.assertNotIn("lora_applied_epoch", plain)
        self.assertIn("vae_object_epoch", plain)
        self.assertIn("lora_applied_epoch", functional)

    def test_fast_interrupt_decode_bypasses(self):
        # An interrupted request with live_preview_fast_interrupt decodes with the preview method; a full-VAE graph
        # replay would silently change that output.
        self.shared.opts.live_preview_fast_interrupt = True
        self.assertIsNone(self.graphs._bypass_reason(self.model, self.x, 0))
        self.shared.state = types.SimpleNamespace(interrupted=True)
        self.assertEqual(self.graphs._bypass_reason(self.model, self.x, 0), "fast_interrupt")

    def test_key_tracks_attention_implementation_and_upcast(self):
        class AttnBlock(torch.nn.Module):
            def forward(self, x):
                return x

        self.vae.decoder.mid = torch.nn.Module()
        self.vae.decoder.mid.attn_1 = AttnBlock()
        x = torch.zeros(1, 4, 8, 8)

        def key(backend="cudnn"):
            optimizations = types.SimpleNamespace(active_sdpa_backend=lambda: backend)
            with mock.patch.dict(sys.modules, {"modules.sd_hijack_optimizations": optimizations}):
                return self.graphs._key(self.model, x)

        base = key()
        self.assertEqual(key(), base)
        self.assertNotEqual(key("flash"), base)  # runtime SDPA backend switch
        with mock.patch.object(AttnBlock, "forward", lambda block, x: x * 1):
            self.assertNotEqual(key(), base)  # sd_hijack swaps AttnBlock.forward on the class
        self.shared.opts.upcast_attn = True
        self.assertNotEqual(key(), base)  # sdp_attnblock_forward upcasts q/k/v per call

    def test_key_tracks_the_nhwc_group_norm_switch(self):
        x = torch.zeros(1, 4, 8, 8)
        switch = types.SimpleNamespace(state_key=lambda: ())
        with mock.patch.dict(sys.modules, {"modules.openclaw_nhwc_groupnorm": switch}):
            off = self.graphs._key(self.model, x)
            switch.state_key = lambda: ("silu", "vae")  # the VAE GroupNorm kernel and fused swish a capture freezes
            self.assertNotEqual(self.graphs._key(self.model, x), off)

    def test_vae_captures_share_one_pool(self):
        pools = []
        graph_pools = []

        class Context:
            def __enter__(ctx):
                return None

            def __exit__(ctx, *args):
                return False

        class Static:
            def __init__(static, value=0.0):
                static.value = value

            def copy_(static, other, non_blocking=False):
                static.value = other.value

            def clone(static):
                return Static(static.value)

        def source(value):
            item = Static(value)
            item.detach = lambda: item
            item.contiguous = lambda: item
            item.device = "cuda:0"
            return item

        stream = types.SimpleNamespace(wait_stream=lambda other: None)
        with mock.patch.object(self.graphs, "_bypass_reason", return_value=None), \
             mock.patch.object(self.graphs, "_execute", side_effect=lambda model, static: static), \
             mock.patch.object(self.graphs.torch.cuda, "Stream", return_value=stream), \
             mock.patch.object(self.graphs.torch.cuda, "current_stream", return_value=stream), \
             mock.patch.object(self.graphs.torch.cuda, "stream", return_value=Context()), \
             mock.patch.object(self.graphs.torch.cuda, "CUDAGraph", return_value=types.SimpleNamespace(replay=lambda: None)), \
             mock.patch.object(self.graphs.torch.cuda, "graph", side_effect=lambda graph, pool=None: graph_pools.append(pool) or Context()), \
             mock.patch.object(self.graphs.torch.cuda, "graph_pool_handle", side_effect=lambda: pools.append(object()) or pools[-1]), \
             mock.patch.object(self.graphs.torch.cuda, "synchronize"):
            for index in range(3):
                with mock.patch.object(self.graphs, "_key", return_value=("key", index)):
                    self.assertEqual(self.graphs.run(self.model, source(float(index))).value, float(index))
        self.assertEqual(len(pools), 1)
        self.assertEqual(graph_pools, [pools[0]] * 3)

    def test_key_probe_failure_fails_closed_without_capture(self):
        # An identity probe that raises must not collapse to a key shared by every model/VAE (a replay would then run
        # another VAE's captured graph); the decode runs eagerly and the error stays visible.
        before = self.graphs.status()["bypass_reasons"].get("key_probe_failed", 0)
        with mock.patch.object(self.graphs, "_module_revision", side_effect=RuntimeError("probe exploded")), \
             mock.patch.object(self.graphs, "_execute", side_effect=AssertionError("must not capture")):
            self.assertIsNone(self.graphs.run(self.model, self.x))
        status = self.graphs.status()
        self.assertEqual(status["bypass_reasons"].get("key_probe_failed", 0), before + 1)
        self.assertEqual(status["cache_size"], 0)
        self.assertIn("probe exploded", status["last_error"])

    def test_inference_mode_vae_parameters_key_by_identity(self):
        # A checkpoint constructed under torch.inference_mode() (processing reloads models inside it) owns inference
        # tensors, whose ._version raises; the revision keys them by identity/address instead of failing.
        with torch.inference_mode():
            vae = torch.nn.Module()
            vae.decoder = torch.nn.Linear(1, 1)
            other = torch.nn.Module()
            other.decoder = torch.nn.Linear(1, 1)
        first = self.graphs._module_revision(vae)
        self.assertEqual(first, self.graphs._module_revision(vae))
        self.assertNotEqual(first, self.graphs._module_revision(other))
        self.assertTrue(first[4] and all(entry[2] is None for entry in first[4]))

    def test_manual_reset_is_reported_under_the_graph_family(self):
        from modules import openclaw_cache_epochs

        def family(family_id):
            return next(item for item in openclaw_cache_epochs.snapshot()["families"] if item["id"] == family_id)

        e10_before = family("E10")["events"]["invalidate"]
        e11_before = family("E11")["reason_counts"].get("manual", 0)
        self.graphs.set_enabled(None, clear_cache=True)
        self.assertEqual(family("E10")["events"]["invalidate"], e10_before)
        self.assertEqual(family("E11")["reason_counts"].get("manual", 0), e11_before + 1)


def _load_module_copy(module, name, environ):
    """Execute a fresh copy of `module`'s source under `name` (sys.modules is untouched) with `environ` applied."""
    spec = importlib.util.spec_from_file_location(name, Path(module.__file__))
    copy = importlib.util.module_from_spec(spec)
    with mock.patch.dict(os.environ, environ):
        spec.loader.exec_module(copy)
    return copy


class GraphCacheStateContractTests(unittest.TestCase):
    def setUp(self):
        from modules import openclaw_vae_decode_graphs

        self.vae_graphs = openclaw_vae_decode_graphs
        openclaw_cuda_graphs.clear()

    def tearDown(self):
        openclaw_cuda_graphs.clear()
        self.vae_graphs.invalidate("test_cleanup")
        openclaw_cuda_graphs.publish_graph_cache_size()

    def test_cuda_graph_knob_is_read_once_at_import(self):
        self.assertTrue(_load_module_copy(openclaw_cuda_graphs, "openclaw_cuda_graphs_env_on", {"OPENCLAW_CUDA_GRAPHS": "1"})._ENABLED)
        self.assertFalse(_load_module_copy(openclaw_cuda_graphs, "openclaw_cuda_graphs_env_off", {"OPENCLAW_CUDA_GRAPHS": "0"})._ENABLED)
        with self.assertRaises(ValueError):  # fail-fast at startup, not a logged-and-ignored setting
            _load_module_copy(openclaw_cuda_graphs, "openclaw_cuda_graphs_env_bad", {"OPENCLAW_CUDA_GRAPHS": "maybe"})

    def test_vae_import_records_no_invalidation(self):
        fresh = _load_module_copy(self.vae_graphs, "openclaw_vae_decode_graphs_env_on", {"OPENCLAW_VAE_DECODE_GRAPHS": "1"})
        status = fresh.status()
        self.assertTrue(status["enabled"])
        self.assertEqual((status["invalidations"], status["invalidation_reasons"]), (0, {}))

    def test_status_is_a_snapshot(self):
        openclaw_cuda_graphs._CACHE[("stale",)] = {"dummy": True}
        openclaw_cuda_graphs.invalidate("model_changed")
        snapshot = openclaw_cuda_graphs.status()
        openclaw_cuda_graphs._CACHE[("stale",)] = {"dummy": True}
        openclaw_cuda_graphs.invalidate("vae_changed")
        openclaw_cuda_graphs._record_bypass("hypernetworks")
        self.assertEqual(snapshot["invalidation_reasons"], {"model_changed": 1})
        self.assertEqual(snapshot["bypass_reasons"], {})

    def assert_waits_for_runtime_lock(self, action):
        held, release, done = threading.Event(), threading.Event(), threading.Event()

        def hold_runtime_lock():  # stands in for a replay/capture in flight
            with openclaw_cuda_graphs._RUNTIME_LOCK:
                held.set()
                release.wait(1)

        holder = threading.Thread(target=hold_runtime_lock)
        holder.start()
        self.assertTrue(held.wait(1))
        worker = threading.Thread(target=lambda: (action(), done.set()))
        worker.start()
        self.assertFalse(done.wait(0.05))
        release.set()
        holder.join(1)
        worker.join(1)
        self.assertTrue(done.is_set())

    def test_clear_waits_for_inflight_replay_or_capture(self):
        self.assert_waits_for_runtime_lock(openclaw_cuda_graphs.clear)

    def test_disable_waits_for_inflight_replay_or_capture(self):
        self.assert_waits_for_runtime_lock(lambda: openclaw_cuda_graphs.set_enabled(False))

    def test_e11_size_is_the_sum_of_both_graph_caches(self):
        from modules import openclaw_cache_epochs

        def e11():
            return next(item for item in openclaw_cache_epochs.snapshot()["families"] if item["id"] == "E11")

        openclaw_cuda_graphs.publish_graph_cache_size()
        self.assertEqual(e11()["capacity"], openclaw_cuda_graphs._MAX_CACHE_SIZE + self.vae_graphs._CACHE_MAX)
        self.assertEqual(e11()["current_size"], 0)
        openclaw_cuda_graphs._CACHE[("unet",)] = {"dummy": True}
        self.vae_graphs._CACHE[("vae",)] = {"dummy": True}
        openclaw_cuda_graphs.publish_graph_cache_size()
        self.assertEqual(e11()["current_size"], 2)


if __name__ == "__main__":
    unittest.main()
