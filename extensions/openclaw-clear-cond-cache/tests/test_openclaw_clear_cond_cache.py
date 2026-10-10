from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
import types
from pathlib import Path
import unittest
from unittest import mock
import uuid

import torch

EXT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = EXT_ROOT / "scripts" / "openclaw_clear_cond_cache.py"
CURRENT_STATUS_MODULE = None
_saved_modules = {}


def setUpModule():
    # install_a1111_stubs() replaces the `modules` package; restore it when this file finishes so later
    # test files import the real A1111 modules.
    _saved_modules.update({key: value for key, value in sys.modules.items() if key.split(".")[0] == "modules"})


def tearDownModule():
    for key in [key for key in sys.modules if key.split(".")[0] == "modules"]:
        del sys.modules[key]
    sys.modules.update(_saved_modules)
    _saved_modules.clear()


def _original_reload_model_weights(sd_model=None, info=None, forced_reload=False):
    del sd_model, info, forced_reload
    return CURRENT_STATUS_MODULE._backend_status_payload()


def _noop(*args, **kwargs):
    del args, kwargs


def _load_core_module(name: str) -> types.ModuleType:
    """Load a dependency-free core module (no A1111 imports) from the checkout under its real name."""
    spec = importlib.util.spec_from_file_location(f"modules.{name}", EXT_ROOT.parents[1] / "modules" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def install_a1111_stubs() -> None:
    modules_pkg = types.ModuleType("modules")
    call_queue_mod = types.ModuleType("modules.call_queue")
    extra_networks_mod = types.ModuleType("modules.extra_networks")
    extra_networks_mod.parse_prompt = lambda text: (text, [])
    extras_mod = types.ModuleType("modules.extras")
    prompt_parser_mod = types.ModuleType("modules.prompt_parser")
    prompt_parser_mod.get_multicond_prompt_list = lambda prompts: (None, prompts, None)
    prompt_parser_mod.get_learned_conditioning_prompt_schedules = lambda prompts, steps: [[[steps, prompt] for prompt in prompts]]
    script_callbacks_mod = types.ModuleType("modules.script_callbacks")
    script_callbacks_mod.on_app_started = lambda _callback: None
    script_callbacks_mod.on_model_loaded = lambda _callback: None
    sd_models_mod = types.ModuleType("modules.sd_models")
    sd_models_mod.reload_model_weights = _original_reload_model_weights
    # every function the backend-status hooks wrap must exist, or the install fails
    for name in ("load_model", "get_checkpoint_state_dict", "load_model_weights", "instantiate_from_config", "send_model_to_device", "get_empty_cond", "apply_weight_quantization"):
        setattr(sd_models_mod, name, _noop)
    sd_models_mod.model_data = types.SimpleNamespace(was_loaded_at_least_once=True, sd_model=None)
    sd_vae_mod = types.ModuleType("modules.sd_vae")
    sd_vae_mod.load_vae = lambda model, vae_file=None, vae_source="from unknown source": None
    sd_hijack_mod = types.ModuleType("modules.sd_hijack")
    sd_hijack_mod.model_hijack = types.SimpleNamespace(get_prompt_lengths=lambda prompt: (len(prompt), 75))
    processing_mod = types.ModuleType("modules.processing")

    class StableDiffusionProcessing:
        conditioning_cache_lock = threading.RLock()
        cached_c = [None]
        cached_uc = [None]
        cached_img2img_init = [None]

    class StableDiffusionProcessingImg2Img:
        @staticmethod
        def clear_img2img_init_cache():
            StableDiffusionProcessing.cached_img2img_init = [None, None]

        @staticmethod
        def img2img_init_cache_status():
            return {}

    class StableDiffusionProcessingTxt2Img:
        cached_hr_c = [None]
        cached_hr_uc = [None]

    processing_mod.StableDiffusionProcessing = StableDiffusionProcessing
    processing_mod.StableDiffusionProcessingImg2Img = StableDiffusionProcessingImg2Img
    processing_mod.StableDiffusionProcessingTxt2Img = StableDiffusionProcessingTxt2Img
    # the real epoch registry and boolean grammar, fresh per call
    openclaw_cache_epochs_mod = _load_core_module("openclaw_cache_epochs")
    openclaw_env_mod = _load_core_module("openclaw_env")
    textual_inversion_pkg = types.ModuleType("modules.textual_inversion")
    textual_inversion_mod = types.ModuleType("modules.textual_inversion.textual_inversion")
    textual_inversion_pkg.textual_inversion = textual_inversion_mod

    modules_pkg.call_queue = call_queue_mod
    modules_pkg.extra_networks = extra_networks_mod
    modules_pkg.extras = extras_mod
    modules_pkg.prompt_parser = prompt_parser_mod
    modules_pkg.script_callbacks = script_callbacks_mod
    modules_pkg.sd_models = sd_models_mod
    modules_pkg.openclaw_cache_epochs = openclaw_cache_epochs_mod
    modules_pkg.openclaw_env = openclaw_env_mod

    sys.modules.update(
        {
            "modules": modules_pkg,
            "modules.call_queue": call_queue_mod,
            "modules.extra_networks": extra_networks_mod,
            "modules.extras": extras_mod,
            "modules.openclaw_cache_epochs": openclaw_cache_epochs_mod,
            "modules.openclaw_env": openclaw_env_mod,
            "modules.prompt_parser": prompt_parser_mod,
            "modules.processing": processing_mod,
            "modules.script_callbacks": script_callbacks_mod,
            "modules.sd_hijack": sd_hijack_mod,
            "modules.sd_models": sd_models_mod,
            "modules.sd_vae": sd_vae_mod,
            "modules.textual_inversion": textual_inversion_pkg,
            "modules.textual_inversion.textual_inversion": textual_inversion_mod,
        }
    )


def import_extension_module():
    module_name = f"openclaw_clear_cond_cache_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class BackendStatusReloadTests(unittest.TestCase):
    def setUp(self):
        install_a1111_stubs()

    def test_backend_status_wrappers_rebind_to_reloaded_module_state(self):
        global CURRENT_STATUS_MODULE
        sd_models = sys.modules["modules.sd_models"]

        first_module = import_extension_module()
        first_wrapper = sd_models.reload_model_weights
        self.assertIs(first_wrapper.__openclaw_backend_status_original__, _original_reload_model_weights)

        second_module = import_extension_module()
        second_wrapper = sd_models.reload_model_weights
        self.assertIsNot(second_wrapper, first_wrapper)
        self.assertIs(second_wrapper.__openclaw_backend_status_original__, _original_reload_model_weights)

        CURRENT_STATUS_MODULE = second_module
        try:
            status = second_wrapper(info=types.SimpleNamespace(name="model.safetensors"))
        finally:
            CURRENT_STATUS_MODULE = None

        self.assertTrue(status["active"])
        self.assertEqual(status["phase"], "model_load")
        self.assertEqual(status["label"], "Reloading checkpoint")
        self.assertFalse(second_module._backend_status_payload()["active"])
        self.assertIs(first_module._backend_status_payload()["active"], False)

    def test_missing_wrap_target_fails_the_install(self):
        for module_name, attr in (("modules.sd_models", "get_empty_cond"), ("modules.sd_vae", "load_vae")):
            with self.subTest(target=f"{module_name}.{attr}"):
                install_a1111_stubs()
                delattr(sys.modules[module_name], attr)
                with self.assertRaisesRegex(AttributeError, attr):
                    import_extension_module()


class _FakeApp:
    def __init__(self):
        self.routes = {}

    def _register(self, method, path):
        def decorator(fn):
            self.routes[(method, path)] = fn
            return fn

        return decorator

    def post(self, path):
        return self._register("POST", path)

    def get(self, path):
        return self._register("GET", path)


class _FakeRequest:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


class _ThreadRecordingLock:
    def __init__(self):
        self.held_by = None

    def __enter__(self):
        self.held_by = threading.get_ident()

    def __exit__(self, exc_type, exc, tb):
        self.held_by = None


class BlockingHandlersRunOffTheEventLoopTests(unittest.TestCase):
    def setUp(self):
        install_a1111_stubs()
        self.module = import_extension_module()
        self.lock = _ThreadRecordingLock()
        sys.modules["modules.call_queue"].queue_lock = self.lock
        self.app = _FakeApp()
        self.module.on_app_started(None, self.app)
        self.worker_threads = []

    def _record(self, result, *, expect_lock):
        def worker(*args, **kwargs):
            thread = threading.get_ident()
            self.worker_threads.append(thread)
            self.assertEqual(self.lock.held_by, thread if expect_lock else None)
            return result

        return worker

    def _call(self, method, path, payload):
        loop_thread = threading.get_ident()
        result = asyncio.run(self.app.routes[(method, path)](_FakeRequest(payload)))
        self.assertTrue(self.worker_threads)
        self.assertNotIn(loop_thread, self.worker_threads)
        return result

    def test_clear_cond_cache_takes_queue_lock_in_the_threadpool(self):
        self.module.clear_cond_cache = self._record({"ok": True}, expect_lock=True)
        self.assertEqual(self._call("POST", "/sdapi/v1/openclaw/clear-cond-cache", {"targets": ["c"]}), {"ok": True})

    def test_cudnn_benchmark_takes_queue_lock_in_the_threadpool(self):
        calls = []

        def apply(enabled):
            calls.append(enabled)
            return self._record({"ok": True, "cudnn_benchmark": enabled}, expect_lock=True)()

        self.module.apply_cudnn_benchmark = apply
        self.assertEqual(self._call("POST", "/sdapi/v1/openclaw/cudnn-benchmark", {"enabled": True}), {"ok": True, "cudnn_benchmark": True})
        self.assertEqual(calls, [True])

    def test_boolean_fields_parse_text_instead_of_truth_testing_it(self):
        cudnn_calls = []

        def apply_cudnn(enabled):
            cudnn_calls.append(enabled)
            return self._record({"ok": True}, expect_lock=True)()

        self.module.apply_cudnn_benchmark = apply_cudnn
        self._call("POST", "/sdapi/v1/openclaw/cudnn-benchmark", {"enabled": "false"})
        self._call("POST", "/sdapi/v1/openclaw/cudnn-benchmark", {"enabled": "on"})
        # bool("false") was True: this turned cuDNN benchmark on.
        self.assertEqual(cudnn_calls, [False, True])
        compile_route = self.app.routes[("POST", "/sdapi/v1/openclaw/torch-compile")]
        self.assertEqual(asyncio.run(compile_route(_FakeRequest({"vae": "false"})))["requested"], {"vae": False})

        rejected = asyncio.run(self.app.routes[("POST", "/sdapi/v1/openclaw/cudnn-benchmark")](_FakeRequest({"enabled": "maybe"})))
        self.assertEqual(rejected["ok"], False)
        self.assertIn("not a boolean", rejected["error"])
        rejected = asyncio.run(compile_route(_FakeRequest({"enabled": [1]})))
        self.assertEqual(rejected["ok"], False)
        self.assertIn("not a boolean", rejected["error"])
        self.assertEqual(cudnn_calls, [False, True])

    def test_model_merge_flags_parse_text_and_keep_their_defaults(self):
        merges = []
        self.module.extras.run_modelmerger = lambda *args: merges.append(args[6:7] + args[12:15]) or ["merged"]
        self.module.call_queue.queue_lock = _ThreadRecordingLock()
        request = {"primary_model_name": "a", "secondary_model_name": "b"}
        self.assertEqual(self.module._run_openclaw_model_merge(request)["ok"], True)
        self.assertEqual(self.module._run_openclaw_model_merge({**request, "save_as_half": "true", "save_metadata": "false", "add_merge_recipe": 0, "copy_metadata_fields": None})["ok"], True)
        # save_as_half, save_metadata, add_merge_recipe, copy_metadata_fields
        self.assertEqual(merges, [(False, True, True, True), (True, False, False, False)])

        rejected = self.module._run_openclaw_model_merge({**request, "save_metadata": "maybe"})
        self.assertEqual(rejected["ok"], False)
        self.assertIn("not a boolean", rejected["error"])
        self.assertEqual(len(merges), 2)

    def test_model_merge_runs_in_the_threadpool(self):
        self.module._run_openclaw_model_merge = self._record({"ok": True, "message": "merged"}, expect_lock=False)
        self.assertEqual(self._call("POST", "/sdapi/v1/openclaw/model-merge", {"primary_model_name": "a"}), {"ok": True, "message": "merged"})

    def test_token_count_runs_in_the_threadpool(self):
        self.module.estimate_token_count = self._record({"ok": True, "token_count": 3}, expect_lock=False)
        self.assertEqual(self._call("POST", "/sdapi/v1/openclaw/token-count", {"text": "a b c"}), {"ok": True, "token_count": 3})
        self.assertEqual(self._call("POST", "/sdapi/v1/openclaw/token_counter", {"text": "a b c"}), {"ok": True, "token_count": 3})


class _TinyVae(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Conv2d(3, 4, 1)
        self.decoder = torch.nn.Conv2d(4, 3, 1)


class TorchCompileRouteTests(unittest.TestCase):
    """The VAE compile slot is gone: the route answers without ever wrapping first_stage_model."""

    def setUp(self):
        install_a1111_stubs()
        self.module = import_extension_module()
        self.vae = _TinyVae()
        self.model = types.SimpleNamespace(first_stage_model=self.vae)
        sys.modules["modules.sd_models"].model_data.sd_model = self.model
        self.app = _FakeApp()
        self.module.on_app_started(None, self.app)

    def _route(self, method, payload=None):
        handler = self.app.routes[(method, "/sdapi/v1/openclaw/torch-compile")]
        return asyncio.run(handler(_FakeRequest(payload)) if method == "POST" else handler())

    def test_vae_request_is_answered_as_disabled_and_never_compiles(self):
        reason = "VAE decode uses CUDA graphs; module compile does not reach decode/encode"
        status = {
            "ok": True, "desired": {"vae": False}, "status": {"vae": False, "last_error": None},
            "disabled_reason": {"vae": reason},
        }
        with mock.patch.object(torch, "compile", side_effect=AssertionError("torch.compile called")):
            for payload in ({"vae": True}, {"target": "vae-only"}, {"enabled": "true"}):
                self.assertEqual(self._route("POST", payload), {**status, "requested": {"vae": True}})
            self.assertEqual(self._route("POST", {"vae": False}), {**status, "requested": {"vae": False}})
            self.assertEqual(self._route("GET"), status)
            self.module.on_model_loaded(None)
        self.assertIs(self.model.first_stage_model, self.vae)

    def test_in_place_checkpoint_load_reaches_every_vae_weight(self):
        # sd_models.load_model_weights loads a switched checkpoint with load_state_dict(strict=False).
        self._route("POST", {"vae": True})
        fresh = _TinyVae().state_dict()
        result = self.model.first_stage_model.load_state_dict(fresh, strict=False)
        self.assertEqual((result.missing_keys, result.unexpected_keys), ([], []))
        for key, value in self.model.first_stage_model.state_dict().items():
            self.assertTrue(torch.equal(value, fresh[key]), key)

    def test_compile_wrapper_would_skip_every_vae_weight(self):
        # Why the slot was removed: the OptimizedModule wrapper (built lazily, nothing compiles here) renames the
        # keys, so the same strict=False load matched none of them.
        wrapped = torch.compile(_TinyVae())
        result = wrapped.load_state_dict(_TinyVae().state_dict(), strict=False)
        self.assertEqual(sorted(result.unexpected_keys), sorted(_TinyVae().state_dict()))
        self.assertTrue(all(key.startswith("_orig_mod.") for key in result.missing_keys))


class ClearCondCacheTests(unittest.TestCase):
    def setUp(self):
        install_a1111_stubs()
        self.module = import_extension_module()
        processing = sys.modules["modules.processing"]
        self.base = processing.StableDiffusionProcessing
        self.txt2img = processing.StableDiffusionProcessingTxt2Img
        self.base.cached_c = [("c",), "c"]
        self.base.cached_uc = [("uc",), "uc"]
        self.txt2img.cached_hr_c = [("hr_c",), "hr_c"]
        self.txt2img.cached_hr_uc = [("hr_uc",), "hr_uc"]
        self.base.cached_img2img_init = [("img2img",), {"init_latent": object()}]

    def _cached_keys(self):
        return [self.base.cached_c[0], self.base.cached_uc[0], self.txt2img.cached_hr_c[0], self.txt2img.cached_hr_uc[0], self.base.cached_img2img_init[0]]

    def _conditioning_epochs(self):
        return self.module.openclaw_cache_epochs.epoch_subset(("conditioner_epoch", "conditioning_hook_epoch"))

    def test_clear_cond_cache_supports_granular_targets(self):
        result = self.module.clear_cond_cache(["img2img_init"])

        self.assertEqual(result["targets"], ["img2img_init"])
        self.assertEqual(result["cleared"], ["StableDiffusionProcessing.cached_img2img_init"])
        self.assertEqual(self._cached_keys(), [("c",), ("uc",), ("hr_c",), ("hr_uc",), None])
        self.assertEqual(self._conditioning_epochs(), (("conditioner_epoch", 0), ("conditioning_hook_epoch", 0)))

    def test_clear_cond_cache_default_preserves_legacy_clear_all_behavior(self):
        result = self.module.clear_cond_cache()

        self.assertEqual(result["targets"], ["c", "hr_c", "hr_uc", "img2img_init", "uc"])
        self.assertEqual(self._cached_keys(), [None] * 5)
        self.assertEqual(self._conditioning_epochs(), (("conditioner_epoch", 1), ("conditioning_hook_epoch", 1)))


class TokenCountTests(unittest.TestCase):
    def setUp(self):
        install_a1111_stubs()
        self.module = import_extension_module()

    def test_token_count_falls_back_to_input_text_when_prompt_schedule_is_empty(self):
        self.module.prompt_parser.get_learned_conditioning_prompt_schedules = lambda _prompts, _steps: []

        result = self.module.estimate_token_count("fallback prompt", 20)

        self.assertEqual(result, {"ok": True, "token_count": len("fallback prompt"), "max_length": 75})

    def test_token_count_uses_longest_scheduled_prompt(self):
        schedules = [
            [[5, "short"], [10, "medium prompt"]],
            [[20, "the longest scheduled prompt"]],
        ]
        self.module.prompt_parser.get_learned_conditioning_prompt_schedules = lambda _prompts, _steps: schedules

        result = self.module.estimate_token_count("ignored base prompt", 20)

        self.assertEqual(result, {"ok": True, "token_count": len("the longest scheduled prompt"), "max_length": 75})

    def test_missing_text_encoder_is_not_reported_as_a_count(self):
        self.module.model_hijack = types.SimpleNamespace(get_prompt_lengths=lambda prompt: ("-", "-"))
        result = self.module.estimate_token_count("a b c", 20)
        self.assertFalse(result["ok"])
        self.assertIsNone(result["token_count"])
        self.assertIsNone(result["max_length"])


if __name__ == "__main__":
    unittest.main()
