from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
import types
from pathlib import Path
import unittest
import uuid

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
    sd_models_mod.model_data = types.SimpleNamespace(was_loaded_at_least_once=True, sd_model=None)
    sd_vae_mod = types.ModuleType("modules.sd_vae")
    sd_vae_mod.load_vae = lambda model, vae_file=None, vae_source="from unknown source": None
    sd_hijack_mod = types.ModuleType("modules.sd_hijack")
    sd_hijack_mod.model_hijack = types.SimpleNamespace(get_prompt_lengths=lambda prompt: (len(prompt), 75))
    processing_mod = types.ModuleType("modules.processing")

    class StableDiffusionProcessing:
        cached_c = [None]
        cached_uc = [None]
        cached_img2img_init = [None]

    class StableDiffusionProcessingImg2Img:
        @staticmethod
        def img2img_init_cache_status():
            return {}

    class StableDiffusionProcessingTxt2Img:
        cached_hr_c = [None]
        cached_hr_uc = [None]

    processing_mod.StableDiffusionProcessing = StableDiffusionProcessing
    processing_mod.StableDiffusionProcessingImg2Img = StableDiffusionProcessingImg2Img
    processing_mod.StableDiffusionProcessingTxt2Img = StableDiffusionProcessingTxt2Img
    openclaw_cache_epochs_mod = types.ModuleType("modules.openclaw_cache_epochs")
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

    sys.modules.update(
        {
            "modules": modules_pkg,
            "modules.call_queue": call_queue_mod,
            "modules.extra_networks": extra_networks_mod,
            "modules.extras": extras_mod,
            "modules.openclaw_cache_epochs": openclaw_cache_epochs_mod,
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
        self.assertTrue(getattr(first_wrapper, "__openclaw_backend_status_wrapped__", False))

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

    def test_torch_compile_takes_queue_lock_in_the_threadpool(self):
        calls = []

        def apply(vae=False):
            calls.append(vae)
            return self._record({"ok": True, "vae": vae}, expect_lock=True)()

        self.module.apply_torch_compile_settings = apply
        self.assertEqual(self._call("POST", "/sdapi/v1/openclaw/torch-compile", {"target": "vae-only"}), {"ok": True, "vae": True})
        self.assertEqual(self._call("POST", "/sdapi/v1/openclaw/torch-compile", {"enabled": False}), {"ok": True, "vae": False})
        self.assertEqual(calls, [True, False])

    def test_cudnn_benchmark_takes_queue_lock_in_the_threadpool(self):
        calls = []

        def apply(enabled):
            calls.append(enabled)
            return self._record({"ok": True, "cudnn_benchmark": enabled}, expect_lock=True)()

        self.module.apply_cudnn_benchmark = apply
        self.assertEqual(self._call("POST", "/sdapi/v1/openclaw/cudnn-benchmark", {"enabled": True}), {"ok": True, "cudnn_benchmark": True})
        self.assertEqual(calls, [True])

    def test_model_merge_runs_in_the_threadpool(self):
        self.module._run_openclaw_model_merge = self._record({"ok": True, "message": "merged"}, expect_lock=False)
        self.assertEqual(self._call("POST", "/sdapi/v1/openclaw/model-merge", {"primary_model_name": "a"}), {"ok": True, "message": "merged"})

    def test_token_count_runs_in_the_threadpool(self):
        self.module.estimate_token_count = self._record({"ok": True, "token_count": 3}, expect_lock=False)
        self.assertEqual(self._call("POST", "/sdapi/v1/openclaw/token-count", {"text": "a b c"}), {"ok": True, "token_count": 3})
        self.assertEqual(self._call("POST", "/sdapi/v1/openclaw/token_counter", {"text": "a b c"}), {"ok": True, "token_count": 3})


class TokenCountTests(unittest.TestCase):
    def setUp(self):
        install_a1111_stubs()
        self.module = import_extension_module()

    def test_counts_the_longest_scheduled_prompt(self):
        self.assertEqual(self.module.estimate_token_count("a b c", 20), {"ok": True, "token_count": 5, "max_length": 75})

    def test_missing_text_encoder_is_not_reported_as_a_count(self):
        self.module.model_hijack = types.SimpleNamespace(get_prompt_lengths=lambda prompt: ("-", "-"))
        result = self.module.estimate_token_count("a b c", 20)
        self.assertFalse(result["ok"])
        self.assertIsNone(result["token_count"])
        self.assertIsNone(result["max_length"])


if __name__ == "__main__":
    unittest.main()
