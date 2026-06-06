from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
import unittest
import uuid

EXT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = EXT_ROOT / "scripts" / "openclaw_clear_cond_cache.py"
CURRENT_STATUS_MODULE = None


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
    textual_inversion_pkg = types.ModuleType("modules.textual_inversion")
    textual_inversion_mod = types.ModuleType("modules.textual_inversion.textual_inversion")
    textual_inversion_pkg.textual_inversion = textual_inversion_mod

    modules_pkg.call_queue = call_queue_mod
    modules_pkg.extra_networks = extra_networks_mod
    modules_pkg.extras = extras_mod
    modules_pkg.prompt_parser = prompt_parser_mod
    modules_pkg.script_callbacks = script_callbacks_mod
    modules_pkg.sd_models = sd_models_mod

    sys.modules.update(
        {
            "modules": modules_pkg,
            "modules.call_queue": call_queue_mod,
            "modules.extra_networks": extra_networks_mod,
            "modules.extras": extras_mod,
            "modules.prompt_parser": prompt_parser_mod,
            "modules.processing": processing_mod,
            "modules.script_callbacks": script_callbacks_mod,
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


if __name__ == "__main__":
    unittest.main()
