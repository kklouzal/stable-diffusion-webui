from __future__ import annotations

import importlib.util
import ast
import sys
import tempfile
import types
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "modules" / "generation_last.py"


def load_generation_last(data_path: Path):
    modules_pkg = types.ModuleType("modules")
    modules_pkg.__path__ = []
    paths = types.ModuleType("modules.paths")
    paths.data_path = str(data_path)
    shared = types.ModuleType("modules.shared")
    shared.opts = types.SimpleNamespace(CLIP_stop_at_last_layers=2)
    shared.state = types.SimpleNamespace(interrupted=False, stopping_generation=False)

    previous = {name: sys.modules.get(name) for name in ("modules", "modules.paths", "modules.shared", "modules.generation_last")}
    sys.modules["modules"] = modules_pkg
    sys.modules["modules.paths"] = paths
    sys.modules["modules.shared"] = shared
    spec = importlib.util.spec_from_file_location("modules.generation_last", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["modules.generation_last"] = module
    spec.loader.exec_module(module)
    return module, shared, previous


def restore_modules(previous):
    for name, value in previous.items():
        if value is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = value


class StableDiffusionProcessingTxt2Img:
    def __init__(self):
        self.prompt = "a red apple on a wooden table"
        self.negative_prompt = ""
        self.styles = []
        self.subseed_strength = 0.0
        self.seed_resize_from_h = 0
        self.seed_resize_from_w = 0
        self.sampler_name = "Euler"
        self.scheduler = "Automatic"
        self.batch_size = 1
        self.n_iter = 1
        self.steps = 20
        self.cfg_scale = 7.0
        self.width = 512
        self.height = 512
        self.restore_faces = False
        self.tiling = False
        self.eta = None
        self.denoising_strength = 0.75
        self.ddim_discretize = None
        self.s_min_uncond = 0.0
        self.s_churn = 0.0
        self.s_tmax = 0.0
        self.s_tmin = 0.0
        self.s_noise = 1.0
        self.refiner_checkpoint = None
        self.refiner_switch_at = 0.8
        self.disable_extra_networks = False
        self.seed = -1
        self.subseed = -1
        self.token_merging_ratio = 0.0
        self.token_merging_ratio_hr = 0.0
        self.enable_hr = True
        self.hr_scale = 2.0
        self.hr_upscaler = "Latent"
        self.hr_second_pass_steps = 12
        self.hr_resize_x = 0
        self.hr_resize_y = 0
        self.hr_checkpoint_name = None
        self.hr_sampler_name = None
        self.hr_scheduler = None
        self.hr_prompt = ""
        self.hr_negative_prompt = ""
        self.override_settings = {}
        self.override_settings_restore_afterwards = False
        self.sd_model = types.SimpleNamespace(sd_checkpoint_info=types.SimpleNamespace(
            title="example-model [abc123]", name_for_extra="example-model", sha256="abc123", filename="/models/example.safetensors"
        ), sd_model_hash="abc123")
        self.sd_model_name = "example-model"
        self.sd_model_hash = "abc123"
        self.sd_vae_name = "example-vae.safetensors"
        self.sd_vae_hash = "def456"
        self.all_seeds = [123456]
        self.all_subseeds = [654321]
        self.scripts = types.SimpleNamespace(alwayson_scripts=[], selectable_scripts=[])
        self.script_args = [0]


class StableDiffusionProcessingImg2Img(StableDiffusionProcessingTxt2Img):
    pass


class GenerationLastTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.module, self.shared, self.previous = load_generation_last(Path(self.temp.name))
        self.processed = types.SimpleNamespace(images=[object()], all_seeds=[123456], all_subseeds=[654321])

    def tearDown(self):
        restore_modules(self.previous)
        self.temp.cleanup()

    def test_txt2img_snapshot_is_persistent_and_replayable(self):
        p = StableDiffusionProcessingTxt2Img()
        snapshot = self.module.capture_completed_generation(p, self.processed)

        self.assertTrue(snapshot["replayable"])
        self.assertEqual(snapshot["schema_version"], 1)
        self.assertEqual(snapshot["generation_type"], "txt2img")
        self.assertEqual(snapshot["parameters"]["seed"], 123456)
        self.assertTrue(snapshot["parameters"]["enable_hr"])
        self.assertEqual(snapshot["checkpoint"]["sha256"], "abc123")
        self.assertEqual(snapshot["parameters"]["override_settings"]["sd_model_checkpoint"], "example-model [abc123]")
        self.assertEqual(self.module.get_last_snapshot(), snapshot)
        path = self.module.snapshot_path()
        self.assertEqual(path, Path(self.temp.name) / "generation-last" / "generation-last.json")
        self.assertTrue(path.is_file())
        self.assertEqual(list(path.parent.glob(".*.tmp")), [])

    def test_cancelled_generation_does_not_replace_previous_snapshot(self):
        p = StableDiffusionProcessingTxt2Img()
        first = self.module.capture_completed_generation(p, self.processed)
        self.shared.state.interrupted = True
        replacement = StableDiffusionProcessingTxt2Img()
        replacement.prompt = "must not replace"

        self.assertIsNone(self.module.capture_completed_generation(replacement, self.processed))
        self.assertEqual(self.module.get_last_snapshot(), first)

    def test_img2img_and_controlnet_are_explicitly_non_replayable_without_inputs(self):
        img2img = StableDiffusionProcessingImg2Img()
        img2img.mask = object()
        snapshot = self.module.build_snapshot(img2img, self.processed)
        self.assertFalse(snapshot["replayable"])
        self.assertTrue(any("init_images" in item for item in snapshot["limitations"]))
        self.assertTrue(any("mask" in item for item in snapshot["limitations"]))

        txt2img = StableDiffusionProcessingTxt2Img()
        txt2img.control_net_enabled = True
        txt2img.control_net_image = "data:image/png;base64,input-is-not-persisted"
        snapshot = self.module.build_snapshot(txt2img, self.processed)
        self.assertFalse(snapshot["replayable"])
        self.assertTrue(any("ControlNet" in item for item in snapshot["limitations"]))
        self.assertNotIn("control_net_image", snapshot["parameters"])

    def test_alwayson_script_arguments_are_replay_payloads(self):
        p = StableDiffusionProcessingTxt2Img()
        script = types.SimpleNamespace(title=lambda: "Example Extension", args_from=1, args_to=3)
        p.scripts = types.SimpleNamespace(alwayson_scripts=[script], selectable_scripts=[])
        p.script_args = [0, True, 0.25]

        snapshot = self.module.build_snapshot(p, self.processed)
        self.assertEqual(snapshot["parameters"]["alwayson_scripts"], {"Example Extension": {"args": [True, 0.25]}})

    def test_prompt_list_uses_the_resolved_first_prompt(self):
        p = StableDiffusionProcessingTxt2Img()
        p.prompt = ["not replayable as a list"]
        p.negative_prompt = ["not replayable as a list"]
        p.all_prompts = ["resolved prompt"]
        p.all_negative_prompts = ["resolved negative"]

        snapshot = self.module.build_snapshot(p, self.processed)
        self.assertEqual(snapshot["parameters"]["prompt"], "resolved prompt")
        self.assertEqual(snapshot["parameters"]["negative_prompt"], "resolved negative")

    def test_missing_snapshot_returns_none(self):
        self.assertIsNone(self.module.get_last_snapshot())

    def test_api_getter_returns_snapshot_or_explicit_404(self):
        source = (MODULE_PATH.parents[0] / "api" / "api.py").read_text(encoding="utf8")
        tree = ast.parse(source)
        api_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Api")
        method = next(node for node in api_class.body if isinstance(node, ast.FunctionDef) and node.name == "get_last_generation")
        subset = ast.Module(body=[ast.ClassDef(name="Api", bases=[], keywords=[], body=[method], decorator_list=[])], type_ignores=[])
        ast.fix_missing_locations(subset)

        class HTTPException(Exception):
            def __init__(self, status_code, detail):
                self.status_code = status_code
                self.detail = detail

        namespace = {"HTTPException": HTTPException, "generation_last": types.SimpleNamespace(get_last_snapshot=lambda: {"schema_version": 1})}
        exec(compile(subset, "<generation-last-api>", "exec"), namespace)
        api = namespace["Api"]()
        self.assertEqual(api.get_last_generation(), {"schema_version": 1})

        namespace["Api"].get_last_generation.__globals__["generation_last"] = types.SimpleNamespace(get_last_snapshot=lambda: None)
        with self.assertRaises(HTTPException) as raised:
            api.get_last_generation()
        self.assertEqual(raised.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
