from __future__ import annotations

import importlib.util
import ast
import os
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from PIL import Image


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


class ControlNetUnit:
    def __init__(self, *, enabled=False, image=None, mask=None):
        self.enabled = enabled
        self.input_mode = "simple"
        self.module = "canny"
        self.model = "control_v11p_sd15_canny"
        self.weight = 0.75
        self.resize_mode = "Crop and Resize"
        self.low_vram = False
        self.processor_res = 512
        self.threshold_a = 64.0
        self.threshold_b = 128.0
        self.guidance_start = 0.1
        self.guidance_end = 0.9
        self.pixel_perfect = True
        self.control_mode = "Balanced"
        self.inpaint_crop_input_image = True
        self.hr_option = "Both"
        self.save_detected_map = True
        self.advanced_weighting = None
        self.pulid_mode = "Fidelity"
        self.union_control_type = "Unknown"
        self.image = image
        self.mask = mask
        self.effective_region_mask = None
        self.ipadapter_input = None
        self.batch_images = []
        self.batch_image_files = []


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
        self.assertEqual(snapshot["schema_version"], 2)
        self.assertEqual(snapshot["generation_type"], "txt2img")
        self.assertEqual(snapshot["parameters"]["seed"], 123456)
        self.assertNotIn("prompt", snapshot["parameters"])
        self.assertNotIn("negative_prompt", snapshot["parameters"])
        self.assertNotIn("width", snapshot["parameters"])
        self.assertNotIn("height", snapshot["parameters"])
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

    def test_img2img_assets_and_missing_assets_control_replayability(self):
        img2img = StableDiffusionProcessingImg2Img()
        img2img.init_images = [Image.new("RGB", (16, 16), "red")]
        img2img.mask = Image.new("L", (16, 16), 255)
        snapshot = self.module.build_snapshot(img2img, self.processed)
        self.assertTrue(snapshot["replayable"])
        self.assertTrue(snapshot["parameters"]["init_images"][0])
        self.assertTrue(snapshot["parameters"]["mask"])

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

        p.script_args = [0, {"prompt": "old prompt", "strength": 0.25}]
        script.args_to = 2
        redacted = self.module.build_snapshot(p, self.processed)
        args = redacted["parameters"]["alwayson_scripts"]["Example Extension"]["args"]
        self.assertNotIn("prompt", args[0])
        self.assertTrue(any("Harness must supply" in item for item in redacted["limitations"]))

    def test_controlnet_units_serialize_inputs_and_report_missing_assets(self):
        p = StableDiffusionProcessingTxt2Img()
        script = types.SimpleNamespace(title=lambda: "ControlNet", args_from=1, args_to=2)
        p.scripts = types.SimpleNamespace(alwayson_scripts=[script], selectable_scripts=[])
        p.script_args = [0, ControlNetUnit(enabled=False, image=object())]

        disabled = self.module.build_snapshot(p, self.processed)
        unit = disabled["parameters"]["alwayson_scripts"]["ControlNet"]["args"][0]
        self.assertTrue(disabled["replayable"])
        self.assertEqual(unit["enabled"], False)
        self.assertEqual(unit["module"], "canny")
        self.assertNotIn("image", unit)

        p.script_args[1] = ControlNetUnit(enabled=True, image=Image.new("RGB", (16, 16), "blue"), mask=Image.new("L", (16, 16), 255))
        enabled = self.module.build_snapshot(p, self.processed)
        unit = enabled["parameters"]["alwayson_scripts"]["ControlNet"]["args"][0]
        self.assertTrue(enabled["replayable"])
        self.assertEqual(unit["enabled"], True)
        self.assertEqual(unit["guidance_start"], 0.1)
        self.assertTrue(unit["image"])
        self.assertTrue(unit["mask"])

        p.script_args[1] = ControlNetUnit(enabled=True)
        missing = self.module.build_snapshot(p, self.processed)
        self.assertFalse(missing["replayable"])
        self.assertTrue(any("args[0].image" in item for item in missing["limitations"]))

    def test_api_controlnet_snapshot_roundtrips_without_losing_images(self):
        import base64
        import io
        output = io.BytesIO()
        Image.new("RGB", (32, 32), "blue").save(output, format="PNG")
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        p = StableDiffusionProcessingImg2Img()
        p.init_images = [Image.new("RGB", (32, 32), "red")]
        script = types.SimpleNamespace(title=lambda: "ControlNet", args_from=1, args_to=3)
        p.scripts = types.SimpleNamespace(alwayson_scripts=[script], selectable_scripts=[])
        p.script_args = [0, {"enabled": True, "module": "depth_zoe", "model": "depth", "image": encoded},
                         {"enabled": False, "image": "unused-invalid-image"}]
        for _ in range(3):
            snapshot = self.module.build_snapshot(p, self.processed)
            self.assertTrue(snapshot["replayable"], snapshot["limitations"])
            units = snapshot["parameters"]["alwayson_scripts"]["ControlNet"]["args"]
            self.assertTrue(units[0]["image"])
            self.assertNotIn("image", units[1])
            p.script_args = [0, *units]
        p.script_args[1].pop("image")
        missing = self.module.build_snapshot(p, self.processed)
        self.assertFalse(missing["replayable"])
        self.assertTrue(any("requires" in item and ".image" in item for item in missing["limitations"]))

    def test_prompts_and_dimensions_are_not_retained(self):
        p = StableDiffusionProcessingTxt2Img()
        p.prompt = ["not replayable as a list"]
        p.negative_prompt = ["not replayable as a list"]
        p.all_prompts = ["resolved prompt"]
        p.all_negative_prompts = ["resolved negative"]

        snapshot = self.module.build_snapshot(p, self.processed)
        self.assertNotIn("prompt", snapshot["parameters"])
        self.assertNotIn("negative_prompt", snapshot["parameters"])
        self.assertNotIn("width", snapshot["parameters"])
        self.assertNotIn("height", snapshot["parameters"])

    def test_concurrent_successes_publish_one_complete_snapshot(self):
        snapshots = []

        def capture(seed):
            p = StableDiffusionProcessingTxt2Img()
            processed = types.SimpleNamespace(images=[object()], all_seeds=[seed], all_subseeds=[seed])
            snapshots.append(self.module.capture_completed_generation(p, processed))

        threads = [threading.Thread(target=capture, args=(seed,)) for seed in (11, 22)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        persisted = self.module.get_last_snapshot()
        self.assertIn(persisted["parameters"]["seed"], {11, 22})
        self.assertEqual(persisted["parameters"]["batch_size"], 1)

    def test_failed_generation_and_image_limits_do_not_create_replayable_snapshot(self):
        p = StableDiffusionProcessingTxt2Img()
        self.assertIsNone(self.module.capture_completed_generation(p, types.SimpleNamespace(images=[], all_seeds=[1], all_subseeds=[1])))

        img2img = StableDiffusionProcessingImg2Img()
        img2img.init_images = [Image.frombytes("RGB", (1800, 1800), os.urandom(1800 * 1800 * 3))]
        snapshot = self.module.build_snapshot(img2img, self.processed)
        self.assertFalse(snapshot["replayable"])
        self.assertTrue(any("per-image retention limit" in item for item in snapshot["limitations"]))

    def test_lossless_1024_inputs_above_old_limit_survive_snapshot(self):
        import base64
        import io
        import random
        pixels = random.Random(42).randbytes(1024 * 1024 * 4)
        source = Image.frombytes("RGBA", (1024, 1024), pixels)
        p = StableDiffusionProcessingImg2Img()
        p.init_images = [source]
        p.control_net_enabled = True
        p.control_net_image = source
        script = types.SimpleNamespace(title=lambda: "ControlNet", args_from=1, args_to=2)
        p.scripts = types.SimpleNamespace(alwayson_scripts=[script], selectable_scripts=[])
        p.script_args = [0, ControlNetUnit(enabled=True, image=source)]
        snapshot = self.module.build_snapshot(p, self.processed)
        self.assertTrue(snapshot["replayable"], snapshot["limitations"])
        parameters = snapshot["parameters"]
        encoded = parameters["init_images"][0]
        self.assertGreater(len(encoded), 2 * 1024 * 1024)
        self.assertLessEqual(len(encoded), self.module._MAX_IMAGE_BYTES)
        self.assertEqual(encoded, parameters["control_net_image"])
        self.assertEqual(encoded, parameters["alwayson_scripts"]["ControlNet"]["args"][0]["image"])
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as decoded:
            self.assertEqual(decoded.tobytes(), pixels)

    def test_credential_filter_keeps_token_merging_settings(self):
        p = StableDiffusionProcessingTxt2Img()
        p.token_merging_ratio = 0.5
        p.token_merging_ratio_hr = 0.25
        p.override_settings = {"token_merging_ratio": 0.5, "token_merging_ratio_img2img": 0.35, "api_token": "must-not-leak"}
        snapshot = self.module.build_snapshot(p, self.processed)
        settings = snapshot["parameters"]["override_settings"]
        self.assertEqual(settings["token_merging_ratio"], 0.5)
        self.assertEqual(settings["token_merging_ratio_hr"], 0.25)
        self.assertEqual(settings["token_merging_ratio_img2img"], 0.35)
        self.assertNotIn("api_token", settings)

    def test_inline_api_images_are_retained_without_network_or_path_reads(self):
        import base64
        import io
        encoded = io.BytesIO()
        Image.new("RGB", (8, 8), "red").save(encoded, format="PNG")
        data = base64.b64encode(encoded.getvalue()).decode("ascii")
        for source in (data, "data:image/png;base64," + data):
            limitations = []
            result = self.module._image_to_api_base64(source, limitations, "ControlNet.image", {"images": 0})
            self.assertFalse(limitations)
            with Image.open(io.BytesIO(base64.b64decode(result))) as image:
                self.assertEqual(image.size, (8, 8))
                self.assertEqual(image.convert("RGB").getpixel((0, 0)), (255, 0, 0))
        for source in ("/tmp/private.png", "https://example.invalid/image.png", "data:image/png;base64,invalid!", "A" * (self.module._MAX_IMAGE_BYTES + 1)):
            limitations = []
            self.assertIs(self.module._image_to_api_base64(source, limitations, "ControlNet.image", {"images": 0}), self.module._OMIT)
            self.assertTrue(limitations)

    def test_version_one_snapshot_is_available_without_new_generation(self):
        legacy = {
            "schema_version": 1,
            "completed_at": "2026-01-01T00:00:00Z",
            "generation_type": "txt2img",
            "replayable": True,
            "limitations": [],
            "checkpoint": {},
            "parameters": {"prompt": "old", "negative_prompt": "old", "width": 512, "height": 512, "steps": 20},
        }
        self.module.persist_snapshot(legacy)
        snapshot = self.module.get_last_snapshot()
        self.assertEqual(snapshot["schema_version"], 2)
        self.assertTrue(snapshot["replayable"])
        self.assertEqual(snapshot["parameters"], {"steps": 20})

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
