from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path
import unittest
from unittest import mock

import torch
from safetensors.torch import load_file, save_file

EXT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT_ROOT))

# Package roots install_a1111_stubs() and the scripts.* imports replace; restored when this file finishes so
# later test files import the real A1111 modules.
_STUBBED_PACKAGES = ("modules", "scripts")
_saved_modules = {}


def setUpModule():
    _saved_modules.update({key: value for key, value in sys.modules.items() if key.split(".")[0] in _STUBBED_PACKAGES})


def tearDownModule():
    for key in [key for key in sys.modules if key.split(".")[0] in _STUBBED_PACKAGES]:
        del sys.modules[key]
    sys.modules.update(_saved_modules)
    _saved_modules.clear()


def install_a1111_stubs():
    modules_pkg = types.ModuleType("modules")
    sd_models_mod = types.ModuleType("modules.sd_models")
    sd_models_mod.checkpoints_list = {}
    sd_models_mod.list_models = lambda: None
    sd_vae_mod = types.ModuleType("modules.sd_vae")
    sd_vae_mod.vae_dict = {}
    sd_vae_mod.refresh_vae_list = lambda: None
    shared_mod = types.ModuleType("modules.shared")
    shared_mod.state = types.SimpleNamespace(begin=lambda: None, end=lambda: None, job=None, textinfo=None)
    shared_mod.cmd_opts = types.SimpleNamespace(lora_dir="/tmp/Lora", lyco_dir_backcompat="/tmp/LyCORIS")
    shared_mod.opts = types.SimpleNamespace(list_hidden_files=True)
    paths_internal_mod = types.ModuleType("modules.paths_internal")
    paths_internal_mod.cwd = os.getcwd()
    call_queue_mod = types.ModuleType("modules.call_queue")
    call_queue_mod.queue_lock = RecordingLock()
    script_callbacks_mod = types.ModuleType("modules.script_callbacks")
    script_callbacks_mod.on_app_started = lambda callback: None
    modules_pkg.call_queue = call_queue_mod
    modules_pkg.script_callbacks = script_callbacks_mod

    sys.modules.update(
        {
            "modules": modules_pkg,
            "modules.call_queue": call_queue_mod,
            "modules.paths_internal": paths_internal_mod,
            "modules.script_callbacks": script_callbacks_mod,
            "modules.sd_models": sd_models_mod,
            "modules.sd_vae": sd_vae_mod,
            "modules.shared": shared_mod,
        }
    )
    # A1111's real walker (modules/util.py), which the Lora extension lists LoRAs with
    util_spec = importlib.util.spec_from_file_location("modules.util", EXT_ROOT.parents[1] / "modules" / "util.py")
    util_mod = importlib.util.module_from_spec(util_spec)
    sys.modules["modules.util"] = util_mod
    util_spec.loader.exec_module(util_mod)
    shared_mod.walk_files = util_mod.walk_files


class RecordingLock:
    def __init__(self):
        self.held = False

    def __enter__(self):
        self.held = True

    def __exit__(self, exc_type, exc, tb):
        self.held = False


class LoraDoctorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_a1111_stubs()
        cls.convert = importlib.import_module("scripts.convert")

    def test_lora_payload_keys_are_not_counted_as_known_junk(self):
        model = {
            "lora_unet_down_blocks_0_attentions_0.lora_down.weight": torch.zeros(4, 8),
            "lora_unet_down_blocks_0_attentions_0.lora_up.weight": torch.zeros(8, 4),
            "optimizer.state": torch.zeros(1),
        }
        info = self.convert.MockModelInfo("/tmp/test-lora.safetensors")

        doctor = self.convert.lora_doctor(model, info, {})

        self.assertEqual(doctor["known_junk_count"], 1)
        self.assertEqual(doctor["missing_up_examples"], [])
        self.assertEqual(doctor["missing_down_examples"], [])


class ConverterSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_a1111_stubs()
        cls.convert = importlib.import_module("scripts.convert")

    def test_output_name_rejects_paths(self):
        for name in ("/tmp/out", "../out", "nested/out", "nested\\out", ".", ".."):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "filename"):
                    self.convert.safe_output_name(name)

    def test_output_path_refuses_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.safetensors"
            path.write_bytes(b"existing")

            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                self.convert.safe_output_path(tmpdir, "model", ".safetensors")



    def test_convert_single_coerces_string_booleans(self):
        info = self.convert.MockModelInfo("/tmp/model.safetensors")
        with mock.patch.object(self.convert, "resolve_model_info", return_value=info), mock.patch.object(self.convert, "do_convert", return_value="ok") as do_convert:
            self.assertEqual(self.convert.convert_single({
                "model": "model",
                "formats": ["safetensors"],
                "fix_clip": "false",
                "force_position_id": "false",
                "delete_known_junk_data": "true",
            }), "ok")

        args = do_convert.call_args.args
        self.assertIs(args[10], False)
        self.assertIs(args[11], False)
        self.assertIs(args[12], True)

    def test_legacy_checkpoint_load_rejects_non_mapping_payload(self):
        with mock.patch.object(self.convert.torch, "load", return_value=["not", "a", "state_dict"]):
            with self.assertRaisesRegex(RuntimeError, "state_dict mapping"):
                self.convert.load_model("/tmp/model.ckpt")

    def test_legacy_checkpoint_load_uses_weights_only(self):
        with mock.patch.object(
            self.convert.torch,
            "load",
            return_value={"state_dict": {"x": torch.zeros(1)}},
        ) as load:
            loaded = self.convert.load_model("/tmp/model.ckpt")

        self.assertEqual(set(loaded), {"x"})
        load.assert_called_once_with("/tmp/model.ckpt", map_location="cpu", weights_only=True)


class RouteLockTests(unittest.TestCase):
    """Both routes rebuild the shared checkpoint/VAE maps in place, so they must run under queue_lock."""

    @classmethod
    def setUpClass(cls):
        install_a1111_stubs()
        cls.convert = importlib.import_module("scripts.convert")
        cls.ui = importlib.import_module("scripts.ui")

    def setUp(self):
        self.lock = self.ui.call_queue.queue_lock
        self.routes = {}
        app = types.SimpleNamespace(get=self._register, post=self._register)
        self.ui.on_app_started(None, app)

    def _register(self, path):
        def decorator(fn):
            self.routes[path] = fn
            return fn

        return decorator

    def test_options_lists_models_under_queue_lock(self):
        def converter_options():
            self.assertTrue(self.lock.held)
            return {"models": [], "vaes": ["None"]}

        with mock.patch.object(self.ui.convert, "converter_options", side_effect=converter_options):
            result = self.routes["/sdapi/v1/openclaw/model-converter/options"]()

        self.assertEqual(result, {"ok": True, "models": [], "vaes": ["None"]})
        self.assertFalse(self.lock.held)

    def test_convert_runs_under_queue_lock_and_releases_it_on_failure(self):
        def convert_single(payload):
            self.assertTrue(self.lock.held)
            raise ValueError("selected model was not found")

        with mock.patch.object(self.ui.convert, "convert_single", side_effect=convert_single):
            result = self.routes["/sdapi/v1/openclaw/model-converter/convert"](self.ui.ConvertRequest(model="missing"))

        self.assertEqual((result["ok"], result["error"]), (False, "selected model was not found"))
        self.assertFalse(self.lock.held)


class ConversionCorrectnessTests(unittest.TestCase):
    POSITION_IDS = "conditioner.embedders.0.transformer.text_model.embeddings.position_ids"

    @classmethod
    def setUpClass(cls):
        install_a1111_stubs()
        cls.convert = importlib.import_module("scripts.convert")

    def _source(self, tmpdir):
        path = os.path.join(tmpdir, "model.safetensors")
        save_file(
            {
                "model.diffusion_model.w": torch.tensor([0.5, -1.0, 3.0]),
                "first_stage_model.encoder.conv_in.weight": torch.ones(3),
                self.POSITION_IDS: torch.arange(77).unsqueeze(0),
            },
            path,
        )
        return path

    def _convert(self, source_path, **overrides):
        args = dict(
            checkpoint_formats=["safetensors"], precision="fp16", conv_type="disabled", custom_name="out",
            bake_in_vae="None", unet_conv="convert", text_encoder_conv="convert", vae_conv="convert",
            others_conv="convert", fix_clip=False, force_position_id=True, delete_known_junk_data=False,
        )
        args.update(overrides)
        return self.convert.do_convert(self.convert.MockModelInfo(source_path), **args)

    def test_bake_in_vae_replaces_first_stage_model_weights(self):
        vae = {"encoder.conv_in.weight": torch.full((3,), 2.0)}
        with tempfile.TemporaryDirectory() as tmpdir:
            source = self._source(tmpdir)
            with mock.patch.dict(self.convert.sd_vae.vae_dict, {"baked.safetensors": "/vae/baked.safetensors"}), \
                    mock.patch.object(self.convert.sd_vae, "load_vae_dict", create=True, return_value=vae) as load_vae:
                self._convert(source, bake_in_vae="baked.safetensors")
            out = load_file(os.path.join(tmpdir, "out.safetensors"))
            self.assertEqual(sorted(os.listdir(tmpdir)), ["model.safetensors", "out.safetensors"])

        load_vae.assert_called_once_with("/vae/baked.safetensors", map_location="cpu")
        self.assertNotIn("encoder.conv_in.weight", out)
        self.assertTrue(torch.equal(out["first_stage_model.encoder.conv_in.weight"], torch.full((3,), 2.0, dtype=torch.float16)))

    def test_float8_unet_export_scans_float8_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._convert(
                self._source(tmpdir), precision="float8_e4m3fn",
                clip_precision="fp16", vae_precision="fp16", other_precision="fp16",
            )
            out = load_file(os.path.join(tmpdir, "out.safetensors"))

        self.assertEqual(out["model.diffusion_model.w"].dtype, torch.float8_e4m3fn)
        self.assertEqual(out["model.diffusion_model.w"].float().tolist(), [0.5, -1.0, 3.0])
        self.assertEqual(out["first_stage_model.encoder.conv_in.weight"].dtype, torch.float16)
        self.assertEqual(out[self.POSITION_IDS].dtype, torch.int64)

    def test_nonfinite_scan_and_repair_handle_float8(self):
        model = {
            "a": torch.tensor([1.0, float("nan"), -2.0]).to(torch.float8_e4m3fn),
            "b": torch.tensor([float("inf"), float("-inf"), 0.5]).to(torch.float8_e5m2),
        }

        report = self.convert.scan_and_repair_nonfinite(model, repair=True)

        self.assertEqual((report["nan_values"], report["posinf_values"], report["neginf_values"]), (1, 1, 1))
        self.assertEqual((model["a"].dtype, model["a"].float().tolist()), (torch.float8_e4m3fn, [1.0, 0.0, -2.0]))
        self.assertEqual((model["b"].dtype, model["b"].float().tolist()), (torch.float8_e5m2, [0.0, 0.0, 0.5]))

    def test_fp16_and_bf16_widen_float8_sources_exactly(self):
        bits = torch.arange(256, dtype=torch.uint8)
        for fp8 in (torch.float8_e4m3fn, torch.float8_e5m2):
            source = bits.view(fp8)
            finite = torch.isfinite(source.float())
            for conv, target in ((self.convert.conv_fp16, torch.float16), (self.convert.conv_bf16, torch.bfloat16)):
                with self.subTest(source=fp8, target=target):
                    out = conv(source)
                    self.assertEqual(out.dtype, target)
                    self.assertTrue(torch.equal(out.float()[finite], source.float()[finite]))

    def test_duplicate_formats_are_written_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._convert(self._source(tmpdir), checkpoint_formats=["safetensors", "ckpt", "safetensors"])
            self.assertEqual(sorted(os.listdir(tmpdir)), ["model.safetensors", "out.ckpt", "out.safetensors"])
            # torch.serialization.load: when the real A1111 modules are imported in this process, modules.safe
            # replaces torch.load with a checker that needs the real shared.cmd_opts.
            ckpt = torch.serialization.load(os.path.join(tmpdir, "out.ckpt"), map_location="cpu", weights_only=True)

        self.assertEqual(ckpt["state_dict"]["model.diffusion_model.w"].dtype, torch.float16)

    def test_failed_save_leaves_no_file_under_final_name(self):
        def partial_write(*args, **kwargs):
            Path(args[1]).write_bytes(b"truncated")
            raise OSError("disk full")

        for fmt, target in (("safetensors", "safetensors.torch.save_file"), ("ckpt", "torch.save")):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory() as tmpdir:
                source = self._source(tmpdir)
                module_name, attr = target.rsplit(".", 1)
                module = self.convert.safetensors.torch if module_name == "safetensors.torch" else self.convert.torch
                with mock.patch.object(module, attr, side_effect=partial_write):
                    with self.assertRaisesRegex(OSError, "disk full"):
                        self._convert(source, checkpoint_formats=[fmt])
                self.assertEqual(os.listdir(tmpdir), ["model.safetensors"])

    def test_atomic_save_refuses_file_created_during_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            final = Path(tmpdir) / "out.safetensors"

            def write(path):
                Path(path).write_bytes(b"new")
                final.write_bytes(b"other")

            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                self.convert.save_atomically(str(final), write)
            self.assertEqual((os.listdir(tmpdir), final.read_bytes()), (["out.safetensors"], b"other"))

    def test_invalid_options_raise_instead_of_success_shaped_result(self):
        info = self.convert.MockModelInfo("/nonexistent/model.safetensors")
        cases = (
            {"checkpoint_formats": ["bogus"]},
            {"checkpoint_formats": []},
            {"precision": "fp12"},
            {"conv_type": "noema"},
            {"vae_conv": "remove"},
            {"unet_precision": "fp9"},
            {"precision": "float8_e4m3fn"},
        )
        for case in cases:
            args = dict(
                checkpoint_formats=["safetensors"], precision="fp16", conv_type="disabled", custom_name="",
                bake_in_vae="None", unet_conv="convert", text_encoder_conv="convert", vae_conv="convert",
                others_conv="convert", fix_clip=False, force_position_id=True, delete_known_junk_data=False,
            )
            args.update(case)
            with self.subTest(**case), self.assertRaises(ValueError):
                self.convert.do_convert(info, **args)

    def test_resolvers_only_accept_listed_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outside = Path(tmpdir) / "outside.safetensors"
            outside.write_bytes(b"x")
            listed = Path(tmpdir) / "models" / "Lora" / "sub" / "style.safetensors"
            listed.parent.mkdir(parents=True)
            listed.write_bytes(b"x")
            cmd_opts = types.SimpleNamespace(
                lora_dir=str(Path(tmpdir) / "models" / "Lora"), lyco_dir_backcompat=str(Path(tmpdir) / "models" / "LyCORIS")
            )
            with mock.patch.object(self.convert.shared, "cmd_opts", cmd_opts):
                self.assertIsNone(self.convert.resolve_model_info(str(outside)))
                self.assertIsNone(self.convert.resolve_lora_info(str(outside)))
                self.assertEqual(self.convert.resolve_lora_info("sub/style").filepath, str(listed))
                self.assertEqual(self.convert.resolve_lora_info(str(listed)).filepath, str(listed))

    def test_lora_resolver_accepts_every_root_and_symlinked_directory_a1111_lists(self):
        # networks.list_available_networks walks --lora-dir and --lyco-dir-backcompat with walk_files
        # (os.walk followlinks=True), and the controller sends the path /sdapi/v1/loras reports.
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            lora_dir = root / "custom-lora-dir"
            linked_target = root / "shared-loras"
            linked_target.mkdir()
            (linked_target / "linked.safetensors").write_bytes(b"x")
            lora_dir.mkdir()
            (lora_dir / "top.safetensors").write_bytes(b"x")
            os.symlink(linked_target, lora_dir / "linked-dir")
            lyco_dir = root / "models" / "LyCORIS"
            lyco_dir.mkdir(parents=True)
            (lyco_dir / "lyco.safetensors").write_bytes(b"x")
            cmd_opts = types.SimpleNamespace(lora_dir=str(lora_dir), lyco_dir_backcompat=str(lyco_dir))
            a1111_paths = [
                str(lora_dir / "top.safetensors"),
                str(lora_dir / "linked-dir" / "linked.safetensors"),
                str(lyco_dir / "lyco.safetensors"),
            ]
            with mock.patch.object(self.convert.shared, "cmd_opts", cmd_opts):
                for path in a1111_paths:
                    with self.subTest(path=path):
                        info = self.convert.resolve_lora_info(path)
                        self.assertIsNotNone(info)
                        self.assertEqual(info.filepath, path)
                self.assertEqual(self.convert.resolve_lora_info("linked-dir/linked").filepath, a1111_paths[1])
                self.assertIsNone(self.convert.resolve_lora_info(str(linked_target / "linked.safetensors")))

    def test_lora_listing_skips_hidden_directories_as_a1111_does(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lora_dir = Path(tmpdir) / "Lora"
            hidden = lora_dir / ".hidden" / "style.safetensors"
            hidden.parent.mkdir(parents=True)
            hidden.write_bytes(b"x")
            cmd_opts = types.SimpleNamespace(lora_dir=str(lora_dir), lyco_dir_backcompat=str(Path(tmpdir) / "LyCORIS"))
            with mock.patch.object(self.convert.shared, "cmd_opts", cmd_opts):
                for list_hidden_files, expected in ((True, [str(hidden)]), (False, [])):
                    with self.subTest(list_hidden_files=list_hidden_files), \
                            mock.patch.object(self.convert.shared.opts, "list_hidden_files", list_hidden_files):
                        self.assertEqual([item["path"] for item in self.convert.list_loras()], expected)

    def test_lora_metadata_drops_source_content_hashes(self):
        original = {"sshs_model_hash": "aa", "sshs_legacy_hash": "bb", "modelspec.hash_sha256": "0xcc", "ss_output_name": "style"}

        metadata = self.convert.lora_metadata(
            self.convert.MockModelInfo("/tmp/style.safetensors"), original,
            precision="bf16", doctor={}, cleanup=False, nonfinite={},
        )

        self.assertFalse(set(metadata) & {"sshs_model_hash", "sshs_legacy_hash", "modelspec.hash_sha256"})
        self.assertEqual(metadata["ss_output_name"], "style")

    def test_fix_clip_adds_position_ids_only_to_hf_clip_text_models(self):
        sd1_key = "cond_stage_model.transformer.text_model.embeddings.position_ids"
        cases = (
            ({"conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight": torch.zeros(1),
              "conditioner.embedders.1.model.token_embedding.weight": torch.zeros(1)}, {self.POSITION_IDS}),
            ({"cond_stage_model.transformer.text_model.embeddings.token_embedding.weight": torch.zeros(1)}, {sd1_key}),
            ({"cond_stage_model.model.token_embedding.weight": torch.zeros(1)}, set()),
        )
        for model, expected in cases:
            with self.subTest(keys=sorted(model)):
                before = set(model)
                self.convert.fix_model(model, fix_clip=True)
                self.assertEqual(set(model) - before, expected)
                for key in expected:
                    self.assertTrue(torch.equal(model[key], torch.arange(77).unsqueeze(0)))

    def test_safetensors_detection_is_case_insensitive_and_metadata_errors_propagate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "Style.SafeTensors")
            save_file({"x": torch.ones(1)}, path, metadata={"ss_output_name": "style"})

            self.assertEqual(set(self.convert.load_model(path)), {"x"})
            self.assertEqual(self.convert.safetensors_metadata(path), {"ss_output_name": "style"})
            with mock.patch.object(self.convert.safetensors, "safe_open", side_effect=OSError("unreadable")):
                with self.assertRaisesRegex(OSError, "unreadable"):
                    self.convert.safetensors_metadata(path)


if __name__ == "__main__":
    unittest.main()
