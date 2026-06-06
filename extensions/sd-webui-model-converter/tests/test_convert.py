from __future__ import annotations

import importlib
import sys
import tempfile
import types
from pathlib import Path
import unittest
from unittest import mock

import torch

EXT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT_ROOT))


def install_a1111_stubs():
    modules_pkg = types.ModuleType("modules")
    paths_mod = types.ModuleType("modules.paths")
    paths_mod.models_path = "/tmp"
    sd_models_mod = types.ModuleType("modules.sd_models")
    sd_models_mod.checkpoints_list = {}
    sd_models_mod.list_models = lambda: None
    sd_vae_mod = types.ModuleType("modules.sd_vae")
    sd_vae_mod.vae_dict = {}
    sd_vae_mod.refresh_vae_list = lambda: None
    shared_mod = types.ModuleType("modules.shared")
    shared_mod.state = types.SimpleNamespace(begin=lambda: None, end=lambda: None, job=None, textinfo=None)

    sys.modules.update(
        {
            "modules": modules_pkg,
            "modules.paths": paths_mod,
            "modules.sd_models": sd_models_mod,
            "modules.sd_vae": sd_vae_mod,
            "modules.shared": shared_mod,
        }
    )


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

    def test_legacy_checkpoint_load_uses_weights_only(self):
        with mock.patch.object(
            self.convert.torch,
            "load",
            return_value={"state_dict": {"x": torch.zeros(1)}},
        ) as load:
            loaded = self.convert.load_model("/tmp/model.ckpt")

        self.assertEqual(set(loaded), {"x"})
        load.assert_called_once_with("/tmp/model.ckpt", map_location="cpu", weights_only=True)


if __name__ == "__main__":
    unittest.main()
