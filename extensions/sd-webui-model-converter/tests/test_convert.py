from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
import unittest

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


if __name__ == "__main__":
    unittest.main()
