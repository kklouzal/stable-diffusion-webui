"""LoRA key -> model layer resolution (networks.load_network) against real torch/transformers module names."""
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch


def _layer_mapping(**roots):
    """network_layer_mapping as assign_network_names_to_compvis_modules names it: dotted module names with "_"."""
    mapping = {}
    for name, module in torch.nn.ModuleDict(roots).named_modules():
        mapping[name.replace(".", "_")] = module
        module.network_layer_name = name.replace(".", "_")
    return mapping


class _Block(torch.nn.Sequential):
    """An output block: resnet stand-in, optional attention stand-in, then the upsampler (Upsample has .conv)."""

    def __init__(self, attention):
        upsample = torch.nn.Module()
        upsample.conv = torch.nn.Conv2d(4, 4, 3, padding=1)
        super().__init__(torch.nn.Identity(), *([torch.nn.Identity()] if attention else []), upsample)


def _unet(first_block_has_attention):
    """output_blocks 0..5 with the upsamplers of up blocks 0 and 1 where SD1 (no attention in up block 0) and SDXL
    (attention in up blocks 0 and 1) put them."""
    return torch.nn.ModuleDict({"output_blocks": torch.nn.ModuleList([
        torch.nn.Identity(), torch.nn.Identity(), _Block(first_block_has_attention),
        torch.nn.Identity(), torch.nn.Identity(), _Block(True),
    ])})


def _lora_file(tmp_path, bases, out=4, inp=4, kernel=()):
    tensors = {}
    for base in bases:
        tensors[f"{base}.lora_down.weight"] = torch.full((1, inp, *kernel), 0.5)
        tensors[f"{base}.lora_up.weight"] = torch.full((out, 1, *([1] * len(kernel))), 0.25)
        tensors[f"{base}.alpha"] = torch.tensor(1.0)
    path = tmp_path / "net.safetensors"
    safetensors.torch.save_file(tensors, str(path))
    return SimpleNamespace(filename=str(path))


@pytest.fixture
def load(lora_networks, monkeypatch):
    networks = lora_networks
    monkeypatch.setattr(networks.devices, "dtype", torch.float32)
    monkeypatch.setattr(networks.shared.opts, "disable_mmap_load_safetensors", False, raising=False)

    def load(mapping, on_disk):
        monkeypatch.setattr(networks.shared, "sd_model", SimpleNamespace(network_layer_mapping=mapping), raising=False)
        return networks.load_network("net", on_disk)
    return load


@pytest.mark.parametrize("sdxl", [False, True])
def test_diffusers_upsampler_keys_resolve_to_the_upsampler_of_either_unet_layout(load, tmp_path, sdxl):
    mapping = _layer_mapping(diffusion_model=_unet(first_block_has_attention=sdxl))
    on_disk = _lora_file(tmp_path, ["lora_unet_up_blocks_0_upsamplers_0_conv", "lora_unet_up_blocks_1_upsamplers_0_conv"], kernel=(3, 3))

    net = load(mapping, on_disk)

    first = "diffusion_model_output_blocks_2_2_conv" if sdxl else "diffusion_model_output_blocks_2_1_conv"
    assert sorted(net.modules) == sorted([first, "diffusion_model_output_blocks_5_2_conv"])
    assert net.modules[first].sd_module is mapping[first]

