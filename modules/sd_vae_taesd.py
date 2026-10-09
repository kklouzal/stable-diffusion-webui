"""
Tiny AutoEncoder for Stable Diffusion
(DNN for encoding / decoding SD's latent space)

https://github.com/madebyollin/taesd
"""
import os
import torch
import torch.nn as nn

from modules import devices, paths_internal, shared
from modules.util import load_file_from_url

sd_vae_taesd_models = {}


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    @staticmethod
    def forward(x):
        return torch.tanh(x / 3) * 3


class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()

    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))


def decoder(latent_channels=4):
    return nn.Sequential(
        Clamp(), conv(latent_channels, 64), nn.ReLU(),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3),
    )


def encoder(latent_channels=4):
    return nn.Sequential(
        conv(3, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, latent_channels),
    )


def _taesd_model(kind, build):
    """The TAESD `kind` ("decoder" or "encoder") network built by `build` for the current model family, its weights
    downloaded on first use; cached per weights file."""
    if shared.sd_model.is_sd3:
        model_name, latent_channels = f"taesd3_{kind}.pth", 16
    elif shared.sd_model.is_sdxl:
        model_name, latent_channels = f"taesdxl_{kind}.pth", 4
    else:
        model_name, latent_channels = f"taesd_{kind}.pth", 4

    loaded_model = sd_vae_taesd_models.get(model_name)

    if loaded_model is None:
        model_path = load_file_from_url(
            'https://github.com/madebyollin/taesd/raw/main/' + model_name,
            model_dir=os.path.join(paths_internal.models_path, "VAE-taesd"),
            file_name=model_name,
        )
        loaded_model = build(latent_channels)
        loaded_model.load_state_dict(torch.load(model_path, map_location='cpu' if devices.device.type != 'cuda' else None, weights_only=True))
        loaded_model.eval()
        loaded_model.to(devices.device, devices.dtype)
        sd_vae_taesd_models[model_name] = loaded_model

    return loaded_model


def decoder_model():
    return _taesd_model("decoder", decoder)


def encoder_model():
    return _taesd_model("encoder", encoder)
