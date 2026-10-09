"""SDXL VAE encodes return the posterior mean: independent of the global RNG, float32, equal to an independent oracle."""

from types import SimpleNamespace

import pytest
import torch

from test.helpers import init_shared

shared = init_shared()

from modules import paths, sd_models_xl  # noqa: E402,F401  (paths puts repositories/ on sys.path for sgm)
import sgm.models.diffusion  # noqa: E402
from sgm.models.autoencoder import AutoencoderKL  # noqa: E402

SCALE = 0.13025


def _tiny_sdxl_vae(dtype):
    torch.manual_seed(0)
    # GroupNorm uses 32 groups: channel counts must be multiples of 32.
    ddconfig = dict(attn_type="vanilla", double_z=True, z_channels=4, resolution=16, in_channels=3, out_ch=3, ch=32,
                    ch_mult=[1, 1], num_res_blocks=1, attn_resolutions=[], dropout=0.0)
    vae = AutoencoderKL(embed_dim=4, ddconfig=ddconfig, lossconfig={"target": "torch.nn.Identity"}).eval().to(dtype)
    engine = SimpleNamespace(first_stage_model=vae, scale_factor=SCALE, en_and_decode_n_samples_a_time=None, disable_first_stage_autocast=True)
    return vae, lambda x: sgm.models.diffusion.DiffusionEngine.encode_first_stage(engine, x)


def _encode_after(encode, x, seed):
    torch.manual_seed(seed)
    torch.randn(17)  # earlier requests consumed the global generator differently
    return encode(x)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sdxl_encode_is_the_posterior_mean_whatever_the_global_rng(dtype):
    vae, encode = _tiny_sdxl_vae(dtype)
    x = (torch.rand(2, 3, 16, 16, generator=torch.Generator().manual_seed(1)) * 2 - 1).to(dtype)

    sampled = [_encode_after(encode, x, seed) for seed in (1, 2)]
    assert not torch.equal(sampled[0], sampled[1])  # stock AutoencoderKL: a draw from the global CPU generator

    sd_models_xl.encode_to_posterior_mode(vae)
    first, second = (_encode_after(encode, x, seed) for seed in (1, 2))

    assert torch.equal(first, second)
    with torch.no_grad():
        mean = torch.chunk(vae.quant_conv(vae.encoder(x)), 2, dim=1)[0]
    assert first.dtype == torch.float32  # as the sampled path was (its noise was float32)
    assert torch.equal(first, SCALE * mean.float())
    sd_models_xl.encode_to_posterior_mode(vae)  # idempotent (extend_sdxl runs on every weight load)
    assert isinstance(vae.regularization, sd_models_xl.DiagonalGaussianMode)
