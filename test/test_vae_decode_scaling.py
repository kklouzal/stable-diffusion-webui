"""SDXL VAE decode input: the sampled float32 latent is scaled by 1/scale_factor in float32 and rounded to the VAE dtype
once (sd_models_xl.decode_first_stage), on the eager path (sd_samplers_common.decode_first_stage ->
samples_to_images_tensor) and the VAE CUDA-graph path (openclaw_vae_decode_graphs._execute). A float32 VAE gets
exactly what upstream sgm computes; models without the float32 contract still get the latent in the VAE dtype."""

import functools
from types import SimpleNamespace

import pytest
import torch

from test.helpers import init_shared

shared = init_shared()

from modules import paths, sd_models_xl  # noqa: E402,F401  (paths puts repositories/ on sys.path for sgm)
# Import order: sd_samplers before sd_samplers_common, openclaw_cuda_graphs before openclaw_vae_decode_graphs (cycles).
from modules import sd_samplers, sd_samplers_common, openclaw_cuda_graphs, openclaw_vae_decode_graphs  # noqa: E402,F401
import sgm.models.diffusion  # noqa: E402

SCALE = 0.13025


class CapturingVAE(torch.nn.Module):
    """first_stage_model stand-in: records the latent decode() receives."""

    def __init__(self, dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros((), dtype=dtype))
        self.decoder = torch.nn.Identity()
        self.received = []

    @property
    def dtype(self):
        return self.weight.dtype

    def decode(self, z):
        self.received.append(z)
        return torch.zeros(z.shape[0], 3, z.shape[2] * 8, z.shape[3] * 8, dtype=z.dtype)


def sdxl_model(dtype, n_samples=None):
    model = SimpleNamespace(first_stage_model=CapturingVAE(dtype), scale_factor=SCALE, en_and_decode_n_samples_a_time=n_samples,
                            disable_first_stage_autocast=False, decode_first_stage_takes_float32=True)
    model.decode_first_stage = functools.partial(sd_models_xl.decode_first_stage, model)
    return model


def latent(seed=0):
    # Sampled latents span roughly +-4 sigma_data / scale_factor; enough elements to hit both rounding outcomes.
    return torch.randn(2, 4, 16, 16, generator=torch.Generator().manual_seed(seed)) * 30


@pytest.fixture
def full_decode(monkeypatch):
    monkeypatch.setitem(shared.opts.data, "sd_vae_decode_method", "Full")
    monkeypatch.setattr(sd_samplers_common.devices, "dtype_vae", torch.bfloat16)


def test_sdxl_engine_patch_is_installed():
    assert sgm.models.diffusion.DiffusionEngine.decode_first_stage is sd_models_xl.decode_first_stage
    assert sgm.models.diffusion.DiffusionEngine.decode_first_stage_takes_float32 is True


def test_eager_decode_rounds_the_scaled_float32_latent_once(full_decode):
    model = sdxl_model(torch.bfloat16)
    z = latent()

    sd_samplers_common.decode_first_stage(model, z)

    (received,) = model.first_stage_model.received
    assert received.dtype == torch.bfloat16
    assert torch.equal(received, (1.0 / SCALE * z).to(torch.bfloat16))
    # The previous order (cast, then scale in bf16) differs by one bf16 ulp in a sizeable share of the elements.
    twice = 1.0 / SCALE * z.to(torch.bfloat16)
    assert (twice != received).float().mean() > 0.05


def test_graph_path_executes_the_same_decode():
    model = sdxl_model(torch.bfloat16)
    z = latent(1)

    openclaw_vae_decode_graphs._execute(model, z)

    (received,) = model.first_stage_model.received
    assert torch.equal(received, (1.0 / SCALE * z).to(torch.bfloat16))


def test_live_preview_full_decode_takes_the_same_path():
    model = sdxl_model(torch.bfloat16)
    z = latent(2)

    sd_samplers_common.samples_to_images_tensor(z, 0, model)

    (received,) = model.first_stage_model.received
    assert torch.equal(received, (1.0 / SCALE * z).to(torch.bfloat16))


def test_chunked_decode_keeps_upstream_chunks():
    model = sdxl_model(torch.bfloat16, n_samples=1)
    z = latent(3)

    out = model.decode_first_stage(z)

    assert [r.shape[0] for r in model.first_stage_model.received] == [1, 1]
    assert torch.equal(torch.cat(model.first_stage_model.received), (1.0 / SCALE * z).to(torch.bfloat16))
    assert out.shape == (2, 3, 128, 128)


def test_float32_vae_is_bitwise_upstream(full_decode, monkeypatch):
    monkeypatch.setattr(sd_samplers_common.devices, "dtype_vae", torch.float32)
    model = sdxl_model(torch.float32)
    z = latent(4)

    sd_samplers_common.decode_first_stage(model, z)

    (received,) = model.first_stage_model.received
    assert torch.equal(received, 1.0 / SCALE * z)  # sgm's own formula on the float32 latent


def test_models_without_the_float32_contract_still_get_the_vae_dtype(full_decode):
    vae = CapturingVAE(torch.bfloat16)
    model = SimpleNamespace(first_stage_model=vae, decode_first_stage=lambda x: vae.decode(1.0 / SCALE * x))
    z = latent(5)

    sd_samplers_common.decode_first_stage(model, z)
    openclaw_vae_decode_graphs._execute(model, z)

    first, second = vae.received
    assert first.dtype == second.dtype == torch.bfloat16
    assert torch.equal(first, 1.0 / SCALE * z.to(torch.bfloat16))
    assert torch.equal(second, first)
