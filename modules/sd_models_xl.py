from __future__ import annotations

import math

import torch

import sgm.models.diffusion
import sgm.modules.autoencoding.regularizers
import sgm.modules.diffusionmodules.denoiser_scaling
import sgm.modules.diffusionmodules.discretizer
import sgm.util
from modules import devices, shared, prompt_parser
from modules import torch_utils


def get_learned_conditioning(self: sgm.models.diffusion.DiffusionEngine, batch: prompt_parser.SdConditioning | list[str]):
    for embedder in self.conditioner.embedders:
        embedder.ucg_rate = 0.0

    width = getattr(batch, 'width', 1024) or 1024
    height = getattr(batch, 'height', 1024) or 1024
    is_negative_prompt = getattr(batch, 'is_negative_prompt', False)
    aesthetic_score = shared.opts.sdxl_refiner_low_aesthetic_score if is_negative_prompt else shared.opts.sdxl_refiner_high_aesthetic_score

    # SDXL embeds these values with sinusoidal timestep embeddings computed in float32, as in the reference
    # implementation. They must not pass through the model dtype first: bfloat16 holds only 8 significant
    # bits, so e.g. a 1500 px size would be embedded as 1504.
    devices_args = dict(device=devices.device, dtype=torch.float32)

    sdxl_conds = {
        "txt": batch,
        "original_size_as_tuple": torch.tensor([height, width], **devices_args).repeat(len(batch), 1),
        "crop_coords_top_left": torch.tensor([shared.opts.sdxl_crop_top, shared.opts.sdxl_crop_left], **devices_args).repeat(len(batch), 1),
        "target_size_as_tuple": torch.tensor([height, width], **devices_args).repeat(len(batch), 1),
        "aesthetic_score": torch.tensor([aesthetic_score], **devices_args).repeat(len(batch), 1),
    }

    force_zero_negative_prompt = is_negative_prompt and all(x == '' for x in batch)
    c = self.conditioner(sdxl_conds, force_zero_embeddings=['txt'] if force_zero_negative_prompt else [])

    return c


def apply_model(self: sgm.models.diffusion.DiffusionEngine, x, t, cond):
    """WARNING: This function is called once per denoising iteration. DO NOT add
    expensive functionc calls such as `model.state_dict`. """
    if self.is_sdxl_inpaint:
        x = torch.cat([x] + cond['c_concat'], dim=1)

    return self.model(x, t, cond)


def get_first_stage_encoding(self, x):  # SDXL's encode_first_stage does everything so get_first_stage_encoding is just there for compatibility
    return x


@torch.no_grad()
def decode_first_stage(self: sgm.models.diffusion.DiffusionEngine, z):
    """sgm's DiffusionEngine.decode_first_stage with the latent scaled in float32 and rounded to the VAE dtype once.

    Upstream scales the latent in the dtype it is given. Callers used to hand it over already cast to the VAE dtype,
    so a bf16 VAE decoded round_bf16(1/scale_factor * round_bf16(z)): two roundings, a quarter of the elements one
    bf16 ulp off round_bf16(1/scale_factor * z). Callers now pass the sampled latent as it is (float32;
    sd_samplers_common.vae_decode_input). Chunking and autocast are upstream's. A float32 VAE is unchanged bitwise."""
    z = (1.0 / self.scale_factor * z.float()).to(self.first_stage_model.dtype)
    n_samples = sgm.util.default(self.en_and_decode_n_samples_a_time, z.shape[0])

    n_rounds = math.ceil(z.shape[0] / n_samples)
    all_out = []
    with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
        for n in range(n_rounds):
            if isinstance(self.first_stage_model.decoder, sgm.models.diffusion.VideoDecoder):
                kwargs = {"timesteps": len(z[n * n_samples : (n + 1) * n_samples])}
            else:
                kwargs = {}
            out = self.first_stage_model.decode(z[n * n_samples : (n + 1) * n_samples], **kwargs)
            all_out.append(out)
    return torch.cat(all_out, dim=0)


sgm.models.diffusion.DiffusionEngine.get_learned_conditioning = get_learned_conditioning
sgm.models.diffusion.DiffusionEngine.apply_model = apply_model
sgm.models.diffusion.DiffusionEngine.get_first_stage_encoding = get_first_stage_encoding
sgm.models.diffusion.DiffusionEngine.decode_first_stage = decode_first_stage
sgm.models.diffusion.DiffusionEngine.decode_first_stage_takes_float32 = True  # see sd_samplers_common.vae_decode_input


def encode_embedding_init_text(self: sgm.modules.GeneralConditioner, init_text, nvpt):
    res = []

    for embedder in [embedder for embedder in self.embedders if hasattr(embedder, 'encode_embedding_init_text')]:
        encoded = embedder.encode_embedding_init_text(init_text, nvpt)
        res.append(encoded)

    return torch.cat(res, dim=1)


def first_embedder_with_attr(self: sgm.modules.GeneralConditioner, attr: str):
    for embedder in self.embedders:
        if hasattr(embedder, attr):
            return embedder

    return None


def tokenize(self: sgm.modules.GeneralConditioner, texts):
    embedder = first_embedder_with_attr(self, 'tokenize')
    if embedder is not None:
        return embedder.tokenize(texts)

    raise AssertionError('no tokenizer available')



def process_texts(self, texts):
    embedder = first_embedder_with_attr(self, 'process_texts')
    if embedder is not None:
        return embedder.process_texts(texts)


def get_target_prompt_token_count(self, token_count):
    embedder = first_embedder_with_attr(self, 'get_target_prompt_token_count')
    if embedder is not None:
        return embedder.get_target_prompt_token_count(token_count)


# those additions to GeneralConditioner make it possible to use it as model.cond_stage_model from SD1.5 in exist
sgm.modules.GeneralConditioner.encode_embedding_init_text = encode_embedding_init_text
sgm.modules.GeneralConditioner.tokenize = tokenize
sgm.modules.GeneralConditioner.process_texts = process_texts
sgm.modules.GeneralConditioner.get_target_prompt_token_count = get_target_prompt_token_count


def extend_sdxl(model):
    """this adds a bunch of parameters to make SDXL model look a bit more like SD1.5 to the rest of the codebase."""

    dtype = torch_utils.get_param(model.model.diffusion_model).dtype
    model.model.diffusion_model.dtype = dtype
    model.model.conditioning_key = 'crossattn'
    model.cond_stage_key = 'txt'
    # model.cond_stage_model will be set in sd_hijack

    model.parameterization = "v" if isinstance(model.denoiser.scaling, sgm.modules.diffusionmodules.denoiser_scaling.VScaling) else "eps"

    discretization = sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization()
    model.alphas_cumprod = torch.asarray(discretization.alphas_cumprod, device=devices.device, dtype=torch.float32)

    model.conditioner.wrapped = torch.nn.Module()

    encode_to_posterior_mode(model.first_stage_model)


class DiagonalGaussianMode(torch.nn.Module):
    """sgm DiagonalGaussianRegularizer(sample=False) (as in AutoencoderKLModeOnly), returning the mean in float32.

    float32 keeps the encode_first_stage result (scale_factor * z) as precise as the sampled path was, where the
    float32 noise promoted it; the unused KL term and std/var exponentials are not computed.
    """

    def forward(self, z):
        mean, _logvar = torch.chunk(z, 2, dim=1)
        return mean.float(), {}


def encode_to_posterior_mode(first_stage_model):
    """Make encode_first_stage return the posterior mean instead of a sample.

    AutoencoderKL samples mean + std * torch.randn(...) from the hidden global CPU generator, so img2img / hires /
    inpaint init latents depended on earlier requests and the img2img init cache froze whichever draw came first.
    For the SDXL VAE the scaled posterior std is ~3e-6 (max ~6e-5), far below the sampler noise added to these latents.
    """
    if isinstance(getattr(first_stage_model, "regularization", None), sgm.modules.autoencoding.regularizers.DiagonalGaussianRegularizer):
        first_stage_model.regularization = DiagonalGaussianMode()


sgm.modules.attention.print = shared.ldm_print
sgm.modules.diffusionmodules.model.print = shared.ldm_print
sgm.modules.diffusionmodules.openaimodel.print = shared.ldm_print
sgm.modules.encoders.modules.print = shared.ldm_print

# this gets the code to load the vanilla attention that we override
sgm.modules.attention.SDP_IS_AVAILABLE = True
sgm.modules.attention.XFORMERS_IS_AVAILABLE = False
