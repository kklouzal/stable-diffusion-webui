import torch
from einops import repeat
import math
import sgm.modules.diffusionmodules.model as _SGM_VAE
import sgm.modules.diffusionmodules.util as _sgm_util

from modules import devices, openclaw_fused_geglu, openclaw_layout_folds, shared, openclaw_nhwc_groupnorm as nhwc_group_norm, openclaw_gn_transpose as gn_transpose
from modules.sd_hijack_utils import CondFunc


class TorchHijackForUnet:
    """
    This is torch, but with cat that resizes tensors to appropriate dimensions if they do not match;
    this makes it possible to create pictures with dimensions that are multiples of 8 rather than 64
    """

    def __getattr__(self, item):
        if item == 'cat':
            return self.cat

        if hasattr(torch, item):
            return getattr(torch, item)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def cat(self, tensors, *args, **kwargs):
        if len(tensors) == 2:
            a, b = tensors
            if a.shape[-2:] != b.shape[-2:]:
                a = torch.nn.functional.interpolate(a, b.shape[-2:], mode="nearest")

            tensors = (a, b)

        return torch.cat(tensors, *args, **kwargs)


th = TorchHijackForUnet()


# Below are monkey patches to enable upcasting a float16 UNet for float32 sampling
def apply_model(orig_func, self, x_noisy, t, cond, **kwargs):
    """Always make sure inputs to unet are in correct dtype."""
    if isinstance(cond, dict):
        for y in cond.keys():
            if isinstance(cond[y], list):
                cond[y] = [x.to(devices.dtype_unet) if isinstance(x, torch.Tensor) else x for x in cond[y]]
            else:
                cond[y] = cond[y].to(devices.dtype_unet) if isinstance(cond[y], torch.Tensor) else cond[y]

    # Timesteps stay float32: the UNet embeds them with float32 sinusoids, and casting them to a half-precision
    # UNet dtype first quantizes them (bfloat16 steps by 2 from t=256 and by 4 from t=512).
    with devices.autocast():
        result = orig_func(self, x_noisy.to(devices.dtype_unet), t.to(torch.float32), cond, **kwargs)
        if devices.unet_needs_upcast:
            return result.float()
        else:
            return result


# Monkey patch to create timestep embed tensor on device, avoiding a block.
def timestep_embedding(_, timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


# Monkey patch to SpatialTransformer removing unnecessary contiguous calls.
# Prevents a lot of unnecessary aten::copy_ calls
def spatial_transformer_forward(_, self, x: torch.Tensor, context=None):
    # note: if no context is given, cross-attention defaults to self-attention
    if not isinstance(context, list):
        context = [context]
    b, c, h, w = x.shape
    x_in = x
    x = self.norm(x)
    if not self.use_linear:
        x = self.proj_in(x)
    x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
    if self.use_linear:
        x = self.proj_in(x)
    for i, block in enumerate(self.transformer_blocks):
        x = block(x, context=context[i])
    if self.use_linear:
        x = self.proj_out(x)
    x = x.view(b, h, w, c).permute(0, 3, 1, 2)
    if not self.use_linear:
        x = self.proj_out(x)
    return x + x_in


# sgm (SDXL) SpatialTransformer.forward without its output .contiguous() copy, keeping sgm's single-context rule (the
# ldm hijack above indexes context per block). With use_linear, proj_out's (b, h*w, c) result is added to x_in as a
# permuted view straight into an NCHW output: the same values and the same result layout as the original
# `.contiguous() + x_in`, minus writing and re-reading the transposed copy. The result layout is kept on purpose: an
# NHWC result would move the copy into the next GroupNorm and change the reduction order of extension code that
# reduces block outputs (ControlNet reference/adain var_mean). The input-side copy stays: the GroupNorm output is NCHW,
# so the permuted view is not contiguous, and Linear fuses its bias into the GEMM only for a contiguous input. The conv
# (not use_linear) variant and autograd (out= does not differentiate) keep the original code.
# With the NHWC GroupNorm switch (modules/openclaw_nhwc_groupnorm.py, unet scope) a channels_last input stays
# channels_last end to end: the norm writes NHWC, so the token view needs no copy, and the residual sum is written
# channels_last (same operand pairs, same values) for the next NHWC GroupNorm/conv; that layout is what moves the
# reduction order of the extension code above, inside the switch's numerically-equivalent class.
def sgm_spatial_transformer_forward(_, self, x, context=None):
    # note: if no context is given, cross-attention defaults to self-attention
    if not isinstance(context, list):
        context = [context]
    b, c, h, w = x.shape
    x_in = x
    x = self.norm(x)
    if not self.use_linear:
        x = self.proj_in(x)
    x = x.permute(0, 2, 3, 1).reshape(b, h * w, -1).contiguous()
    if self.use_linear:
        x = self.proj_in(x)
    for i, block in enumerate(self.transformer_blocks):
        if i > 0 and len(context) == 1:
            i = 0  # use same context for each block
        x = block(x, context=context[i])
    if self.use_linear:
        x = self.proj_out(x)
    x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2)
    if self.use_linear and not torch.is_grad_enabled():
        if nhwc_group_norm.UNET in nhwc_group_norm.scopes() and nhwc_group_norm.is_nhwc(x_in):
            out = torch.empty(x.shape, dtype=torch.result_type(x, x_in), device=x.device, memory_format=torch.channels_last)
            return torch.add(x, x_in, out=out)
        return torch.add(x, x_in, out=torch.empty(x.shape, dtype=torch.result_type(x, x_in), device=x.device))
    x = x.contiguous()
    if not self.use_linear:
        x = self.proj_out(x)
    return x + x_in


# bf16-native UNet norms. Under --precision autocast, CUDA autocast runs group_norm and layer_norm in fp32: every UNet
# GroupNorm/LayerNorm reads an fp32 copy of its bf16 input and writes fp32, which the next Linear/conv casts back to
# bf16 (GroupNorm32 adds its own x.float()/.type(x.dtype) round trip). ATen's bf16 CUDA norm kernels compute in fp32 from
# the exactly widened bf16 values and round once on store, so with bf16 input and bf16 weights the UNet norms run with
# autocast off for the call and skip the cast kernels and the fp32 traffic. Verified on GB10 (torch 2.14; CUDA tests in
# test/test_sd_hijack_unet.py, timing/error in tools/benchmark_unet_norms.py at SDXL batch-2 shapes):
# - LayerNorm (BasicTransformerBlock norm1/2/3): bitwise identical to the autocast path. eps, mean and rstd stay fp32,
#   and the vectorized kernel uses one vec_size for every dtype, so the reduction order matches as long as both paths
#   vectorize; UnetLayerNorm keeps the bf16 operands aligned like the fp32 copies the autocast path makes. 6.7-12.7x
#   faster including the consumers' casts (autocast's fp32 output is cast once per consuming Linear).
# - GroupNorm (GroupNorm32, SpatialTransformer.norm): the CUDA GroupNorm kernel takes eps in the input dtype, so eps becomes
#   bf16(eps) (1e-5 -> 1.0014e-5, 1e-6 -> 9.984e-7): results equal an fp32 run with that eps, rounded once. Not bitwise
#   versus autocast, but the error against a float64 reference equals the bf16 rounding floor, as on the autocast path
#   (max/RMS within one rounding, mean/RMS +0.1% at most even with group variances far below eps). 1.8-4.8x faster.
# CLIP/open_clip and VAE norms are not touched. Set to False to restore the autocast fp32 norms everywhere.
UNET_BF16_NATIVE_NORMS = True


def bf16_native_norm_eligible(module, x):
    """Whether a UNet norm can run natively in bf16 with the results described above, instead of autocast's fp32."""
    # Training (grad enabled) keeps the fp32 norm backward.
    if not UNET_BF16_NATIVE_NORMS or not x.is_cuda or x.dtype != torch.bfloat16 or torch.is_grad_enabled():
        return False
    if not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != torch.bfloat16:
        return False
    # Upcast sampling and quantized weight storage keep their own precision handling. Functional LoRA (lora_functional)
    # adds each network's norm output to this norm's output, and that sum must keep happening in the autocast fp32 dtype.
    if devices.unet_needs_upcast or devices.fp8 or devices.mxfp8 or devices.nvfp4 or getattr(shared.opts, "lora_functional", False):
        return False
    weight, bias = module.weight, module.bias
    return (weight is None or weight.dtype == torch.bfloat16) and (bias is None or bias.dtype == torch.bfloat16)


def unet_nhwc_group_norm(module, x, act=None):
    """NHWC GroupNorm (+SiLU) kernel for a bf16-native-eligible UNet norm when the unet scope is on, else None.

    Same precision class as the bf16-native path it replaces (bf16 in, fp32 statistics, bf16 out); see
    modules/openclaw_nhwc_groupnorm.py for scopes and numerics. NCHW input always returns None (no copies are added)."""
    if nhwc_group_norm.UNET not in nhwc_group_norm.scopes():
        return None
    return nhwc_group_norm.group_norm(module, x, act)


def group_norm32_bf16_forward(orig_func, self, x):
    # Class-level GroupNorm.forward is looked up per call so the Lora extension's patch stays in the chain.
    y = unet_nhwc_group_norm(self, x)
    if y is not None:
        return y
    # ATen's CUDA group_norm copies a channels_last input to NCHW; gn_transpose makes that same copy faster (switch).
    with torch.autocast("cuda", enabled=False):
        return torch.nn.GroupNorm.forward(self, gn_transpose.for_group_norm(x))


class UnetGroupNorm(torch.nn.GroupNorm):
    """SpatialTransformer.norm (a plain GroupNorm): takes the bf16-native path when eligible.

    The class is swapped onto the instances at construction, so CLIP and VAE norms never see it; state_dict keys and
    isinstance checks are unchanged, and the class-level GroupNorm.forward (with the Lora patch) is what runs."""

    def forward(self, x):
        if bf16_native_norm_eligible(self, x):
            return group_norm32_bf16_forward(None, self, x)
        return torch.nn.GroupNorm.forward(self, x)


class UnetLayerNorm(torch.nn.LayerNorm):
    """BasicTransformerBlock norm1/2/3: take the bf16-native path when eligible; see UnetGroupNorm."""

    def forward(self, x):
        # A loaded hypernetwork adds its output to attn1's context (norm1's output) in that tensor's dtype.
        # ATen vectorizes LayerNorm only on aligned operands; the autocast path's fp32 copies always are, and a
        # non-contiguous input is copied (aligned) before the kernel.
        if (
            bf16_native_norm_eligible(self, x)
            and not shared.loaded_hypernetworks
            and all(t.data_ptr() % 16 == 0 for t in (x, self.weight, self.bias) if t is not None and t.is_contiguous())
        ):
            with torch.autocast("cuda", enabled=False):
                return torch.nn.LayerNorm.forward(self, x)
        return torch.nn.LayerNorm.forward(self, x)


_TORCH_SILU_FORWARD = torch.nn.SiLU.forward


def _fusable_norm_silu(norm, silu, norm_forward):
    """norm runs norm_forward (no subclass or later patch) and silu is a plain torch SiLU."""
    return type(norm).forward is norm_forward and type(silu) is torch.nn.SiLU and torch.nn.SiLU.forward is _TORCH_SILU_FORWARD


def sgm_resblock_forward(orig_func, self, x, emb):
    """sgm ResBlock._forward with each GroupNorm32+SiLU pair on the fused NHWC kernel (unet + silu scopes).

    Op for op the upstream forward otherwise, for the SDXL configuration only (no up/down, no scale-shift norm,
    timestep embedding added). The fused pairs skip just the GroupNorm32 and SiLU module calls, so those modules and
    their Sequentials must carry no hooks or forward overrides and GroupNorm32.forward must be the hijack installed
    below; conv/dropout/emb/skip modules are still called as modules. Anything else, or NCHW input, runs the upstream
    forward (which still dispatches each GroupNorm32 itself)."""
    in_layers, out_layers = self.in_layers, self.out_layers
    if (
        self.updown or self.use_scale_shift_norm or getattr(self, "skip_t_emb", False) or getattr(self, "exchange_temb_dims", False)
        or len(in_layers) != 3 or len(out_layers) != 4
        or not _fusable_norm_silu(in_layers[0], in_layers[1], _SGM_GROUPNORM32_FORWARD)
        or not _fusable_norm_silu(out_layers[0], out_layers[1], _SGM_GROUPNORM32_FORWARD)
        or nhwc_group_norm.has_forward_hooks(in_layers, out_layers, in_layers[0], in_layers[1], out_layers[0], out_layers[1])
        or not bf16_native_norm_eligible(in_layers[0], x)
    ):
        return orig_func(self, x, emb)
    h = nhwc_group_norm.group_norm(in_layers[0], x, act="silu")
    if h is None:
        return orig_func(self, x, emb)
    h = in_layers[2](h)
    emb_out = self.emb_layers(emb).type(h.dtype)
    while len(emb_out.shape) < len(h.shape):
        emb_out = emb_out[..., None]
    h = h + emb_out
    fused = nhwc_group_norm.group_norm(out_layers[0], h, act="silu") if bf16_native_norm_eligible(out_layers[0], h) else None
    h = out_layers(h) if fused is None else out_layers[3](out_layers[2](fused))
    return self.skip_connection(x) + h


class VaeGroupNorm(torch.nn.GroupNorm):
    """sgm VAE GroupNorm (Normalize): the NHWC kernel for eligible channels_last input while the vae scope is on.

    Swapped onto the instances at construction (state_dict keys and isinstance unchanged). Off, or for any other input,
    it runs the class-level GroupNorm.forward exactly as a plain GroupNorm instance does, on the NCHW copy of
    modules/openclaw_gn_transpose.py when that switch is on (the same tensor ATen's CUDA group_norm copies to itself)."""

    def forward(self, x):
        if nhwc_group_norm.VAE in nhwc_group_norm.scopes() and nhwc_group_norm.vae_norm_eligible(self, x):
            y = nhwc_group_norm.group_norm(self, x)
            if y is not None:
                return y
        return torch.nn.GroupNorm.forward(self, gn_transpose.for_group_norm(x))


def vae_normalize(orig_func, *args, **kwargs):
    norm = orig_func(*args, **kwargs)
    if type(norm) is torch.nn.GroupNorm:
        norm.__class__ = VaeGroupNorm
    return norm


def sgm_vae_resnet_block_forward(orig_func, self, x, temb, **kwargs):
    """sgm VAE ResnetBlock.forward with GroupNorm+swish on the fused NHWC kernel (vae + silu scopes).

    Op for op the upstream forward otherwise. sgm's nonlinearity is swish, x * sigmoid(x): the upstream function, or
    torch's silu once sd_hijack.apply_optimizations has swapped it in (the production state); the kernel applies it to
    the fp32 normalized value before its single rounding. Any other nonlinearity, norms that are not plain VaeGroupNorm
    instances without hooks or overrides, or NCHW input run the upstream forward."""
    norm1, norm2 = self.norm1, self.norm2
    nonlinearity = _SGM_VAE.nonlinearity  # read per call: sd_hijack.apply_optimizations swaps it
    if (
        kwargs or type(norm1) is not VaeGroupNorm or type(norm2) is not VaeGroupNorm
        or nonlinearity not in _VAE_SWISH_FUNCTIONS
        or nhwc_group_norm.has_forward_hooks(norm1, norm2)
        or not nhwc_group_norm.vae_norm_eligible(norm1, x)
    ):
        return orig_func(self, x, temb, **kwargs)
    h = nhwc_group_norm.group_norm(norm1, x, act="silu")
    if h is None:
        return orig_func(self, x, temb, **kwargs)
    h = self.conv1(h)
    if temb is not None:
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
    fused = nhwc_group_norm.group_norm(norm2, h, act="silu") if nhwc_group_norm.vae_norm_eligible(norm2, h) else None
    h = nonlinearity(norm2(h)) if fused is None else fused
    h = self.dropout(h)
    h = self.conv2(h)
    if self.in_channels != self.out_channels:
        x = self.conv_shortcut(x) if self.use_conv_shortcut else self.nin_shortcut(x)
    return x + h


def transformer_block_init(orig_func, self, *args, **kwargs):
    orig_func(self, *args, **kwargs)
    for name in ("norm1", "norm2", "norm3"):
        norm = getattr(self, name, None)
        if type(norm) is torch.nn.LayerNorm:
            norm.__class__ = UnetLayerNorm


def spatial_transformer_init(orig_func, self, *args, **kwargs):
    orig_func(self, *args, **kwargs)
    if type(self.norm) is torch.nn.GroupNorm:
        self.norm.__class__ = UnetGroupNorm


class GELUHijack(torch.nn.GELU, torch.nn.Module):
    """OpenCLIP's MLP activation (text encoders only), upcast under --upcast-sampling; returns the input's dtype,
    which is float32 for a float32 text encoder (sd_models.float32_text_encoder_names)."""
    def __init__(self, *args, **kwargs):
        torch.nn.GELU.__init__(self, *args, **kwargs)
    def forward(self, x):
        if devices.unet_needs_upcast:
            return torch.nn.GELU.forward(self.float(), x.float()).to(x.dtype)
        else:
            return torch.nn.GELU.forward(self, x)


ddpm_edit_hijack = None
def hijack_ddpm_edit():
    global ddpm_edit_hijack
    if not ddpm_edit_hijack:
        CondFunc('modules.models.diffusion.ddpm_edit.LatentDiffusion.decode_first_stage', first_stage_sub, first_stage_cond)
        CondFunc('modules.models.diffusion.ddpm_edit.LatentDiffusion.encode_first_stage', first_stage_sub, first_stage_cond)
        ddpm_edit_hijack = CondFunc('modules.models.diffusion.ddpm_edit.LatentDiffusion.apply_model', apply_model)


unet_needs_upcast = lambda *args, **kwargs: devices.unet_needs_upcast
CondFunc('ldm.modules.diffusionmodules.openaimodel.timestep_embedding', timestep_embedding)
CondFunc('ldm.modules.attention.SpatialTransformer.forward', spatial_transformer_forward)
CondFunc('sgm.modules.attention.SpatialTransformer.forward', sgm_spatial_transformer_forward)

if torch.cuda.is_available():
    CondFunc('ldm.modules.diffusionmodules.util.GroupNorm32.forward', lambda orig_func, self, *args, **kwargs: orig_func(self.float(), *args, **kwargs), unet_needs_upcast)
    CondFunc('ldm.modules.attention.GEGLU.forward', lambda orig_func, self, x: orig_func(self.float(), x.float()).to(devices.dtype_unet), unet_needs_upcast)
    CondFunc('open_clip.transformer.ResidualAttentionBlock.__init__', lambda orig_func, *args, **kwargs: kwargs.update({'act_layer': GELUHijack}) and False or orig_func(*args, **kwargs), lambda _, *args, **kwargs: kwargs.get('act_layer') is None or kwargs['act_layer'] == torch.nn.GELU)

bf16_native_norm_cond = lambda orig_func, self, x: bf16_native_norm_eligible(self, x)
CondFunc('ldm.modules.diffusionmodules.util.GroupNorm32.forward', group_norm32_bf16_forward, bf16_native_norm_cond)
CondFunc('sgm.modules.diffusionmodules.util.GroupNorm32.forward', group_norm32_bf16_forward, bf16_native_norm_cond)
CondFunc('ldm.modules.attention.BasicTransformerBlock.__init__', transformer_block_init)
CondFunc('sgm.modules.attention.BasicTransformerBlock.__init__', transformer_block_init)
CondFunc('ldm.modules.attention.SpatialTransformer.__init__', spatial_transformer_init)
CondFunc('sgm.modules.attention.SpatialTransformer.__init__', spatial_transformer_init)

# NHWC GroupNorm switch (modules/openclaw_nhwc_groupnorm.py). The fused forwards check the switch first: off, the
# upstream forwards run unchanged. VAE norms are always VaeGroupNorm, which is the plain GroupNorm path while off.
# sgm is what SDXL (and ControlNet, whose cldm builds from sgm when it imports) runs; the ldm GroupNorm32 and
# SpatialTransformer.norm share the unet-scope GroupNorm dispatch above.
# The GroupNorm32.forward hijack installed above (read after the CondFunc so _fusable_norm_silu compares against it).
_SGM_GROUPNORM32_FORWARD = _sgm_util.GroupNorm32.forward
# Swish as sgm defines it, and torch's silu that sd_hijack.apply_optimizations installs in its place.
_VAE_SWISH_FUNCTIONS = (_SGM_VAE.nonlinearity, torch.nn.functional.silu)
_UNET_SILU_SCOPES = frozenset((nhwc_group_norm.UNET, nhwc_group_norm.SILU))
_VAE_SILU_SCOPES = frozenset((nhwc_group_norm.VAE, nhwc_group_norm.SILU))
CondFunc('sgm.modules.diffusionmodules.openaimodel.ResBlock._forward', sgm_resblock_forward, lambda *args, **kwargs: _UNET_SILU_SCOPES <= nhwc_group_norm.scopes())
CondFunc('sgm.modules.diffusionmodules.model.Normalize', vae_normalize)
CondFunc('sgm.modules.diffusionmodules.model.ResnetBlock.forward', sgm_vae_resnet_block_forward, lambda *args, **kwargs: _VAE_SILU_SCOPES <= nhwc_group_norm.scopes())
# Layout folds (modules/openclaw_layout_folds.py, switch): installed after the NHWC GroupNorm forwards, so they run
# first and defer to them while that switch's unet/vae scope is on.
openclaw_layout_folds.install()

first_stage_cond = lambda _, self, *args, **kwargs: devices.unet_needs_upcast and self.model.diffusion_model.dtype in (torch.float16, torch.bfloat16)
first_stage_sub = lambda orig_func, self, x, **kwargs: orig_func(self, x.to(devices.dtype_vae), **kwargs)
CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.decode_first_stage', first_stage_sub, first_stage_cond)
CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.encode_first_stage', first_stage_sub, first_stage_cond)
CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.get_first_stage_encoding', lambda orig_func, *args, **kwargs: orig_func(*args, **kwargs).float(), first_stage_cond)

CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.apply_model', apply_model)
CondFunc('sgm.modules.diffusionmodules.wrappers.OpenAIWrapper.forward', apply_model)

# After the upcast GEGLU CondFunc above, so the fused kernel's CondFunc wraps it and defers to it under upcast sampling.
openclaw_fused_geglu.install()


def timestep_embedding_cast_result(orig_func, timesteps, *args, **kwargs):
    if devices.unet_needs_upcast and timesteps.dtype == torch.int64:
        dtype = torch.float32
    else:
        dtype = devices.dtype_unet
    return orig_func(timesteps, *args, **kwargs).to(dtype=dtype)


CondFunc('ldm.modules.diffusionmodules.openaimodel.timestep_embedding', timestep_embedding_cast_result)
CondFunc('sgm.modules.diffusionmodules.openaimodel.timestep_embedding', timestep_embedding_cast_result)
