"""Layout folds: elementwise ops write the memory format their consumer reads, so the consumer does not copy (H-2).

Why: under --opt-channelslast the convolutions read and write channels_last, while ATen's CUDA group_norm reads and
writes NCHW. Today each GroupNorm -> SiLU -> conv chain copies the SiLU's NCHW output to channels_last inside the conv,
and the ResBlock's `h + emb_out` (channels_last, like the conv output h) is copied to NCHW inside the next GroupNorm.
Both copies run on ATen's generic strided copy kernel. The folds compute the same elementwise op straight into a tensor
of the layout the consumer would have copied it to:
- silu_for_conv: SiLU written channels_last when the following conv reads channels_last. ATen's convolution (cuDNN's
  cudnn_conv_suggest_memory_format, oneDNN likewise) computes channels_last whenever its weight is channels_last and
  copies an NCHW input with input.contiguous(channels_last) first; on a channels_last input that is a no-op, so the conv
  runs the same call on the same values. Stands aside under functional LoRA (lora_functional; see silu_for_conv).
- add_for_group_norm: `h + other` written NCHW when a GroupNorm on CUDA reads it next (ATen's CUDA group_norm copies a
  channels_last input to NCHW; see modules/openclaw_gn_transpose.py). The GroupNorm then normalizes the same NCHW values.
Bitwise contract: the folded op is the same ATen elementwise kernel (silu, add) on the same operands, so every value is
identical; only the output strides differ, and the consumer would have produced exactly those strides itself. Block
outputs keep their layouts. The layout policy (--opt-channelslast, the parked NHWC GroupNorm kernels) is unchanged; the
folds stand aside wherever that NHWC GroupNorm switch's unet (ResBlock) or vae (ResnetBlock) scope is on.

Where: sgm ResBlock._forward (SDXL UNet and the ControlNets built from sgm) and the sgm VAE ResnetBlock.forward, as
CondFuncs installed by sd_hijack_unet after the NHWC GroupNorm ones (so these run first and defer to them). Only for the
configurations that run in production, op for op the upstream forward otherwise: no up/down sampling, no scale-shift
norm, timestep embedding added, plain Sequential/SiLU modules without hooks (their module calls are the ones skipped;
every GroupNorm, conv, dropout, embedding and skip module is still called as a module), torch's silu as the VAE
nonlinearity (what sd_hijack.apply_optimizations installs), no autograd, dropout inactive. Anything else runs the forward
installed below these.

Switch (default on since the 2026-10-10 GPU A/B, see openclaw_gn_transpose): OPENCLAW_LAYOUT_FOLDS, read once at import (startup) with the openclaw_env grammar; an invalid
value fails the import. Off, nothing is installed. The state is fixed for the process, so CUDA graph keys need not
include it.
"""

from __future__ import annotations

import sgm.modules.diffusionmodules.model as sgm_vae
import torch
import torch.nn.functional as F

from modules import openclaw_env, openclaw_nhwc_groupnorm as nhwc_group_norm, shared
from modules.sd_hijack_utils import CondFunc

ENV_NAME = "OPENCLAW_LAYOUT_FOLDS"
ENABLED = openclaw_env.env_bool(ENV_NAME, True)

_CL = torch.channels_last
_TORCH_SILU_FORWARD = torch.nn.SiLU.forward


def _gn_reads_nchw(x: torch.Tensor) -> bool:
    """Whether ATen's group_norm reads x as an NCHW copy: on CUDA (CPU keeps channels_last). A seam for CPU tests."""
    return x.is_cuda


def silu_for_conv(x: torch.Tensor, conv: torch.nn.Module) -> torch.Tensor:
    """F.silu(x), written channels_last when `conv` (called next on it) reads a channels_last input.

    Not under functional LoRA (lora_functional): the Lora extension's Conv2d forward then also runs each network's
    down/up convs on this input, and those keep NCHW weights, so a channels_last input would pick another conv kernel
    for them than the NCHW input they get upstream."""
    if (
        x.dim() == 4 and not x.is_contiguous(memory_format=_CL) and isinstance(conv, torch.nn.Conv2d)
        and nhwc_group_norm.is_nhwc(conv.weight) and not getattr(shared.opts, "lora_functional", False)
    ):
        return torch.ops.aten.silu.out(x, out=torch.empty_like(x, memory_format=_CL))
    return F.silu(x)


def add_for_group_norm(h: torch.Tensor, other: torch.Tensor, norm: torch.nn.Module) -> torch.Tensor:
    """h + other, written NCHW when GroupNorm `norm` (called next on it) reads an NCHW copy of a channels_last h."""
    if (
        isinstance(norm, torch.nn.GroupNorm)
        and nhwc_group_norm.is_nhwc(h)
        and _gn_reads_nchw(h)
        and torch.broadcast_shapes(h.shape, other.shape) == h.shape
    ):
        return torch.add(h, other, out=torch.empty(h.shape, dtype=torch.result_type(h, other), device=h.device))
    return h + other


def _plain_silu(module: torch.nn.Module) -> bool:
    return type(module) is torch.nn.SiLU and not module.inplace and torch.nn.SiLU.forward is _TORCH_SILU_FORWARD


def _inactive_dropout(module: torch.nn.Module) -> bool:
    return type(module) is torch.nn.Dropout and (not module.training or module.p == 0)


def sgm_resblock_forward(orig_func, self, x, emb):
    """sgm ResBlock._forward with both SiLUs written channels_last for their convs and `h + emb_out` written NCHW for
    the out GroupNorm. Skips only the in/out Sequential and SiLU module calls (checked for hooks and overrides)."""
    in_layers, out_layers = self.in_layers, self.out_layers
    if (
        self.updown or self.use_scale_shift_norm or getattr(self, "skip_t_emb", False) or getattr(self, "exchange_temb_dims", False)
        or torch.is_grad_enabled()
        or type(in_layers) is not torch.nn.Sequential or type(out_layers) is not torch.nn.Sequential
        or len(in_layers) != 3 or len(out_layers) != 4
        or not _plain_silu(in_layers[1]) or not _plain_silu(out_layers[1]) or not _inactive_dropout(out_layers[2])
        or nhwc_group_norm.has_forward_hooks(in_layers, out_layers, in_layers[1], out_layers[1])
    ):
        return orig_func(self, x, emb)
    h = in_layers[2](silu_for_conv(in_layers[0](x), in_layers[2]))
    emb_out = self.emb_layers(emb).type(h.dtype)
    while len(emb_out.shape) < len(h.shape):
        emb_out = emb_out[..., None]
    h = add_for_group_norm(h, emb_out, out_layers[0])
    h = out_layers[3](out_layers[2](silu_for_conv(out_layers[0](h), out_layers[3])))
    return self.skip_connection(x) + h


def sgm_vae_resnet_block_forward(orig_func, self, x, temb, **kwargs):
    """sgm VAE ResnetBlock.forward with each nonlinearity written channels_last for the conv after it. Needs torch's
    silu as the nonlinearity (production; sgm's own swish is a different op) and no timestep embedding (the VAE)."""
    # sgm_vae.nonlinearity is read per call: sd_hijack.apply_optimizations swaps it.
    if kwargs or temb is not None or sgm_vae.nonlinearity is not F.silu or torch.is_grad_enabled() or not _inactive_dropout(self.dropout):
        return orig_func(self, x, temb, **kwargs)
    h = self.conv1(silu_for_conv(self.norm1(x), self.conv1))
    h = self.conv2(self.dropout(silu_for_conv(self.norm2(h), self.conv2)))
    if self.in_channels != self.out_channels:
        x = self.conv_shortcut(x) if self.use_conv_shortcut else self.nin_shortcut(x)
    return x + h


def install():
    """Install the folds (called once from sd_hijack_unet, after the NHWC GroupNorm CondFuncs); nothing while off."""
    if not ENABLED:
        return
    CondFunc(
        'sgm.modules.diffusionmodules.openaimodel.ResBlock._forward', sgm_resblock_forward,
        lambda *args, **kwargs: nhwc_group_norm.UNET not in nhwc_group_norm.scopes(),
    )
    CondFunc(
        'sgm.modules.diffusionmodules.model.ResnetBlock.forward', sgm_vae_resnet_block_forward,
        lambda *args, **kwargs: nhwc_group_norm.VAE not in nhwc_group_norm.scopes(),
    )
