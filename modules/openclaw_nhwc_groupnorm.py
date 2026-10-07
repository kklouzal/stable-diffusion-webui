"""NHWC-native GroupNorm (optionally fused with SiLU) for channels_last SDXL UNet, ControlNet and VAE activations.

Why: with --opt-channelslast the convolutions produce NHWC activations, but ATen's CUDA group_norm computes NCHW only, so
every GroupNorm copies its input to NCHW and the next convolution copies the result back (GB10 profile of the img2img
1280² full-stack request: direct_copy 1.68 s = 12% of GPU time, 0.95 s of it in the VAE decode; torch GroupNorm on a
channels_last tensor incl. those copies costs 3.4-4x its NCHW time, perf2 audit runs/gn-bench.json). The kernels in
modules/openclaw_nhwc_groupnorm_kernel.py read and write channels_last directly and can fuse the following SiLU, so a
GroupNorm moves the bytes of torch's NCHW kernel (x read twice, y written once) and no layout copy.

Why not apex.contrib.group_norm (in the NGC image): its two-pass forward, which every SDXL GroupNorm above 32x32 pixels
and the whole VAE decode would take, adds the per-CTA group sums with float atomicAdd, so outputs change from run to
run (this deployment keeps same-request outputs bit-identical); and it takes var = E[x^2] - mean^2 in float32 and
normalizes by 1 when that is <= 0, which loses the variance of groups with a large mean.

Switch (default off): OPENCLAW_NHWC_GROUPNORM at import (an invalid value, or no Triton, fails the import, i.e.
startup), or set_scopes() at runtime (POST /sdapi/v1/openclaw/nhwc-groupnorm). The value is "all", or a comma list of:
  unet        UNet GroupNorm32 / SpatialTransformer.norm (ldm and sgm) on channels_last input take the kernel, and the
              sgm SpatialTransformer keeps a channels_last input channels_last end to end. Only where the bf16-native
              norm path applies (sd_hijack_unet.bf16_native_norm_eligible). ControlNet builds from the same sgm classes.
  silu        additionally fuse GroupNorm+SiLU in sgm ResBlock in/out layers (with unet) and GroupNorm+swish in the sgm
              VAE ResnetBlock (with vae).
  vae         the sgm VAE's GroupNorms (Normalize) on bf16 channels_last input with autocast off (how the decode runs).
  controlnet  ControlNet models are kept channels_last on the device while the SD model is (--opt-channelslast), so their
              activations are NHWC too (cldm.PlugableControlModel.fullvram).
Off, every call takes the pre-existing code path and outputs are bitwise unchanged. Graph caches key on state_key().

Numerical contract when on (not bitwise against the torch paths; the numerically-equivalent class):
- statistics in float32 from the exactly widened inputs: per (tile, channel) mean and centered M2, merged in a fixed
  order (Chan et al.) into the group mean and M2 -- Welford-class accuracy at any mean/std ratio, like torch's kernel;
- eps added in float32 (the torch bf16-native path rounds eps to bf16 first; the autocast path keeps it float32);
- y = (x - mean) * (rstd * w) + b and the SiLU y / (1 + exp(-y)) in float32, rounded once to the input dtype (torch
  rounds the GroupNorm output before SiLU);
- deterministic: no atomics, fixed reduction order and tiling for a given shape, so a request reproduces bit for bit.
NHWC activations and channels_last ControlNet weights also change the reduction order of code that reduces them
(cuDNN conv algorithms, ControlNet reference_adain var_mean, TeaCache's relative-L1 gate, whose thresholded skip
decisions can then differ in borderline steps), within the same class.

Thread-safety: the scope set is an immutable frozenset swapped atomically; set_scopes callers serialize against
generations (the API takes queue_lock). The dispatch counters are diagnostics: eager calls only (a CUDA graph replay
skips Python), approximate under concurrency.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any

import torch

UNET = "unet"
SILU = "silu"
VAE = "vae"
CONTROLNET = "controlnet"
ALL_SCOPES = frozenset((UNET, SILU, VAE, CONTROLNET))
ENV_NAME = "OPENCLAW_NHWC_GROUPNORM"

_DTYPES = (torch.bfloat16, torch.float16)
_MAX_BLOCK_C = 128  # channels per program: one 256-byte bf16 run per pixel
_MIN_BLOCK_C = 16  # C must be a multiple of 16 (every SDXL/VAE width is a multiple of 64)
_STATS_TILE_ELEMENTS = 2048  # BLOCK_HW x BLOCK_C loaded per statistics step
_APPLY_TILE_ELEMENTS = 4096  # BLOCK_HW x BLOCK_C per apply program
_MIN_STATS_PROGRAMS = 256  # tile the pixels finer until the statistics grid has this many programs (GB10: 48 SMs)
_MAX_TILE_HW = 2048
_COMBINE_BLOCK = 256

_LOCK = threading.Lock()
_SCOPES: frozenset[str] = frozenset()
_STATE_KEY: tuple[str, ...] = ()
_KERNELS: Any = None
_ERROR: str | None = None
_COUNTS = {"group_norm": 0, "group_norm_silu": 0}
_FALLBACKS: dict[str, int] = {}

# torch's own GroupNorm.forward, recognized by identity of definition: the Lora extension replaces the class attribute.
_TORCH_GROUP_NORM_FORWARD_ID = ("torch.nn.modules.normalization", "GroupNorm.forward")


def parse_scopes(value: Any) -> frozenset[str]:
    """Scope set from an env/API value: None/''/0/false/off/none -> off; 1/true/on/all -> every scope; else a list."""
    if value is None:
        return frozenset()
    if isinstance(value, (list, tuple, set, frozenset)):
        tokens = [str(item).strip().lower() for item in value]
    else:
        tokens = [item.strip().lower() for item in str(value).split(",")]
    tokens = [token for token in tokens if token]
    if not tokens or tokens in (["0"], ["false"], ["off"], ["none"], ["no"]):
        return frozenset()
    if tokens in (["1"], ["true"], ["on"], ["all"], ["yes"]):
        return ALL_SCOPES
    unknown = sorted(set(tokens) - ALL_SCOPES)
    if unknown:
        raise ValueError(f"unknown {ENV_NAME} scope(s) {unknown}; expected 'all' or a comma list of {sorted(ALL_SCOPES)}")
    return frozenset(tokens)


def _load_kernels() -> Any:
    from modules import openclaw_nhwc_groupnorm_kernel

    return openclaw_nhwc_groupnorm_kernel


def set_scopes(value: Any) -> dict[str, Any]:
    """Enable the given scopes (see parse_scopes); raises ValueError/RuntimeError and keeps the old state on failure."""
    global _SCOPES, _STATE_KEY, _KERNELS, _ERROR
    scopes = parse_scopes(value)
    with _LOCK:
        if scopes and _KERNELS is None:
            try:
                _KERNELS = _load_kernels()
            except Exception as exc:
                _ERROR = f"NHWC GroupNorm kernels unavailable: {type(exc).__name__}: {exc}"[:500]
                raise RuntimeError(_ERROR) from exc
        _ERROR = None
        _SCOPES = scopes
        _STATE_KEY = tuple(sorted(scopes))
    return status()


def status() -> dict[str, Any]:
    return {
        "scopes": list(_STATE_KEY),
        "kernels_loaded": _KERNELS is not None,
        "error": _ERROR,
        "calls": dict(_COUNTS),
        "fallbacks": dict(_FALLBACKS),
    }


def reset_counters() -> None:
    _COUNTS.update(dict.fromkeys(_COUNTS, 0))
    _FALLBACKS.clear()


def scopes() -> frozenset[str]:
    return _SCOPES


def state_key() -> tuple[str, ...]:
    """Hashable switch state for graph-cache keys: a captured graph freezes which GroupNorm path ran."""
    return _STATE_KEY


def is_nhwc(x: torch.Tensor) -> bool:
    """4-D and laid out channels_last, and not also NCHW-contiguous (C == 1 or H*W == 1 is both; torch never copies it)."""
    return x.dim() == 4 and x.is_contiguous(memory_format=torch.channels_last) and not x.is_contiguous()


def _on_cuda(x: torch.Tensor) -> bool:
    """The kernels' device requirement (one seam, so CPU tests can drive the dispatch through Triton's interpreter)."""
    return x.is_cuda


def _fallback(reason: str) -> None:
    _FALLBACKS[reason] = _FALLBACKS.get(reason, 0) + 1


def _is_torch_group_norm_forward(function: Any) -> bool:
    return (getattr(function, "__module__", None), getattr(function, "__qualname__", None)) == _TORCH_GROUP_NORM_FORWARD_ID


def class_chain_ready(module: torch.nn.Module) -> bool:
    """Run what the class-level GroupNorm.forward chain does before its kernel; False when the chain is not known.

    The GroupNorm subclasses reach the kernel through torch.nn.GroupNorm.forward, which the Lora extension patches: in
    merged mode it first applies (or restores) the loaded networks' weights on the module, then calls torch's forward.
    That step must keep happening when the NHWC kernel replaces torch's. Functional LoRA adds per-network terms to the
    output (never fusable here) and any other patch is unknown: both fall back to the class chain.
    """
    forward = torch.nn.GroupNorm.forward
    if _is_torch_group_norm_forward(forward):
        return True
    networks = sys.modules.get("networks")
    prepare = getattr(networks, "network_GroupNorm_prepare", None)
    if (
        prepare is not None
        and forward is getattr(networks, "network_GroupNorm_forward", None)
        and _is_torch_group_norm_forward(getattr(getattr(networks, "originals", None), "GroupNorm_forward", None))
    ):
        return bool(prepare(module))
    return False


def _pow2_at_most(value: int) -> int:
    return 1 << (max(1, value).bit_length() - 1)


def launch_config(n: int, c: int, hw: int) -> dict[str, int]:
    """Tiling for an [n, c, h, w] input with hw = h * w: channel chunk, statistics tile and sub-tile, apply tile.

    A pure function of the shape, so a shape always runs the same reduction order (determinism across runs)."""
    block_c = min(c & -c, _MAX_BLOCK_C)  # largest power of two dividing c
    stats_block_hw = max(1, _STATS_TILE_ELEMENTS // block_c)
    units = n * (c // block_c)
    target_tiles = -(-_MIN_STATS_PROGRAMS // units)
    tile_hw = min(_MAX_TILE_HW, max(stats_block_hw, _pow2_at_most(-(-hw // target_tiles))))
    return {
        "block_c": block_c,
        "stats_block_hw": stats_block_hw,
        "tile_hw": tile_hw,
        "tiles": -(-hw // tile_hw),
        "apply_block_hw": _pow2_at_most(_APPLY_TILE_ELEMENTS // block_c),
    }


def _launch(kernels: Any, x: torch.Tensor, groups: int, weight: torch.Tensor, bias: torch.Tensor, eps: float, silu: bool) -> torch.Tensor:
    n, c, h, w = x.shape
    hw = h * w
    config = launch_config(n, c, hw)
    block_c, tiles = config["block_c"], config["tiles"]
    partial = torch.empty((2, n, tiles, c), device=x.device, dtype=torch.float32)
    stats = torch.empty((n, groups, 2), device=x.device, dtype=torch.float32)
    y = torch.empty_like(x)  # channels_last, like x
    chunks = n * (c // block_c)
    kernels.channel_stats_kernel[(tiles, chunks)](
        x, partial[0], partial[1], hw, c, tiles,
        TILE_HW=config["tile_hw"], BLOCK_HW=config["stats_block_hw"], BLOCK_C=block_c, num_warps=4,
    )
    kernels.group_stats_kernel[(n, groups)](
        partial[0], partial[1], stats, hw, c, tiles, c // groups, eps,
        TILE_HW=config["tile_hw"], BLOCK_K=_COMBINE_BLOCK, num_warps=4,
    )
    apply_block_hw = config["apply_block_hw"]
    kernels.apply_kernel[(-(-hw // apply_block_hw), chunks)](
        x, y, weight, bias, stats, hw, c, c // groups, groups,
        BLOCK_HW=apply_block_hw, BLOCK_C=block_c, SILU=silu, num_warps=8,
    )
    return y


def group_norm(module: torch.nn.GroupNorm, x: torch.Tensor, act: str | None = None) -> torch.Tensor | None:
    """GroupNorm `module` (then SiLU when act == "silu") of channels_last `x` on the NHWC kernels, or None.

    The caller owns the precision decision (the torch path it replaces must compute in x's dtype from x's dtype weights)
    and the scope check. None means: take the torch path, nothing was computed. Output is channels_last, x's dtype.
    """
    kernels = _KERNELS
    if kernels is None:
        _fallback("kernels_not_loaded")
        return None
    weight, bias = module.weight, module.bias
    if not _on_cuda(x) or torch.is_grad_enabled() or weight is None or bias is None or x.numel() == 0:
        _fallback("unsupported_call")
        return None
    if not is_nhwc(x):
        _fallback("not_nhwc")
        return None
    channels = x.shape[1]
    if x.dtype not in _DTYPES or weight.dtype != x.dtype or bias.dtype != x.dtype or channels % _MIN_BLOCK_C:
        _fallback("unsupported_shape_or_dtype")
        return None
    if not class_chain_ready(module):
        _fallback("groupnorm_forward_chain")
        return None
    # weight/bias are read after class_chain_ready: the Lora merge may have replaced them (same dtype and shape).
    y = _launch(kernels, x, module.num_groups, module.weight, module.bias, float(module.eps), act == "silu")
    _COUNTS["group_norm_silu" if act else "group_norm"] += 1
    return y


def vae_norm_eligible(module: torch.nn.GroupNorm, x: torch.Tensor) -> bool:
    """Whether the torch path this replaces computes a bf16 GroupNorm of bf16 input with bf16 weights.

    The VAE decode/encode runs with autocast off (sd_samplers_common/processing), so torch's kernel then reads bf16 and
    writes bf16. Under CUDA autocast torch would compute in fp32 instead: keep that path.
    """
    weight, bias = module.weight, module.bias
    return (
        x.dtype == torch.bfloat16
        and _on_cuda(x)
        and not torch.is_autocast_enabled("cuda")
        and weight is not None and weight.dtype == torch.bfloat16
        and bias is not None and bias.dtype == torch.bfloat16
    )


def has_forward_hooks(*modules: torch.nn.Module) -> bool:
    """Any module or global forward hook, or an instance-level forward override, that a fused call would skip."""
    from torch.nn.modules import module as torch_module

    if torch_module._global_forward_hooks or torch_module._global_forward_pre_hooks:
        return True
    return any(m._forward_hooks or m._forward_pre_hooks or "forward" in m.__dict__ for m in modules)


def apply_controlnet_layout(model: torch.nn.Module, channels_last: bool) -> None:
    """Keep a ControlNet model's weights channels_last while the controlnet scope is on and the caller asks for it, and
    restore the checkpoint layout otherwise.

    channels_last: the caller's placement allows it (cldm.PlugableControlModel.fullvram: full VRAM, not Control-LoRA,
    and the SD model is channels_last; aggressive_lowvram: never). Converting is lossless, so restoring
    contiguous_format gives the original weights bit for bit; a model this never converted is left untouched.
    """
    wanted = channels_last and CONTROLNET in _SCOPES
    converted = bool(getattr(model, "_openclaw_channels_last", False))
    if wanted and not converted:
        model.to(memory_format=torch.channels_last)
        model._openclaw_channels_last = True
    elif converted and not wanted:
        model.to(memory_format=torch.contiguous_format)
        model._openclaw_channels_last = False


def _configure_from_env() -> None:
    """Startup configuration: a configured switch that cannot be honoured stops the import (and the server)."""
    value = os.environ.get(ENV_NAME)
    if not value:
        return
    try:
        set_scopes(value)
    except (ValueError, RuntimeError) as exc:
        raise RuntimeError(f"{ENV_NAME}={value!r}: {exc}") from exc


_configure_from_env()
