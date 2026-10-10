"""Fast NHWC -> NCHW copy of a GroupNorm input on CUDA, bitwise identical to what ATen's group_norm does today.

Why: with --opt-channelslast the convolutions write channels_last activations, but ATen's CUDA group_norm computes NCHW
only: it calls input.contiguous() and normalizes that copy (its output is NCHW). That copy runs on ATen's generic strided
copy kernel, which reaches about 10-20 GB/s at the VAE's full-resolution sizes (n-w1 nsys profile: 1.74 s of
direct_copy per request). The kernel in modules/openclaw_gn_transpose_kernel.py writes the same NCHW tensor with a tiled
transpose (coalesced loads along C, coalesced stores along H*W), and the unchanged ATen group_norm then reads it; its
own contiguous() is a no-op on an NCHW tensor.

Contract: for_group_norm(x) returns a tensor equal to x.contiguous() element for element and bit for bit (a copy, no
arithmetic: NaN payloads and signed zeros are kept), NCHW-contiguous, same dtype and device. GroupNorm outputs are
therefore unchanged in values and layout. The layout policy (--opt-channelslast, the parked NHWC GroupNorm kernels) is
unchanged too.

Scope: CUDA tensors that are channels_last and not also NCHW-contiguous (nhwc_group_norm.is_nhwc), bf16/fp16/fp32, no
autograd. Anything else is returned unchanged and takes ATen's own copy. On CPU the switch does nothing: ATen's CPU
group_norm keeps a channels_last input channels_last (its output too), so an NCHW copy would change its output layout
and reduction order.

Callers: sd_hijack_unet.group_norm32_bf16_forward (UNet/ControlNet GroupNorm32 and SpatialTransformer.norm on the
bf16-native path) and VaeGroupNorm.forward (every sgm VAE GroupNorm), right before the class-level GroupNorm.forward
(the Lora extension's patch stays in the chain and receives the NCHW tensor; its merged mode only touches weights).

Switch (default off): OPENCLAW_GN_FAST_TRANSPOSE, read once at import (startup) with the openclaw_env grammar; an
invalid value, or a missing Triton while it is on, fails the import. The state is fixed for the process, so CUDA graph
keys need not include it.
"""

from __future__ import annotations

import torch

from modules import openclaw_env
from modules.openclaw_nhwc_groupnorm import is_nhwc

ENV_NAME = "OPENCLAW_GN_FAST_TRANSPOSE"
ENABLED = openclaw_env.env_bool(ENV_NAME, False)

_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
# Tile: 64 channels x 64 pixels (8 KiB bf16, 16 KiB fp32 through shared memory), 4 warps. tools/benchmark_gn_transpose.py
# sweeps alternatives on the GPU.
BLOCK_C = 64
BLOCK_HW = 64
NUM_WARPS = 4
_MAX_GRID_YZ = 65535  # CUDA grid y/z limit: channel tiles and samples


def _load_kernel():
    from modules.openclaw_gn_transpose_kernel import nhwc_to_nchw_kernel

    return nhwc_to_nchw_kernel


_KERNEL = _load_kernel() if ENABLED else None


def _on_cuda(x: torch.Tensor) -> bool:
    """The kernel's device requirement (one seam, so CPU tests can drive the wrapper through Triton's interpreter)."""
    return x.is_cuda


def transpose(x: torch.Tensor, block_c: int | None = None, block_hw: int | None = None, num_warps: int | None = None) -> torch.Tensor:
    """x.contiguous() of a channels_last [N, C, H, W] tensor (is_nhwc(x)) on the kernel; the tile defaults to the module
    constants and is a parameter for the benchmark's sweep. Requires the kernel to be loaded (the switch on)."""
    block_c, block_hw, num_warps = block_c or BLOCK_C, block_hw or BLOCK_HW, num_warps or NUM_WARPS
    n, c, h, w = x.shape
    hw = h * w
    y = torch.empty((n, c, h, w), dtype=x.dtype, device=x.device)
    if y.numel():
        # is_nhwc: every dimension of size > 1 has its dense channels_last stride and the others are only ever indexed
        # at 0, so the kernel can address x as a dense [N, HW, C] array from its data pointer.
        _KERNEL[(-(-hw // block_hw), -(-c // block_c), n)](x, y, c, hw, BLOCK_C=block_c, BLOCK_HW=block_hw, num_warps=num_warps)
    return y


def for_group_norm(x: torch.Tensor) -> torch.Tensor:
    """The tensor to hand ATen's group_norm instead of x: x.contiguous() on the kernel when the switch covers x, else x."""
    if (
        not ENABLED
        or not _on_cuda(x)
        or not is_nhwc(x)
        or x.dtype not in _DTYPES
        or (x.requires_grad and torch.is_grad_enabled())
        or x.shape[0] > _MAX_GRID_YZ
        or -(-x.shape[1] // BLOCK_C) > _MAX_GRID_YZ
    ):
        return x
    return transpose(x)
