"""Fused GEGLU for the sgm/ldm transformer feed-forward on CUDA, bitwise identical to the eager path.

GEGLU.forward is `x, gate = self.proj(x).chunk(2, dim=-1); return x * F.gelu(gate)`. On CUDA that runs two elementwise
kernels over the strided halves of proj's output: gelu writes a temporary that mul reads back together with the other
half, both on PyTorch's non-vectorized strided path. SDXL runs it 70 times per UNet call, plus the PAG hidden pass and the
ControlNet. The fused kernel reads each half once and writes the product once; proj (autocast, LoRA hooks) is unchanged.

Bitwise contract for bf16/fp16 proj outputs, from ATen's CUDA kernels:
- GeluCUDAKernelImpl (approximate='none') computes in float `x * 0.5f * (1.0f + erff(x * (float)M_SQRT1_2))` and rounds
  to the tensor dtype. nvcc lowers erff to libdevice __nv_erff and builds without FTZ.
- mul_kernel_cuda multiplies the two values as floats and rounds the product once.
The kernel (modules/openclaw_fused_geglu_kernel.py) evaluates the same expression in the same order with libdevice's
__nv_erff and libdevice FTZ off. It rounds the gelu to the tensor dtype, multiplies in float32 and rounds once. Triton's
down-conversions round to nearest even, like c10's float to bf16/fp16 conversions. test/test_openclaw_fused_geglu.py
pins the generated PTX on CPU and, on the GPU, compares every bf16 and fp16 gate value against the eager path.

Scope: CUDA bf16/fp16 tensors that do not need autograd (inference). Everything else, including the upcast-sampling
path and CPU, runs the original forward. OPENCLAW_FUSED_GEGLU=0 (read once at import) restores the original forward
everywhere.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from modules import devices, openclaw_env
from modules.sd_hijack_utils import CondFunc

ENABLED = openclaw_env.env_bool("OPENCLAW_FUSED_GEGLU", True)

_FUSED_DTYPES = (torch.bfloat16, torch.float16)


def _kernel():
    from modules.openclaw_fused_geglu_kernel import geglu_kernel
    return geglu_kernel


def _block_size(inner: int) -> int:
    for block in (1024, 512, 256):
        if inner % block == 0:
            return block
    return min(1024, 1 << max(0, (inner - 1).bit_length()))


def fusable(h: torch.Tensor) -> bool:
    """Whether fused_geglu reproduces `a * F.gelu(gate)` for proj output `h` bit for bit (and needs no autograd)."""
    return (
        h.is_cuda
        and h.dtype in _FUSED_DTYPES
        and h.dim() >= 1
        and h.shape[-1] % 2 == 0
        and h.is_contiguous()
        and not (h.requires_grad and torch.is_grad_enabled())
    )


def fused_geglu(h: torch.Tensor) -> torch.Tensor:
    """`a, gate = h.chunk(2, dim=-1); return a * F.gelu(gate)` in one kernel; `h` must satisfy fusable()."""
    inner = h.shape[-1] // 2
    out = torch.empty(h.shape[:-1] + (inner,), dtype=h.dtype, device=h.device)
    rows = h.numel() // h.shape[-1] if h.shape[-1] else 0
    if rows and inner:
        block = _block_size(inner)
        # BLOCK // 256 warps: 8 elements per thread, one 16-byte vector per half for bf16/fp16.
        _kernel()[(rows, -(-inner // block))](h, out, inner, BLOCK=block, num_warps=max(1, block // 256), enable_reflect_ftz=False)
    return out


def geglu_forward(orig_func, self, x):
    h = self.proj(x)
    if fusable(h):
        return fused_geglu(h)
    # The original GEGLU.forward (sgm and ldm are identical), for proj outputs the kernel does not cover.
    x, gate = h.chunk(2, dim=-1)
    return x * F.gelu(gate)


def geglu_cond(orig_func, self, x):
    # Upcast sampling runs GEGLU in float32 through its own CondFunc (sd_hijack_unet); leave that path untouched.
    return x.is_cuda and not devices.unet_needs_upcast


def install():
    """Wrap sgm/ldm GEGLU.forward (called once from sd_hijack_unet, after its upcast CondFunc)."""
    if not ENABLED:
        return
    CondFunc('sgm.modules.attention.GEGLU.forward', geglu_forward, geglu_cond)
    CondFunc('ldm.modules.attention.GEGLU.forward', geglu_forward, geglu_cond)
