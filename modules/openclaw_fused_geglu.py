"""Fused GEGLU for the sgm/ldm transformer feed-forward on CUDA, bitwise identical to the eager path.

GEGLU.forward is `x, gate = self.proj(x).chunk(2, dim=-1); return x * F.gelu(gate)`. On CUDA that runs two elementwise
kernels over the strided halves of proj's output: gelu writes a temporary that mul reads back together with the other
half, both on PyTorch's non-vectorized strided path. SDXL runs it 70 times per UNet call, plus the PAG hidden pass and the
ControlNet. The fused kernel reads each half once and writes the product once; proj (autocast, LoRA hooks) is unchanged.

Bitwise contract for bf16/fp16 proj outputs, from ATen's CUDA kernels:
- GeluCUDAKernelImpl (approximate='none') computes in float `x * 0.5f * (1.0f + erff(x * (float)M_SQRT1_2))` and rounds
  to the tensor dtype. nvcc builds it without FTZ and links erff as __nv_erff from the CUDA toolkit's libdevice
  (nvvm/libdevice/libdevice.10.bc of the toolkit torch was built with).
- mul_kernel_cuda multiplies the two values as floats and rounds the product once.
The kernel (modules/openclaw_fused_geglu_kernel.py) evaluates the same expression in the same order with libdevice FTZ
off. It rounds the gelu to the tensor dtype, multiplies in float32 and rounds once. Triton's down-conversions round to
nearest even, like c10's float to bf16/fp16 conversions.
erff must come from that same libdevice. By default Triton links its own bundled libdevice, whose __nv_erff is a
different implementation (exp branch from |x| >= 1.00296 instead of 1.00376, other coefficients). On GB10 that moved
the gelu of 4 of the 65536 bf16 and 98 of the 65536 fp16 gate values by one ULP (small negative outputs, where 1 + erf
cancels). So every launch links the toolkit's libdevice, resolved once at import: CUDA_HOME (else CUDA_PATH, else
/usr/local/cuda) must hold include/cuda.h with torch.version.cuda's CUDA_VERSION and nvvm/libdevice/libdevice.10.bc,
or the import (startup) fails while the fused path is enabled.
test/test_openclaw_fused_geglu.py pins the generated PTX on CPU, checks that the kernel's erff constants are those of
ATen's own GELU kernels in libtorch_cuda (the sm_120 SASS that GB10 runs) and, on the GPU, compares every bf16 and fp16
gate and multiplicand value against the eager path.

Scope: CUDA bf16/fp16 tensors that do not need autograd (inference). Everything else, including the upcast-sampling
path and CPU, runs the original forward. OPENCLAW_FUSED_GEGLU=0 (read once at import) restores the original forward
everywhere.
"""

from __future__ import annotations

import os
import re

import torch
import torch.nn.functional as F

from modules import devices, openclaw_env
from modules.sd_hijack_utils import CondFunc

ENABLED = openclaw_env.env_bool("OPENCLAW_FUSED_GEGLU", True)


def _toolkit_libdevice() -> str:
    """libdevice.10.bc of the CUDA toolkit torch was built with: the library nvcc linked ATen's erff from."""
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    major, minor = (int(part) for part in torch.version.cuda.split(".")[:2])
    expected = major * 1000 + minor * 10
    header = os.path.join(home, "include", "cuda.h")
    path = os.path.join(home, "nvvm", "libdevice", "libdevice.10.bc")
    found = None
    try:
        with open(header, encoding="utf-8") as f:
            for line in f:
                if (match := re.match(r"#define CUDA_VERSION (\d+)", line)) is not None:
                    found = int(match.group(1))
                    break
    except OSError:
        pass  # reported below as CUDA_VERSION None
    if found != expected or not os.path.isfile(path):
        raise RuntimeError(
            f"fused GEGLU: erff must come from the libdevice of CUDA {torch.version.cuda} (torch's build toolkit) to "
            f"match ATen bit for bit, but {header} has CUDA_VERSION {found} (expected {expected}) and {path} "
            f"{'exists' if os.path.isfile(path) else 'is missing'}. Point CUDA_HOME at that toolkit, or set "
            "OPENCLAW_FUSED_GEGLU=0."
        )
    return path


# Resolved once at import (startup); None when the fused path is off or torch has no CUDA.
_EXTERN_LIBS = (("libdevice", _toolkit_libdevice()),) if ENABLED and torch.version.cuda else None

_FUSED_DTYPES = (torch.bfloat16, torch.float16)


def _kernel():
    from modules.openclaw_fused_geglu_kernel import geglu_kernel
    return geglu_kernel


def _block_size(inner: int) -> int:
    for block in (1024, 512, 256):
        if inner % block == 0:
            return block
    return min(1024, 1 << max(0, (inner - 1).bit_length()))


def _launch_options(block: int) -> dict:
    """Triton options of every fused launch (the PTX/SASS tests compile with these too)."""
    if _EXTERN_LIBS is None:
        raise RuntimeError("fused GEGLU is off (OPENCLAW_FUSED_GEGLU=0) or torch has no CUDA")
    # BLOCK // 256 warps: 8 elements per thread, one 16-byte vector per half for bf16/fp16.
    return {"num_warps": max(1, block // 256), "enable_reflect_ftz": False, "extern_libs": _EXTERN_LIBS}


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
        _kernel()[(rows, -(-inner // block))](h, out, inner, BLOCK=block, **_launch_options(block))
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
