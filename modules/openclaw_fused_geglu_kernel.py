"""Triton kernel behind modules/openclaw_fused_geglu.py (imported on the first fused CUDA call, so startup and CPU runs
never import Triton). The bitwise contract is documented there; it holds only with the launch options that module
passes, which link the CUDA toolkit's libdevice (ATen's erff) instead of Triton's bundled one."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def geglu_kernel(h_ptr, out_ptr, inner, BLOCK: tl.constexpr):
    # One program per (row, BLOCK columns): an h row is [a | gate], each `inner` wide; the out row is a * gelu(gate).
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < inner
    src = h_ptr + row * (2 * inner) + cols
    a = tl.load(src, mask=mask).to(tl.float32)
    g = tl.load(src + inner, mask=mask).to(tl.float32)
    # ATen GeluCUDAKernelImpl: x * 0.5f * (1.0f + erff(x * kAlpha)), kAlpha = (float)M_SQRT1_2 written out exactly.
    # The *_rn calls pin every float op as its own IEEE round-to-nearest instruction: plain operators let LLVM contract
    # into FMAs or turn the final float multiply of two widened bf16 values into a bf16 multiply (one rounding instead
    # of ATen's two: float product, then bf16; they differ for subnormal products).
    erf = libdevice.erf(libdevice.mul_rn(g, 0.707106769084930419921875))
    gelu = libdevice.mul_rn(libdevice.mul_rn(g, 0.5), libdevice.add_rn(1.0, erf))
    gelu = gelu.to(h_ptr.dtype.element_ty).to(tl.float32)
    tl.store(out_ptr + row * inner + cols, libdevice.mul_rn(a, gelu).to(out_ptr.dtype.element_ty), mask=mask)
