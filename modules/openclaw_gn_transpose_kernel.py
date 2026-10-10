"""Triton kernel behind modules/openclaw_gn_transpose.py (imported at startup only while the switch is on, so CPU runs
and switch-off starts never import Triton)."""

import triton
import triton.language as tl


@triton.jit
def nhwc_to_nchw_kernel(x_ptr, y_ptr, C, HW, BLOCK_C: tl.constexpr, BLOCK_HW: tl.constexpr):
    # One program per (pixel tile, channel tile, sample): x is a dense [N, HW, C] array (channels_last), y a dense
    # [N, C, HW] one (NCHW). The tile is loaded along C and stored along HW; Triton moves it between the two register
    # layouts through shared memory, so both global accesses are coalesced. Values are copied, never converted.
    hw = tl.program_id(0).to(tl.int64) * BLOCK_HW + tl.arange(0, BLOCK_HW)
    c = tl.program_id(1).to(tl.int64) * BLOCK_C + tl.arange(0, BLOCK_C)
    base = tl.program_id(2).to(tl.int64) * C * HW
    mask = (hw[:, None] < HW) & (c[None, :] < C)
    tile = tl.load(x_ptr + base + hw[:, None] * C + c[None, :], mask=mask)
    tl.store(y_ptr + base + c[None, :] * HW + hw[:, None], tile, mask=mask)
