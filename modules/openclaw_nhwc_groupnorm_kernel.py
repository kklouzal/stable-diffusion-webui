"""Triton kernels behind modules/openclaw_nhwc_groupnorm.py (imported when the switch is first turned on, so startup and
CPU runs never import Triton). The contract and the launch parameters are documented there.

x is a channels_last [N, C, H, W] tensor, i.e. a dense [N, HW, C] array; channels are split into chunks of BLOCK_C
(a power of two dividing C), so every load and store covers BLOCK_C contiguous elements per pixel. No kernel uses
atomics and every reduction runs in a fixed order: results are identical from run to run.
"""

import triton
import triton.language as tl


@triton.jit
def channel_stats_kernel(x_ptr, mean_ptr, m2_ptr, HW, C, T, TILE_HW: tl.constexpr, BLOCK_HW: tl.constexpr, BLOCK_C: tl.constexpr):
    # One program per (pixel tile t of TILE_HW pixels, sample n and channel chunk): each channel's mean and centered sum
    # of squares (M2) over the tile's pixels. Sub-tiles of BLOCK_HW pixels are centered on their own mean and merged in
    # order with Chan et al.'s update, so no sum of squares is ever taken around zero.
    t = tl.program_id(0)
    chunks = C // BLOCK_C
    n = (tl.program_id(1) // chunks).to(tl.int64)
    c = (tl.program_id(1) % chunks) * BLOCK_C + tl.arange(0, BLOCK_C)
    rows = tl.arange(0, BLOCK_HW)
    tile_start = t * TILE_HW
    tile_pixels = tl.minimum(HW - tile_start, TILE_HW)
    mean = tl.zeros([BLOCK_C], dtype=tl.float32)
    m2 = tl.zeros([BLOCK_C], dtype=tl.float32)
    for start in range(0, TILE_HW, BLOCK_HW):
        hw = tile_start + start + rows
        valid = (start + rows) < tile_pixels
        x = tl.load(x_ptr + (n * HW + hw[:, None]) * C + c[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        n_before = tl.minimum(tile_pixels, start).to(tl.float32)
        n_sub = tl.maximum(tl.minimum(tile_pixels - start, BLOCK_HW), 0).to(tl.float32)
        sub_mean = tl.sum(x, axis=0) / tl.maximum(n_sub, 1.0)
        d = tl.where(valid[:, None], x - sub_mean[None, :], 0.0)
        sub_m2 = tl.sum(d * d, axis=0)
        ratio = n_sub / tl.maximum(n_before + n_sub, 1.0)
        delta = sub_mean - mean
        mean = mean + delta * ratio
        m2 = m2 + sub_m2 + delta * delta * (n_before * ratio)
    out = (n * T + t) * C + c
    tl.store(mean_ptr + out, mean)
    tl.store(m2_ptr + out, m2)


@triton.jit
def group_stats_kernel(mean_ptr, m2_ptr, stats_ptr, HW, C, T, CPG, eps, TILE_HW: tl.constexpr, BLOCK_K: tl.constexpr):
    # One program per (sample n, group g): combine the group's T x CPG (tile, channel) statistics, each over n_t pixels,
    # into mean = sum(n_t * m) / count and M2 = sum(M2 + n_t * (m - mean)^2), in index order; write mean and
    # 1 / sqrt(M2 / count + eps).
    n = tl.program_id(0).to(tl.int64)
    g = tl.program_id(1)
    groups = tl.num_programs(1)
    entries = T * CPG
    count = HW.to(tl.float32) * CPG
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for start in range(0, entries, BLOCK_K):
        k = start + tl.arange(0, BLOCK_K)
        valid = k < entries
        t = k // CPG
        m = tl.load(mean_ptr + (n * T + t) * C + g * CPG + k % CPG, mask=valid, other=0.0)
        n_t = tl.where(valid, tl.minimum(HW - t * TILE_HW, TILE_HW).to(tl.float32), 0.0)
        acc += n_t * m
    mean = tl.sum(acc, axis=0) / count
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for start in range(0, entries, BLOCK_K):
        k = start + tl.arange(0, BLOCK_K)
        valid = k < entries
        t = k // CPG
        index = (n * T + t) * C + g * CPG + k % CPG
        m = tl.load(mean_ptr + index, mask=valid, other=0.0)
        q = tl.load(m2_ptr + index, mask=valid, other=0.0)
        n_t = tl.where(valid, tl.minimum(HW - t * TILE_HW, TILE_HW).to(tl.float32), 0.0)
        dm = m - mean
        acc += q + n_t * dm * dm
    rstd = 1.0 / tl.sqrt_rn(tl.sum(acc, axis=0) / count + eps)
    tl.store(stats_ptr + (n * groups + g) * 2, mean)
    tl.store(stats_ptr + (n * groups + g) * 2 + 1, rstd)


@triton.jit
def apply_kernel(x_ptr, y_ptr, w_ptr, b_ptr, stats_ptr, HW, C, CPG, G, BLOCK_HW: tl.constexpr, BLOCK_C: tl.constexpr, SILU: tl.constexpr):
    # One program per (BLOCK_HW pixels, sample n and channel chunk): y = (x - mean) * (rstd * w) + b in float32, then
    # SiLU y / (1 + exp(-y)) when SILU, rounded once to y's dtype (round to nearest even).
    chunks = C // BLOCK_C
    n = (tl.program_id(1) // chunks).to(tl.int64)
    c = (tl.program_id(1) % chunks) * BLOCK_C + tl.arange(0, BLOCK_C)
    hw = tl.program_id(0) * BLOCK_HW + tl.arange(0, BLOCK_HW)
    valid = hw < HW
    offsets = (n * HW + hw[:, None]) * C + c[None, :]
    x = tl.load(x_ptr + offsets, mask=valid[:, None], other=0.0).to(tl.float32)
    stats = stats_ptr + (n * G + c // CPG) * 2
    mean = tl.load(stats)
    scale = tl.load(stats + 1) * tl.load(w_ptr + c).to(tl.float32)
    y = (x - mean[None, :]) * scale[None, :] + tl.load(b_ptr + c).to(tl.float32)[None, :]
    if SILU:
        y = y / (1.0 + tl.exp(-y))
    tl.store(y_ptr + offsets, y.to(y_ptr.dtype.element_ty), mask=valid[:, None])
