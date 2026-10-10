#!/usr/bin/env python3
"""NHWC -> NCHW transpose kernel (modules/openclaw_gn_transpose.py) vs ATen's .contiguous() on CUDA, production shapes.

Per shape and dtype:
  contiguous      x.contiguous(): the copy ATen's group_norm makes of a channels_last input today
  kernel          the Triton transpose at the module's tile (BLOCK_C x BLOCK_HW, NUM_WARPS)
  kernel <tile>   with --sweep, every tile in TILES as well (the fastest is printed per shape)
  gn_today        F.group_norm(x_cl): ATen's copy + the NCHW GroupNorm kernel
  gn_switch       F.group_norm(transpose(x_cl)): what the switch runs
GB/s counts the bytes moved (read + write of the tensor). Every variant is checked bit for bit against .contiguous()
(and the GroupNorm outputs against each other) before it is timed.
Timing: CUDA events around every iteration, enqueued back to back after a warm-up, median and p10-p90 over --iters.
Usage (GPU host, inside the image): python tools/benchmark_gn_transpose.py [--iters 50] [--sweep] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules import openclaw_gn_transpose as gn_transpose  # noqa: E402

# n-w1 (img2img 1280²): the VAE encode/decode at full, 1/2 and 1/4 resolution, the SDXL UNet/ControlNet at 160²
# latents with CFG batch 2; plain (txt2img 1024²): the VAE decode at 1024².
SHAPES = [
    (1, 128, 1280, 1280), (1, 256, 1280, 1280), (1, 256, 640, 640), (1, 512, 640, 640), (1, 512, 320, 320),
    (1, 512, 160, 160), (1, 128, 1024, 1024), (1, 256, 512, 512),
    (2, 320, 160, 160), (2, 640, 80, 80), (2, 960, 80, 80), (2, 1280, 40, 40), (2, 1920, 40, 40), (2, 2560, 40, 40),
]
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
TILES = [(32, 32, 4), (64, 64, 4), (64, 64, 8), (64, 128, 4), (64, 128, 8), (128, 64, 8), (128, 128, 8), (32, 128, 4)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--sweep", action="store_true", help="also time every tile in TILES")
    parser.add_argument("--dtypes", default="bf16,fp32")
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def timings_ms(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for start, end in events:
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    values = sorted(start.elapsed_time(end) for start, end in events)
    return {
        "median_ms": round(statistics.median(values), 4),
        "p10_ms": round(values[len(values) // 10], 4),
        "p90_ms": round(values[(len(values) * 9) // 10], 4),
    }


def measure(shape, dtype, args):
    n, c, h, w = shape
    generator = torch.Generator(device="cuda").manual_seed(c * h)
    x = (torch.randn(shape, generator=generator, device="cuda") * 2 + 0.5).to(dtype).contiguous(memory_format=torch.channels_last)
    weight = torch.randn(c, device="cuda", dtype=dtype)
    bias = torch.randn(c, device="cuda", dtype=dtype)
    bits = torch.int16 if dtype.itemsize == 2 else torch.int32
    reference = x.contiguous()
    gigabytes = 2 * x.numel() * dtype.itemsize / 1e9
    tiles = TILES if args.sweep else [(gn_transpose.BLOCK_C, gn_transpose.BLOCK_HW, gn_transpose.NUM_WARPS)]
    row = {"shape": list(shape), "dtype": str(dtype)}
    row["contiguous"] = timings_ms(lambda: x.contiguous(), args.iters, args.warmup)
    for block_c, block_hw, warps in tiles:
        name = f"kernel_{block_c}x{block_hw}w{warps}"
        y = gn_transpose.transpose(x, block_c, block_hw, warps)
        assert torch.equal(y.view(bits), reference.view(bits)), name
        row[name] = timings_ms(lambda bc=block_c, bh=block_hw, nw=warps: gn_transpose.transpose(x, bc, bh, nw), args.iters, args.warmup)
    for value in row.values():
        if isinstance(value, dict):
            value["GB_per_s"] = round(gigabytes / (value["median_ms"] / 1e3), 1)
    if c % 32 == 0:
        today = F.group_norm(x, 32, weight, bias, 1e-6)
        switch = F.group_norm(gn_transpose.transpose(x), 32, weight, bias, 1e-6)
        assert today.stride() == switch.stride() and torch.equal(today, switch), "GroupNorm outputs differ"
        row["gn_today"] = timings_ms(lambda: F.group_norm(x, 32, weight, bias, 1e-6), args.iters, args.warmup)
        row["gn_switch"] = timings_ms(lambda: F.group_norm(gn_transpose.transpose(x), 32, weight, bias, 1e-6), args.iters, args.warmup)
    return row


def main():
    args = parse_args()
    if gn_transpose._KERNEL is None:
        gn_transpose._KERNEL = gn_transpose._load_kernel()
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name()}, tile {gn_transpose.BLOCK_C}x{gn_transpose.BLOCK_HW} "
          f"w{gn_transpose.NUM_WARPS}, iters {args.iters}")
    rows = []
    with torch.inference_mode():
        for dtype_name in args.dtypes.split(","):
            for shape in SHAPES:
                row = measure(shape, DTYPES[dtype_name], args)
                rows.append(row)
                kernels = {k: v for k, v in row.items() if k.startswith("kernel_")}
                best = min(kernels, key=lambda k: kernels[k]["median_ms"])
                line = (f"{dtype_name} {str(tuple(shape)):22s} contiguous {row['contiguous']['median_ms']:8.3f} ms "
                        f"{row['contiguous']['GB_per_s']:7.1f} GB/s | {best} {kernels[best]['median_ms']:8.3f} ms "
                        f"{kernels[best]['GB_per_s']:7.1f} GB/s")
                if "gn_today" in row:
                    line += f" | GN today {row['gn_today']['median_ms']:8.3f} ms, switch {row['gn_switch']['median_ms']:8.3f} ms"
                print(line, flush=True)
    if args.json:
        args.json.write_text(json.dumps({"torch": torch.__version__, "device": torch.cuda.get_device_name(), "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
