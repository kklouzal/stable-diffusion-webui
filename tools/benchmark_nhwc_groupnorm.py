#!/usr/bin/env python3
"""NHWC GroupNorm(+SiLU) kernels (modules/openclaw_nhwc_groupnorm.py) vs torch's GroupNorm on CUDA, SDXL shapes.

Variants per shape (bf16 activations and weights, 32 groups):
  torch_nchw       F.group_norm on an NCHW tensor: torch's floor, no layout copies
  torch_nhwc       F.group_norm on a channels_last tensor + .contiguous(channels_last) of the result: what a channels_last
                   UNet pays today (GroupNorm copies its input to NCHW, the next conv copies the output back)
  kernel_nhwc      the NHWC kernels, channels_last in and out
  *_silu           the same followed by SiLU (ResBlock in/out layers, VAE ResnetBlock); the kernel fuses it
Timing: CUDA events around every iteration, enqueued back to back after a warm-up, median over --iters.
Error: max and mean |output - float64 reference| / rms(reference); the reference's own bf16 rounding is the floor.
Usage (GPU host, inside the image): python tools/benchmark_nhwc_groupnorm.py [--iters 100] [--json out.json]
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

from modules import openclaw_nhwc_groupnorm as nhwc  # noqa: E402

# (batch, channels, size, eps): UNet/ControlNet GroupNorm32 and SpatialTransformer.norm at 1024-1536 px, hires 1280 px
# with CFG batch 3, and the VAE decoder at 1024 px.
SHAPES = [
    (2, 320, 128, 1e-5), (2, 640, 64, 1e-5), (2, 960, 64, 1e-5), (2, 1280, 32, 1e-5), (2, 1920, 64, 1e-5),
    (2, 2560, 32, 1e-5), (2, 640, 96, 1e-6), (2, 1280, 48, 1e-6), (3, 320, 160, 1e-5), (3, 640, 80, 1e-5),
    (1, 512, 128, 1e-6), (1, 512, 256, 1e-6), (1, 256, 512, 1e-6), (1, 128, 1024, 1e-6),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def median_ms(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for start, end in events:
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(start.elapsed_time(end) for start, end in events)


def error_over_rms(output, reference):
    diff = (output.double() - reference).abs()
    rms = reference.pow(2).mean().sqrt()
    return round((diff.max() / rms).item(), 6), round((diff.mean() / rms).item(), 8)


def variants(kernels, x, weight, bias, eps):
    """{name: (fn, silu)} over one input; x is NCHW, the NHWC variants read its channels_last copy."""
    x_cl = x.contiguous(memory_format=torch.channels_last)
    cl = torch.channels_last
    return {
        "torch_nchw": (lambda: F.group_norm(x, 32, weight, bias, eps), False),
        "torch_nhwc": (lambda: F.group_norm(x_cl, 32, weight, bias, eps).contiguous(memory_format=cl), False),
        "kernel_nhwc": (lambda: nhwc._launch(kernels, x_cl, 32, weight, bias, eps, False), False),
        "torch_nchw_silu": (lambda: F.silu(F.group_norm(x, 32, weight, bias, eps)), True),
        "torch_nhwc_silu": (lambda: F.silu(F.group_norm(x_cl, 32, weight, bias, eps)).contiguous(memory_format=cl), True),
        "kernel_nhwc_silu": (lambda: nhwc._launch(kernels, x_cl, 32, weight, bias, eps, True), True),
    }


def measure(kernels, batch, channels, size, eps, iters, warmup):
    generator = torch.Generator(device="cuda").manual_seed(channels * size)
    scale = torch.logspace(-3, 1, 32, device="cuda").repeat_interleave(channels // 32).view(1, channels, 1, 1)
    x = (torch.randn((batch, channels, size, size), device="cuda", generator=generator) * scale + scale * 3).to(torch.bfloat16)
    weight = (torch.randn(channels, device="cuda", generator=generator) * 0.2 + 1).to(torch.bfloat16)
    bias = (torch.randn(channels, device="cuda", generator=generator) * 0.1).to(torch.bfloat16)
    reference = F.group_norm(x.double(), 32, weight.double(), bias.double(), eps)
    references = {False: reference, True: F.silu(reference)}
    row = {"shape": [batch, channels, size, size], "eps": eps, "config": nhwc.launch_config(batch, channels, size * size)}
    with torch.inference_mode():
        for name, (fn, silu) in variants(kernels, x, weight, bias, eps).items():
            row[name] = {"ms": round(median_ms(fn, iters, warmup), 4), "error_max_mean": error_over_rms(fn(), references[silu])}
            if name.startswith("kernel"):
                row[name]["repeat_bitwise"] = bool(torch.equal(fn(), fn()))
    row["floor_max_mean"] = error_over_rms(reference.to(torch.bfloat16), reference)
    return row


def main():
    args = parse_args()
    nhwc.set_scopes("all")
    rows = []
    for batch, channels, size, eps in SHAPES:
        row = measure(nhwc._KERNELS, batch, channels, size, eps, args.iters, args.warmup)
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
