#!/usr/bin/env python3
"""Layout folds (modules/openclaw_layout_folds.py) and the GroupNorm fast transpose (modules/openclaw_gn_transpose.py)
on whole sgm blocks at production sizes, channels_last bf16 on CUDA.

Blocks: SDXL UNet/ControlNet ResBlocks at n-w1 latent sizes (160² latents, CFG batch 2) under bf16 autocast with the
bf16-native norms, and sgm VAE ResnetBlocks at the 1280² decode sizes with autocast off. Variants per block:
  base        both switches off (what production runs)
  transpose   OPENCLAW_GN_FAST_TRANSPOSE behaviour only
  folds       OPENCLAW_LAYOUT_FOLDS behaviour only
  both        both
Every variant's output is checked bitwise against base before it is timed. Per variant: median and p10-p90 ms over
--iters (CUDA events, back to back after a warm-up), and from one profiled call the CUDA kernel time split into copy
kernels (name contains "copy") and the rest, and the copy kernel count.
The folds are installed into the sgm classes once (irreversibly for this process), so base and transpose run first.
Usage (GPU host, inside the image): python tools/benchmark_layout_folds.py [--iters 30] [--json out.json]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


ARGS = parse_args()
sys.argv = sys.argv[:1]  # modules.shared parses the command line on import
os.environ.setdefault("IGNORE_CMD_ARGS_ERRORS", "1")
sys.path.insert(0, str(ROOT))
for repo in ("generative-models", "stable-diffusion-stability-ai"):
    sys.path.insert(0, str(ROOT / "repositories" / repo))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from modules import devices, openclaw_gn_transpose as gn_transpose, openclaw_layout_folds as folds, sd_hijack_unet  # noqa: E402
import sgm.modules.diffusionmodules.model as sgm_vae  # noqa: E402
import sgm.modules.diffusionmodules.openaimodel as sgm_openaimodel  # noqa: E402

BF16 = torch.bfloat16
CL = torch.channels_last
UNET = [(320, 320, 160), (320, 640, 80), (640, 640, 80), (640, 1280, 40), (1280, 1280, 40), (2560, 1280, 40), (1920, 1280, 40)]
VAE = [(128, 128, 1280), (256, 128, 1280), (256, 256, 640), (512, 256, 640), (512, 512, 320), (512, 512, 160)]


def production_runtime():
    """The state sd_hijack_unet's bf16-native norm path expects (as test/conftest.py's default_runtime)."""
    sd_hijack_unet.UNET_BF16_NATIVE_NORMS = True
    sd_hijack_unet.shared = SimpleNamespace(opts=SimpleNamespace(lora_functional=False), loaded_hypernetworks=[])
    for flag in ("unet_needs_upcast", "fp8", "mxfp8", "nvfp4"):
        setattr(devices, flag, False)
    sgm_vae.nonlinearity = F.silu  # what sd_hijack.apply_optimizations installs


def randomized(module):
    generator = torch.Generator().manual_seed(4321)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05 + (1.0 if parameter.dim() == 1 else 0.0))
    return module.eval().to("cuda", BF16).to(memory_format=CL)


def cases():
    generator = torch.Generator(device="cuda").manual_seed(0)
    for channels, out_channels, size in UNET:
        block = randomized(sgm_openaimodel.ResBlock(channels, 1280, 0.0, out_channels=out_channels))
        x = torch.randn((2, channels, size, size), generator=generator, device="cuda").to(BF16).contiguous(memory_format=CL)
        emb = torch.randn((2, 1280), generator=generator, device="cuda").to(BF16)
        yield f"unet_resblock {channels}->{out_channels} 2x{size}²", block, (x, emb), True
    for channels, out_channels, size in VAE:
        block = randomized(sgm_vae.ResnetBlock(in_channels=channels, out_channels=out_channels, dropout=0.0, temb_channels=0))
        x = torch.randn((1, channels, size, size), generator=generator, device="cuda").to(BF16).contiguous(memory_format=CL)
        yield f"vae_resnetblock {channels}->{out_channels} 1x{size}²", block, (x, None), False


def run(block, args, autocast):
    context = torch.autocast("cuda", dtype=BF16) if autocast else contextlib.nullcontext()
    with context:
        return block(*args)


def timings(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for start, end in events:
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    values = sorted(start.elapsed_time(end) for start, end in events)
    return {"median_ms": round(statistics.median(values), 4), "p10_ms": round(values[len(values) // 10], 4), "p90_ms": round(values[(len(values) * 9) // 10], 4)}


def kernel_split(fn):
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    copy_us = other_us = 0.0
    copies = 0
    for event in prof.key_averages():
        total = getattr(event, "device_time_total", None) or getattr(event, "cuda_time_total", 0.0)
        if event.device_type != torch.autograd.DeviceType.CUDA or not total:
            continue
        if "copy" in event.key.lower():
            copy_us += total
            copies += event.count
        else:
            other_us += total
    return {"copy_kernels": copies, "copy_ms": round(copy_us / 1e3, 4), "other_kernels_ms": round(other_us / 1e3, 4)}


def measure(variant, rows, references, args):
    for name, block, inputs, autocast in cases():
        fn = lambda block=block, inputs=inputs, autocast=autocast: run(block, inputs, autocast)  # noqa: E731
        out = fn()
        if variant == "base":
            references[name] = out
        elif not (out.stride() == references[name].stride() and torch.equal(out, references[name])):
            raise SystemExit(f"{variant}: {name} differs from base")
        row = {"variant": variant, "block": name, **timings(fn, args.iters, args.warmup), **kernel_split(fn)}
        rows.append(row)
        print(f"{variant:9s} {name:36s} {row['median_ms']:8.3f} ms (p10 {row['p10_ms']:.3f} p90 {row['p90_ms']:.3f}) "
              f"copies {row['copy_kernels']:3d} = {row['copy_ms']:7.3f} ms, other {row['other_kernels_ms']:7.3f} ms", flush=True)
        del out


def main():
    args = ARGS
    production_runtime()
    if gn_transpose._KERNEL is None:
        gn_transpose._KERNEL = gn_transpose._load_kernel()
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name()}, iters {args.iters}")
    rows, references = [], {}
    with torch.inference_mode():
        for variant, transpose, fold in (("base", False, False), ("transpose", True, False), ("folds", False, True), ("both", True, True)):
            gn_transpose.ENABLED = transpose
            if fold and not folds.ENABLED:
                folds.ENABLED = True
                folds.install()
            measure(variant, rows, references, args)
    if args.json:
        args.json.write_text(json.dumps({"torch": torch.__version__, "device": torch.cuda.get_device_name(), "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
