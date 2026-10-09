#!/usr/bin/env python3
"""N1 evidence: bf16-native UNet norms vs the autocast fp32 path on CUDA.

For each SDXL norm site, runs the production module (CondFunc/class-swap dispatch from modules/sd_hijack_unet.py) under
torch.autocast("cuda", bfloat16) + no_grad with UNET_BF16_NATIVE_NORMS off (autocast path) and on (native path), plus the
consumer-side work each path causes (autocast leaves fp32 outputs that every consuming Linear casts back to bf16):
- GroupNorm32 (ResBlock in/out_layers): output is bf16 on both paths;
- SpatialTransformer.norm (plain GroupNorm, eps 1e-6): output -> (b, h*w, c) contiguous copy -> proj_in's bf16 cast;
- BasicTransformerBlock LayerNorm: norm1 feeds to_q/to_k/to_v (3 casts), norm2/norm3 feed one Linear (1 cast).
Timing: CUDA events around every iteration, all enqueued back to back after a warm-up, median over --iters.
Error: max |consumed output - float64 reference| / rms(reference), with the float64 reference's own bf16 rounding error
as the floor. Usage (GPU host, inside the image): python tools/benchmark_unet_norms.py [--iters 100] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


ARGS = parse_args()
sys.argv = sys.argv[:1]  # modules.shared parses the command line on import
os.environ.setdefault("IGNORE_CMD_ARGS_ERRORS", "1")
sys.path.insert(0, str(ROOT))
for repo in ("generative-models", "stable-diffusion-stability-ai"):
    if (ROOT / "repositories" / repo).is_dir():
        sys.path.insert(0, str(ROOT / "repositories" / repo))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from modules import sd_hijack_unet  # noqa: E402  (installs the CondFuncs before the modules below are built)
import sgm.modules.attention as sgm_attention  # noqa: E402
import sgm.modules.diffusionmodules.util as sgm_util  # noqa: E402

BF16 = torch.bfloat16


def activations(shape, distribution, memory_format=torch.contiguous_format, seed=1234):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    channels = shape[1] if len(shape) == 4 else shape[-1]
    view = (1, channels, 1, 1) if len(shape) == 4 else (channels,)
    if distribution == "randn":
        x = torch.randn(shape, device="cuda", generator=generator)
    else:  # "wide": per-channel scales 1e-3..10 with offsets, including groups whose variance is far below eps
        scale = torch.logspace(-3, 1, channels, device="cuda")
        offset = torch.randn(channels, device="cuda", generator=generator) * scale * 3
        x = torch.randn(shape, device="cuda", generator=generator) * scale.view(view) + offset.view(view)
    return x.to(BF16).contiguous(memory_format=memory_format)


def randomize(module, seed=4321):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        module.weight.copy_(torch.randn(module.weight.shape, generator=generator) * 0.5 + 1.0)
        module.bias.copy_(torch.randn(module.bias.shape, generator=generator) * 0.5)
    return module.to("cuda", BF16)


def cuda_median_ms(fn):
    for _ in range(ARGS.warmup):
        fn()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(ARGS.iters)]
    wall = time.perf_counter()
    for start, end in events:
        start.record()
        fn()
        end.record()
    cpu_us = (time.perf_counter() - wall) / ARGS.iters * 1e6
    torch.cuda.synchronize()
    samples = sorted(start.elapsed_time(end) for start, end in events)
    return {"median_ms": statistics.median(samples), "p10_ms": samples[len(samples) // 10], "p90_ms": samples[len(samples) * 9 // 10], "cpu_us_per_call": cpu_us}


def error_vs(reference, output):
    rms = reference.pow(2).mean().sqrt()
    diff = (output.double() - reference).abs()
    return {"max_over_rms": (diff.max() / rms).item(), "mean_over_rms": (diff.mean() / rms).item()}


def run_case(name, module, x, reference, consume, consumed_reference=None):
    consumed_reference = reference if consumed_reference is None else consumed_reference
    result = {"case": name}
    for path, native in (("autocast", False), ("native", True)):
        sd_hijack_unet.UNET_BF16_NATIVE_NORMS = native

        def step():
            return consume(module(x))

        with torch.no_grad(), torch.autocast("cuda", dtype=BF16):
            raw = module(x)
            eligible = sd_hijack_unet.bf16_native_norm_eligible(module, x)
            consumed = step()
            result[path] = {
                "raw_dtype": str(raw.dtype).replace("torch.", ""),
                "eligible": eligible,
                **cuda_median_ms(step),
                "error_consumed": error_vs(consumed_reference, consumed),
                "error_raw": error_vs(reference, raw),
            }
    sd_hijack_unet.UNET_BF16_NATIVE_NORMS = True
    result["bf16_rounding_floor"] = error_vs(consumed_reference, consumed_reference.to(BF16))
    auto, native = result["autocast"], result["native"]
    result["speedup"] = auto["median_ms"] / native["median_ms"]
    result["native_error_within_autocast_plus_rounding"] = (
        native["error_consumed"]["max_over_rms"] <= auto["error_consumed"]["max_over_rms"] + result["bf16_rounding_floor"]["max_over_rms"]
    )
    return result


def bf16_casts(count):
    def consume(y):
        for _ in range(count):
            out = y.to(BF16)  # autocast's per-Linear cast; a no-op on a bf16 tensor
        return out
    return consume


def main():
    assert torch.cuda.is_available(), "needs a CUDA device"
    results = []
    for distribution in ("randn", "wide"):
        for channels, size in ((320, 128), (640, 64), (1280, 32)):
            for fmt_name, fmt in (("nchw", torch.contiguous_format), ("nhwc", torch.channels_last)):
                x = activations((2, channels, size, size), distribution, fmt)
                group_norm32 = randomize(sgm_util.normalization(channels))
                st_norm = sgm_attention.Normalize(channels)
                st_norm.__class__ = sd_hijack_unet.UnetGroupNorm
                st_norm = randomize(st_norm)
                for name, module in (("GroupNorm32", group_norm32), ("SpatialTransformer.norm", st_norm)):
                    with torch.no_grad():
                        reference = F.group_norm(x.double(), 32, module.weight.double(), module.bias.double(), module.eps)
                    consumed_reference = None
                    if name == "GroupNorm32":
                        consume = bf16_casts(1)
                    else:
                        def consume(y, b=2, c=channels, hw=size * size):
                            return y.permute(0, 2, 3, 1).reshape(b, hw, c).contiguous().to(BF16)
                        consumed_reference = reference.permute(0, 2, 3, 1).reshape(2, size * size, channels)
                    case = run_case(f"{name} {distribution} {fmt_name} [2,{channels},{size},{size}]", module, x, reference, consume, consumed_reference)
                    results.append(case)
                    print(json.dumps(case), flush=True)
                del x
        for channels, tokens in ((640, 4096), (1280, 1024)):
            x = activations((2, tokens, channels), distribution)
            norm = torch.nn.LayerNorm(channels)
            norm.__class__ = sd_hijack_unet.UnetLayerNorm
            norm = randomize(norm)
            with torch.no_grad():
                reference = F.layer_norm(x.double(), (channels,), norm.weight.double(), norm.bias.double(), norm.eps)
            for casts, site in ((3, "norm1"), (1, "norm2/3")):
                case = run_case(f"LayerNorm {site} {distribution} [2,{tokens},{channels}]", norm, x, reference, bf16_casts(casts))
                results.append(case)
                print(json.dumps(case), flush=True)
    summary = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "iters": ARGS.iters,
        "results": results,
    }
    if ARGS.json:
        ARGS.json.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
