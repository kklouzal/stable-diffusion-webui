"""Run the NHWC GroupNorm kernels (modules/openclaw_nhwc_groupnorm_kernel.py) on CPU in Triton's interpreter.

Started by test/test_openclaw_nhwc_groupnorm.py in a fresh process with TRITON_INTERPRET=1: the variable must be set
before triton is imported, or triton.language's own @jit helpers (tl.sum, tl.minimum, ...) stay compiled-mode functions
the interpreter cannot call. argv[1] is a JSON object {case: [shape, groups, module constants, silu, offset]}; prints
one JSON object {case: result} on the last stdout line.
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import openclaw_nhwc_groupnorm as nhwc, openclaw_nhwc_groupnorm_kernel as kernels  # noqa: E402

TUNABLES = ("_MIN_STATS_PROGRAMS", "_STATS_TILE_ELEMENTS", "_APPLY_TILE_ELEMENTS")


def run_case(index, shape, groups, silu, offset):
    generator = torch.Generator().manual_seed(index)
    channels = shape[1]
    if offset:  # offset + noise of std offset / 64, a few bf16 ulps: a large mean over std
        x = torch.randn(shape, generator=generator) * offset / 64 + offset
    else:  # per-channel scales 0.01..10 around 3: groups mixing very different channels
        x = torch.randn(shape, generator=generator) * torch.logspace(-2, 1, channels).view(1, -1, 1, 1) + 3
    x = x.to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
    weight = (torch.randn(channels, generator=generator) * 0.5 + 1).to(torch.bfloat16)
    bias = (torch.randn(channels, generator=generator) * 0.2).to(torch.bfloat16)
    y = nhwc._launch(kernels, x, groups, weight, bias, 1e-5, silu)
    reference = F.group_norm(x.double(), groups, weight.double(), bias.double(), 1e-5)
    if silu:
        reference = F.silu(reference)
    # The interpreter casts float32 -> bf16 toward zero (the compiled kernel rounds to nearest even; the PTX test pins
    # cvt.rn): up to one bf16 ulp of the output, plus the float32 statistics.
    excess = ((y.double() - reference).abs() - (reference.abs() * 2.0 ** -7 + 1e-3)).max().item()
    return {
        "excess": excess,
        "layout": [str(y.dtype), list(y.shape), y.is_contiguous(memory_format=torch.channels_last)],
        "repeat_bitwise": bool(torch.equal(y, nhwc._launch(kernels, x, groups, weight, bias, 1e-5, silu))),
        "tiles": nhwc.launch_config(shape[0], channels, shape[2] * shape[3])["tiles"],
    }


def main():
    defaults = {name: getattr(nhwc, name) for name in TUNABLES}
    results = {}
    for index, (name, (shape, groups, overrides, silu, offset)) in enumerate(json.loads(sys.argv[1]).items()):
        for key, value in {**defaults, **overrides}.items():
            setattr(nhwc, key, value)
        results[name] = run_case(index, shape, groups, silu, offset)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
