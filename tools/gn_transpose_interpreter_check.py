"""Run modules/openclaw_gn_transpose.py's for_group_norm, with its real Triton kernel, on CPU in Triton's interpreter.

Started by test/test_openclaw_gn_transpose.py in a fresh process with TRITON_INTERPRET=1 (it must be set before triton
is imported) and OPENCLAW_GN_FAST_TRANSPOSE=1 (the switch loads the kernel at import). argv[1] is a JSON object
{case: [shape, dtype name, source]}, source "dense" (a channels_last tensor), "offset" (one at a nonzero storage offset)
or "size1-strides" (size-1 dimensions with non-dense strides, which channels_last contiguity allows);
the tile is 16 x 16 so small shapes still have several tiles and tails. Prints one JSON object {case: result} on the
last stdout line.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import openclaw_gn_transpose as gn_transpose  # noqa: E402

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
BITS = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}


def make_input(index, shape, dtype, source):
    generator = torch.Generator().manual_seed(index)
    n, c, h, w = shape
    numel = n * c * h * w
    offset = 3 if source == "offset" else 0
    flat = torch.randn(numel + offset, generator=generator) * 100
    flat[offset:offset + 4] = torch.tensor([float("inf"), float("-inf"), -0.0, 0.0])  # kept bit for bit (no arithmetic)
    # a dense [N, H, W, C] array viewed as [N, C, H, W]: channels_last, at storage offset `offset`
    x = flat.to(dtype)[offset:].view(n, h, w, c).permute(0, 3, 1, 2)
    if source == "size1-strides":
        # arbitrary strides on the size-1 dimensions, which channels_last contiguity ignores
        strides = [7919 if size == 1 else stride for size, stride in zip(shape, x.stride())]
        x = x.as_strided(shape, strides, x.storage_offset())
    return x


def run_case(index, shape, dtype_name, source):
    dtype = DTYPES[dtype_name]
    x = make_input(index, shape, dtype, source)
    assert gn_transpose.is_nhwc(x), (shape, x.stride())
    y = gn_transpose.for_group_norm(x)
    reference = x.contiguous()
    return {
        "transposed": y is not x,
        "contiguous": y.is_contiguous(),
        "bitwise": bool(torch.equal(y.view(BITS[dtype]), reference.view(BITS[dtype]))),
        "shape_dtype": [list(y.shape), str(y.dtype)],
    }


def main():
    gn_transpose._on_cuda = lambda x: True
    gn_transpose.BLOCK_C = gn_transpose.BLOCK_HW = 16
    results = {}
    for index, (name, (shape, dtype_name, source)) in enumerate(json.loads(sys.argv[1]).items()):
        results[name] = run_case(index, shape, dtype_name, source)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
