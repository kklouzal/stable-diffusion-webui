"""Fast NHWC -> NCHW GroupNorm input copy: modules/openclaw_gn_transpose.py and its two call sites in sd_hijack_unet.

CPU tests:
- the switch: off by default, read with the openclaw_env grammar at import, an invalid value fails the import;
- the wrapper: which tensors it covers (CUDA, channels_last and not also NCHW-contiguous, bf16/fp16/fp32, no autograd)
  and that it returns everything else unchanged (the same object), CPU included;
- the call sites: GroupNorm32 / SpatialTransformer.norm on the bf16-native path and VaeGroupNorm hand ATen the copy,
  with outputs bitwise equal to the switch-off path (CPU group_norm made to copy like ATen's CUDA one);
- the real kernel in Triton's interpreter (tools/gn_transpose_interpreter_check.py): tails, storage offsets, size-1
  dimensions with odd strides, bf16/fp16/fp32 bit for bit against x.contiguous();
- the kernel's sm_121 PTX: no conversion or arithmetic on the values, vectorized global loads and stores.
CUDA tests (GPU host): bit-for-bit equality with x.contiguous() and of the GroupNorm outputs at the production shapes,
odd shapes and every dtype, the ATen layout facts the contract relies on, and the hijacked modules end to end:
    OPENCLAW_GN_FAST_TRANSPOSE=1 python -m pytest -q test/test_openclaw_gn_transpose.py -k cuda
Timing: tools/benchmark_gn_transpose.py.
"""
import copy
import json
import os
import re
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from test.helpers import ROOT as _ROOT, add_repositories_to_sys_path, randomize

add_repositories_to_sys_path("generative-models", "stable-diffusion-stability-ai")
pytest.importorskip("sgm.modules.attention")

from modules import openclaw_gn_transpose as gn_transpose, sd_hijack_unet  # noqa: E402
import sgm.modules.attention as sgm_attention  # noqa: E402
import sgm.modules.diffusionmodules.model as sgm_vae  # noqa: E402
import sgm.modules.diffusionmodules.util as sgm_util  # noqa: E402

BF16 = torch.bfloat16
CL = torch.channels_last
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the transpose kernel runs on CUDA only")


def _import_with(value):
    env = {key: item for key, item in os.environ.items() if key != gn_transpose.ENV_NAME}
    if value is not None:
        env[gn_transpose.ENV_NAME] = value
    code = "from modules import openclaw_gn_transpose as t; print(t.ENABLED, t._KERNEL is not None)"
    return subprocess.run([sys.executable, "-c", code], cwd=_ROOT, env=env, capture_output=True, text=True, timeout=300)


def test_switch_is_off_by_default_and_an_invalid_value_fails_the_import():
    for value, expected in ((None, "False False"), ("", "False False"), ("0", "False False"), ("off", "False False")):
        process = _import_with(value)
        assert process.returncode == 0 and process.stdout.split("\n")[-2] == expected, process.stderr[-2000:]
    process = _import_with("fast")
    assert process.returncode != 0 and "OPENCLAW_GN_FAST_TRANSPOSE='fast' is not a boolean" in process.stderr


def test_switch_on_loads_the_kernel_at_import():
    pytest.importorskip("triton")
    process = _import_with("1")
    assert process.returncode == 0 and process.stdout.split("\n")[-2] == "True True", process.stderr[-2000:]


@pytest.fixture
def stand_in(monkeypatch):
    """Switch on with a stand-in kernel that records the calls and returns x.contiguous() (what the kernel computes)."""
    calls = []

    def transpose(x):
        calls.append(x)
        return x.contiguous()

    monkeypatch.setattr(gn_transpose, "ENABLED", True)
    monkeypatch.setattr(gn_transpose, "transpose", transpose)
    yield calls


def _cl(shape, dtype=BF16, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=generator) * 2 + 0.5).to(dtype).contiguous(memory_format=CL)


def test_wrapper_returns_the_input_unchanged_when_off_on_cpu_or_out_of_scope(stand_in, monkeypatch):
    x = _cl((2, 64, 5, 7))
    assert gn_transpose.for_group_norm(x) is x  # CPU: ATen's CPU group_norm keeps channels_last, nothing to copy
    monkeypatch.setattr(gn_transpose, "_on_cuda", lambda t: True)
    nchw = x.contiguous()
    both = _cl((2, 64, 1, 1))  # channels_last and NCHW-contiguous at once: ATen does not copy it
    single_channel = _cl((2, 1, 5, 7))
    wide = _cl((2, 64, 5, 7), torch.float64)
    grad = _cl((2, 64, 5, 7), torch.float32).requires_grad_()
    for tensor in (nchw, both, single_channel, wide, grad):
        assert gn_transpose.for_group_norm(tensor) is tensor
    assert stand_in == []
    with torch.no_grad():
        assert gn_transpose.for_group_norm(grad) is not grad
    for dtype in (BF16, torch.float16, torch.float32):
        covered = _cl((1, 32, 3, 1), dtype)
        y = gn_transpose.for_group_norm(covered)
        assert y.is_contiguous() and torch.equal(y, covered)
    monkeypatch.setattr(gn_transpose, "ENABLED", False)
    assert gn_transpose.for_group_norm(x) is x
    assert len(stand_in) == 4


@pytest.fixture
def cuda_like_group_norm(monkeypatch, default_runtime):
    """CPU group_norm as ATen's CUDA one computes it (NCHW copy of the input, NCHW output) and the bf16-native path's
    eligibility minus its CUDA conditions, so the call sites run on CPU as they do on the GPU."""
    group_norm = F.group_norm
    monkeypatch.setattr(F, "group_norm", lambda x, *args, **kwargs: group_norm(x.contiguous(), *args, **kwargs))
    monkeypatch.setattr(gn_transpose, "_on_cuda", lambda t: True)
    monkeypatch.setattr(sd_hijack_unet, "bf16_native_norm_eligible", lambda module, x: x.dtype == BF16 and not torch.is_grad_enabled())


def _off_and_on(module, x, monkeypatch):
    with torch.no_grad():
        monkeypatch.setattr(gn_transpose, "ENABLED", False)
        off = module(x)
        monkeypatch.setattr(gn_transpose, "ENABLED", True)
        on = module(x)
    return off, on


@pytest.mark.parametrize("kind", ["groupnorm32", "spatial-transformer-norm", "vae-normalize"])
def test_call_sites_hand_aten_the_copy_and_outputs_are_bitwise_unchanged(stand_in, cuda_like_group_norm, monkeypatch, kind):
    if kind == "groupnorm32":
        norm = sgm_util.GroupNorm32(32, 64)
    elif kind == "spatial-transformer-norm":
        norm = sgm_attention.SpatialTransformer(64, 2, 32, depth=1, context_dim=48, use_linear=True).norm
        assert type(norm) is sd_hijack_unet.UnetGroupNorm
    else:
        norm = sgm_vae.Normalize(64)
        assert type(norm) is sd_hijack_unet.VaeGroupNorm
    norm = randomize(norm).eval().to(BF16)
    x = _cl((2, 64, 6, 5))
    off, on = _off_and_on(norm, x, monkeypatch)
    assert torch.equal(off, on) and off.stride() == on.stride() and on.is_contiguous()
    assert len(stand_in) == 1 and stand_in[0] is x


def test_vae_norm_with_the_nhwc_kernel_scope_on_still_prefers_the_kernel(stand_in, cuda_like_group_norm, monkeypatch):
    """The parked NHWC GroupNorm switch keeps precedence: the copy is only made where ATen's group_norm runs."""
    from modules import openclaw_nhwc_groupnorm as nhwc

    norm = randomize(sgm_vae.Normalize(64)).eval().to(BF16)
    marker = torch.zeros(1)
    monkeypatch.setattr(nhwc, "_SCOPES", frozenset({nhwc.VAE}))
    monkeypatch.setattr(nhwc, "vae_norm_eligible", lambda module, x: True)
    monkeypatch.setattr(nhwc, "group_norm", lambda module, x, act=None: marker)
    with torch.no_grad():
        assert norm(_cl((1, 64, 4, 4))) is marker
    assert stand_in == []


# --- the real kernel on CPU: Triton's interpreter and the sm_121 PTX ------------------------------------------------

# (shape, dtype, source); the interpreter run uses 16 x 16 tiles: tails in both directions, several tiles per program axis
_INTERPRETER_CASES = {
    "bf16-tails": ((2, 40, 5, 7), "bfloat16", "dense"),
    "fp16-tails": ((3, 17, 3, 11), "float16", "dense"),
    "fp32-tails": ((2, 33, 4, 9), "float32", "dense"),
    "bf16-exact-tiles": ((1, 32, 4, 8), "bfloat16", "dense"),
    "bf16-offset": ((2, 24, 3, 5), "bfloat16", "offset"),
    "bf16-h1": ((1, 48, 1, 37), "bfloat16", "size1-strides"),
    "fp32-w1": ((2, 20, 19, 1), "float32", "size1-strides"),
    "fp16-n1": ((1, 50, 6, 6), "float16", "size1-strides"),
}


@pytest.fixture(scope="module")
def interpreter_results():
    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET="1", **{gn_transpose.ENV_NAME: "1"})
    process = subprocess.run(
        [sys.executable, str(_ROOT / "tools" / "gn_transpose_interpreter_check.py"), json.dumps(_INTERPRETER_CASES)],
        cwd=_ROOT, env=env, capture_output=True, text=True, timeout=900,
    )
    assert process.returncode == 0, process.stderr[-6000:]
    return json.loads(process.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("case", list(_INTERPRETER_CASES))
def test_kernel_in_the_interpreter_copies_bit_for_bit(interpreter_results, case):
    shape, dtype, _ = _INTERPRETER_CASES[case]
    assert interpreter_results[case] == {
        "transposed": True, "contiguous": True, "bitwise": True, "shape_dtype": [list(shape), f"torch.{dtype}"],
    }


def _ptx(dtype):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from modules.openclaw_gn_transpose_kernel import nhwc_to_nchw_kernel

    signature = {"x_ptr": f"*{dtype}", "y_ptr": f"*{dtype}", "C": "i32", "HW": "i32", "BLOCK_C": "constexpr", "BLOCK_HW": "constexpr"}
    names = list(signature)
    source = ASTSource(
        fn=nhwc_to_nchw_kernel,
        signature=signature,
        constexprs={(names.index("BLOCK_C"),): gn_transpose.BLOCK_C, (names.index("BLOCK_HW"),): gn_transpose.BLOCK_HW},
        attrs={(i,): [["tt.divisibility", 16]] for i in range(4)},
    )
    ptx = triton.compile(source, target=GPUTarget("cuda", 121, 32), options={"num_warps": gn_transpose.NUM_WARPS}).asm["ptx"]
    return "\n".join(line for line in ptx.splitlines() if not line.lstrip().startswith(("//", ".loc", ".b8")))


@pytest.mark.parametrize("dtype", ["bf16", "fp16", "fp32"])
def test_kernel_ptx_moves_values_without_touching_them(dtype):
    """A copy cannot change bits only if no instruction converts or computes on the values: no cvt between float
    types and no float arithmetic; the global loads and stores are vectorized (both sides coalesced)."""
    pytest.importorskip("triton")
    body = _ptx(dtype)
    assert not re.search(r"\bcvt(\.\w+)*\.(bf16|f16|f32|f64)\b", body)
    assert not re.search(r"\b(add|sub|mul|fma|div|min|max)\.(rn\.)?(f32|f16|bf16|f16x2|bf16x2)\b", body)
    assert re.search(r"ld\.global(\.\w+)*\.v[248]", body) and re.search(r"st\.global(\.\w+)*\.v[248]", body)


# --- CUDA ---------------------------------------------------------------------------------------------------------

# Production shapes (n-w1: VAE encode/decode at 1280², SDXL UNet at 160² latents with CFG batch 2) and odd ones.
_CUDA_SHAPES = [
    (1, 128, 1280, 1280), (1, 256, 640, 640), (1, 512, 320, 320), (1, 512, 160, 160),
    (2, 320, 160, 160), (2, 640, 80, 80), (2, 1280, 40, 40), (2, 960, 80, 80), (2, 1920, 40, 40), (3, 640, 96, 96),
    (1, 3, 7, 5), (2, 33, 17, 13), (1, 65, 1, 129), (5, 100, 9, 1), (1, 4097, 3, 3), (2, 64, 63, 65),
]
_CUDA_DTYPES = [BF16, torch.float16, torch.float32]


@pytest.fixture
def cuda_switch_on(monkeypatch):
    if gn_transpose._KERNEL is None:
        monkeypatch.setattr(gn_transpose, "_KERNEL", gn_transpose._load_kernel())
    monkeypatch.setattr(gn_transpose, "ENABLED", True)


def _cuda_cl(shape, dtype, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(shape, generator=generator, device="cuda") * 3 + 0.25
    flat = x.view(-1)
    if flat.numel() >= 5:
        flat[:5] = torch.tensor([float("inf"), float("-inf"), float("nan"), -0.0, 0.0], device="cuda")
    return x.to(dtype).contiguous(memory_format=CL)


_BITS = {BF16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}


@needs_cuda
@pytest.mark.parametrize("dtype", _CUDA_DTYPES, ids=str)
@pytest.mark.parametrize("shape", _CUDA_SHAPES, ids=str)
def test_cuda_transpose_equals_contiguous_bit_for_bit(cuda_switch_on, shape, dtype):
    x = _cuda_cl(shape, dtype)
    if not gn_transpose.is_nhwc(x):
        pytest.skip("C == 1 or H*W == 1 is NCHW-contiguous too")
    with torch.inference_mode():
        y = gn_transpose.for_group_norm(x)
        reference = x.contiguous()
    assert y is not x and y.stride() == reference.stride() and y.dtype == dtype
    assert torch.equal(y.view(_BITS[dtype]), reference.view(_BITS[dtype]))  # NaN payloads and signed zeros included


@needs_cuda
def test_cuda_aten_group_norm_copies_channels_last_input_to_nchw():
    """The fact the switch relies on: ATen's CUDA group_norm on channels_last input equals its result on the NCHW copy,
    bit for bit, and returns NCHW (if a torch upgrade adds an NHWC kernel this fails and the switch must be revisited)."""
    for dtype in _CUDA_DTYPES:
        x = _cuda_cl((2, 320, 40, 40), dtype)
        weight = torch.randn(320, device="cuda", dtype=dtype)
        bias = torch.randn(320, device="cuda", dtype=dtype)
        with torch.inference_mode():
            y = F.group_norm(x, 32, weight, bias, 1e-5)
            assert y.is_contiguous() and torch.equal(y, F.group_norm(x.contiguous(), 32, weight, bias, 1e-5))


@needs_cuda
@pytest.mark.parametrize("dtype", _CUDA_DTYPES, ids=str)
@pytest.mark.parametrize("shape", [s for s in _CUDA_SHAPES if s[1] % 32 == 0], ids=str)
def test_cuda_group_norm_outputs_are_bitwise_unchanged(cuda_switch_on, shape, dtype):
    x = _cuda_cl(shape, dtype, seed=1)
    x.view(-1)[:5] = 0.5  # finite statistics
    channels = shape[1]
    weight = torch.randn(channels, device="cuda", dtype=dtype)
    bias = torch.randn(channels, device="cuda", dtype=dtype)
    with torch.inference_mode():
        expected = F.group_norm(x, 32, weight, bias, 1e-6)
        actual = F.group_norm(gn_transpose.for_group_norm(x), 32, weight, bias, 1e-6)
    assert actual.stride() == expected.stride() and torch.equal(actual, expected)


@needs_cuda
@pytest.mark.parametrize("kind", ["vae-normalize", "groupnorm32-bf16-native"])
def test_cuda_hijacked_norms_are_bitwise_unchanged(cuda_switch_on, default_runtime, monkeypatch, kind):
    if kind == "vae-normalize":
        norm, shape, autocast = sgm_vae.Normalize(128), (1, 128, 1280, 1280), False
    else:
        norm, shape, autocast = sgm_util.GroupNorm32(32, 640), (2, 640, 80, 80), True
    norm = randomize(norm).eval().to("cuda", BF16)
    reference = copy.deepcopy(norm)
    x = _cuda_cl(shape, BF16, seed=2)
    x.view(-1)[:5] = 0.5
    with torch.inference_mode(), torch.autocast("cuda", dtype=BF16, enabled=autocast):
        on = norm(x)
        monkeypatch.setattr(gn_transpose, "ENABLED", False)
        off = reference(x)
    assert on.stride() == off.stride() and torch.equal(on, off)
