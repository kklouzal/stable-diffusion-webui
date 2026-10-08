"""Fused GEGLU (modules/openclaw_fused_geglu.py): dispatch on CPU, bitwise equality and timing on CUDA.

CPU tests pin the CondFunc installation, the dispatch/fallback and, by compiling the Triton kernel for sm_121 without a
GPU, the instruction-level properties the bitwise argument relies on (round-to-nearest float32 ops, libdevice erff
without FTZ, no narrowed bf16/fp16 arithmetic, 16-byte vector loads). They also check that the kernel's erff constants
are the ones in ATen's own GELU kernels (libtorch_cuda sm_120 SASS, read with cuobjdump), which fails for Triton's
bundled libdevice. The CUDA tests compare the fused kernel with the eager `a * F.gelu(gate)` bit for bit over every
bf16/fp16 gate and multiplicand value and over SDXL feed-forward shapes run through the hijacked sgm GEGLU under bf16
autocast, and print eager vs fused timings. Run them on the GPU host with:
    python -m pytest -q -s test/test_openclaw_fused_geglu.py -k cuda
OPENCLAW_FUSED_GEGLU_TIMING_JSON=<path> also writes the timings as JSON.
"""
import json
import os
import re
import shutil
import statistics
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
for _repo in ("generative-models", "stable-diffusion-stability-ai"):
    _path = str(_ROOT / "repositories" / _repo)
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)  # what modules/paths.py does at startup

pytest.importorskip("sgm.modules.attention")
pytest.importorskip("ldm.modules.attention")


def _import_runtime_modules():
    """Import devices/sd_hijack_unet against the real modules.shared (see test_sd_hijack_unet.py: another test module
    may leave a minimal modules.shared stub in sys.modules, which devices cannot import against)."""
    import importlib
    import modules

    stub = sys.modules.get("modules.shared")
    if stub is None or getattr(stub, "__file__", None) is not None:
        from modules import devices, openclaw_fused_geglu, sd_hijack_unet  # noqa: F401
        return devices, openclaw_fused_geglu
    package_attribute = modules.__dict__.pop("shared", None)
    del sys.modules["modules.shared"]
    try:
        importlib.import_module("modules.shared")
        from modules import devices, openclaw_fused_geglu, sd_hijack_unet  # noqa: F401
    finally:
        sys.modules["modules.shared"] = stub
        if package_attribute is None:
            modules.__dict__.pop("shared", None)
        else:
            modules.shared = package_attribute
    return devices, openclaw_fused_geglu


devices, fused = _import_runtime_modules()
import ldm.modules.attention as ldm_attention  # noqa: E402
import sgm.modules.attention as sgm_attention  # noqa: E402

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused GEGLU kernel runs on CUDA only")
BF16 = torch.bfloat16


def _condfunc(forward):
    """The CondFunc behind a hijacked forward (CondFunc installs `lambda *args, **kwargs: self(*args, **kwargs)`)."""
    for cell in forward.__closure__ or ():
        if hasattr(cell.cell_contents, "_CondFunc__sub_func"):
            return cell.cell_contents
    return None


def _outermost_original(cls):
    """GEGLU.forward as the repository defines it, below every CondFunc."""
    forward = cls.forward
    while (condfunc := _condfunc(forward)) is not None:
        forward = condfunc._CondFunc__orig_func
    return forward


def _eager(h):
    a, gate = h.chunk(2, dim=-1)
    return a * F.gelu(gate)


def test_block_size_divides_sdxl_widths():
    assert fused._block_size(2560) == 512  # SDXL level-1 feed-forward (640 * 4)
    assert fused._block_size(5120) == 1024  # SDXL level-2 feed-forward (1280 * 4)
    assert fused._block_size(1280) == 256  # SD1.5 level-0 feed-forward (320 * 4)
    assert fused._block_size(48) == 64
    assert fused._block_size(1) == 1


def test_fused_condfunc_is_the_outermost_geglu_forward_for_sgm_and_ldm():
    if not fused.ENABLED:
        pytest.skip("OPENCLAW_FUSED_GEGLU is off in this environment")
    for cls in (sgm_attention.GEGLU, ldm_attention.GEGLU):
        condfunc = _condfunc(cls.forward)
        assert condfunc is not None
        assert condfunc._CondFunc__sub_func is fused.geglu_forward
        assert condfunc._CondFunc__cond_func is fused.geglu_cond


def test_install_is_a_no_op_when_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(fused, "ENABLED", False)
    monkeypatch.setattr(fused, "CondFunc", lambda *args: calls.append(args))
    fused.install()
    assert calls == []


@pytest.mark.parametrize("dtype", [torch.float32, BF16])
@pytest.mark.parametrize("cls", [sgm_attention.GEGLU, ldm_attention.GEGLU])
def test_cpu_forward_is_the_original(cls, dtype):
    torch.manual_seed(0)
    module = cls(16, 24).to(dtype)
    x = torch.randn(2, 5, 16, dtype=dtype)
    with torch.inference_mode():
        assert torch.equal(module(x), _outermost_original(cls)(module, x))
        assert torch.equal(module(x), _eager(module.proj(x)))


def test_cond_takes_cuda_inputs_unless_upcast_sampling(monkeypatch):
    monkeypatch.setattr(devices, "unet_needs_upcast", False)
    assert fused.geglu_cond(None, None, SimpleNamespace(is_cuda=True))
    assert not fused.geglu_cond(None, None, SimpleNamespace(is_cuda=False))
    monkeypatch.setattr(devices, "unet_needs_upcast", True)
    assert not fused.geglu_cond(None, None, SimpleNamespace(is_cuda=True))


def test_forward_runs_proj_once_and_hands_its_output_to_the_kernel(monkeypatch):
    torch.manual_seed(0)
    module = sgm_attention.GEGLU(16, 24)
    x = torch.randn(2, 5, 16)
    proj_outputs, kernel_inputs = [], []
    module.proj.register_forward_hook(lambda mod, args, out: proj_outputs.append(out))
    monkeypatch.setattr(fused, "fusable", lambda h: True)
    monkeypatch.setattr(fused, "fused_geglu", lambda h: kernel_inputs.append(h) or "fused")
    with torch.inference_mode():
        assert fused.geglu_forward(None, module, x) == "fused"
    assert len(proj_outputs) == 1 and kernel_inputs == proj_outputs

    monkeypatch.setattr(fused, "fusable", lambda h: False)
    with torch.inference_mode():
        out = fused.geglu_forward(None, module, x)
        assert torch.equal(out, _eager(module.proj(x)))
    assert len(proj_outputs) == 3


def test_fusable_needs_cuda():
    assert not fused.fusable(torch.zeros(2, 8, dtype=BF16))


def _compile_ptx(dtype, block, **overrides):
    """The fused kernel's PTX for GB10 (sm_121) with fused_geglu's launch options (and `overrides`), without a GPU."""
    triton = pytest.importorskip("triton")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from modules.openclaw_fused_geglu_kernel import geglu_kernel

    source = ASTSource(
        fn=geglu_kernel,
        signature={"h_ptr": f"*{dtype}", "out_ptr": f"*{dtype}", "inner": "i32", "BLOCK": "constexpr"},
        constexprs={(3,): block},
        attrs={(0,): [["tt.divisibility", 16]], (1,): [["tt.divisibility", 16]], (2,): [["tt.divisibility", 16]]},
    )
    options = {**fused._launch_options(block), **overrides}
    return triton.compile(source, target=GPUTarget("cuda", 121, 32), options=options).asm["ptx"]


def test_toolkit_libdevice_matches_torch_cuda(monkeypatch, tmp_path):
    if torch.version.cuda is None:
        pytest.skip("CPU-only torch build")
    path = fused._toolkit_libdevice()
    assert path.endswith(os.path.join("nvvm", "libdevice", "libdevice.10.bc")) and os.path.isfile(path)
    if fused.ENABLED:
        assert fused._EXTERN_LIBS == (("libdevice", path),)
        assert fused._launch_options(1024)["extern_libs"] == fused._EXTERN_LIBS

    monkeypatch.setattr(torch.version, "cuda", "12.1")
    with pytest.raises(RuntimeError, match="expected 12010"):
        fused._toolkit_libdevice()
    monkeypatch.undo()
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    with pytest.raises(RuntimeError, match="CUDA_VERSION None"):
        fused._toolkit_libdevice()


def test_launch_options_refuse_when_disabled(monkeypatch):
    monkeypatch.setattr(fused, "_EXTERN_LIBS", None)
    with pytest.raises(RuntimeError, match="OPENCLAW_FUSED_GEGLU=0"):
        fused._launch_options(1024)


def test_kernel_compiles_for_sm121_with_aten_rounding():
    """The PTX the bitwise argument relies on, compiled for GB10 (sm_121) without a GPU."""
    if fused._EXTERN_LIBS is None:
        pytest.skip("fused GEGLU is off in this environment")
    for dtype, cvt in (("bf16", "cvt.rn.bf16"), ("fp16", "cvt.rn.f16")):
        for block in (512, 1024):
            ptx = _compile_ptx(dtype, block)
            body = "\n".join(line for line in ptx.splitlines() if not line.lstrip().startswith(("//", ".loc", ".b8")))
            # Float32 arithmetic outside libdevice's erff is explicitly round-to-nearest, nothing is flushed except
            # erff's own ex2.approx.ftz (which it also uses without FTZ), and no op is narrowed to bf16/fp16 math.
            assert re.search(r"\bmul\.rn\.f32\b", body) and re.search(r"\badd\.rn\.f32\b", body)
            assert not re.search(r"\.ftz\.f32", body.replace("ex2.approx.ftz.f32", ""))
            assert not re.search(r"\b(mul|fma|add|sub)(\.rn)?\.(bf16|f16)(x2)?\b", body)
            assert cvt in body
            assert "ld.global.v4" in body and "st.global.v4" in body
            assert "0f3F3504F3" in body  # kAlpha = (float)M_SQRT1_2; erff's own constants: the ATen SASS test below


# Constants the kernel itself adds around erff: 0, 0.5, 1 and kAlpha.
_KERNEL_OWN_CONSTANTS = {0x00000000, 0x3F000000, 0x3F800000, 0x3F3504F3}
_MAGNITUDE = 0x7FFFFFFF


def _ptx_constant_magnitudes(ptx):
    return {int(v, 16) & _MAGNITUDE for v in re.findall(r"\b0f([0-9A-Fa-f]{8})\b", ptx)} - _KERNEL_OWN_CONSTANTS


def _sass_constant_magnitudes(sass):
    """The 32-bit float constants of a SASS listing, sign dropped: float immediates of FSETP/FSEL/FFMA/FMUL/FADD and
    the constants ptxas materializes into registers with MOV, IMAD.MOV.U32 or HFMA2 (two halves)."""
    def f32_bits(text):
        return struct.unpack("<I", struct.pack("<f", float(text)))[0] & _MAGNITUDE

    def f16_bits(text):
        return struct.unpack("<H", struct.pack("<e", float(text)))[0]

    found = set()
    for operands in re.findall(r"\b(?:FSETP\.\S+|FSEL|FFMA|FMUL|FADD)\s+([^;]*);", sass):
        found.update(f32_bits(v) for v in re.findall(r"(?<![\w.])-?\d+\.\d+(?:e[-+]\d+)?", operands))
    found.update(int(v, 16) & _MAGNITUDE for v in re.findall(r"\bMOV\S*\s+R\d+,\s*(0x[0-9a-f]{1,8})\b", sass))
    found.update(int(v, 16) & _MAGNITUDE for v in re.findall(r"\bIMAD\.MOV\.U32\s+R\d+,\s*RZ,\s*RZ,\s*(0x[0-9a-f]{1,8})\b", sass))
    for high, low in re.findall(r"\bHFMA2(?:\.MMA)?\s+R\d+,\s*-RZ,\s*RZ,\s*([-0-9.e+]+),\s*([-0-9.e+]+)\s*;", sass):
        found.add(((f16_bits(high) << 16) | f16_bits(low)) & _MAGNITUDE)
    return found


def _aten_gelu_kernels(aten_dtype):
    """SASS of ATen's GELU kernels for one dtype from the sm_120 cubins in libtorch_cuda (GB10, sm_121, runs those)."""
    cuobjdump = (os.environ.get("TRITON_CUOBJDUMP_PATH") or shutil.which("cuobjdump")
                 or os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "cuobjdump"))
    lib = Path(torch.__file__).parent / "lib" / "libtorch_cuda.so"
    if not os.path.isfile(cuobjdump) or not lib.is_file():
        pytest.skip("needs cuobjdump and libtorch_cuda.so")
    symbols = subprocess.run([cuobjdump, "-symbols", "-arch", "sm_120", str(lib)], capture_output=True, text=True, check=True).stdout
    tag = f"N3c10{len(aten_dtype)}{aten_dtype}E"
    names = sorted({name for name in re.findall(r"\b_Z\w*GeluCUDAKernelImpl\w*", symbols) if tag in name})
    assert names, f"no sm_120 {aten_dtype} GeluCUDAKernelImpl kernels in {lib}"
    sass = subprocess.run([cuobjdump, "-sass", "-arch", "sm_120", "-fun", ",".join(names), str(lib)],
                          capture_output=True, text=True, check=True).stdout
    return dict(block.split("\n", 1) for block in re.split(r"\n\s*Function : ", sass)[1:])


@pytest.mark.parametrize("dtype,aten_dtype", [("bf16", "BFloat16"), ("fp16", "Half")])
def test_kernel_erff_constants_are_those_of_atens_gelu(dtype, aten_dtype):
    """Every float constant of the fused kernel's erff (branch threshold, both polynomials) is in one of ATen's GELU
    kernels, i.e. both link the same libdevice __nv_erff. Triton's bundled libdevice must fail this check: that is the
    one-ULP mismatch the toolkit libdevice fixes (if it ever passes, the extern_libs override has become redundant)."""
    if fused._EXTERN_LIBS is None:
        pytest.skip("fused GEGLU is off in this environment")
    ours = _ptx_constant_magnitudes(_compile_ptx(dtype, 1024))
    bundled = _ptx_constant_magnitudes(_compile_ptx(dtype, 1024, extern_libs=None))
    assert len(ours) >= 10 and len(bundled) >= 10 and ours != bundled
    kernels = {name: _sass_constant_magnitudes(body) for name, body in _aten_gelu_kernels(aten_dtype).items()}
    assert any(ours <= constants for constants in kernels.values()), {
        name[:120]: sorted(hex(c) for c in ours - constants) for name, constants in kernels.items()}
    assert not any(bundled <= constants for constants in kernels.values()), "Triton's bundled libdevice matches ATen's erff"


def test_kernel_indexing_in_triton_interpreter(monkeypatch):
    """fused_geglu's grid and launch, and the kernel's rows, column blocks, tail masks and output layout, run on CPU by
    Triton's interpreter.

    The interpreter has no libdevice, so a private copy of the kernel module gets stand-ins: erf -> 0 and plain float32
    ops for mul_rn/add_rn. The kernel then computes round(float(a) * float(round((g * 0.5) * 1.0))), which torch's CPU
    float32 ops reproduce exactly. The interpreter casts float32 -> bf16 toward zero (the compiled kernel's
    cvt.rn.bf16.f32 is pinned by the PTX test), so the bf16 reference truncates the same way; fp16 rounds to nearest.
    """
    pytest.importorskip("triton")
    import importlib.util

    monkeypatch.setenv("TRITON_INTERPRET", "1")
    spec = importlib.util.spec_from_file_location("_geglu_kernel_interpreted", _ROOT / "modules" / "openclaw_fused_geglu_kernel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.libdevice = SimpleNamespace(erf=lambda x: x * 0.0, mul_rn=lambda a, b: a * b, add_rn=lambda a, b: a + b)
    monkeypatch.setattr(fused, "_kernel", lambda: module.geglu_kernel)
    if fused._EXTERN_LIBS is None:  # the interpreter ignores launch options; fused_geglu only needs them present
        monkeypatch.setattr(fused, "_EXTERN_LIBS", (("libdevice", "unused by the interpreter"),))

    def truncate_to_bf16(t):
        return (t.contiguous().view(torch.int32) & -65536).view(torch.float32).to(BF16)

    generator = torch.Generator().manual_seed(0)
    for shape in [(2, 3, 2 * 2560), (1, 2, 2 * 5120), (3, 2 * 48), (1, 2 * 3), (2, 2 * 1000)]:
        for dtype in (BF16, torch.float16):
            h = (torch.randn(shape, generator=generator) * 3).to(dtype)
            out = fused.fused_geglu(h)
            a, g = h.float().chunk(2, dim=-1)
            if dtype == BF16:
                reference = truncate_to_bf16(a * truncate_to_bf16((g * 0.5) * 1.0).float())
            else:
                reference = (a * ((g * 0.5) * 1.0).to(dtype).float()).to(dtype)
            assert torch.equal(out.view(torch.int16), reference.view(torch.int16)), (shape, dtype)


def _all_values(dtype, device):
    return torch.arange(-(2 ** 15), 2 ** 15, dtype=torch.int32, device=device).to(torch.int16).view(dtype)


def _mismatch_report(out, ref):
    """None when `out` equals `ref` bit for bit (NaN payloads aside), else what differs."""
    assert out.shape == ref.shape and out.dtype == ref.dtype and out.stride() == ref.stride()
    nan = torch.isnan(ref)
    if not torch.equal(torch.isnan(out), nan):
        return f"NaN positions differ ({int((torch.isnan(out) != nan).sum())})"
    mismatch = (out.view(torch.int16) != ref.view(torch.int16)) & ~nan
    if not bool(mismatch.any()):
        return None
    return f"{int(mismatch.sum())} mismatches, e.g. {out[mismatch][:4].tolist()} vs {ref[mismatch][:4].tolist()}"


def _assert_bitwise_equal(out, ref, label=""):
    report = _mismatch_report(out, ref)
    assert report is None, f"{label}: {report}"


@needs_cuda
@pytest.mark.parametrize("dtype", [BF16, torch.float16])
def test_cuda_every_gate_value_is_bitwise_equal_to_eager(dtype):
    generator = torch.Generator(device="cuda").manual_seed(1234)
    every = _all_values(dtype, "cuda").view(64, 1024)
    finite = lambda t: torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)  # noqa: E731
    others = {
        "ones": torch.ones_like(every),
        "randn": torch.randn(every.shape, device="cuda", generator=generator).to(dtype),
        "wide": finite(torch.randn(every.shape, device="cuda", generator=generator)
                       * torch.exp2(torch.randint(-40, 40, every.shape, device="cuda", generator=generator).float())).to(dtype),
    }
    failures = []
    with torch.inference_mode():
        for name, other in others.items():
            for role, (a, gate) in (("gate", (other, every)), ("multiplicand", (every, other))):
                h = torch.cat([a, gate], dim=-1)
                assert fused.fusable(h)
                report = _mismatch_report(fused.fused_geglu(h), _eager(h))
                if report is not None:
                    failures.append(f"every {dtype} {role} value, other half {name}: {report}")
    assert not failures, "\n".join(failures)


# SDXL feed-forward GEGLUs: (tokens, model dim) at 1280² (latent 160²: 80² tokens at 640, 40² at 1280) and 1024²,
# batch 2 (cond+uncond, ControlNet) and 1 (PAG's cond-row pass).
SDXL_SHAPES = [(2, 6400, 640), (1, 6400, 640), (2, 1600, 1280), (1, 1600, 1280), (2, 4096, 640), (2, 1024, 1280)]
_TIMINGS = []


def _median_ms(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(start.elapsed_time(end) for start, end in zip(starts, ends))


@needs_cuda
@pytest.mark.parametrize("batch,tokens,dim", SDXL_SHAPES)
def test_cuda_sdxl_geglu_is_bitwise_equal_under_bf16_autocast(batch, tokens, dim):
    generator = torch.Generator(device="cuda").manual_seed(batch * 7 + tokens + dim)
    module = sgm_attention.GEGLU(dim, dim * 4).to("cuda", BF16)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, device="cuda", generator=generator) * 0.03)
    x = torch.randn(batch, tokens, dim, device="cuda", generator=generator, dtype=BF16)
    original = _outermost_original(sgm_attention.GEGLU)
    calls = []
    real = fused.fused_geglu

    def counting(h):
        calls.append(h.shape)
        return real(h)

    with torch.inference_mode(), torch.autocast("cuda", dtype=BF16):
        reference = original(module, x)
        fused.fused_geglu = counting
        try:
            out = module(x)
        finally:
            fused.fused_geglu = real
        assert calls == [torch.Size([batch, tokens, dim * 8])]
        _assert_bitwise_equal(out, reference)

        h = module.proj(x)
        eager_ms = _median_ms(lambda: _eager(h))
        fused_ms = _median_ms(lambda: real(h))
    row = {"shape": [batch, tokens, dim * 8], "eager_ms": round(eager_ms, 4), "fused_ms": round(fused_ms, 4),
           "speedup": round(eager_ms / fused_ms, 3)}
    _TIMINGS.append(row)
    print(f"\nfused GEGLU {row}")
    path = os.environ.get("OPENCLAW_FUSED_GEGLU_TIMING_JSON")
    if path:
        Path(path).write_text(json.dumps(_TIMINGS, indent=2))


@needs_cuda
def test_cuda_float32_and_grad_keep_the_eager_path():
    module = sgm_attention.GEGLU(64, 128).to("cuda")
    x = torch.randn(2, 8, 64, device="cuda")
    calls = []
    real = fused.fused_geglu
    fused.fused_geglu = lambda h: calls.append(h) or real(h)
    try:
        with torch.inference_mode():
            assert torch.equal(module(x), _eager(module.proj(x)))  # float32: not fusable
        module = module.to(BF16)
        out = module(x.to(BF16).requires_grad_())  # autograd: not fusable
        assert out.requires_grad
    finally:
        fused.fused_geglu = real
    assert calls == []
