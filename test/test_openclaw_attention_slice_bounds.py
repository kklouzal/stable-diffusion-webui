import contextlib
import warnings
from types import SimpleNamespace

import pytest
import torch

from test.helpers import load_source, module


def _load_sd_hijack_optimizations():
    """modules/sd_hijack_optimizations.py as a private module on stand-ins for shared/devices/hypernetworks and for
    the ldm/sgm attention classes its optimizers patch (the tests keep set_sdpa_backend from patching them)."""
    def default(value, default_value):
        return value if value is not None else (default_value() if callable(default_value) else default_value)

    class StubCrossAttention:
        forward = staticmethod(lambda self, x, context=None, mask=None, **kwargs: x)

    class StubAttnBlock:
        forward = staticmethod(lambda self, x, **kwargs: x)

    hypernetwork = module(
        "modules.hypernetworks.hypernetwork",
        attention_CrossAttention_forward=lambda self, x, context=None, mask=None, **kwargs: x,
        apply_hypernetworks=lambda _nets, context: (context, context),
    )
    stubs = {
        "modules": module("modules", package=True),
        "modules.shared": module(
            "modules.shared",
            opts=SimpleNamespace(upcast_attn=False),
            cmd_opts=SimpleNamespace(sub_quad_q_chunk_size=1024, sub_quad_kv_chunk_size=None, sub_quad_chunk_threshold=None),
            device=torch.device("cpu"),
            loaded_hypernetworks=[],
        ),
        "modules.devices": module("modules.devices", without_autocast=lambda disable=False: contextlib.nullcontext()),
        "modules.sub_quadratic_attention": module(
            "modules.sub_quadratic_attention",
            efficient_dot_product_attention=lambda q, k, v, **_kwargs: torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0),
        ),
        "modules.hypernetworks": module("modules.hypernetworks", package=True, hypernetwork=hypernetwork),
        "modules.hypernetworks.hypernetwork": hypernetwork,
        "ldm.util": module("ldm.util", default=default),
    }
    for root in ("ldm", "sgm"):
        attention = module(f"{root}.modules.attention", CrossAttention=StubCrossAttention)
        model = module(f"{root}.modules.diffusionmodules.model", AttnBlock=StubAttnBlock)
        diffusionmodules = module(f"{root}.modules.diffusionmodules", package=True, model=model)
        submodules = module(f"{root}.modules", package=True, attention=attention, diffusionmodules=diffusionmodules)
        stubs.update({
            root: module(root, package=True, modules=submodules),
            f"{root}.modules": submodules,
            f"{root}.modules.attention": attention,
            f"{root}.modules.diffusionmodules": diffusionmodules,
            f"{root}.modules.diffusionmodules.model": model,
        })
    stubs["ldm"].util = stubs["ldm.util"]
    return load_source("sd_hijack_optimizations_under_test", "modules/sd_hijack_optimizations.py", stubs)


opt = _load_sd_hijack_optimizations()


@pytest.fixture(autouse=True)
def attention_options(monkeypatch):
    """Give every test its own shared.opts/loaded_hypernetworks."""
    monkeypatch.setattr(opt.shared, "opts", SimpleNamespace(upcast_attn=False), raising=False)
    monkeypatch.setattr(opt.shared, "loaded_hypernetworks", [], raising=False)


class TinyCrossAttention(torch.nn.Module):
    def __init__(self, dim=8, heads=2):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_q = torch.nn.Linear(dim, dim, bias=False)
        self.to_k = torch.nn.Linear(dim, dim, bias=False)
        self.to_v = torch.nn.Linear(dim, dim, bias=False)
        self.to_out = torch.nn.Sequential(torch.nn.Linear(dim, dim, bias=False))


def test_doggettx_attention_keeps_positive_slice_when_memory_steps_exceed_tokens():
    original_vram = opt.get_available_vram
    original_hyper = opt.hypernetwork.apply_hypernetworks
    original_upcast = getattr(opt.shared.opts, "upcast_attn", False)
    original_loaded_hypernetworks = getattr(opt.shared, "loaded_hypernetworks", [])
    try:
        opt.get_available_vram = lambda: 50
        opt.hypernetwork.apply_hypernetworks = lambda _nets, context: (context, context)
        opt.shared.loaded_hypernetworks = []
        opt.shared.opts.upcast_attn = False

        module = TinyCrossAttention()
        x = torch.randn(1, 4, 8)
        y = opt.split_cross_attention_forward(module, x)

        assert y.shape == x.shape
        assert torch.isfinite(y).all()
    finally:
        opt.get_available_vram = original_vram
        opt.hypernetwork.apply_hypernetworks = original_hyper
        opt.shared.loaded_hypernetworks = original_loaded_hypernetworks
        opt.shared.opts.upcast_attn = original_upcast


def test_sdpa_math_backend_uses_non_deprecated_torch_nn_attention_api():
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        y = opt.run_scaled_dot_product_attention(q, k, v, sdpa_backend_override="math")

    assert y.shape == q.shape
    assert not [warning for warning in caught if issubclass(warning.category, FutureWarning)]


def test_select_optimizer_resolves_an_unknown_setting_like_automatic(capsys):
    # Priority order as sd_hijack.list_optimizers sorts it: Doggettx 90, sdp-no-mem 80, sdp 70.
    available = [opt.SdOptimizationDoggettx(), opt.SdOptimizationSdpNoMem(), opt.SdOptimizationSdp()]
    doggettx, sdp = available[0], available[2]
    flags = SimpleNamespace(opt_sdp_attention=True, disable_opt_split_attention=False)

    # The GB10 config carried SD.Next's "Scaled-Dot-Product", which used to select Doggettx despite --opt-sdp-attention.
    assert opt.select_optimizer("Scaled-Dot-Product", available, flags) is sdp
    assert "'Scaled-Dot-Product' is not available; using Automatic" in capsys.readouterr().out
    assert opt.select_optimizer("Automatic", available, flags) is sdp
    assert opt.select_optimizer("sdp - scaled dot product", available, flags) is sdp
    assert opt.select_optimizer("Doggettx", available, flags) is doggettx
    assert opt.select_optimizer("None", available, flags) is None
    assert opt.select_optimizer("Automatic", available, SimpleNamespace(disable_opt_split_attention=False)) is doggettx
    assert opt.select_optimizer("Automatic", available, SimpleNamespace(disable_opt_split_attention=True)) is None


ALL_FOUR = [opt.SDPBackend.CUDNN_ATTENTION, opt.SDPBackend.FLASH_ATTENTION, opt.SDPBackend.EFFICIENT_ATTENTION, opt.SDPBackend.MATH]


@pytest.fixture
def sdpa_selection(monkeypatch):
    """Restore the active selection afterwards and keep set_sdpa_backend from patching attention classes."""
    monkeypatch.setattr(opt.SdOptimizationSdp, "apply", lambda self: None)
    previous = opt._active_sdpa_backend
    yield
    opt._active_sdpa_backend = previous


@pytest.fixture
def recorded_sdpa_kernel(monkeypatch):
    entered = []
    real_sdpa_kernel = opt.sdpa_kernel

    @contextlib.contextmanager
    def recording_sdpa_kernel(backends, *args, **kwargs):
        entered.append(list(backends))
        with real_sdpa_kernel(backends, *args, **kwargs):
            yield

    monkeypatch.setattr(opt, "sdpa_kernel", recording_sdpa_kernel)
    return entered


@pytest.fixture
def restore_sdp_globals():
    getters = [torch._C._get_cudnn_sdp_enabled, torch._C._get_flash_sdp_enabled, torch._C._get_mem_efficient_sdp_enabled, torch._C._get_math_sdp_enabled, torch._C._get_overrideable_sdp_enabled]
    setters = [torch._C._set_sdp_use_cudnn, torch._C._set_sdp_use_flash, torch._C._set_sdp_use_mem_efficient, torch._C._set_sdp_use_math, torch._C._set_sdp_use_overrideable]
    flags = [get() for get in getters]
    order = torch._C._get_sdp_priority_order()
    yield
    for set_flag, flag in zip(setters, flags):
        set_flag(flag)
    torch._C._set_sdp_priority_order(order)


def _qkv(seed=0):
    generator = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(2, 2, 16, 8, generator=generator) for _ in range(3))


def test_set_sdpa_backend_parses_once_and_keeps_status_contract(sdpa_selection, monkeypatch):
    status = opt.set_sdpa_backend(" CUDNN+flash efficient,math ")
    assert status["sdpa_backend"] == opt.active_sdpa_backend() == "cudnn,flash,efficient,math"
    assert status["sdpa_backend_choices"] == opt._SDPA_BACKEND_CHOICES
    q, k, v = _qkv()
    opt.run_scaled_dot_product_attention(q, k, v, sdpa_backend_override="flash,math")

    def no_parsing(_value):
        raise AssertionError("SDPA backend selection was re-parsed on the attention hot path")

    monkeypatch.setattr(opt, "_normalize_sdpa_backend_choice", no_parsing)
    for _ in range(3):
        opt.run_scaled_dot_product_attention(q, k, v)
        opt.run_scaled_dot_product_attention(q, k, v, sdpa_backend_override="flash,math")


def test_invalid_sdpa_backend_raises_at_set_time_and_keeps_selection(sdpa_selection):
    opt.set_sdpa_backend("flash,math")
    with pytest.raises(ValueError, match="Unsupported SDPA backend: bogus"):
        opt.set_sdpa_backend("flash,bogus")
    assert opt.sdpa_backend_status()["sdpa_backend"] == opt.active_sdpa_backend() == "flash,math"


@pytest.mark.parametrize("selection", ["auto", "cudnn,flash,efficient,math", "math,efficient,flash,cudnn", "flash,cudnn,mem_efficient,math,flash"])
def test_default_backend_set_skips_sdpa_kernel(sdpa_selection, recorded_sdpa_kernel, restore_sdp_globals, selection):
    opt.set_sdpa_backend(selection)
    q, k, v = _qkv()

    out = opt.run_scaled_dot_product_attention(q, k, v)

    assert recorded_sdpa_kernel == []
    with opt.sdpa_kernel(ALL_FOUR):
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    assert torch.equal(out, expected)


@pytest.mark.parametrize(("selection", "override", "expected"), [
    ("flash,math", None, [opt.SDPBackend.FLASH_ATTENTION, opt.SDPBackend.MATH]),
    ("cudnn,flash,math", None, [opt.SDPBackend.CUDNN_ATTENTION, opt.SDPBackend.FLASH_ATTENTION, opt.SDPBackend.MATH]),
    ("math", None, [opt.SDPBackend.MATH]),
    ("auto", "flash,math", [opt.SDPBackend.FLASH_ATTENTION, opt.SDPBackend.MATH]),
    ("cudnn,flash,efficient,math", "math", [opt.SDPBackend.MATH]),
])
def test_backend_subsets_and_overrides_enter_sdpa_kernel(sdpa_selection, recorded_sdpa_kernel, selection, override, expected):
    opt.set_sdpa_backend(selection)
    q, k, v = _qkv(1)

    out = opt.run_scaled_dot_product_attention(q, k, v, sdpa_backend_override=override)

    assert recorded_sdpa_kernel == [expected]
    with opt.sdpa_kernel(expected):
        assert torch.equal(out, torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0))


@pytest.mark.parametrize("change", ["flash_off", "math_off", "cudnn_off", "efficient_off", "overrideable_before_math"])
def test_full_set_keeps_sdpa_kernel_when_global_state_differs_from_default(sdpa_selection, recorded_sdpa_kernel, restore_sdp_globals, change):
    opt.set_sdpa_backend("cudnn,flash,efficient,math")
    if change == "overrideable_before_math":
        order = torch._C._get_sdp_priority_order()
        order.remove(int(opt.SDPBackend.OVERRIDEABLE))
        order.insert(order.index(int(opt.SDPBackend.MATH)), int(opt.SDPBackend.OVERRIDEABLE))
        torch._C._set_sdp_priority_order(order)
    else:
        {
            "flash_off": torch._C._set_sdp_use_flash,
            "math_off": torch._C._set_sdp_use_math,
            "cudnn_off": torch._C._set_sdp_use_cudnn,
            "efficient_off": torch._C._set_sdp_use_mem_efficient,
        }[change](False)
    q, k, v = _qkv(2)

    out = opt.run_scaled_dot_product_attention(q, k, v)

    assert recorded_sdpa_kernel == [ALL_FOUR]
    with opt.sdpa_kernel(ALL_FOUR):
        assert torch.equal(out, torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0))


def test_full_set_skip_is_exact_when_overrideable_is_disabled(sdpa_selection, recorded_sdpa_kernel, restore_sdp_globals):
    torch._C._set_sdp_use_overrideable(False)
    opt.set_sdpa_backend("cudnn,flash,efficient,math")
    q, k, v = _qkv(3)

    out = opt.run_scaled_dot_product_attention(q, k, v)

    assert recorded_sdpa_kernel == []
    with opt.sdpa_kernel(ALL_FOUR):
        assert torch.equal(out, torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0))


class TinyAttnBlock(torch.nn.Module):
    """Same parameters and structure as sgm/ldm diffusionmodules.model.AttnBlock (single head, head_dim = channels)."""

    def __init__(self, channels):
        super().__init__()
        self.norm = torch.nn.GroupNorm(32, channels, eps=1e-6)
        self.q = torch.nn.Conv2d(channels, channels, 1)
        self.k = torch.nn.Conv2d(channels, channels, 1)
        self.v = torch.nn.Conv2d(channels, channels, 1)
        self.proj_out = torch.nn.Conv2d(channels, channels, 1)


def legacy_3d_attnblock_forward(self, x, sdpa_backend_override=None):
    """Oracle: the pre-PV1 sdp_attnblock_forward / sdp_no_mem_attnblock_forward with 3-D q/k/v."""
    h_ = self.norm(x)
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)
    b, c, h, w = q.shape
    q, k, v = (opt.rearrange(t, 'b c h w -> b (h w) c') for t in (q, k, v))
    dtype = q.dtype
    if opt.shared.opts.upcast_attn:
        q, k, v = q.float(), k.float(), v.float()
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = opt.run_scaled_dot_product_attention(q, k, v, is_causal=False, sdpa_backend_override=sdpa_backend_override)
    out = out.to(dtype)
    out = opt.rearrange(out, 'b (h w) c -> b c h w', h=h)
    out = self.proj_out(out)
    return x + out


def _attnblock_case(channels, dtype, channels_last, device="cpu", size=8, seed=0):
    torch.manual_seed(seed)
    block = TinyAttnBlock(channels).to(device=device, dtype=dtype).eval()
    x = torch.randn(2 if device == "cpu" else 1, channels, size, size, device=device, dtype=dtype)
    if channels_last:
        block = block.to(memory_format=torch.channels_last)
        x = x.contiguous(memory_format=torch.channels_last)
    return block, x


# Fused (online-softmax) kernel vs the math path: one rounding of the final bf16 residual add, or fp32 reassociation.
_FUSED_VS_MATH_TOLERANCE = {torch.float32: dict(rtol=1e-5, atol=1e-5), torch.bfloat16: dict(rtol=2 ** -7, atol=2 ** -7)}


@pytest.fixture
def upcast_attn():
    previous = opt.shared.opts.upcast_attn
    yield lambda value: setattr(opt.shared.opts, "upcast_attn", value)
    opt.shared.opts.upcast_attn = previous


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("upcast", [False, True])
def test_vae_attnblock_4d_layout_is_bitwise_identical_on_the_math_backend(sdpa_selection, upcast_attn, dtype, channels_last, upcast):
    # Same backend on both sides isolates the layout change: the 3-D and 4-D math paths run the same batched GEMMs.
    upcast_attn(upcast)
    opt.set_sdpa_backend("math")
    block, x = _attnblock_case(64, dtype, channels_last)
    with torch.no_grad():
        expected = legacy_3d_attnblock_forward(block, x)
        out = opt.sdp_attnblock_forward(block, x)
        out_no_mem = opt.sdp_no_mem_attnblock_forward(block, x)
        expected_no_mem = legacy_3d_attnblock_forward(block, x, sdpa_backend_override="flash,math")

    assert out.dtype == expected.dtype == dtype
    assert out.stride() == expected.stride()
    assert torch.equal(out, expected)
    # "flash,math" ignores the active selection; CPU flash now accepts the 4-D input, so this is close, not bitwise.
    torch.testing.assert_close(out_no_mem, expected_no_mem, **_FUSED_VS_MATH_TOLERANCE[dtype])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("upcast", [False, True])
def test_vae_attnblock_4d_default_selection_stays_close_to_legacy(sdpa_selection, upcast_attn, dtype, channels_last, upcast):
    # With every backend enabled the 4-D input becomes eligible for a fused kernel (CPU flash here, mem-efficient on
    # CUDA at head_dim 512) while the legacy 3-D input always ran math: numerically equivalent, not bitwise.
    upcast_attn(upcast)
    opt.set_sdpa_backend("cudnn,flash,efficient,math")
    block, x = _attnblock_case(64, dtype, channels_last, seed=1)
    with torch.no_grad():
        expected = legacy_3d_attnblock_forward(block, x)
        out = opt.sdp_attnblock_forward(block, x)

    assert out.stride() == expected.stride()
    torch.testing.assert_close(out, expected, **_FUSED_VS_MATH_TOLERANCE[dtype])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA: SDXL-VAE mid-block shape on the fused kernel")
def test_vae_attnblock_sdxl_shape_uses_mem_efficient_kernel_close_to_legacy_math_with_bounded_memory(sdpa_selection, upcast_attn):
    # SDXL VAE mid-block: 512 channels at a 128x128 latent (1024x1024 image), bf16, channels_last as deployed.
    upcast_attn(False)
    opt.set_sdpa_backend("cudnn,flash,efficient,math")
    block, x = _attnblock_case(512, torch.bfloat16, True, device="cuda", size=128, seed=2)
    tokens = x.shape[-2] * x.shape[-1]
    with torch.no_grad():
        h_ = block.norm(x)
        q, k, v = (opt.rearrange(m(h_), 'b c h w -> b 1 (h w) c').contiguous() for m in (block.q, block.k, block.v))
        params = torch.backends.cuda.SDPAParams(q, k, v, None, 0.0, False, False)
        assert torch.backends.cuda.can_use_efficient_attention(params, True)
        assert not torch.backends.cuda.can_use_flash_attention(params, False)  # head_dim 512 > 256
        del h_, q, k, v, params

        expected = legacy_3d_attnblock_forward(block, x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        out = opt.sdp_attnblock_forward(block, x)
        torch.cuda.synchronize()
        peak_delta = torch.cuda.max_memory_allocated() - baseline

    # Bound: at most one bf16 ULP of the residual output per element (rtol = atol = 2^-7), mean error far smaller.
    torch.testing.assert_close(out, expected, rtol=2 ** -7, atol=2 ** -7)
    assert (out.float() - expected.float()).abs().mean().item() < 2e-3
    # The legacy math path materialises fp32 L x L scores (tokens^2 * 4 B = 1 GiB here); the fused path stays O(L).
    assert peak_delta < tokens * tokens * 4 // 4


class _CpuWithoutAutocast:
    """devices.without_autocast for CPU tests: the real one switches CUDA autocast off; this switches CPU autocast off."""

    @staticmethod
    def without_autocast(disable=False):
        return torch.autocast("cpu", enabled=False) if torch.is_autocast_enabled("cpu") and not disable else contextlib.nullcontext()


def _sdpa_cross_attention(dim=64, heads=2):
    """TinyCrossAttention with the (Linear, Dropout) to_out pair scaled_dot_product_attention_forward indexes."""
    attn = TinyCrossAttention(dim=dim, heads=heads)
    attn.to_out.append(torch.nn.Dropout(0.0))
    return attn.to(torch.bfloat16).eval()


@pytest.fixture
def recorded_sdpa_calls(monkeypatch):
    """Record the q dtype and the autocast state at every scaled_dot_product_attention call.

    Autocast lists scaled_dot_product_attention as a lower-precision op on CUDA and CPU alike
    (ATen autocast_mode.h AT_FORALL_LOWER_PRECISION_FP; autocast_mode.cpp KERNEL_CPU): a float32 q/k/v reaching it
    with autocast on is cast back to bfloat16 before the kernel runs.
    """
    calls = []
    real = torch.nn.functional.scaled_dot_product_attention

    def recording(q, k, v, *args, **kwargs):
        calls.append((q.dtype, k.dtype, v.dtype, torch.is_autocast_enabled("cpu")))
        return real(q, k, v, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", recording)
    monkeypatch.setattr(opt, "devices", _CpuWithoutAutocast)
    return calls


@pytest.mark.parametrize("override", [None, "flash,math"])
def test_upcast_attn_runs_cross_attention_sdpa_in_float32_with_autocast_off(sdpa_selection, upcast_attn, recorded_sdpa_calls, override):
    # Before the fix the float32 q/k/v went into the kernel under autocast, which ran it in bfloat16: upcast_attn
    # was a no-op on the SDPA path. Oracle: the same forward without any autocast, where the bfloat16 Linears give
    # the same q/k/v and the attention really runs in float32.
    upcast_attn(True)
    opt.set_sdpa_backend("math")
    torch.manual_seed(0)
    attn = _sdpa_cross_attention()
    x = torch.randn(2, 16, 64, dtype=torch.bfloat16)
    context = torch.randn(2, 5, 64, dtype=torch.bfloat16)
    forward = opt.scaled_dot_product_attention_forward if override is None else opt.scaled_dot_product_no_mem_attention_forward
    with torch.no_grad():
        expected = forward(attn, x, context)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = forward(attn, x, context)
    assert recorded_sdpa_calls[-1] == (torch.float32, torch.float32, torch.float32, False)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, expected)


def test_without_upcast_attn_cross_attention_keeps_autocast(sdpa_selection, upcast_attn, recorded_sdpa_calls):
    upcast_attn(False)
    opt.set_sdpa_backend("math")
    attn = _sdpa_cross_attention()
    x = torch.randn(2, 16, 64, dtype=torch.bfloat16)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        opt.scaled_dot_product_attention_forward(attn, x)
    assert recorded_sdpa_calls == [(torch.bfloat16, torch.bfloat16, torch.bfloat16, True)]


@pytest.mark.parametrize("forward", ["sdp_attnblock_forward", "sdp_no_mem_attnblock_forward"])
@pytest.mark.parametrize("upcast", [False, True])
def test_vae_attnblock_upcast_attn_turns_autocast_off_for_the_kernel(sdpa_selection, upcast_attn, recorded_sdpa_calls, forward, upcast):
    upcast_attn(upcast)
    opt.set_sdpa_backend("math")
    block, x = _attnblock_case(64, torch.bfloat16, channels_last=False)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = getattr(opt, forward)(block, x)
    if upcast:
        assert recorded_sdpa_calls == [(torch.float32, torch.float32, torch.float32, False)]
    else:
        assert recorded_sdpa_calls == [(torch.bfloat16, torch.bfloat16, torch.bfloat16, True)]
    assert out.shape == x.shape
