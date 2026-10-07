import contextlib
import importlib.util
import sys
import types
import warnings
from types import SimpleNamespace

import pytest
import torch


_ATTENTION_STUB_MODULES = []


def _has_module_spec(name):
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def _setdefault_attention_stub(name, module):
    if name not in sys.modules:
        sys.modules[name] = module
        _ATTENTION_STUB_MODULES.append(name)
    return sys.modules[name]


def _cleanup_attention_import_stubs():
    for name in reversed(_ATTENTION_STUB_MODULES):
        module = sys.modules.pop(name, None)
        if "." not in name:
            continue
        parent_name, attr = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None and getattr(parent, attr, None) is module:
            delattr(parent, attr)


def _install_attention_import_stubs():
    shared = types.ModuleType("modules.shared")
    shared.opts = SimpleNamespace(upcast_attn=False)
    shared.cmd_opts = SimpleNamespace(sub_quad_q_chunk_size=1024, sub_quad_kv_chunk_size=None, sub_quad_chunk_threshold=None)
    shared.device = torch.device("cpu")
    shared.loaded_hypernetworks = []

    devices = types.ModuleType("modules.devices")
    devices.without_autocast = lambda disable=False: contextlib.nullcontext()

    sub_quadratic_attention = types.ModuleType("modules.sub_quadratic_attention")
    sub_quadratic_attention.efficient_dot_product_attention = lambda q, k, v, **_kwargs: torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

    hypernetworks_pkg = types.ModuleType("modules.hypernetworks")
    hypernetwork = types.ModuleType("modules.hypernetworks.hypernetwork")
    hypernetwork.attention_CrossAttention_forward = lambda self, x, context=None, mask=None, **kwargs: x
    hypernetwork.apply_hypernetworks = lambda _nets, context: (context, context)
    hypernetworks_pkg.hypernetwork = hypernetwork

    if not _has_module_spec("ldm.modules.attention") or not _has_module_spec("sgm.modules.attention"):
        ldm_pkg = types.ModuleType("ldm")
        ldm_pkg.__path__ = []
        ldm_util = types.ModuleType("ldm.util")
        ldm_util.default = lambda value, default_value: value if value is not None else (default_value() if callable(default_value) else default_value)
        ldm_modules = types.ModuleType("ldm.modules")
        ldm_modules.__path__ = []
        ldm_attention = types.ModuleType("ldm.modules.attention")
        ldm_diffusionmodules = types.ModuleType("ldm.modules.diffusionmodules")
        ldm_diffusionmodules.__path__ = []
        ldm_diffusion_model = types.ModuleType("ldm.modules.diffusionmodules.model")
        sgm_pkg = types.ModuleType("sgm")
        sgm_pkg.__path__ = []
        sgm_modules = types.ModuleType("sgm.modules")
        sgm_modules.__path__ = []
        sgm_attention = types.ModuleType("sgm.modules.attention")
        sgm_diffusionmodules = types.ModuleType("sgm.modules.diffusionmodules")
        sgm_diffusionmodules.__path__ = []
        sgm_diffusion_model = types.ModuleType("sgm.modules.diffusionmodules.model")

        class StubCrossAttention:
            forward = staticmethod(lambda self, x, context=None, mask=None, **kwargs: x)

        class StubAttnBlock:
            forward = staticmethod(lambda self, x, **kwargs: x)

        ldm_attention.CrossAttention = StubCrossAttention
        ldm_diffusion_model.AttnBlock = StubAttnBlock
        sgm_attention.CrossAttention = StubCrossAttention
        sgm_diffusion_model.AttnBlock = StubAttnBlock
        ldm_pkg.util = ldm_util
        ldm_pkg.modules = ldm_modules
        ldm_modules.attention = ldm_attention
        ldm_modules.diffusionmodules = ldm_diffusionmodules
        ldm_diffusionmodules.model = ldm_diffusion_model
        sgm_pkg.modules = sgm_modules
        sgm_modules.attention = sgm_attention
        sgm_modules.diffusionmodules = sgm_diffusionmodules
        sgm_diffusionmodules.model = sgm_diffusion_model

        _setdefault_attention_stub("ldm", ldm_pkg)
        _setdefault_attention_stub("ldm.util", ldm_util)
        _setdefault_attention_stub("ldm.modules", ldm_modules)
        _setdefault_attention_stub("ldm.modules.attention", ldm_attention)
        _setdefault_attention_stub("ldm.modules.diffusionmodules", ldm_diffusionmodules)
        _setdefault_attention_stub("ldm.modules.diffusionmodules.model", ldm_diffusion_model)
        _setdefault_attention_stub("sgm", sgm_pkg)
        _setdefault_attention_stub("sgm.modules", sgm_modules)
        _setdefault_attention_stub("sgm.modules.attention", sgm_attention)
        _setdefault_attention_stub("sgm.modules.diffusionmodules", sgm_diffusionmodules)
        _setdefault_attention_stub("sgm.modules.diffusionmodules.model", sgm_diffusion_model)
    _setdefault_attention_stub("modules.shared", shared)
    _setdefault_attention_stub("modules.devices", devices)
    _setdefault_attention_stub("modules.sub_quadratic_attention", sub_quadratic_attention)
    _setdefault_attention_stub("modules.hypernetworks", hypernetworks_pkg)
    _setdefault_attention_stub("modules.hypernetworks.hypernetwork", hypernetwork)


_install_attention_import_stubs()
from modules import sd_hijack_optimizations as opt
_cleanup_attention_import_stubs()


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


if __name__ == "__main__":
    test_doggettx_attention_keeps_positive_slice_when_memory_steps_exceed_tokens()
    test_sdpa_math_backend_uses_non_deprecated_torch_nn_attention_api()
    print("PASS attention slice bounds and SDPA deprecation smoke")
