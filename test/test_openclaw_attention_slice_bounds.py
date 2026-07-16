import contextlib
import importlib.util
import sys
import types
import warnings
from types import SimpleNamespace

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


if __name__ == "__main__":
    test_doggettx_attention_keeps_positive_slice_when_memory_steps_exceed_tokens()
    test_sdpa_math_backend_uses_non_deprecated_torch_nn_attention_api()
    print("PASS attention slice bounds and SDPA deprecation smoke")
