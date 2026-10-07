"""sd_hijack_unet: bf16-native UNet norms and the sgm SpatialTransformer.forward hijack.

CPU tests cover which norms take the bf16-native path and when, and the SpatialTransformer hijack against the upstream
forward. The CUDA tests check the numerical claims in
modules/sd_hijack_unet.py on device; run them on the GPU host with:
    python -m pytest -q test/test_sd_hijack_unet.py -k cuda
"""
import contextlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

_ROOT = Path(__file__).resolve().parents[1]
for _repo in ("generative-models", "stable-diffusion-stability-ai"):
    _path = str(_ROOT / "repositories" / _repo)
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)  # what modules/paths.py does at startup

pytest.importorskip("sgm.modules.attention")
pytest.importorskip("ldm.modules.attention")


def _import_runtime_modules():
    """Import devices/sd_hijack_unet against the real modules.shared.

    Some test modules (test_openclaw_cuda_graphs.py) leave a minimal modules.shared stub in sys.modules; devices cannot
    import against it. Load the real module for these imports only and put the stub back for its owner."""
    import importlib
    import modules

    stub = sys.modules.get("modules.shared")
    if stub is None or getattr(stub, "__file__", None) is not None:
        from modules import devices, sd_hijack_unet
        return devices, sd_hijack_unet
    package_attribute = modules.__dict__.pop("shared", None)
    del sys.modules["modules.shared"]
    try:
        importlib.import_module("modules.shared")
        from modules import devices, sd_hijack_unet
    finally:
        sys.modules["modules.shared"] = stub
        if package_attribute is None:
            modules.__dict__.pop("shared", None)
        else:
            modules.shared = package_attribute
    return devices, sd_hijack_unet


devices, sd_hijack_unet = _import_runtime_modules()
import ldm.modules.attention as ldm_attention  # noqa: E402
import ldm.modules.diffusionmodules.model as ldm_vae  # noqa: E402
import ldm.modules.diffusionmodules.util as ldm_util  # noqa: E402
import sgm.modules.attention as sgm_attention  # noqa: E402
import sgm.modules.diffusionmodules.model as sgm_vae  # noqa: E402
import sgm.modules.diffusionmodules.util as sgm_util  # noqa: E402

BF16 = torch.bfloat16
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16-native norms only run on CUDA")


@pytest.fixture
def default_runtime(monkeypatch):
    """The production state the bf16-native path expects: no upcast, no quantized storage, no functional LoRA,
    no hypernetworks, switch on."""
    monkeypatch.setattr(sd_hijack_unet, "UNET_BF16_NATIVE_NORMS", True)
    monkeypatch.setattr(sd_hijack_unet, "shared", SimpleNamespace(opts=SimpleNamespace(lora_functional=False), loaded_hypernetworks=[]))
    for flag in ("unet_needs_upcast", "fp8", "mxfp8", "nvfp4"):
        monkeypatch.setattr(devices, flag, False)


def _cuda_autocast_state(enabled=True, dtype=BF16):
    return mock.patch.multiple(torch, is_autocast_enabled=mock.Mock(return_value=enabled), get_autocast_dtype=mock.Mock(return_value=dtype))


class _RecordingAutocast:
    """Stands in for torch.autocast so CPU tests can see whether the bf16-native path disabled CUDA autocast."""

    def __init__(self):
        self.calls = []
        self.active = False

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))

        @contextlib.contextmanager
        def context():
            self.active = True
            try:
                yield
            finally:
                self.active = False

        return context()


def _record_class_forward(monkeypatch, cls, autocast):
    """Replace cls.forward the way the Lora extension does and record each call's input dtype and autocast state."""
    seen = []
    original = cls.forward

    def lora_like_forward(self, input):
        seen.append((self, input.dtype, autocast.active))
        return original(self, input)

    monkeypatch.setattr(cls, "forward", lora_like_forward)
    return seen


def test_unet_norms_get_the_dispatching_classes_without_changing_state_dict_or_isinstance():
    for attention in (sgm_attention, ldm_attention):
        block = attention.BasicTransformerBlock(64, 2, 32, context_dim=32, checkpoint=False)
        assert [type(block.norm1), type(block.norm2), type(block.norm3)] == [sd_hijack_unet.UnetLayerNorm] * 3
        assert all(isinstance(norm, torch.nn.LayerNorm) for norm in (block.norm1, block.norm2, block.norm3))

        transformer = attention.SpatialTransformer(64, 2, 32, depth=2, context_dim=[32, 32], use_linear=True, use_checkpoint=False)
        assert type(transformer.norm) is sd_hijack_unet.UnetGroupNorm
        assert isinstance(transformer.norm, torch.nn.GroupNorm)
        assert all(type(b.norm1) is sd_hijack_unet.UnetLayerNorm for b in transformer.transformer_blocks)
        assert {"norm.weight", "norm.bias", "transformer_blocks.1.norm3.weight"} <= set(transformer.state_dict())

    for util in (sgm_util, ldm_util):
        assert type(util.normalization(64)) is util.GroupNorm32


def test_vae_and_clip_norms_keep_their_classes():
    for vae in (sgm_vae, ldm_vae):
        encoder = vae.Encoder(ch=32, out_ch=3, ch_mult=(1, 2), num_res_blocks=1, attn_resolutions=[4], in_channels=3, resolution=8, z_channels=4, double_z=False)
        norms = [m for m in encoder.modules() if isinstance(m, torch.nn.GroupNorm)]
        assert norms and all(type(m) is torch.nn.GroupNorm for m in norms)

    open_clip_transformer = pytest.importorskip("open_clip.transformer")
    block = open_clip_transformer.ResidualAttentionBlock(64, 2)
    layer_norms = [m for m in block.modules() if isinstance(m, torch.nn.LayerNorm)]
    assert layer_norms and not any(isinstance(m, sd_hijack_unet.UnetLayerNorm) for m in layer_norms)


def test_eligibility_requires_every_condition(default_runtime, monkeypatch):
    norm = torch.nn.GroupNorm(32, 64).to(BF16)
    x = SimpleNamespace(is_cuda=True, dtype=BF16)

    def eligible():
        with _cuda_autocast_state(), torch.no_grad():
            return sd_hijack_unet.bf16_native_norm_eligible(norm, x)

    assert eligible()

    with _cuda_autocast_state(enabled=False), torch.no_grad():
        assert not sd_hijack_unet.bf16_native_norm_eligible(norm, x)
    with _cuda_autocast_state(dtype=torch.float16), torch.no_grad():
        assert not sd_hijack_unet.bf16_native_norm_eligible(norm, x)
    with _cuda_autocast_state(), torch.enable_grad():
        assert not sd_hijack_unet.bf16_native_norm_eligible(norm, x)

    for attribute, value in (("is_cuda", False), ("dtype", torch.float32), ("dtype", torch.float16)):
        changed = SimpleNamespace(**{**vars(x), attribute: value})
        with _cuda_autocast_state(), torch.no_grad():
            assert not sd_hijack_unet.bf16_native_norm_eligible(norm, changed), attribute

    for flag in ("unet_needs_upcast", "fp8", "mxfp8", "nvfp4"):
        with monkeypatch.context() as patch:
            patch.setattr(devices, flag, True)
            assert not eligible(), flag

    with monkeypatch.context() as patch:
        patch.setattr(sd_hijack_unet.shared.opts, "lora_functional", True)
        assert not eligible()

    with monkeypatch.context() as patch:
        patch.setattr(sd_hijack_unet, "UNET_BF16_NATIVE_NORMS", False)
        assert not eligible()

    for name in ("weight", "bias"):
        mixed = torch.nn.GroupNorm(32, 64).to(BF16)
        getattr(mixed, name).data = getattr(mixed, name).data.float()
        with _cuda_autocast_state(), torch.no_grad():
            assert not sd_hijack_unet.bf16_native_norm_eligible(mixed, x), name

    no_affine = torch.nn.GroupNorm(32, 64, affine=False)
    with _cuda_autocast_state(), torch.no_grad():
        assert sd_hijack_unet.bf16_native_norm_eligible(no_affine, x)


def test_group_norm32_native_path_runs_class_forward_in_bf16_with_autocast_off(default_runtime, monkeypatch):
    autocast = _RecordingAutocast()
    seen = _record_class_forward(monkeypatch, torch.nn.GroupNorm, autocast)
    monkeypatch.setattr(torch, "autocast", autocast)
    x = torch.randn(2, 64, 4, 4, dtype=BF16)

    for util in (sgm_util, ldm_util):
        norm = util.normalization(64).to(BF16)
        torch.nn.init.normal_(norm.weight)
        torch.nn.init.normal_(norm.bias)
        expected = F.group_norm(x, 32, norm.weight, norm.bias, norm.eps)

        seen.clear()
        with mock.patch.object(sd_hijack_unet, "bf16_native_norm_eligible", return_value=True):
            out = norm(x)
        assert seen == [(norm, BF16, True)]
        assert autocast.calls[-1] == (("cuda",), {"enabled": False})
        assert out.dtype == BF16 and torch.equal(out, expected)

        # Not eligible: the original GroupNorm32 path (fp32 input, result cast back), still through the class forward.
        # (fp32 weights: on CPU nothing casts the bf16 weights for the fp32 input the way CUDA autocast does.)
        norm.float()
        seen.clear()
        calls_before = len(autocast.calls)
        with mock.patch.object(sd_hijack_unet, "bf16_native_norm_eligible", return_value=False):
            out = norm(x)
        assert seen == [(norm, torch.float32, False)]
        assert len(autocast.calls) == calls_before
        assert out.dtype == BF16


@pytest.mark.parametrize("attention", [sgm_attention, ldm_attention], ids=["sgm", "ldm"])
def test_spatial_transformer_and_block_norms_dispatch(default_runtime, monkeypatch, attention):
    autocast = _RecordingAutocast()
    monkeypatch.setattr(torch, "autocast", autocast)
    group_seen = _record_class_forward(monkeypatch, torch.nn.GroupNorm, autocast)
    layer_seen = _record_class_forward(monkeypatch, torch.nn.LayerNorm, autocast)
    transformer = attention.SpatialTransformer(64, 2, 32, depth=1, context_dim=32, use_linear=True, use_checkpoint=False).to(BF16)
    layer_norm = transformer.transformer_blocks[0].norm2
    x4 = torch.randn(2, 64, 4, 4, dtype=BF16)
    x3 = torch.randn(2, 16, 64, dtype=BF16)

    with mock.patch.object(sd_hijack_unet, "bf16_native_norm_eligible", return_value=True):
        assert torch.equal(transformer.norm(x4), F.group_norm(x4, 32, transformer.norm.weight, transformer.norm.bias, 1e-6))
        assert torch.equal(layer_norm(x3), F.layer_norm(x3, (64,), layer_norm.weight, layer_norm.bias, 1e-5))
    assert group_seen == [(transformer.norm, BF16, True)]
    assert layer_seen == [(layer_norm, BF16, True)]

    group_seen.clear()
    layer_seen.clear()
    with mock.patch.object(sd_hijack_unet, "bf16_native_norm_eligible", return_value=False):
        transformer.norm(x4)
        layer_norm(x3)
    assert group_seen == [(transformer.norm, BF16, False)]
    assert layer_seen == [(layer_norm, BF16, False)]


def test_layer_norm_keeps_autocast_path_with_hypernetworks_or_misaligned_operands(default_runtime, monkeypatch):
    autocast = _RecordingAutocast()
    monkeypatch.setattr(torch, "autocast", autocast)
    layer_seen = _record_class_forward(monkeypatch, torch.nn.LayerNorm, autocast)
    norm = torch.nn.LayerNorm(64).to(BF16)
    norm.__class__ = sd_hijack_unet.UnetLayerNorm
    storage = torch.randn(4 * 64 + 8, dtype=BF16)
    aligned = storage[:4 * 64].view(4, 64)
    misaligned = storage[1:1 + 4 * 64].view(4, 64)  # 2-byte offset
    non_contiguous_misaligned = storage[1:1 + 4 * 64].view(64, 4).t()  # LayerNorm copies it to an aligned buffer first

    def native(x):
        layer_seen.clear()
        with mock.patch.object(sd_hijack_unet, "bf16_native_norm_eligible", return_value=True):
            out = norm(x)
        assert torch.equal(out, F.layer_norm(x, (64,), norm.weight, norm.bias, norm.eps))
        return layer_seen[0][2]

    assert native(aligned)
    assert not native(misaligned)
    assert native(non_contiguous_misaligned)
    monkeypatch.setattr(sd_hijack_unet.shared, "loaded_hypernetworks", [object()])
    assert not native(aligned)


def upstream_sgm_spatial_transformer_forward(self, x, context=None):
    """Verbatim generative-models sgm/modules/attention.py SpatialTransformer.forward: the oracle for the hijack."""
    # note: if no context is given, cross-attention defaults to self-attention
    if not isinstance(context, list):
        context = [context]
    b, c, h, w = x.shape
    x_in = x
    x = self.norm(x)
    if not self.use_linear:
        x = self.proj_in(x)
    x = rearrange(x, "b c h w -> b (h w) c").contiguous()
    if self.use_linear:
        x = self.proj_in(x)
    for i, block in enumerate(self.transformer_blocks):
        if i > 0 and len(context) == 1:
            i = 0  # use same context for each block
        x = block(x, context=context[i])
    if self.use_linear:
        x = self.proj_out(x)
    x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
    if not self.use_linear:
        x = self.proj_out(x)
    return x + x_in


@pytest.mark.parametrize("use_linear", [True, False], ids=["linear", "conv"])
@pytest.mark.parametrize("memory_format", [torch.contiguous_format, torch.channels_last], ids=["nchw", "nhwc"])
def test_sgm_spatial_transformer_forward_matches_upstream(use_linear, memory_format):
    torch.manual_seed(0)
    transformer = sgm_attention.SpatialTransformer(64, 2, 32, depth=2, context_dim=48, use_linear=use_linear, use_checkpoint=False)
    with torch.no_grad():
        for parameter in transformer.parameters():
            parameter.normal_(0.0 if parameter.dim() > 1 else 1.0, 0.2)  # proj_out starts zeroed
    transformer = transformer.eval().to(memory_format=memory_format)
    x = torch.randn(2, 64, 6, 10).contiguous(memory_format=memory_format)
    first, second = torch.randn(2, 7, 48), torch.randn(2, 5, 48)

    for context in (first, [first], [first, second]):
        with torch.no_grad():
            hijacked = transformer(x, context=context)
            upstream = upstream_sgm_spatial_transformer_forward(transformer, x, context=context)
        assert torch.equal(hijacked, upstream)
        assert hijacked.stride() == upstream.stride()  # same result layout for whatever comes next

    # Autograd (training) takes the original expression; out= does not differentiate.
    with torch.enable_grad():
        x_grad = x.clone().requires_grad_(True)
        hijacked = transformer(x_grad, context=[first])
        hijacked.sum().backward()
    assert torch.equal(hijacked.detach(), upstream_sgm_spatial_transformer_forward(transformer, x, context=[first]).detach())
    assert x_grad.grad is not None


def test_sgm_spatial_transformer_single_context_feeds_every_block():
    transformer = sgm_attention.SpatialTransformer(64, 2, 32, depth=3, context_dim=48, use_linear=True, use_checkpoint=False).eval()
    seen = []
    for block in transformer.transformer_blocks:
        block.register_forward_pre_hook(lambda _module, _args, kwargs: seen.append(kwargs["context"]), with_kwargs=True)
    context = torch.randn(1, 3, 48)

    with torch.no_grad():
        transformer(torch.randn(1, 64, 4, 4), context=[context])

    assert len(seen) == 3 and all(c is context for c in seen)


# --- CUDA: numerical contract (path taken, dtype, error vs a float64 reference; LayerNorm bitwise) ---
# Distances in bf16 ULPs are meaningless for outputs near zero, where the affine terms cancel; errors are measured
# against a float64 reference relative to the output RMS, as consumed downstream (rounded to bf16).

def _error_over_rms(output, reference):
    """(max, mean) |output - reference| / rms(reference)."""
    diff = (output.double() - reference).abs()
    rms = reference.pow(2).mean().sqrt()
    return (diff.max() / rms).item(), (diff.mean() / rms).item()


def _assert_native_error_within_autocast_plus_rounding(native, autocast, reference):
    """The native output may not be worse than the autocast path's (both as consumed, i.e. rounded to bf16) by more
    than the reference's own bf16 rounding error."""
    native_max, native_mean = _error_over_rms(native, reference)
    autocast_max, autocast_mean = _error_over_rms(autocast.to(BF16), reference)
    floor_max, floor_mean = _error_over_rms(reference.to(BF16), reference)
    assert native_max <= autocast_max + floor_max, (native_max, autocast_max, floor_max)
    assert native_mean <= autocast_mean + 0.01 * floor_mean, (native_mean, autocast_mean, floor_mean)


def _bf16_eps(eps):
    return torch.tensor(eps, dtype=BF16).item()


def _random_activations(shape, memory_format=torch.contiguous_format):
    # Per-channel scales from 1e-3 to 10 and offsets of a few scales. For NCHW the scale is shared by each run of
    # channels / 32 channels (one GroupNorm group), so group variances span 1e-6..100, including the range where eps
    # matters most.
    generator = torch.Generator(device="cuda").manual_seed(1234)
    if len(shape) == 4:
        channels = shape[1]
        scale = torch.logspace(-3, 1, 32, device="cuda").repeat_interleave(channels // 32)
        view = (1, channels, 1, 1)
    else:
        channels = shape[-1]
        scale = torch.logspace(-3, 1, channels, device="cuda")
        view = (channels,)
    offset = torch.randn(channels, device="cuda", generator=generator) * scale * 3
    x = torch.randn(shape, device="cuda", generator=generator) * scale.view(view) + offset.view(view)
    return x.to(BF16).contiguous(memory_format=memory_format)


def _randomize(module):
    generator = torch.Generator().manual_seed(4321)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.5 + (1.0 if parameter.dim() == 1 else 0.0))
    return module


def _run(module, *args, native, **kwargs):
    with mock.patch.object(sd_hijack_unet, "UNET_BF16_NATIVE_NORMS", native), torch.no_grad(), torch.autocast("cuda", dtype=BF16):
        return module(*args, **kwargs)


@needs_cuda
@pytest.mark.parametrize("channels,size", [(320, 128), (640, 64), (1280, 32)])
@pytest.mark.parametrize("memory_format", [torch.contiguous_format, torch.channels_last], ids=["nchw", "nhwc"])
@pytest.mark.parametrize("kind", ["GroupNorm32", "SpatialTransformer.norm"])
def test_cuda_bf16_group_norm_native_path_matches_float64_like_autocast(default_runtime, channels, size, memory_format, kind):
    if kind == "GroupNorm32":
        norm = sgm_util.normalization(channels)
    else:
        norm = sgm_attention.Normalize(channels)
        norm.__class__ = sd_hijack_unet.UnetGroupNorm
    norm = _randomize(norm).to("cuda", BF16).to(memory_format=memory_format)
    x = _random_activations((2, channels, size, size), memory_format)

    with torch.no_grad(), torch.autocast("cuda", dtype=BF16):
        assert sd_hijack_unet.bf16_native_norm_eligible(norm, x)
    native = _run(norm, x, native=True)
    autocast = _run(norm, x, native=False)
    with torch.no_grad():
        oracle = F.group_norm(x.float(), 32, norm.weight.float(), norm.bias.float(), _bf16_eps(norm.eps)).to(BF16)
        reference = F.group_norm(x.double(), 32, norm.weight.double(), norm.bias.double(), norm.eps)

    assert native.dtype == BF16  # the native path ran (autocast leaves fp32 for SpatialTransformer.norm)
    assert torch.equal(native, oracle)  # ATen's bf16 kernel: fp32 math with eps cast to bf16, one rounding
    _assert_native_error_within_autocast_plus_rounding(native, autocast, reference)


@needs_cuda
@pytest.mark.parametrize("channels,tokens", [(640, 4096), (1280, 1024)])
@pytest.mark.parametrize("offset", [0, 2, 4, 8])
def test_cuda_bf16_layer_norm_is_bitwise_equal_to_autocast_when_aligned(default_runtime, channels, tokens, offset):
    norm = torch.nn.LayerNorm(channels)
    norm.__class__ = sd_hijack_unet.UnetLayerNorm
    norm = _randomize(norm).to("cuda", BF16)
    full = _random_activations((2, tokens, channels))
    storage = torch.empty(full.numel() + offset, device="cuda", dtype=BF16)
    x = storage[offset:].view_as(full)
    x.copy_(full)
    aligned = x.data_ptr() % 16 == 0

    native = _run(norm, x, native=True)
    autocast = _run(norm, x, native=False)
    with torch.no_grad():
        reference = F.layer_norm(x.double(), (channels,), norm.weight.double(), norm.bias.double(), norm.eps)

    assert autocast.dtype == torch.float32
    if aligned:
        assert native.dtype == BF16  # the native path ran
        assert torch.equal(native, autocast.to(BF16))  # the consumer Linear's autocast cast
        _assert_native_error_within_autocast_plus_rounding(native, autocast, reference)
    else:
        # A misaligned bf16 operand would vectorize differently from autocast's aligned fp32 copy: keep autocast.
        assert native.dtype == torch.float32 and torch.equal(native, autocast)


@needs_cuda
def test_cuda_transformer_block_is_bitwise_equal_to_autocast(default_runtime):
    block = _randomize(sgm_attention.BasicTransformerBlock(640, 10, 64, context_dim=2048, checkpoint=False)).to("cuda", BF16)
    x = _random_activations((2, 1024, 640))
    context = _random_activations((2, 77, 2048))

    assert torch.equal(_run(block, x, context=context, native=True), _run(block, x, context=context, native=False))


@needs_cuda
@pytest.mark.parametrize("memory_format", [torch.contiguous_format, torch.channels_last], ids=["nchw", "nhwc"])
def test_cuda_spatial_transformer_and_resblock_differ_only_by_group_norm_eps(default_runtime, memory_format):
    import copy
    import importlib

    # Earlier test files may leave a stub openaimodel module in sys.modules; import the real one for this test only.
    with mock.patch.dict(sys.modules):
        if not getattr(sys.modules.get("sgm.modules.diffusionmodules.openaimodel"), "__file__", None):
            for name in [name for name in sys.modules if name == "sgm" or name.startswith("sgm.")]:
                if not getattr(sys.modules[name], "__file__", None) and not getattr(sys.modules[name], "__path__", None):
                    del sys.modules[name]
            sys.modules.pop("sgm.modules.diffusionmodules.openaimodel", None)
        ResBlock = importlib.import_module("sgm.modules.diffusionmodules.openaimodel").ResBlock

    transformer = sgm_attention.SpatialTransformer(640, 10, 64, depth=2, context_dim=2048, use_linear=True, use_checkpoint=False)
    resblock = ResBlock(640, 1280, 0.0, out_channels=640)
    x = _random_activations((2, 640, 32, 32), memory_format)
    context = _random_activations((2, 77, 2048))
    emb = _random_activations((2, 1280))

    for module, args in ((transformer, (x, [context])), (resblock, (x, emb))):
        module = _randomize(module).to("cuda", BF16).to(memory_format=memory_format)
        reference = copy.deepcopy(module)
        for norm in reference.modules():
            if isinstance(norm, torch.nn.GroupNorm):
                norm.eps = _bf16_eps(norm.eps)

        assert torch.equal(_run(module, *args, native=True), _run(reference, *args, native=False))
