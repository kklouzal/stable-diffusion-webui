"""Layout folds (modules/openclaw_layout_folds.py): SiLU written channels_last for the conv after it, ResBlock
`h + emb_out` written NCHW for the GroupNorm after it, in the sgm ResBlock and the sgm VAE ResnetBlock.

CPU tests: the switch (default off, invalid value fails the import, off installs nothing); with the folds installed,
the sgm ResBlock (identity and conv skip), VAE ResnetBlock (identity and nin shortcut) and a VAE decoder in
channels_last give torch.equal outputs with the same strides as the upstream forwards, while the convs now receive
channels_last input and the out GroupNorm NCHW input (CPU group_norm made to copy like ATen's CUDA one); every fallback
(NCHW conv weights, autograd, hooks, sgm's own swish, the NHWC GroupNorm scopes) runs the upstream forward.
CUDA tests (GPU host): the same blocks at SDXL UNet and VAE production sizes, bitwise against the switch-off forwards:
    python -m pytest -q test/test_openclaw_layout_folds.py -k cuda
Timing and copy-kernel counts: tools/benchmark_layout_folds.py.
"""
import copy
import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from test.helpers import ROOT as _ROOT, add_repositories_to_sys_path, randomize

add_repositories_to_sys_path("generative-models", "stable-diffusion-stability-ai")
pytest.importorskip("sgm.modules.attention")

from modules import openclaw_gn_transpose as gn_transpose, openclaw_layout_folds as folds, openclaw_nhwc_groupnorm as nhwc, sd_hijack_unet  # noqa: E402
import sgm.modules.diffusionmodules.model as sgm_vae  # noqa: E402
import sgm.modules.diffusionmodules.openaimodel as sgm_openaimodel  # noqa: E402

BF16 = torch.bfloat16
CL = torch.channels_last
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the production layouts are CUDA's")


def _import_with(value):
    env = {key: item for key, item in os.environ.items() if key != folds.ENV_NAME}
    if value is not None:
        env[folds.ENV_NAME] = value
    code = (
        "import sys; sys.path.insert(0, 'repositories/generative-models'); "
        "from modules import openclaw_layout_folds as f; print(f.ENABLED)"
    )
    return subprocess.run([sys.executable, "-c", code], cwd=_ROOT, env=env, capture_output=True, text=True, timeout=300)


def test_switch_is_off_by_default_and_an_invalid_value_fails_the_import():
    for value, expected in ((None, "False"), ("", "False"), ("0", "False"), ("1", "True"), ("yes", "True")):
        process = _import_with(value)
        assert process.returncode == 0 and process.stdout.split("\n")[-2] == expected, process.stderr[-2000:]
    process = _import_with("2")
    assert process.returncode != 0 and "OPENCLAW_LAYOUT_FOLDS='2' is not a boolean" in process.stderr


def test_off_installs_nothing(monkeypatch):
    monkeypatch.setattr(sgm_openaimodel.ResBlock, "_forward", _below_folds(sgm_openaimodel.ResBlock._forward))
    monkeypatch.setattr(sgm_vae.ResnetBlock, "forward", _below_folds(sgm_vae.ResnetBlock.forward))
    monkeypatch.setattr(folds, "ENABLED", False)
    before = (sgm_openaimodel.ResBlock._forward, sgm_vae.ResnetBlock.forward)
    folds.install()
    assert (sgm_openaimodel.ResBlock._forward, sgm_vae.ResnetBlock.forward) == before


def _below_folds(forward):
    """The forward below the folds' CondFunc when OPENCLAW_LAYOUT_FOLDS=1 installed it at import, else `forward`."""
    for cell in forward.__closure__ or ():
        condfunc = cell.cell_contents
        if getattr(condfunc, "_CondFunc__sub_func", None) in (folds.sgm_resblock_forward, folds.sgm_vae_resnet_block_forward):
            return condfunc._CondFunc__orig_func
    return forward


@pytest.fixture
def upstream(monkeypatch):
    """The forwards installed without the folds (restored after the test), to compare the folds against."""
    monkeypatch.setattr(sgm_openaimodel.ResBlock, "_forward", _below_folds(sgm_openaimodel.ResBlock._forward))
    monkeypatch.setattr(sgm_vae.ResnetBlock, "forward", _below_folds(sgm_vae.ResnetBlock.forward))
    monkeypatch.setattr(sgm_vae, "nonlinearity", F.silu)  # what sd_hijack.apply_optimizations installs
    for name, value in (("_SCOPES", frozenset()), ("_STATE_KEY", ())):
        monkeypatch.setattr(nhwc, name, value)
    return monkeypatch


def _install(monkeypatch):
    monkeypatch.setattr(folds, "ENABLED", True)
    folds.install()
    assert _below_folds(sgm_openaimodel.ResBlock._forward) is not sgm_openaimodel.ResBlock._forward


@pytest.fixture
def cuda_like(upstream, default_runtime):
    """CPU group_norm as ATen's CUDA one computes it (an NCHW copy of the input, NCHW output), the fold's device seam,
    and the bf16-native norm eligibility minus its CUDA conditions."""
    group_norm = F.group_norm
    upstream.setattr(F, "group_norm", lambda x, *args, **kwargs: group_norm(x.contiguous(), *args, **kwargs))
    upstream.setattr(folds, "_gn_reads_nchw", lambda x: True)
    upstream.setattr(sd_hijack_unet, "bf16_native_norm_eligible", lambda module, x: x.dtype == BF16 and not torch.is_grad_enabled())
    return upstream


def _layout(t):
    if t.is_contiguous(memory_format=CL) and not t.is_contiguous():
        return "NHWC"
    return "NCHW" if t.is_contiguous() else "other"


def _record_inputs(*modules):
    """Pre-hooks recording each module's input layout; returns (log, remove)."""
    log = []
    handles = [m.register_forward_pre_hook(lambda m, args, name=name: log.append((name, _layout(args[0])))) for name, m in modules]
    return log, lambda: [h.remove() for h in handles]


def _input(shape, memory_format=CL, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=generator) * 2 + 0.5).to(BF16).contiguous(memory_format=memory_format)


def _resblock(channels=64, out_channels=128, memory_format=CL):
    block = randomize(sgm_openaimodel.ResBlock(channels, 128, 0.0, out_channels=out_channels)).eval().to(BF16)
    return block.to(memory_format=memory_format)


def _vae_block(out_channels, memory_format=CL):
    block = sgm_vae.ResnetBlock(in_channels=64, out_channels=out_channels, dropout=0.0, temb_channels=0)
    return randomize(block).eval().to(BF16).to(memory_format=memory_format)


def _resblock_watch(block):
    return (("in_norm", block.in_layers[0]), ("in_conv", block.in_layers[2]), ("out_norm", block.out_layers[0]), ("out_conv", block.out_layers[3]))


def _run(module, args, watch):
    log, remove = _record_inputs(*watch)
    try:
        with torch.no_grad():
            return module(*args), log
    finally:
        remove()


@pytest.mark.parametrize("out_channels", [64, 128], ids=["identity-skip", "conv-skip"])
def test_resblock_folds_are_bitwise_and_remove_both_copies(cuda_like, out_channels):
    block = _resblock(out_channels=out_channels)
    args = (_input((2, 64, 8, 12)), _input((2, 128), torch.contiguous_format, 1))
    off, off_log = _run(block, args, _resblock_watch(block))
    _install(cuda_like)
    on, on_log = _run(block, args, _resblock_watch(block))
    assert torch.equal(off, on) and off.stride() == on.stride()
    assert off_log == [("in_norm", "NHWC"), ("in_conv", "NCHW"), ("out_norm", "NHWC"), ("out_conv", "NCHW")]
    assert on_log == [("in_norm", "NHWC"), ("in_conv", "NHWC"), ("out_norm", "NCHW"), ("out_conv", "NHWC")]


@pytest.mark.parametrize("out_channels", [64, 128], ids=["identity", "nin-shortcut"])
def test_vae_resnet_block_folds_are_bitwise(cuda_like, out_channels):
    block = _vae_block(out_channels)
    watch = (("conv1", block.conv1), ("conv2", block.conv2))
    x = _input((1, 64, 8, 12))
    off, off_log = _run(block, (x, None), watch)
    _install(cuda_like)
    on, on_log = _run(block, (x, None), watch)
    assert torch.equal(off, on) and off.stride() == on.stride()
    assert off_log == [("conv1", "NCHW"), ("conv2", "NCHW")] and on_log == [("conv1", "NHWC"), ("conv2", "NHWC")]


def test_vae_decoder_end_to_end(cuda_like):
    decoder = sgm_vae.Decoder(ch=32, out_ch=3, ch_mult=(1, 2), num_res_blocks=1, attn_resolutions=[8], in_channels=3, resolution=16, z_channels=4)
    decoder = randomize(decoder).eval().to(BF16).to(memory_format=CL)
    x = _input((1, 4, 8, 8))
    with torch.no_grad():
        off = decoder(x)
        _install(cuda_like)
        on = decoder(x)
    assert torch.equal(off, on)


def test_folds_compose_with_the_fast_transpose(cuda_like):
    """With both switches the out GroupNorm already gets NCHW input: only in_layers[0] still needs the transpose."""
    calls = []
    cuda_like.setattr(gn_transpose, "ENABLED", True)
    cuda_like.setattr(gn_transpose, "_on_cuda", lambda x: True)
    cuda_like.setattr(gn_transpose, "transpose", lambda x: calls.append(x.shape) or x.contiguous())
    block = _resblock()
    args = (_input((2, 64, 8, 12)), _input((2, 128), torch.contiguous_format, 1))
    with torch.no_grad():
        off = block(*args)
        assert calls == [(2, 64, 8, 12), (2, 128, 8, 12)]
        _install(cuda_like)
        on = block(*args)
    assert torch.equal(off, on) and calls[2:] == [(2, 64, 8, 12)]


def test_nchw_conv_weights_keep_the_upstream_layouts(cuda_like):
    block = _resblock(memory_format=torch.contiguous_format)
    args = (_input((2, 64, 8, 12), torch.contiguous_format), _input((2, 128), torch.contiguous_format, 1))
    off, off_log = _run(block, args, _resblock_watch(block))
    _install(cuda_like)
    on, on_log = _run(block, args, _resblock_watch(block))
    assert torch.equal(off, on) and on_log == off_log and all(layout == "NCHW" for _, layout in on_log)


def test_cpu_group_norm_keeps_the_upstream_add(upstream, default_runtime):
    """On CPU ATen's group_norm reads channels_last as it is: the add stays channels_last (the SiLU fold still applies)."""
    upstream.setattr(sd_hijack_unet, "bf16_native_norm_eligible", lambda module, x: x.dtype == BF16 and not torch.is_grad_enabled())
    block = _resblock()
    args = (_input((2, 64, 8, 12)), _input((2, 128), torch.contiguous_format, 1))
    off, _ = _run(block, args, ())
    _install(upstream)
    on, log = _run(block, args, _resblock_watch(block))
    assert torch.equal(off, on) and ("out_norm", "NHWC") in log


def _assert_upstream(block, args, monkeypatch, expected_log):
    off, _ = _run(block, args, ())
    _install(monkeypatch)
    on, log = _run(block, args, _resblock_watch(block))
    assert torch.equal(off, on) and log == expected_log


_UPSTREAM_LOG = [("in_norm", "NHWC"), ("in_conv", "NCHW"), ("out_norm", "NHWC"), ("out_conv", "NCHW")]


def test_hooked_silu_runs_the_upstream_forward(cuda_like):
    block = _resblock()
    seen = []
    block.in_layers[1].register_forward_hook(lambda m, args, out: seen.append(out.shape))
    _assert_upstream(block, (_input((2, 64, 8, 12)), _input((2, 128), torch.contiguous_format, 1)), cuda_like, _UPSTREAM_LOG)
    assert len(seen) == 2


def test_nhwc_groupnorm_unet_scope_runs_the_upstream_forward(cuda_like):
    block = _resblock()
    args = (_input((2, 64, 8, 12)), _input((2, 128), torch.contiguous_format, 1))
    cuda_like.setattr(nhwc, "_SCOPES", frozenset({nhwc.UNET}))
    cuda_like.setattr(nhwc, "group_norm", lambda module, x, act=None: None)  # the kernels' "not taken" answer
    _assert_upstream(block, args, cuda_like, _UPSTREAM_LOG)


def test_autograd_runs_the_upstream_forward(cuda_like):
    block = _resblock().float()
    x = _input((2, 64, 8, 12)).float().requires_grad_()
    emb = _input((2, 128), torch.contiguous_format, 1).float()
    off = block(x, emb)
    _install(cuda_like)
    log, remove = _record_inputs(*_resblock_watch(block))
    on = block(x, emb)
    remove()
    assert torch.equal(off, on) and on.requires_grad and log == _UPSTREAM_LOG


def test_vae_block_with_sgm_swish_runs_the_upstream_forward(cuda_like):
    swish = sd_hijack_unet._VAE_SWISH_FUNCTIONS[0]
    cuda_like.setattr(sgm_vae, "nonlinearity", swish)
    block = _vae_block(64)
    watch = (("conv1", block.conv1), ("conv2", block.conv2))
    x = _input((1, 64, 8, 12))
    off, _ = _run(block, (x, None), watch)
    _install(cuda_like)
    on, log = _run(block, (x, None), watch)
    assert torch.equal(off, on) and log == [("conv1", "NCHW"), ("conv2", "NCHW")]


# --- CUDA ---------------------------------------------------------------------------------------------------------

def _cuda(module):
    return randomize(module).eval().to("cuda", BF16).to(memory_format=CL)


def _cuda_input(shape, seed=0, memory_format=CL):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(shape, generator=generator, device="cuda") * 2 + 0.5).to(BF16).contiguous(memory_format=memory_format)


@needs_cuda
@pytest.mark.parametrize("channels,out_channels,size", [(320, 320, 160), (320, 640, 80), (640, 640, 80), (640, 1280, 40), (1280, 1280, 40), (2560, 1280, 40), (1920, 1280, 40)])
def test_cuda_resblock_is_bitwise_unchanged(upstream, default_runtime, channels, out_channels, size):
    """SDXL UNet/ControlNet ResBlocks at n-w1 latent sizes (160² latents, CFG batch 2), bf16 autocast, bf16-native norms."""
    block = _cuda(sgm_openaimodel.ResBlock(channels, 1280, 0.0, out_channels=out_channels))
    reference = copy.deepcopy(block)
    x = _cuda_input((2, channels, size, size))
    emb = _cuda_input((2, 1280), 1, torch.contiguous_format)
    with torch.inference_mode(), torch.autocast("cuda", dtype=BF16):
        off = reference(x, emb)
        _install(upstream)
        on = block(x, emb)
    assert on.stride() == off.stride() and torch.equal(on, off)


@needs_cuda
@pytest.mark.parametrize("channels,out_channels,size", [(128, 128, 1280), (256, 128, 1280), (256, 256, 640), (512, 256, 640), (512, 512, 320), (512, 512, 160)])
def test_cuda_vae_resnet_block_is_bitwise_unchanged(upstream, channels, out_channels, size):
    """sgm VAE ResnetBlocks at the n-w1 1280² decode sizes, autocast off as the VAE runs."""
    block = _cuda(sgm_vae.ResnetBlock(in_channels=channels, out_channels=out_channels, dropout=0.0, temb_channels=0))
    reference = copy.deepcopy(block)
    x = _cuda_input((1, channels, size, size))
    with torch.inference_mode():
        off = reference(x, None)
        _install(upstream)
        on = block(x, None)
    assert on.stride() == off.stride() and torch.equal(on, off)
