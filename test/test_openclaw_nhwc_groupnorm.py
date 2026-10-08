"""NHWC GroupNorm switch: modules/openclaw_nhwc_groupnorm.py, its kernels and its hijacks in modules/sd_hijack_unet.py.

CPU tests:
- dispatch, with a stand-in launch that computes exactly the torch ops it replaces (F.group_norm of the same tensor,
  then F.silu), so a fused forward must reproduce the upstream forward bitwise: switch off changes nothing, where the
  kernels run and with which activation, output layouts, every fallback, the Lora chain, the ControlNet weight layout,
  the scope parsing;
- the real kernels in Triton's interpreter: indexing, tails, channel chunks, sub-tile merging, accuracy against float64
  at any mean/std ratio, run-to-run identity; and their PTX for GB10 (sm_121): no atomics, vector loads, rounding.
CUDA tests (GPU host): accuracy within torch's bf16 path at SDXL shapes and at any mean/std ratio, bitwise run-to-run
and CUDA-graph replay identity, SDXL-sized blocks against the switch-off path:
    python -m pytest -q test/test_openclaw_nhwc_groupnorm.py -k cuda
Timing: test/benchmark_nhwc_groupnorm.py.
"""
import copy
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from test.test_sd_hijack_unet import (  # noqa: F401  (default_runtime is a fixture)
    BF16,
    _error_over_rms,
    _random_activations,
    _randomize,
    default_runtime,
    sd_hijack_unet,
    sgm_attention,
    sgm_util,
    sgm_vae,
)
from test.test_openclaw_lora_network_identity import lora_networks  # noqa: F401  (fixture)
from modules import openclaw_nhwc_groupnorm as nhwc

_ROOT = Path(__file__).resolve().parents[1]
sgm_openaimodel = importlib.import_module("sgm.modules.diffusionmodules.openaimodel")
CL = torch.channels_last
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the NHWC GroupNorm kernels run on CUDA only")


def _cpu_eligible(module, x):
    """sd_hijack_unet.bf16_native_norm_eligible minus its CUDA-device and CUDA-autocast conditions."""
    return x.dtype == BF16 and not torch.is_grad_enabled() and all(t is None or t.dtype == BF16 for t in (module.weight, module.bias))


@pytest.fixture
def switch(monkeypatch):
    """The switch's process state, restored after the test."""
    for name, value in (("_SCOPES", frozenset()), ("_STATE_KEY", ()), ("_ERROR", None)):
        monkeypatch.setattr(nhwc, name, value)
    monkeypatch.setattr(nhwc, "_KERNELS", nhwc._KERNELS)
    nhwc.reset_counters()
    yield
    nhwc.reset_counters()


@pytest.fixture
def kernel(monkeypatch, switch, default_runtime):  # noqa: F811  (the imported fixture)
    """Stand-in launch: records each call and returns what the torch path computes from the same tensor."""
    assert nhwc._is_torch_group_norm_forward(torch.nn.GroupNorm.forward), "an earlier test left GroupNorm.forward patched"
    calls = []

    def launch(kernels, x, groups, weight, bias, eps, silu):
        assert kernels is sentinel and nhwc.is_nhwc(x)
        calls.append(SimpleNamespace(channels=x.shape[1], act="silu" if silu else "", eps=eps, bias=bias))
        y = F.group_norm(x, groups, weight, bias, eps)
        return F.silu(y) if silu else y

    sentinel = object()
    monkeypatch.setattr(nhwc, "_KERNELS", sentinel)
    monkeypatch.setattr(nhwc, "_launch", launch)
    monkeypatch.setattr(nhwc, "_on_cuda", lambda x: True)
    monkeypatch.setattr(sd_hijack_unet, "bf16_native_norm_eligible", _cpu_eligible)
    yield calls


def _bf16_cl(module):
    return _randomize(module).eval().to(BF16).to(memory_format=CL)


def _input(shape, memory_format=CL, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=generator) * 2 + 0.5).to(BF16).contiguous(memory_format=memory_format)


def _off_and_on(module, *args, scopes="all"):
    with torch.no_grad():
        nhwc.set_scopes("")
        off = module(*args)
        nhwc.set_scopes(scopes)
        on = module(*args)
        nhwc.set_scopes("")
    return off, on


def _resblock(channels=64, out_channels=128, memory_format=CL):
    block = _randomize(sgm_openaimodel.ResBlock(channels, 128, 0.0, out_channels=out_channels)).eval().to(BF16)
    return block.to(memory_format=memory_format)


def _transformer(memory_format=CL):
    transformer = _randomize(sgm_attention.SpatialTransformer(64, 2, 32, depth=1, context_dim=48, use_linear=True, use_checkpoint=False))
    return transformer.eval().to(BF16).to(memory_format=memory_format)


def _context(seed=2):
    return [_input((2, 5, 48), torch.contiguous_format, seed)]


# --- registration and switch-off identity -------------------------------------------------------------------------

def test_hijacks_are_installed_on_the_modules_sdxl_runs():
    assert "CondFunc" in sgm_openaimodel.ResBlock._forward.__qualname__
    assert "CondFunc" in sgm_vae.ResnetBlock.forward.__qualname__
    assert sgm_vae is sd_hijack_unet._SGM_VAE
    encoder = sgm_vae.Encoder(ch=32, out_ch=3, ch_mult=(1, 2), num_res_blocks=1, attn_resolutions=[4], in_channels=3, resolution=8, z_channels=4, double_z=False)
    norms = [m for m in encoder.modules() if isinstance(m, torch.nn.GroupNorm)]
    assert norms and all(type(m) is sd_hijack_unet.VaeGroupNorm and m.eps == 1e-6 for m in norms)
    assert set(norms[0].state_dict()) == set(torch.nn.GroupNorm(32, 32, eps=1e-6).state_dict())
    assert sd_hijack_unet._VAE_SWISH_FUNCTIONS[-1] is F.silu  # what sd_hijack.apply_optimizations installs


def test_switch_off_runs_the_upstream_paths_without_calling_the_kernels(kernel):
    x = _input((2, 64, 8, 12))
    vae_block = _bf16_cl(sgm_vae.ResnetBlock(in_channels=64, out_channels=128, dropout=0.0, temb_channels=0))
    with torch.no_grad():
        _resblock()(x, _input((2, 128), torch.contiguous_format, 1))
        _transformer()(x, _context())
        vae_block(x, None)
    assert kernel == [] and nhwc.status()["calls"] == {"group_norm": 0, "group_norm_silu": 0} and nhwc.status()["fallbacks"] == {}

    vae_norm = _bf16_cl(sgm_vae.Normalize(64))
    plain = copy.deepcopy(vae_norm)
    plain.__class__ = torch.nn.GroupNorm
    with torch.no_grad():
        assert torch.equal(vae_norm(x), plain(x))


# --- dispatch with the switch on: bitwise op-for-op equality with the upstream forwards -----------------------------

@pytest.mark.parametrize("out_channels", [64, 128], ids=["identity-skip", "conv-skip"])
def test_resblock_fuses_both_norm_silu_pairs(kernel, out_channels):
    block = _resblock(out_channels=out_channels)
    off, on = _off_and_on(block, _input((2, 64, 8, 12)), _input((2, 128), torch.contiguous_format, 1))
    assert torch.equal(off, on)
    assert [(c.channels, c.act, c.eps) for c in kernel] == [(64, "silu", 1e-5), (out_channels, "silu", 1e-5)]
    assert nhwc.status()["calls"] == {"group_norm": 0, "group_norm_silu": 2}


def test_resblock_with_only_the_unet_scope_dispatches_unfused(kernel):
    off, on = _off_and_on(_resblock(), _input((2, 64, 8, 12)), _input((2, 128), torch.contiguous_format, 1), scopes="unet")
    assert torch.equal(off, on)
    assert [(c.channels, c.act) for c in kernel] == [(64, ""), (128, "")]


def test_spatial_transformer_stays_channels_last_end_to_end(kernel):
    off, on = _off_and_on(_transformer(), _input((2, 64, 8, 12)), _context())
    assert torch.equal(off, on)
    assert off.is_contiguous() and on.is_contiguous(memory_format=CL) and not on.is_contiguous()
    assert [(c.channels, c.act, c.eps) for c in kernel] == [(64, "", 1e-6)]


@pytest.mark.parametrize("nonlinearity", ["upstream", "torch_silu"])
@pytest.mark.parametrize("out_channels", [64, 128], ids=["identity", "nin-shortcut"])
def test_vae_resnet_block_fuses_norm_swish(kernel, monkeypatch, nonlinearity, out_channels):
    """sgm's swish (x * sigmoid(x), two bf16 roundings) or torch's silu (what sd_hijack.apply_optimizations installs):
    the fused path computes the SiLU, so its oracle is the upstream forward with nonlinearity = F.silu."""
    block = _bf16_cl(sgm_vae.ResnetBlock(in_channels=64, out_channels=out_channels, dropout=0.0, temb_channels=0))
    x = _input((1, 64, 8, 12))
    monkeypatch.setattr(sgm_vae, "nonlinearity", F.silu)
    expected, _ = _off_and_on(block, x, None, scopes="")
    if nonlinearity == "upstream":
        swish = sd_hijack_unet._VAE_SWISH_FUNCTIONS[0]  # sgm's own, captured when sd_hijack_unet was imported
        assert (swish.__module__, swish.__name__) == ("sgm.modules.diffusionmodules.model", "nonlinearity")
        monkeypatch.setattr(sgm_vae, "nonlinearity", swish)
    assert kernel == []
    _, on = _off_and_on(block, x, None)
    assert torch.equal(on, expected)
    assert [(c.channels, c.act, c.eps) for c in kernel] == [(64, "silu", 1e-6), (out_channels, "silu", 1e-6)]


def test_vae_resnet_block_with_another_nonlinearity_dispatches_each_norm_alone(kernel, monkeypatch):
    monkeypatch.setattr(sgm_vae, "nonlinearity", torch.tanh)
    block = _bf16_cl(sgm_vae.ResnetBlock(in_channels=64, out_channels=64, dropout=0.0, temb_channels=0))
    off, on = _off_and_on(block, _input((1, 64, 8, 12)), None)
    assert torch.equal(off, on)
    assert [c.act for c in kernel] == ["", ""]


def test_vae_decoder_end_to_end(kernel, monkeypatch):
    monkeypatch.setattr(sgm_vae, "nonlinearity", F.silu)
    decoder = _bf16_cl(sgm_vae.Decoder(ch=32, out_ch=3, ch_mult=(1, 2), num_res_blocks=1, attn_resolutions=[8], in_channels=3, resolution=16, z_channels=4))
    off, on = _off_and_on(decoder, _input((1, 4, 8, 8)))
    assert torch.equal(off, on)
    resnet_blocks = sum(isinstance(m, sgm_vae.ResnetBlock) for m in decoder.modules())
    standalone = sum(isinstance(m, sd_hijack_unet.VaeGroupNorm) for m in decoder.modules()) - 2 * resnet_blocks
    assert standalone > 0 and sorted(c.act for c in kernel) == [""] * standalone + ["silu"] * (2 * resnet_blocks)


def test_vae_norms_keep_the_torch_path_under_autocast(kernel, monkeypatch):
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda device_type=None: True)
    off, on = _off_and_on(_bf16_cl(sgm_vae.Normalize(64)), _input((1, 64, 8, 12)))
    assert torch.equal(off, on) and kernel == []


# --- inputs and modules the kernels must not take ----------------------------------------------------------------

def test_nchw_activations_never_dispatch(kernel):
    x = _input((2, 64, 8, 12), torch.contiguous_format)
    block = _resblock(memory_format=torch.contiguous_format)
    off, on = _off_and_on(block, x, _input((2, 128), torch.contiguous_format, 1))
    assert torch.equal(off, on)
    off, on = _off_and_on(_transformer(torch.contiguous_format), x, _context())
    assert torch.equal(off, on) and on.is_contiguous()
    assert kernel == [] and nhwc.status()["fallbacks"].get("not_nhwc", 0) > 0


def test_group_norm_declines_unsupported_calls(kernel):
    nhwc.set_scopes("all")
    norm = _bf16_cl(sgm_util.normalization(64))
    x = _input((2, 64, 8, 12))
    with torch.enable_grad():
        assert nhwc.group_norm(norm, x) is None
    with torch.no_grad():
        assert nhwc.group_norm(_bf16_cl(torch.nn.GroupNorm(8, 40)), _input((2, 40, 8, 12))) is None  # C % 16
        assert nhwc.group_norm(_randomize(torch.nn.GroupNorm(32, 64)).to(memory_format=CL), x) is None  # fp32 weights
        assert nhwc.group_norm(_bf16_cl(torch.nn.GroupNorm(32, 64, affine=False)), x) is None
        assert nhwc.group_norm(norm, _input((2, 64, 1, 1))) is None  # NHWC and NCHW at once: torch copies nothing
        assert nhwc.group_norm(norm, x) is not None
    assert nhwc.status()["fallbacks"] == {"unsupported_call": 2, "unsupported_shape_or_dtype": 2, "not_nhwc": 1}
    assert len(kernel) == 1


def test_hooks_or_instance_forward_overrides_keep_the_module_calls(kernel):
    block = _resblock()
    x, emb = _input((2, 64, 8, 12)), _input((2, 128), torch.contiguous_format, 1)
    seen = []
    handle = block.in_layers[0].register_forward_hook(lambda module, args, output: seen.append(output.shape))
    off, on = _off_and_on(block, x, emb)
    handle.remove()
    assert torch.equal(off, on) and len(seen) == 2  # the hook ran in both modes
    assert [c.act for c in kernel] == ["", ""]  # each GroupNorm32 on its own, SiLU as a module

    kernel.clear()
    original_silu = block.out_layers[1].forward
    block.out_layers[1].forward = lambda h: seen.append("silu") or original_silu(h)
    off, on = _off_and_on(block, x, emb)
    assert torch.equal(off, on) and seen.count("silu") == 2 and [c.act for c in kernel] == ["", ""]


def test_unknown_group_norm_forward_patch_falls_back_to_the_class_chain(kernel, monkeypatch):
    original = torch.nn.GroupNorm.forward
    patched_calls = []

    def patched(self, input):
        patched_calls.append(self)
        return original(self, input)

    monkeypatch.setattr(torch.nn.GroupNorm, "forward", patched)
    off, on = _off_and_on(_transformer(), _input((2, 64, 8, 12)), _context())
    assert torch.equal(off, on) and len(patched_calls) == 2 and kernel == []
    assert nhwc.status()["fallbacks"] == {"groupnorm_forward_chain": 1}


def test_merged_lora_is_applied_before_the_kernel_and_functional_lora_falls_back(kernel, lora_networks, monkeypatch):  # noqa: F811  (the imported fixture)
    networks = lora_networks
    monkeypatch.setattr(networks, "originals", SimpleNamespace(GroupNorm_forward=torch.nn.GroupNorm.forward), raising=False)
    monkeypatch.setattr(torch.nn.GroupNorm, "forward", networks.network_GroupNorm_forward)
    applied = []

    def apply_weights(module):  # a merged norm delta; may replace the parameter object (networks.py does for bias)
        applied.append(module)
        module.bias = torch.nn.Parameter(module.bias.detach() + 1, requires_grad=False)

    monkeypatch.setattr(networks, "network_apply_weights", apply_weights)
    monkeypatch.setattr(networks.shared.opts, "lora_functional", False, raising=False)
    norm = _bf16_cl(sgm_util.normalization(64))
    x = _input((2, 64, 8, 12))
    nhwc.set_scopes("all")
    with torch.no_grad():
        y = nhwc.group_norm(norm, x)
        assert applied == [norm] and kernel[-1].bias is norm.bias  # the kernel read the merged bias
        assert torch.equal(y, F.group_norm(x, 32, norm.weight, norm.bias, norm.eps))

        functional = []
        monkeypatch.setattr(networks.shared.opts, "lora_functional", True)
        monkeypatch.setattr(networks, "network_forward", lambda module, input, original: functional.append(module) or original(module, input))
        assert nhwc.group_norm(norm, x) is None and len(kernel) == 1 and applied == [norm]
        networks.network_GroupNorm_forward(norm, x)
        assert functional == [norm]
    assert nhwc.status()["fallbacks"] == {"groupnorm_forward_chain": 1}


def test_network_group_norm_forward_keeps_both_paths(lora_networks, monkeypatch):  # noqa: F811  (the imported fixture)
    networks = lora_networks
    calls = []
    monkeypatch.setattr(networks, "originals", SimpleNamespace(GroupNorm_forward=lambda module, input: calls.append("original") or input), raising=False)
    monkeypatch.setattr(networks, "network_apply_weights", lambda module: calls.append("apply"))
    monkeypatch.setattr(networks, "network_forward", lambda module, input, original: calls.append("functional") or original(module, input))
    norm, x = torch.nn.GroupNorm(2, 4), torch.randn(1, 4, 2, 2)
    for functional, expected in ((False, ["apply", "original"]), (True, ["functional", "original"])):
        monkeypatch.setattr(networks.shared.opts, "lora_functional", functional, raising=False)
        calls.clear()
        assert networks.network_GroupNorm_forward(norm, x) is x
        assert calls == expected
        assert networks.network_GroupNorm_prepare(norm) is (not functional)


# --- scopes, configuration, ControlNet layout ----------------------------------------------------------------------

def test_parse_and_set_scopes(switch, monkeypatch):
    for value in (None, "", "0", "off", "false", " none ", [], ","):
        assert nhwc.parse_scopes(value) == frozenset()
    for value in ("1", "all", "ON", True, "yes"):
        assert nhwc.parse_scopes(value) == nhwc.ALL_SCOPES
    assert nhwc.parse_scopes("unet, silu") == {"unet", "silu"}
    assert nhwc.parse_scopes(["vae", "controlnet"]) == {"vae", "controlnet"}
    with pytest.raises(ValueError):
        nhwc.parse_scopes("unet,nchw")

    monkeypatch.setattr(nhwc, "_KERNELS", None)

    def missing():
        raise ImportError("No module named 'triton'")

    monkeypatch.setattr(nhwc, "_load_kernels", missing)
    with pytest.raises(RuntimeError, match="kernels unavailable"):
        nhwc.set_scopes("unet")
    assert nhwc.scopes() == frozenset() and nhwc.state_key() == () and "triton" in nhwc.status()["error"]
    assert nhwc.set_scopes("off")["scopes"] == []  # turning it off needs no kernels

    monkeypatch.setattr(nhwc, "_load_kernels", lambda: SimpleNamespace())
    assert nhwc.set_scopes("vae,unet")["scopes"] == ["unet", "vae"] and nhwc.state_key() == ("unet", "vae")
    assert nhwc.status()["kernels_loaded"]


def test_env_configuration_fails_startup_when_it_cannot_be_honoured(switch, monkeypatch):
    monkeypatch.setattr(nhwc, "_KERNELS", None)
    monkeypatch.setattr(nhwc, "_load_kernels", lambda: (_ for _ in ()).throw(ImportError("no triton")))
    monkeypatch.setenv(nhwc.ENV_NAME, "all")
    with pytest.raises(RuntimeError, match="OPENCLAW_NHWC_GROUPNORM='all'"):
        nhwc._configure_from_env()
    monkeypatch.setenv(nhwc.ENV_NAME, "unet,bogus")
    with pytest.raises(RuntimeError, match="bogus"):
        nhwc._configure_from_env()
    monkeypatch.setenv(nhwc.ENV_NAME, "0")
    nhwc._configure_from_env()
    assert nhwc.scopes() == frozenset()


def test_controlnet_layout_follows_the_scope_and_restores_bit_for_bit(switch, monkeypatch):
    model = torch.nn.Sequential(torch.nn.Conv2d(4, 32, 3), torch.nn.GroupNorm(32, 32), torch.nn.Conv2d(32, 8, 1))
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    def scope(on):
        monkeypatch.setattr(nhwc, "_SCOPES", frozenset({"controlnet"}) if on else frozenset())

    untouched = model[0].weight.data_ptr()
    nhwc.apply_controlnet_layout(model, channels_last=True)  # scope off, never converted: no copy
    scope(True)
    nhwc.apply_controlnet_layout(model, channels_last=False)  # low VRAM or an NCHW SD model: no copy either
    assert model[0].weight.data_ptr() == untouched

    nhwc.apply_controlnet_layout(model, channels_last=True)
    assert model[0].weight.is_contiguous(memory_format=CL) and not model[0].weight.is_contiguous()
    converted = model[0].weight.data_ptr()
    nhwc.apply_controlnet_layout(model, channels_last=True)  # idempotent: no second copy
    assert model[0].weight.data_ptr() == converted

    restores = (
        lambda: nhwc.apply_controlnet_layout(model, channels_last=False),
        lambda: scope(False) or nhwc.apply_controlnet_layout(model, channels_last=True),
    )
    for restore in restores:
        scope(True)
        nhwc.apply_controlnet_layout(model, channels_last=True)
        assert not model[0].weight.is_contiguous()
        restore()
        assert all(t.is_contiguous() for t in model.state_dict().values())
        assert all(torch.equal(model.state_dict()[name], tensor) for name, tensor in before.items())


# --- the kernels on CPU: Triton's interpreter and the sm_121 PTX ---------------------------------------------------

# Cases for the real kernels in Triton's interpreter: (shape, groups, module constants to override, silu, offset).
# offset > 0 makes the input offset + noise of std offset / 64 (a few bf16 ulps): a large mean over std, where an
# E[x^2] - mean^2 variance cancels; the centered tile statistics must not.
_SUB_TILES = {"_MIN_STATS_PROGRAMS": 1, "_STATS_TILE_ELEMENTS": 512}
_INTERPRETER_CASES = {
    "tail-one-chunk-cpg2": ((2, 64, 9, 13), 32, {}, False, 0),
    "five-chunks-cpg10": ((1, 320, 6, 7), 32, {}, True, 0),
    "32-channel-chunks-cpg3": ((2, 96, 5, 5), 32, {}, False, 0),
    "16-channel-chunks-16-groups": ((3, 48, 4, 4), 16, {}, True, 0),
    "many-sub-tiles-tail": ((1, 128, 33, 31), 32, {"_MIN_STATS_PROGRAMS": 1, "_STATS_TILE_ELEMENTS": 1024}, False, 0),
    "small-tiles-cpg20": ((2, 640, 10, 10), 32, {"_STATS_TILE_ELEMENTS": 256, "_APPLY_TILE_ELEMENTS": 512}, True, 0),
    "offset-1e2": ((1, 64, 16, 16), 32, _SUB_TILES, False, 1e2),
    "offset-1e3": ((1, 64, 16, 16), 32, _SUB_TILES, False, 1e3),
    "offset-1e4": ((1, 64, 16, 16), 32, _SUB_TILES, True, 1e4),
}


@pytest.fixture(scope="module")
def interpreter_results():
    """All cases in one fresh process (test/nhwc_groupnorm_interpreter_check.py) with TRITON_INTERPRET=1."""
    pytest.importorskip("triton")
    env = {key: value for key, value in os.environ.items() if key != nhwc.ENV_NAME}
    env["TRITON_INTERPRET"] = "1"
    process = subprocess.run(
        [sys.executable, str(_ROOT / "test" / "nhwc_groupnorm_interpreter_check.py"), json.dumps(_INTERPRETER_CASES)],
        cwd=_ROOT, env=env, capture_output=True, text=True, timeout=900,
    )
    assert process.returncode == 0, process.stderr[-6000:]
    return json.loads(process.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("case", list(_INTERPRETER_CASES))
def test_kernels_in_the_interpreter_match_float64(interpreter_results, case):
    """Indexing, tails, channel chunks crossing groups, sub-tile merging and the group combine against float64,
    output layout, and run-to-run identity."""
    result = interpreter_results[case]
    assert result["layout"] == ["torch.bfloat16", list(_INTERPRETER_CASES[case][0]), True]
    assert result["excess"] <= 0, result
    assert result["repeat_bitwise"]
    if case == "many-sub-tiles-tail":
        assert result["tiles"] == 2  # two 512-pixel tiles of 64 sub-tiles of 8 pixels, the second one short


def test_launch_config_tiles_sdxl_shapes():
    for (n, c, hw), expected in {
        (2, 320, 128 * 128): {"block_c": 64, "stats_block_hw": 32, "tile_hw": 512, "tiles": 32, "apply_block_hw": 64},
        (2, 1280, 32 * 32): {"block_c": 128, "stats_block_hw": 16, "tile_hw": 64, "tiles": 16, "apply_block_hw": 32},
        (2, 960, 64 * 64): {"block_c": 64, "stats_block_hw": 32, "tile_hw": 256, "tiles": 16, "apply_block_hw": 64},
        (1, 128, 1024 * 1024): {"block_c": 128, "stats_block_hw": 16, "tile_hw": 2048, "tiles": 512, "apply_block_hw": 32},
    }.items():
        assert nhwc.launch_config(n, c, hw) == expected
        config = expected
        assert config["tile_hw"] % config["stats_block_hw"] == 0
        assert config["tiles"] * n * (c // config["block_c"]) >= nhwc._MIN_STATS_PROGRAMS or config["tile_hw"] == nhwc._MAX_TILE_HW


def _ptx(kernel, signature, constexprs, num_warps):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    names = list(signature)
    pointer_or_aligned = [i for i, name in enumerate(names) if signature[name].startswith("*") or name == "C"]
    source = ASTSource(
        fn=kernel,
        signature=signature,
        constexprs={(names.index(name),): value for name, value in constexprs.items()},
        attrs={(i,): [["tt.divisibility", 16]] for i in pointer_or_aligned},
    )
    ptx = triton.compile(source, target=GPUTarget("cuda", 121, 32), options={"num_warps": num_warps}).asm["ptx"]
    return "\n".join(line for line in ptx.splitlines() if not line.lstrip().startswith(("//", ".loc", ".b8")))


def test_kernels_compile_for_sm121_without_atomics():
    """Determinism and codegen without a GPU: no atomic or reduction-to-memory instruction anywhere, 16-byte vector
    loads/stores of the activations, IEEE sqrt for rstd, round-to-nearest-even bf16 output."""
    pytest.importorskip("triton")
    from modules import openclaw_nhwc_groupnorm_kernel as kernels

    stats = _ptx(
        kernels.channel_stats_kernel,
        {"x_ptr": "*bf16", "mean_ptr": "*fp32", "m2_ptr": "*fp32", "HW": "i32", "C": "i32", "T": "i32", "TILE_HW": "constexpr", "BLOCK_HW": "constexpr", "BLOCK_C": "constexpr"},
        {"TILE_HW": 512, "BLOCK_HW": 32, "BLOCK_C": 64},
        4,
    )
    combine = _ptx(
        kernels.group_stats_kernel,
        {"mean_ptr": "*fp32", "m2_ptr": "*fp32", "stats_ptr": "*fp32", "HW": "i32", "C": "i32", "T": "i32", "CPG": "i32", "eps": "fp32", "TILE_HW": "constexpr", "BLOCK_K": "constexpr"},
        {"TILE_HW": 512, "BLOCK_K": 256},
        4,
    )
    apply = _ptx(
        kernels.apply_kernel,
        {"x_ptr": "*bf16", "y_ptr": "*bf16", "w_ptr": "*bf16", "b_ptr": "*bf16", "stats_ptr": "*fp32", "HW": "i32", "C": "i32", "CPG": "i32", "G": "i32", "BLOCK_HW": "constexpr", "BLOCK_C": "constexpr", "SILU": "constexpr"},
        {"BLOCK_HW": 64, "BLOCK_C": 64, "SILU": True},
        8,
    )
    for body in (stats, combine, apply):
        assert not re.search(r"\b(atom|red)\.", body)
    assert "ld.global.v4" in stats and "ld.global.v4" in apply and "st.global.v4" in apply
    assert "sqrt.rn.f32" in combine
    assert "cvt.rn.bf16.f32" in apply or "cvt.rn.bf16x2.f32" in apply


# --- CUDA: the compiled kernels --------------------------------------------------------------------------------

@pytest.fixture
def kernels_on(default_runtime, switch):  # noqa: F811  (the imported fixture)
    nhwc.set_scopes("all")  # imports Triton and the kernels
    yield
    nhwc.set_scopes("")


# (channels, size, eps, batch): SDXL UNet/ControlNet GroupNorm32 and SpatialTransformer.norm at 1024-1536 px requests,
# and the VAE decoder's norms at 1024 px.
_CUDA_SHAPES = [
    (320, 128, 1e-5, 2), (640, 64, 1e-5, 2), (960, 64, 1e-5, 2), (1280, 32, 1e-5, 2), (1920, 64, 1e-5, 2),
    (2560, 32, 1e-5, 2), (640, 96, 1e-6, 2), (1280, 48, 1e-6, 2), (320, 160, 1e-5, 3),
    (512, 128, 1e-6, 1), (512, 256, 1e-6, 1), (256, 512, 1e-6, 1), (128, 1024, 1e-6, 1),
]


def _cuda_norm(channels, eps):
    return _randomize(torch.nn.GroupNorm(32, channels, eps=eps)).to("cuda", BF16)


@needs_cuda
@pytest.mark.parametrize("act", ["", "silu"])
@pytest.mark.parametrize("channels,size,eps,batch", _CUDA_SHAPES)
def test_cuda_error_is_within_the_torch_bf16_path_and_runs_repeat_bitwise(kernels_on, channels, size, eps, batch, act):
    norm = _cuda_norm(channels, eps)
    x = _random_activations((batch, channels, size, size), CL)
    with torch.inference_mode():
        y = nhwc.group_norm(norm, x, act=act or None)
        assert y is not None and y.is_contiguous(memory_format=CL) and y.dtype == BF16
        assert torch.equal(y, nhwc.group_norm(norm, x, act=act or None))
        native = F.group_norm(x, 32, norm.weight, norm.bias, eps)  # torch's bf16 path (eps rounded to bf16)
        reference = F.group_norm(x.double(), 32, norm.weight.double(), norm.bias.double(), eps)
        if act:
            native, reference = F.silu(native), F.silu(reference)
    kernel_max, kernel_mean = _error_over_rms(y, reference)
    native_max, native_mean = _error_over_rms(native, reference)
    floor_max, floor_mean = _error_over_rms(reference.to(BF16), reference)
    assert kernel_max <= native_max + floor_max, (kernel_max, native_max, floor_max)
    assert kernel_mean <= native_mean + 0.05 * floor_mean, (kernel_mean, native_mean, floor_mean)


@needs_cuda
@pytest.mark.parametrize("offset", [1, 10, 100, 1000, 10000])
def test_cuda_accuracy_holds_at_any_mean_over_std(kernels_on, offset):
    norm = _cuda_norm(640, 1e-5)
    generator = torch.Generator(device="cuda").manual_seed(offset)
    x = (torch.randn((2, 640, 64, 64), device="cuda", generator=generator) * max(1.0, offset * 2.0 ** -6) + offset)
    x = x.to(BF16).contiguous(memory_format=CL)
    with torch.inference_mode():
        y = nhwc.group_norm(norm, x)
        native = F.group_norm(x, 32, norm.weight, norm.bias, 1e-5)
        reference = F.group_norm(x.double(), 32, norm.weight.double(), norm.bias.double(), 1e-5)
    kernel_max, _ = _error_over_rms(y, reference)
    native_max, _ = _error_over_rms(native, reference)
    floor_max, _ = _error_over_rms(reference.to(BF16), reference)
    assert kernel_max <= native_max + floor_max, (offset, kernel_max, native_max)


_BLOCK_KINDS = ["resblock", "spatial_transformer", "vae_resnet_block"]


def _block_module(kind, small=False):
    """The block of each kind: SDXL-sized, or (small) the same classes at CPU-test size."""
    if kind == "resblock":
        channels, emb_channels = (64, 128) if small else (640, 1280)
        return sgm_openaimodel.ResBlock(channels, emb_channels, 0.0, out_channels=channels)
    if kind == "spatial_transformer":
        if small:
            return sgm_attention.SpatialTransformer(64, 2, 32, depth=1, context_dim=48, use_linear=True, use_checkpoint=False)
        return sgm_attention.SpatialTransformer(640, 10, 64, depth=1, context_dim=2048, use_linear=True, use_checkpoint=False)
    channels = 64 if small else 512
    return sgm_vae.ResnetBlock(in_channels=channels, out_channels=channels, dropout=0.0, temb_channels=0)


def _sdxl_block(kind):
    """(module, args, autocast): SDXL-sized blocks; the UNet ones run under bf16 autocast, the VAE one without (how the
    decode runs)."""
    module = _block_module(kind)
    if kind == "resblock":
        return module, (_random_activations((2, 640, 64, 64), CL), _random_activations((2, 1280))), True
    if kind == "spatial_transformer":
        return module, (_random_activations((2, 640, 64, 64), CL), [_random_activations((2, 77, 2048))]), True
    return module, (_random_activations((1, 512, 128, 128), CL), None), False


def _float64_reference(module, device):
    """A float64 copy of `module` on `device` that computes every norm in float64.

    sgm GroupNorm32.forward normalizes x.float(): in a float64 copy that is a float32 input against float64 weights,
    which F.group_norm rejects (and a float32 norm would not be a float64 reference anyway). A plain GroupNorm computes
    in its input dtype, so the copy's GroupNorm32 instances become plain GroupNorms (same parameters and eps)."""
    reference = copy.deepcopy(module).to(device, torch.float64)
    for submodule in reference.modules():
        if isinstance(submodule, sgm_util.GroupNorm32):
            submodule.__class__ = torch.nn.GroupNorm
    return reference


def _run_block(module, args, autocast, scopes):
    nhwc.set_scopes(scopes)
    with torch.inference_mode(), torch.autocast("cuda", dtype=BF16, enabled=autocast):
        return module(*args)


def _double(arg):
    if isinstance(arg, list):
        return [a.double() for a in arg]
    return arg.double() if torch.is_tensor(arg) else arg


@pytest.mark.parametrize("kind", _BLOCK_KINDS)
def test_float64_block_references_run_and_compute_the_block(switch, kind):
    """The CUDA block test's float64 reference, on CPU at test size: a plain float64 copy of a ResBlock fails in sgm
    GroupNorm32 (float32 input, float64 weights), the reference built by _float64_reference runs every kind and agrees
    with the float32 block."""
    module = _randomize(_block_module(kind, small=True)).eval()
    generator = torch.Generator().manual_seed(5)
    x = torch.randn((2, 64, 8, 8), generator=generator, dtype=torch.float64)
    extra = {"resblock": torch.randn((2, 128), generator=generator, dtype=torch.float64),
             "spatial_transformer": [torch.randn((2, 5, 48), generator=generator, dtype=torch.float64)]}.get(kind)
    args64 = (x, extra)
    args32 = (x.float(), [a.float() for a in extra] if isinstance(extra, list) else None if extra is None else extra.float())

    with torch.inference_mode():
        if kind == "resblock":
            with pytest.raises(RuntimeError):
                copy.deepcopy(module).double()(*args64)
        reference = _float64_reference(module, "cpu")(*args64)
        expected = module(*args32)
    assert reference.dtype == torch.float64 and torch.isfinite(reference).all()
    torch.testing.assert_close(expected.double(), reference, rtol=1e-4, atol=1e-4 * reference.abs().max().item())


@needs_cuda
@pytest.mark.parametrize("kind", _BLOCK_KINDS)
def test_cuda_fused_blocks_match_the_torch_paths_and_replay_in_cuda_graphs(kernels_on, kind):
    """Switch-on error against float64 within 10% of switch-off's, a second run bitwise equal, and a CUDA graph replay
    equal to the eager run."""
    module, args, autocast = _sdxl_block(kind)
    module = _randomize(module).eval()
    reference_module = _float64_reference(module, "cuda")
    module = module.to("cuda", BF16).to(memory_format=CL)

    off, on = _run_block(module, args, autocast, ""), _run_block(module, args, autocast, "all")
    assert torch.equal(on, _run_block(module, args, autocast, "all"))
    with torch.inference_mode():
        reference = reference_module(*[_double(a) for a in args])
    off_max, off_mean = _error_over_rms(off.float(), reference)
    on_max, on_mean = _error_over_rms(on.float(), reference)
    assert on_mean <= 1.1 * off_mean and on_max <= 2 * off_max, (on_mean, off_mean, on_max, off_max)

    nhwc.set_scopes("all")
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.autocast("cuda", dtype=BF16, enabled=autocast):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            module(*args)  # warm-up on a side stream, as torch.cuda.graphs recommends
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph):
            captured = module(*args)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, on)
