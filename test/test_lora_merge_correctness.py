"""LoRA merge numerics: deltas are computed and summed in float32 and rounded to the stored dtype once.

The oracle is the float64 evaluation of W + sum(multiplier * alpha / rank * up @ down) rounded once to the
layer dtype. Test values are chosen so the float32 computation is exact; any intermediate rounding to the
layer dtype (the defect fixed here) shows up as a mismatch.
"""
import sys
import dataclasses
from types import SimpleNamespace

import pytest

from test.test_openclaw_lora_network_identity import lora_networks  # noqa: F401  (fixture)
from modules.torchao_weight_quant import NVFP4

torch = pytest.importorskip("torch")


def _net(networks, name, multiplier=1.0):
    net = networks.network.Network(name, SimpleNamespace(filename=f"{name}.safetensors"))
    net.source_key = (name,)
    net.te_multiplier = net.unet_multiplier = multiplier
    return net


def _add_module(networks, net, layer, w, module_type):
    sd_key = layer.network_layer_name
    weights = networks.network.NetworkWeights(network_key=f"{net.name}.{sd_key}", sd_key=sd_key, w=w, sd_module=layer)
    net.modules[sd_key] = module_type(net, weights)
    return net


def _lora(networks, layer, name, up, down, alpha, multiplier):
    net = _net(networks, name, multiplier)
    w = {"lora_up.weight": up, "lora_down.weight": down, "alpha": torch.tensor(float(alpha))}
    return _add_module(networks, net, layer, w, networks.network_lora.NetworkModuleLora)


def _grid(shape, generator, scale=2.0 ** -6):
    """Small multiples of a power of two: exact in fp16/bf16 and with exact float32 products and sums."""
    return torch.randint(-8, 9, shape, generator=generator).to(torch.float16) * scale


@pytest.fixture
def bf16_lora(lora_networks, monkeypatch):  # noqa: F811  (the imported fixture)
    networks = lora_networks
    monkeypatch.setattr(networks.devices, "dtype", torch.bfloat16)
    monkeypatch.setattr(networks.devices, "device", torch.device("cpu"), raising=False)
    monkeypatch.setattr(networks, "extra_network_lora", SimpleNamespace(errors={}), raising=False)
    return networks


def test_stacked_small_lora_deltas_are_not_swamped_by_bf16_rounding(bf16_lora):
    networks = bf16_lora
    layer = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    layer.network_layer_name = "diffusion_model_layer"
    with torch.no_grad():
        layer.weight.copy_(torch.arange(16).reshape(4, 4) * 2.0 ** -7 + 1.0)  # bf16 ulp = 2**-7 on [1, 2)
    base = layer.weight.detach().clone()
    # Each delta (3 * 2**-10 = 0.375 ulp) alone rounds away; the three together add 1.125 ulp.
    nets = [_lora(networks, layer, f"n{i}", torch.full((4, 1), 2.0 ** -5, dtype=torch.float16), torch.full((1, 4), 3 * 2.0 ** -5, dtype=torch.float16), 1.0, 1.0) for i in range(3)]

    networks._set_loaded_networks(nets)
    networks.network_apply_weights(layer)

    assert torch.equal(layer.weight, (base.double() + 9 * 2.0 ** -10).to(torch.bfloat16))
    assert torch.equal(layer.weight, base + 2.0 ** -7)


def _autocast(enabled):
    # Generation runs merges lazily inside forwards under bf16 autocast; CPU autocast stands in for CUDA's here.
    return torch.autocast("cpu", dtype=torch.bfloat16, enabled=enabled)


@pytest.mark.parametrize("autocast", [False, True])
def test_lora_merge_matches_fp64_oracle_and_restores_exactly(bf16_lora, autocast):
    networks = bf16_lora
    g = torch.Generator().manual_seed(1234)
    layer = torch.nn.Conv2d(16, 32, 3, padding=1, bias=True, dtype=torch.bfloat16)
    layer.network_layer_name = "diffusion_model_conv"
    base_w, base_b = layer.weight.detach().clone(), layer.bias.detach().clone()
    specs = [(4, 2.0, 0.75), (8, 8.0, -1.25), (2, 1.0, 0.5)]  # rank, alpha, multiplier
    nets = []
    expected = base_w.double()
    for i, (rank, alpha, multiplier) in enumerate(specs):
        down = _grid((rank, 16, 3, 3), g)
        up = _grid((32, rank, 1, 1), g)
        nets.append(_lora(networks, layer, f"n{i}", up, down, alpha, multiplier))
        expected = expected + multiplier * alpha / rank * (up.double().reshape(32, rank) @ down.double().reshape(rank, -1)).reshape(base_w.shape)
    expected = expected.to(torch.bfloat16)

    for _ in range(3):  # repeated apply/restore cycles are bit-exact: no drift
        networks._set_loaded_networks(nets)
        with _autocast(autocast):
            networks.network_apply_weights(layer)
        assert torch.equal(layer.weight, expected)
        assert torch.equal(layer.bias, base_b)
        networks._set_loaded_networks([])
        with _autocast(autocast):
            networks.network_apply_weights(layer)
        assert torch.equal(layer.weight, base_w) and torch.equal(layer.bias, base_b)


def test_fp16_master_merge_keeps_every_network(bf16_lora):
    """fp8 storage with an fp16 master: each network used to overwrite the weight with master + its own delta."""
    networks = bf16_lora
    g = torch.Generator().manual_seed(7)
    layer = torch.nn.Linear(8, 8, bias=False)
    with torch.no_grad():
        layer.weight.copy_(_grid((8, 8), g, 2.0 ** -3))
    layer.network_layer_name = "diffusion_model_layer"
    layer.fp16_weight = layer.weight.detach().half()
    layer.to(torch.float8_e4m3fn)
    ups = [_grid((8, 2), g) for _ in range(2)]
    downs = [_grid((2, 8), g) for _ in range(2)]
    nets = [_lora(networks, layer, f"n{i}", ups[i], downs[i], 2.0, 1.0) for i in range(2)]

    networks._set_loaded_networks(nets)
    networks.network_apply_weights(layer)

    deltas = [ups[i].double() @ downs[i].double() for i in range(2)]
    expected = (layer.fp16_weight.double() + deltas[0] + deltas[1]).float().to(torch.float8_e4m3fn)
    last_only = (layer.fp16_weight.double() + deltas[1]).float().to(torch.float8_e4m3fn)
    assert torch.equal(layer.weight.float(), expected.float())
    assert not torch.equal(layer.weight.float(), last_only.float())


def test_mha_in_proj_lora_merges_in_float32(bf16_lora):
    networks = bf16_lora
    g = torch.Generator().manual_seed(3)
    mha = torch.nn.MultiheadAttention(8, 2, bias=False, batch_first=True, dtype=torch.bfloat16)
    mha.network_layer_name = "1_model_transformer_resblocks_0_attn"
    base = mha.in_proj_weight.detach().clone()
    proj = torch.nn.Linear(8, 8, bias=False)  # shape donor for the q/k/v LoRA modules
    nets, expected = [], base.double()
    for i in range(3):
        net = _net(networks, f"n{i}")
        deltas = []
        for part in ("q", "k", "v"):
            proj.network_layer_name = f"{mha.network_layer_name}_{part}_proj"
            up, down = _grid((8, 1), g), _grid((1, 8), g)
            _add_module(networks, net, proj, {"lora_up.weight": up, "lora_down.weight": down}, networks.network_lora.NetworkModuleLora)
            deltas.append(up.double() @ down.double())
        expected = expected + torch.cat(deltas)
        nets.append(net)

    networks._set_loaded_networks(nets)
    with _autocast(True):
        networks.network_apply_weights(mha)

    assert torch.equal(mha.in_proj_weight, expected.to(torch.bfloat16))


def test_oft_merges_into_bf16_layer(bf16_lora):
    """OFT's float32 rotation used to meet the bf16 weight in einsum and raise (the network was skipped)."""
    networks = bf16_lora
    g = torch.Generator().manual_seed(5)
    layer = torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
    layer.network_layer_name = "diffusion_model_layer"
    base = layer.weight.detach().clone()
    blocks = _grid((2, 4, 4), g, 2.0 ** -5)
    net = _add_module(networks, _net(networks, "oft"), layer, {"oft_blocks": blocks, "alpha": torch.tensor(1.0)}, networks.network_oft.NetworkModuleOFT)

    networks._set_loaded_networks([net])
    with _autocast(True):
        networks.network_apply_weights(layer)

    q = blocks.double() - blocks.double().transpose(1, 2)
    eye = torch.eye(4, dtype=torch.float64)
    r = (eye + q) @ torch.linalg.inv(eye - q)
    expected = torch.einsum("k n m, k n i -> k m i", r, base.double().reshape(2, 4, 8)).reshape(8, 8)
    assert networks.extra_network_lora.errors == {}
    assert (layer.weight.double() - expected).abs().max() <= 2.0 ** -8 * expected.abs().max()
    assert not torch.equal(layer.weight, base)


def test_full_diff_bias_on_biasless_bf16_layer_creates_parameter(bf16_lora):
    networks = bf16_lora
    layer = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    layer.network_layer_name = "diffusion_model_layer"
    base = layer.weight.detach().clone()
    diff, diff_b = torch.full((4, 4), 0.25, dtype=torch.float16), torch.arange(4, dtype=torch.float16)
    net = _add_module(networks, _net(networks, "full"), layer, {"diff": diff, "diff_b": diff_b}, networks.network_full.NetworkModuleFull)

    networks._set_loaded_networks([net])
    networks.network_apply_weights(layer)
    assert isinstance(layer.bias, torch.nn.Parameter) and layer.bias.dtype == torch.bfloat16
    assert torch.equal(layer.bias, diff_b.to(torch.bfloat16))
    assert torch.equal(layer.weight, (base.double() + 0.25).to(torch.bfloat16))

    networks._set_loaded_networks([])
    networks.network_apply_weights(layer)
    assert layer.bias is None and torch.equal(layer.weight, base)


def test_functional_generic_forward_uses_input_dtype(bf16_lora):
    networks = bf16_lora
    layer = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    layer.network_layer_name = "diffusion_model_layer"
    diff = torch.full((4, 4), 0.5, dtype=torch.float16)
    module = _add_module(networks, _net(networks, "full"), layer, {"diff": diff}, networks.network_full.NetworkModuleFull).modules[layer.network_layer_name]
    x = torch.ones(2, 4, dtype=torch.bfloat16)
    y = layer(x)

    out = module.forward(x, y)

    assert out.dtype == torch.bfloat16
    assert torch.equal(out, y + 2.0)


def test_quant_managed_merge_rounds_stacked_deltas_once(bf16_lora, monkeypatch):
    networks = bf16_lora
    g = torch.Generator().manual_seed(11)
    layer = torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)
    layer.network_layer_name = "layer"
    layer.network_nvfp4_base_weight = layer.weight.detach().cpu().clone()
    layer.network_nvfp4_base_bias = None
    ups, downs = [_grid((16, 4), g) for _ in range(3)], [_grid((4, 16), g) for _ in range(3)]
    nets = [_lora(networks, layer, f"n{i}", ups[i], downs[i], 4.0, 1.0) for i in range(3)]
    expected = layer.network_nvfp4_base_weight.double() + sum(u.double() @ d.double() for u, d in zip(ups, downs))
    networks._set_loaded_networks(nets)
    seen = []

    def quantize_(module, config, filter_fn, device):
        seen.append(module.weight.detach().clone())

    monkeypatch.setitem(sys.modules, "torchao.quantization", SimpleNamespace(quantize_=quantize_))
    backend = dataclasses.replace(NVFP4, make_config=lambda: "cfg", validate_config=lambda _config: None, tensor_type=lambda: torch.nn.Parameter)

    with _autocast(True):
        assert networks.network_apply_quant_merged_lora(backend, layer)
    assert seen[0].dtype == torch.bfloat16
    assert torch.equal(seen[0], expected.to(torch.bfloat16))


def _merge_fp32(networks, layer, nets):
    networks._set_loaded_networks(nets)
    with _autocast(True):
        networks.network_apply_weights(layer)
    assert networks.extra_network_lora.errors == {}
    return layer.weight.detach().double()


def _close(actual, expected):
    return (actual - expected).abs().max().item() <= 1e-5 * expected.abs().max().item()


@pytest.mark.parametrize("axis", ["output", "input"])
def test_dora_scale_axis_follows_its_shape(bf16_lora, axis):
    """LyCORIS stores dora_scale per output row ([out, 1], wd_on_out, its default since 2025-04) or per input column."""
    networks = bf16_lora
    g = torch.Generator().manual_seed(21)
    layer = torch.nn.Linear(12, 8, bias=False)
    layer.network_layer_name = "diffusion_model_layer"
    base = layer.weight.detach().double()
    up, down = _grid((8, 4), g), _grid((4, 12), g)
    scale_shape = (8, 1) if axis == "output" else (1, 12)
    dora_scale = torch.rand(scale_shape, generator=g) + 0.5
    net = _net(networks, "dora")
    w = {"lora_up.weight": up, "lora_down.weight": down, "alpha": torch.tensor(2.0), "dora_scale": dora_scale}
    _add_module(networks, net, layer, w, networks.network_lora.NetworkModuleLora)

    merged = base + 0.5 * (up.double() @ down.double())
    norm = merged.norm(dim=1, keepdim=True) if axis == "output" else merged.norm(dim=0, keepdim=True)
    assert _close(_merge_fp32(networks, layer, [net]), merged * dora_scale.double() / norm)


def test_lora_a_b_keys_keep_alpha(bf16_lora):
    networks = bf16_lora
    g = torch.Generator().manual_seed(22)
    layer = torch.nn.Linear(8, 8, bias=False)
    layer.network_layer_name = "diffusion_model_layer"
    base = layer.weight.detach().double()
    up, down = _grid((8, 4), g), _grid((4, 8), g)
    net = _net(networks, "peft")
    weights = networks.network.NetworkWeights(network_key="peft.layer", sd_key=layer.network_layer_name, w={"lora_A.weight": down, "lora_B.weight": up, "alpha": torch.tensor(2.0)}, sd_module=layer)
    net.modules[layer.network_layer_name] = networks.network_lora.ModuleTypeLora().create_module(net, weights)

    assert _close(_merge_fp32(networks, layer, [net]), base + 0.5 * (up.double() @ down.double()))


@pytest.mark.parametrize("on_input", [0, 1])
@pytest.mark.parametrize("conv", [False, True])
def test_ia3_scales_the_stored_channel_axis(bf16_lora, on_input, conv):
    networks = bf16_lora
    g = torch.Generator().manual_seed(23)
    layer = torch.nn.Conv2d(4, 6, 3, bias=False) if conv else torch.nn.Linear(4, 6, bias=False)
    layer.network_layer_name = "diffusion_model_layer"
    base = layer.weight.detach().double()
    dim = 4 if on_input else 6
    scale = torch.rand(dim, generator=g)
    stored = scale.reshape(1, dim, 1, 1) if conv else scale  # LyCORIS layouts
    net = _add_module(networks, _net(networks, "ia3"), layer, {"weight": stored, "on_input": torch.tensor(on_input)}, networks.network_ia3.NetworkModuleIa3)

    broadcast = [1] * base.dim()
    broadcast[1 if on_input else 0] = dim
    assert _close(_merge_fp32(networks, layer, [net]), base * (1 + scale.double().reshape(broadcast)))


@pytest.mark.parametrize("layout", ["old", "new", "new-conv"])
def test_glora_layouts_and_alpha_scale(bf16_lora, layout):
    """Oracle: LyCORIS GLoRA, dW = alpha/r * (b2 b1 + W a2 a1) (old layout) or alpha/r * (W a1 a2 + b1 b2) (current)."""
    networks = bf16_lora
    g = torch.Generator().manual_seed(24)
    out_dim, in_dim, r, alpha = 10, 6, 3, 1.5
    conv = layout == "new-conv"
    layer = torch.nn.Conv2d(in_dim, out_dim, 3, bias=False) if conv else torch.nn.Linear(in_dim, out_dim, bias=False)
    layer.network_layer_name = "diffusion_model_layer"
    base = layer.weight.detach().double()

    def rnd(*shape):
        return torch.randn(*shape, generator=g) * 0.1

    if layout == "old":
        a1, a2, b1, b2 = rnd(r, in_dim), rnd(in_dim, r), rnd(r, in_dim), rnd(out_dim, r)
        expected_delta = b2.double() @ b1.double() + (base @ a2.double()) @ a1.double()
    elif layout == "new":
        a1, a2, b1, b2 = rnd(in_dim, r), rnd(r, in_dim), rnd(out_dim, r), rnd(r, in_dim)
        expected_delta = (base @ a1.double()) @ a2.double() + b1.double() @ b2.double()
    else:
        a1, a2, b1, b2 = rnd(in_dim, r, 1, 1), rnd(r, in_dim, 1, 1), rnd(out_dim, r, 1, 1), rnd(r, in_dim, 3, 3)
        wa = torch.einsum("o i k l, i j -> o j k l", base, a1.double().flatten(1))
        expected_delta = torch.einsum("o i k l, i j -> o j k l", wa, a2.double().flatten(1)) + (b1.double().flatten(1) @ b2.double().flatten(1)).reshape(base.shape)
    w = {"a1.weight": a1, "a2.weight": a2, "b1.weight": b1, "b2.weight": b2, "alpha": torch.tensor(alpha)}
    net = _add_module(networks, _net(networks, "glora"), layer, w, networks.network_glora.NetworkModuleGLora)

    assert _close(_merge_fp32(networks, layer, [net]), base + alpha / r * expected_delta)


def test_functional_lora_forward_applies_dyn_dim(bf16_lora):
    networks = bf16_lora
    g = torch.Generator().manual_seed(25)
    layer = torch.nn.Linear(8, 8, bias=False)
    layer.network_layer_name = "diffusion_model_layer"
    up, down = _grid((8, 4), g), _grid((4, 8), g)
    net = _lora(networks, layer, "dyn", up, down, 4.0, 1.0)
    net.dyn_dim = 2
    module = net.modules[layer.network_layer_name]
    x = torch.randn(3, 8, generator=g)
    y = layer(x)

    out = module.forward(x, y)

    delta = up[:, :2].float() @ down[:2].float()
    assert torch.allclose(out, y + x @ delta.T, atol=1e-6)
