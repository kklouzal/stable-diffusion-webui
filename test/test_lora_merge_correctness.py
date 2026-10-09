"""LoRA merge numerics: deltas are computed and summed in float32 and rounded to the stored dtype once.

The oracle is the float64 evaluation of W + sum(multiplier * alpha / rank * up @ down) rounded once to the
layer dtype. Test values are chosen so the float32 computation is exact; any intermediate rounding to the
layer dtype (the defect fixed here) shows up as a mismatch.
"""
import sys
import dataclasses
from types import SimpleNamespace

import pytest
import torch

from modules.torchao_weight_quant import NVFP4


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
def bf16_lora(lora_networks, monkeypatch):
    networks = lora_networks
    monkeypatch.setattr(networks.devices, "dtype", torch.bfloat16)
    monkeypatch.setattr(networks.devices, "device", torch.device("cpu"), raising=False)
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


@pytest.mark.parametrize("layer_kind", ["mha", "sd3_qkv_linear"])
def test_combined_qkv_lora_merges_in_float32(bf16_lora, layer_kind):
    """q/k/v LoRA modules merge into MHA's in_proj_weight or SD3 QkvLinear's weight."""
    networks = bf16_lora
    g = torch.Generator().manual_seed(3)
    if layer_kind == "mha":
        layer = torch.nn.MultiheadAttention(8, 2, bias=False, batch_first=True, dtype=torch.bfloat16)
        layer.network_layer_name = "1_model_transformer_resblocks_0_attn"
        field = "in_proj_weight"
    else:
        from modules.models.sd3.mmdit import QkvLinear
        layer = QkvLinear(8, 24, bias=False, dtype=torch.bfloat16)
        layer.network_layer_name = "diffusion_model_joint_blocks_0_x_block_attn_qkv"
        field = "weight"
    base = getattr(layer, field).detach().clone()
    proj = torch.nn.Linear(8, 8, bias=False)  # shape donor for the q/k/v LoRA modules
    nets, expected = [], base.double()
    for i in range(3):
        net = _net(networks, f"n{i}")
        deltas = []
        for part in ("q", "k", "v"):
            proj.network_layer_name = f"{layer.network_layer_name}_{part}_proj"
            up, down = _grid((8, 1), g), _grid((1, 8), g)
            _add_module(networks, net, proj, {"lora_up.weight": up, "lora_down.weight": down}, networks.network_lora.NetworkModuleLora)
            deltas.append(up.double() @ down.double())
        expected = expected + torch.cat(deltas)
        nets.append(net)

    networks._set_loaded_networks(nets)
    with _autocast(True):
        networks.network_apply_weights(layer)

    assert torch.equal(getattr(layer, field), expected.to(torch.bfloat16))


@pytest.mark.parametrize("autocast", [False, True])
def test_oft_merges_into_bf16_layer(bf16_lora, autocast):
    """OFT's float32 rotation used to meet the bf16 weight in einsum and raise without autocast (layer skipped)."""
    networks = bf16_lora
    g = torch.Generator().manual_seed(5)
    layer = torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
    layer.network_layer_name = "diffusion_model_layer"
    base = layer.weight.detach().clone()
    blocks = _grid((2, 4, 4), g, 2.0 ** -5)
    net = _add_module(networks, _net(networks, "oft"), layer, {"oft_blocks": blocks, "alpha": torch.tensor(1.0)}, networks.network_oft.NetworkModuleOFT)

    networks._set_loaded_networks([net])
    with _autocast(autocast):
        networks.network_apply_weights(layer)

    q = blocks.double() - blocks.double().transpose(1, 2)
    eye = torch.eye(4, dtype=torch.float64)
    r = (eye + q) @ torch.linalg.inv(eye - q)
    expected = torch.einsum("k n m, k n i -> k m i", r, base.double().reshape(2, 4, 8)).reshape(8, 8)
    # float32 math rounded once: within half a bf16 ulp of the fp64 rotation (one ulp of slack for float32 inverse error)
    assert ((layer.weight.double() - expected).abs() <= 2.0 ** -8 * expected.abs() + 1e-12).all()
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


def test_network_created_bias_is_replaced_when_the_network_set_changes(bf16_lora):
    """A bias-less layer's backup is "no bias"; the bias a network created must not be mistaken for a missing backup
    (which raised "no backup bias found" on the next change to another non-empty network set)."""
    networks = bf16_lora
    layer = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    layer.network_layer_name = "diffusion_model_layer"
    base = layer.weight.detach().clone()
    diff, diff_b = torch.full((4, 4), 0.25, dtype=torch.float16), torch.arange(4, dtype=torch.float16)
    first = _add_module(networks, _net(networks, "full_a"), layer, {"diff": diff, "diff_b": diff_b}, networks.network_full.NetworkModuleFull)
    second = _add_module(networks, _net(networks, "full_b", 0.5), layer, {"diff": diff, "diff_b": diff_b}, networks.network_full.NetworkModuleFull)

    networks._set_loaded_networks([first])
    networks.network_apply_weights(layer)
    assert torch.equal(layer.bias, diff_b.to(torch.bfloat16))

    networks._set_loaded_networks([second])
    networks.network_apply_weights(layer)
    assert torch.equal(layer.bias, (diff_b * 0.5).to(torch.bfloat16))
    assert torch.equal(layer.weight, (base.double() + 0.125).to(torch.bfloat16))

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


def _publish_layers(networks, monkeypatch, *layers):
    monkeypatch.setattr(networks.shared, "sd_model", SimpleNamespace(network_layer_mapping={layer.network_layer_name: layer for layer in layers}), raising=False)


def test_mis_shaped_lora_fails_publication_and_keeps_the_previous_state(bf16_lora, monkeypatch):
    """A delta that only broadcasts to the layer (an [out, 1] column) used to be added to every column; one that does
    not broadcast was counted and the layer skipped. Either way the request ran without the requested network."""
    networks = bf16_lora
    g = torch.Generator().manual_seed(31)
    layer = torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
    layer.network_layer_name = "diffusion_model_attn2_to_k"
    _publish_layers(networks, monkeypatch, layer)
    good = _lora(networks, layer, "good", _grid((8, 2), g), _grid((2, 8), g), 2.0, 1.0)
    networks._publish_applied_state([good])
    merged = layer.weight.detach().clone()
    published_key = networks._applied_state_key

    bad = _lora(networks, layer, "bad", _grid((8, 1), g), _grid((1, 1), g), 1.0, 1.0)
    with pytest.raises(RuntimeError, match=r"LoRA bad cannot be applied to layer diffusion_model_attn2_to_k: delta shape \(8, 1\) != weight shape \(8, 8\)"):
        networks._publish_applied_state([good, bad])

    assert torch.equal(layer.weight, merged)
    assert networks.loaded_networks == [good] and networks._applied_state_key == published_key


def test_failing_lora_merge_raises_naming_network_and_layer(bf16_lora):
    networks = bf16_lora
    layer = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    layer.network_layer_name = "diffusion_model_layer"
    base = layer.weight.detach().clone()
    net = _lora(networks, layer, "broken", torch.ones(4, 1), torch.ones(1, 4), 1.0, 1.0)
    net.modules[layer.network_layer_name].calc_updown = lambda _weight: (_ for _ in ()).throw(ValueError("injected"))

    networks._set_loaded_networks([net])
    with pytest.raises(RuntimeError, match="LoRA broken cannot be applied to layer diffusion_model_layer: injected"):
        networks.network_apply_weights(layer)
    assert torch.equal(layer.weight, base) and layer.network_current_names == ()


def test_qkv_projection_keys_on_a_plain_layer_raise(bf16_lora):
    networks = bf16_lora
    layer = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    layer.network_layer_name = "diffusion_model_layer"
    proj = torch.nn.Linear(4, 4, bias=False)
    proj.network_layer_name = "diffusion_model_layer_q_proj"
    net = _lora(networks, proj, "qonly", torch.ones(4, 1), torch.ones(1, 4), 1.0, 1.0)

    networks._set_loaded_networks([net])
    with pytest.raises(RuntimeError, match="q/k/v projection keys for layer diffusion_model_layer, which is a Linear"):
        networks.network_apply_weights(layer)


def _qkv_layer(layer_kind):
    if layer_kind == "mha":
        layer = torch.nn.MultiheadAttention(8, 2, bias=False, batch_first=True, dtype=torch.bfloat16)
        layer.network_layer_name = "1_model_transformer_resblocks_0_attn"
        return layer, "in_proj_weight"
    from modules.models.sd3.mmdit import QkvLinear
    layer = QkvLinear(8, 24, bias=False, dtype=torch.bfloat16)
    layer.network_layer_name = "diffusion_model_joint_blocks_0_x_block_attn_qkv"
    return layer, "weight"


def _partial_qkv_net(networks, layer, generator, parts):
    """A LoRA with modules for the projections in `parts` only; returns it and the expected [q; k; v] float64 delta."""
    net = _net(networks, "partial")
    proj = torch.nn.Linear(8, 8, bias=False)
    deltas = []
    for part in "qkv":
        if part not in parts:
            deltas.append(torch.zeros(8, 8, dtype=torch.float64))
            continue
        proj.network_layer_name = f"{layer.network_layer_name}_{part}_proj"
        up, down = _grid((8, 1), generator), _grid((1, 8), generator)
        _add_module(networks, net, proj, {"lora_up.weight": up, "lora_down.weight": down}, networks.network_lora.NetworkModuleLora)
        deltas.append(up.double() @ down.double())
    return net, torch.cat(deltas)


@pytest.mark.parametrize("layer_kind", ["mha", "sd3_qkv_linear"])
@pytest.mark.parametrize("parts", ["qv", "k"])
def test_lora_with_only_some_qkv_projections_applies_them(bf16_lora, layer_kind, parts):
    """clip_g's MultiheadAttention LoRA with q and v but no k was dropped whole without any error."""
    networks = bf16_lora
    layer, field = _qkv_layer(layer_kind)
    base = getattr(layer, field).detach().clone()
    net, delta = _partial_qkv_net(networks, layer, torch.Generator().manual_seed(33), parts)

    networks._set_loaded_networks([net])
    networks.network_apply_weights(layer)

    assert torch.equal(getattr(layer, field), (base.double() + delta).to(torch.bfloat16))
    assert not torch.equal(getattr(layer, field), base)


def test_quant_managed_qkv_linear_applies_partial_qkv_lora(bf16_lora, monkeypatch):
    networks = bf16_lora
    layer, _field = _qkv_layer("sd3_qkv_linear")
    layer.network_nvfp4_base_weight = layer.weight.detach().clone()
    layer.network_nvfp4_base_bias = None
    net, delta = _partial_qkv_net(networks, layer, torch.Generator().manual_seed(35), "qv")
    networks._set_loaded_networks([net])
    seen = []
    monkeypatch.setitem(sys.modules, "torchao.quantization", SimpleNamespace(quantize_=lambda module, *_a, **_k: seen.append(module.weight.detach().clone())))
    backend = dataclasses.replace(NVFP4, make_config=lambda: "cfg", validate_config=lambda _config: None, tensor_type=lambda: torch.nn.Parameter)

    assert networks.network_apply_quant_merged_lora(backend, layer)
    assert torch.equal(seen[0], (layer.network_nvfp4_base_weight.double() + delta).to(torch.bfloat16))


def test_lora_set_change_touches_only_the_layers_of_changed_networks(bf16_lora, monkeypatch):
    """A layer no old or new network touches was backed up to the CPU and copied back from the backup on every LoRA
    set change; a layer whose networks did not change was restored and re-merged to the same bits."""
    networks = bf16_lora
    g = torch.Generator().manual_seed(41)
    shared = torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
    shared.network_layer_name = "diffusion_model_shared"
    only_b = torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
    only_b.network_layer_name = "diffusion_model_only_b"
    untouched = torch.nn.Linear(8, 8, bias=True, dtype=torch.bfloat16)
    untouched.network_layer_name = "diffusion_model_untouched"
    a = _lora(networks, shared, "a", _grid((8, 2), g), _grid((2, 8), g), 2.0, 0.5)
    b = _lora(networks, only_b, "b", _grid((8, 2), g), _grid((2, 8), g), 2.0, 0.75)
    base_shared, base_b = shared.weight.detach().clone(), only_b.weight.detach().clone()
    untouched_storage = (untouched.weight.data_ptr(), untouched.bias.data_ptr())
    untouched_values = (untouched.weight.detach().clone(), untouched.bias.detach().clone())
    merges, restores = [], []
    calc = a.modules[shared.network_layer_name].calc_updown
    monkeypatch.setattr(a.modules[shared.network_layer_name], "calc_updown", lambda w: merges.append("a") or calc(w))
    real_restore = networks.restore_weights_backup
    monkeypatch.setattr(networks, "restore_weights_backup", lambda obj, field, w: restores.append(obj.network_layer_name) or real_restore(obj, field, w))

    def apply(nets):
        networks._set_loaded_networks(nets)
        for layer in (shared, only_b, untouched):
            networks.network_apply_weights(layer)

    apply([a])
    merged_a = shared.weight.detach().clone()
    apply([a, b])
    apply([b])
    apply([])

    assert merges == ["a"]  # [a] -> [a, b] keeps shared's merge; [a, b] -> [b] restores it once
    assert sorted(set(restores)) == ["diffusion_model_only_b", "diffusion_model_shared"]
    assert not hasattr(untouched, "network_weights_backup") and not hasattr(untouched, "network_current_names")
    assert (untouched.weight.data_ptr(), untouched.bias.data_ptr()) == untouched_storage
    assert torch.equal(untouched.weight, untouched_values[0]) and torch.equal(untouched.bias, untouched_values[1])
    assert torch.equal(merged_a, (base_shared.double() + 0.5 * (a.modules[shared.network_layer_name].up_model.weight.double() @ a.modules[shared.network_layer_name].down_model.weight.double())).to(torch.bfloat16))
    assert torch.equal(shared.weight, base_shared) and torch.equal(only_b.weight, base_b)


def test_switching_lora_functional_republishes_the_same_networks(bf16_lora, monkeypatch):
    """A lora_functional request restores the base weights in its forwards. The next merged request with the same
    networks was an applied-state hit: nothing re-merged before sampling, and CUDA graphs captured on the merged
    weights (which replay without running the lazy merge) read the base weights."""
    networks = bf16_lora
    g = torch.Generator().manual_seed(43)
    layer = torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
    layer.network_layer_name = "diffusion_model_layer"
    _publish_layers(networks, monkeypatch, layer)
    base = layer.weight.detach().clone()
    net = _lora(networks, layer, "a", _grid((8, 2), g), _grid((2, 8), g), 2.0, 1.0)
    notes = []
    monkeypatch.setattr(networks.openclaw_cuda_graphs, "note_lora_loaded", lambda: notes.append(networks.shared.opts.lora_functional))

    monkeypatch.setattr(networks.shared.opts, "lora_functional", False, raising=False)
    assert networks._publish_applied_state([net])
    merged = layer.weight.detach().clone()
    assert not torch.equal(merged, base)

    monkeypatch.setattr(networks.shared.opts, "lora_functional", True)
    assert networks._publish_applied_state([net])
    networks.network_forward(layer, torch.ones(1, 8, dtype=torch.bfloat16), torch.nn.Linear.forward)
    assert torch.equal(layer.weight, base)

    monkeypatch.setattr(networks.shared.opts, "lora_functional", False)
    assert networks._publish_applied_state([net])
    assert torch.equal(layer.weight, merged)  # re-merged at publication, before any forward
    assert notes == [False, True, False]
    assert not networks._publish_applied_state([net])  # an unchanged mode is still a hit


def test_lora_factors_keep_the_file_dtype_and_merge_closer_to_the_fp64_reference(bf16_lora):
    """The factors were rounded to devices.dtype (bf16) at parse: 8 mantissa bits for fp16 files' 11."""
    networks = bf16_lora
    g = torch.Generator().manual_seed(45)
    layer = torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
    layer.network_layer_name = "diffusion_model_layer"
    with torch.no_grad():
        layer.weight.copy_(torch.randn(64, 64, generator=g) * 0.03)
    base = layer.weight.detach().clone()
    up, down = (torch.randn(64, 8, generator=g) * 0.02).half(), (torch.randn(8, 64, generator=g) * 0.05).half()
    net = _lora(networks, layer, "fp16", up, down, 4.0, 0.8)
    module = net.modules[layer.network_layer_name]
    assert module.up_model.weight.dtype == torch.float16 and torch.equal(module.up_model.weight, up)
    assert module.down_model.weight.dtype == torch.float16 and torch.equal(module.down_model.weight, down)

    networks._set_loaded_networks([net])
    networks.network_apply_weights(layer)

    scale = 0.8 * 4.0 / 8
    reference = base.double() + scale * (up.double() @ down.double())
    bf16_factors = (base.float() + scale * (up.bfloat16().float() @ down.bfloat16().float())).bfloat16()
    error = (layer.weight.double() - reference).abs().sum()
    assert error < (bf16_factors.double() - reference).abs().sum()
    assert (layer.weight != reference.bfloat16()).sum() < (bf16_factors != reference.bfloat16()).sum()


def test_functional_lora_forward_casts_file_dtype_factors_to_the_input(bf16_lora):
    networks = bf16_lora
    g = torch.Generator().manual_seed(47)
    layer = torch.nn.Conv2d(4, 4, 3, padding=1, bias=False, dtype=torch.bfloat16)
    layer.network_layer_name = "diffusion_model_conv"
    up, down = _grid((4, 2, 1, 1), g), _grid((2, 4, 3, 3), g)
    module = _lora(networks, layer, "conv", up, down, 2.0, 1.0).modules[layer.network_layer_name]
    x = torch.randn(1, 4, 5, 5, generator=g).bfloat16()
    y = layer(x)

    out = module.forward(x, y)

    expected = y + torch.nn.functional.conv2d(torch.nn.functional.conv2d(x, down.bfloat16(), padding=1), up.bfloat16())
    assert out.dtype == torch.bfloat16 and torch.equal(out, expected)


def test_functional_lora_keeps_float32_text_encoder_inputs_under_upcast_sampling(bf16_lora, monkeypatch):
    # --upcast-sampling: cond_cast_unet casts layer inputs to the bf16 UNet dtype; a float32 text encoder layer
    # (sd_models.float32_text_encoder_names) must get its float32 input, or F.linear sees mixed dtypes.
    networks = bf16_lora
    monkeypatch.setattr(networks.devices, "unet_needs_upcast", True)
    monkeypatch.setattr(networks.devices, "dtype_unet", torch.bfloat16)
    generator = torch.Generator().manual_seed(0)
    layer = torch.nn.Linear(4, 4)
    layer.network_layer_name = "transformer_text_model_encoder_layers_0_mlp_fc1"
    up, down = _grid((4, 1), generator), _grid((1, 4), generator)
    networks._set_loaded_networks([_lora(networks, layer, "te", up, down, 1.0, 0.5)])
    x = torch.randn(3, 4, generator=generator)

    with torch.no_grad():
        y = networks.network_forward(layer, x, torch.nn.Linear.forward)
        expected = x.double() @ layer.weight.double().T + layer.bias.double() + 0.5 * (x.double() @ down.double().T @ up.double().T)

    assert y.dtype == torch.float32
    torch.testing.assert_close(y.double(), expected, rtol=1e-6, atol=1e-6)

    unet_layer = torch.nn.Linear(4, 4, dtype=torch.bfloat16)  # UNet layers still get the bf16 cast
    unet_layer.network_layer_name = "diffusion_model_layer"
    with torch.no_grad():
        assert networks.network_forward(unet_layer, x, torch.nn.Linear.forward).dtype == torch.bfloat16
