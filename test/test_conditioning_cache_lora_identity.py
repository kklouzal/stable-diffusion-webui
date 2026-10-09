"""The conditioning cache keys on the LoRA state the text encoders run with: U-Net-only changes hit, text encoder
changes miss (networks.current_text_encoder_state_identity)."""

from types import SimpleNamespace

import pytest


TE_LAYER = "0_transformer_text_model_encoder_layers_0_mlp_fc1"
TE2_LAYER = "1_model_transformer_resblocks_0_mlp_c_fc"
UNET_LAYER = "diffusion_model_input_blocks_4_1_transformer_blocks_0_attn2_to_k"


@pytest.fixture
def networks(lora_networks, monkeypatch):
    networks = lora_networks
    parsed = {}
    for name, layers in {"te": (TE_LAYER, UNET_LAYER), "te2": (TE2_LAYER,), "unet": (UNET_LAYER,)}.items():
        on_disk = SimpleNamespace(filename=f"{name}.safetensors", shorthash=name, read_hash=lambda: None)
        net = networks.network.Network(name, on_disk)
        net.modules = {layer: SimpleNamespace(network=net) for layer in layers}
        parsed[name] = net
    monkeypatch.setattr(networks, "available_networks", {name: net.network_on_disk for name, net in parsed.items()}, raising=False)
    monkeypatch.setattr(networks, "available_network_aliases", dict(networks.available_networks), raising=False)
    monkeypatch.setattr(networks, "network_file_signature", lambda filename: ("sha256", str(filename)))
    monkeypatch.setattr(networks, "load_network", lambda name, on_disk: parsed[name])
    monkeypatch.setattr(networks.shared.opts, "lora_functional", False, raising=False)
    return networks


def _identity(networks, names, te=None, unet=None, dyn=None):
    networks.load_networks(names, te or [1.0] * len(names), unet or [1.0] * len(names), dyn or [None] * len(names))
    return networks.current_text_encoder_state_identity()


def test_layer_keys_split_text_encoder_from_unet(networks):
    assert networks.network.is_text_encoder_key(TE_LAYER)
    assert networks.network.is_text_encoder_key(TE2_LAYER)
    assert networks.network.is_text_encoder_key("transformer_text_model_encoder_layers_0_mlp_fc1")  # SD1
    assert not networks.network.is_text_encoder_key(UNET_LAYER)


def test_unet_only_changes_keep_the_identity(networks):
    none = _identity(networks, [])
    assert _identity(networks, ["unet"]) == none
    assert _identity(networks, ["unet"], unet=[0.3], dyn=[4]) == none

    base = _identity(networks, ["te"])
    assert base != none
    assert _identity(networks, ["te"], unet=[0.25]) == base
    assert _identity(networks, ["te", "unet"]) == base
    assert _identity(networks, ["unet", "te"], unet=[0.5, 2.0]) == base


def test_text_encoder_changes_change_the_identity(networks, monkeypatch):
    base = _identity(networks, ["te", "te2"])
    seen = [base]
    for identity in (
        _identity(networks, ["te"]),
        _identity(networks, ["te2", "te"]),  # order of the per-layer delta sum
        _identity(networks, ["te", "te2"], te=[0.5, 1.0]),
        _identity(networks, ["te", "te2"], dyn=[4, None]),
    ):
        assert identity not in seen
        seen.append(identity)

    monkeypatch.setattr(networks.shared.opts, "lora_functional", True, raising=False)
    assert _identity(networks, ["te", "te2"]) not in seen
