import os
import sys
from types import SimpleNamespace

import pytest


class _TestOpts(SimpleNamespace):
    def __getattr__(self, name):
        return False


@pytest.fixture
def lora_networks(monkeypatch):
    import importlib
    import modules as modules_pkg
    from modules import shared as imported_shared

    shared = imported_shared
    if getattr(shared, "__file__", None) is None:
        sys.modules.pop("modules.shared", None)
        if hasattr(modules_pkg, "shared"):
            delattr(modules_pkg, "shared")
        shared = importlib.import_module("modules.shared")

    if not hasattr(shared, "cmd_opts"):
        shared.cmd_opts = _TestOpts()
    monkeypatch.setattr(shared.cmd_opts, "use_ipex", False, raising=False)
    monkeypatch.setattr(shared.cmd_opts, "use_cpu", [], raising=False)

    if not hasattr(shared, "state"):
        shared.state = _TestOpts()

    if shared.opts is None:
        shared.opts = _TestOpts()
    monkeypatch.setattr(shared.opts, "hide_samplers", [], raising=False)
    monkeypatch.setattr(shared.opts, "samples_format", "png", raising=False)
    monkeypatch.setattr(shared.opts, "lora_in_memory_limit", 10, raising=False)
    monkeypatch.setattr(shared.opts, "lora_bundled_ti_to_infotext", False, raising=False)
    monkeypatch.setattr(shared.opts, "lora_not_found_warning_console", False, raising=False)
    monkeypatch.setattr(shared.opts, "lora_not_found_gradio_warning", False, raising=False)

    sys.path.insert(0, "extensions-builtin/Lora")
    import networks

    embedding_db = SimpleNamespace(
        word_embeddings={},
        expected_shape=-1,
        skipped_embeddings={},
        register_embedding_by_name=lambda *args, **kwargs: None,
        register_embedding=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(networks.sd_hijack.model_hijack, "embedding_db", embedding_db, raising=False)
    monkeypatch.setattr(networks.sd_hijack.model_hijack, "comments", [], raising=False)
    monkeypatch.setattr(networks.shared, "sd_model", SimpleNamespace(network_layer_mapping={}), raising=False)
    monkeypatch.setattr(networks.devices, "torch_gc", lambda: None)
    monkeypatch.setattr(networks.openclaw_cuda_graphs, "note_lora_loaded", lambda reason="lora_changed": None)

    on_disk = SimpleNamespace(filename="alpha.safetensors", shorthash="abc", read_hash=lambda: None)
    monkeypatch.setattr(networks, "available_networks", {"alpha": on_disk}, raising=False)
    monkeypatch.setattr(networks, "available_network_aliases", {"alpha": on_disk, "alpha-alias": on_disk}, raising=False)
    monkeypatch.setattr(networks, "forbidden_network_aliases", {}, raising=False)
    monkeypatch.setattr(networks, "networks_in_memory", {}, raising=False)
    networks.loaded_networks.clear()
    yield networks
    networks.loaded_networks.clear()
    sys.path = [p for p in sys.path if p != "extensions-builtin/Lora"]


def _base_network(networks, name="alpha"):
    base = networks.network.Network(name, networks.available_networks["alpha"])
    base.mtime = 1
    base.source_signature = (20, 10)
    tensor_payload = object()
    module = SimpleNamespace(network=base, tensor_payload=tensor_payload, marker="shared immutable weights")
    base.modules = {"layer": module}
    return base, module, tensor_payload


def test_duplicate_lora_mentions_keep_independent_multiplier_owners(lora_networks, monkeypatch):
    networks = lora_networks
    base, module, tensor_payload = _base_network(networks)
    monkeypatch.setattr(networks, "load_network", lambda name, on_disk: base)

    networks.load_networks(["alpha", "alpha"], [0.25, 0.75], [0.5, 1.5], [4, 8])

    first, second = networks.loaded_networks
    assert first is not second
    assert first.modules["layer"] is not second.modules["layer"]
    assert first.modules["layer"].network is first
    assert second.modules["layer"].network is second
    assert first.modules["layer"].tensor_payload is tensor_payload
    assert second.modules["layer"].tensor_payload is tensor_payload
    assert first.te_multiplier == 0.25
    assert second.te_multiplier == 0.75
    assert first.unet_multiplier == 0.5
    assert second.unet_multiplier == 1.5
    assert first.dyn_dim == 4
    assert second.dyn_dim == 8


def test_duplicate_alias_mentions_clone_module_backrefs(lora_networks, monkeypatch):
    networks = lora_networks
    base, _module, _payload = _base_network(networks, name="alpha-alias")
    monkeypatch.setattr(networks, "load_network", lambda name, on_disk: base)

    networks.load_networks(["alpha-alias", "alpha-alias"], [0.1, 0.9], [0.2, 1.8], [2, 6])

    first, second = networks.loaded_networks
    assert first is not second
    assert first.mentioned_name == "alpha-alias"
    assert second.mentioned_name == "alpha-alias"
    assert first.modules["layer"].network is first
    assert second.modules["layer"].network is second
    assert first.te_multiplier == 0.1
    assert second.te_multiplier == 0.9


def test_wanted_names_include_source_signature_for_stale_weight_invalidation(lora_networks):
    networks = lora_networks
    on_disk = SimpleNamespace(filename="alpha.safetensors", shorthash="abc")
    net = networks.network.Network("alpha", on_disk)
    net.mentioned_name = "alias-alpha"
    net.te_multiplier = 1.0
    net.unet_multiplier = 1.0
    net.dyn_dim = None
    net.mtime = 111
    net.source_signature = (1234, 5678)
    networks.loaded_networks[:] = [net]

    first_signature = networks.network_wanted_names()
    net.source_signature = (9999, 5678)
    second_signature = networks.network_wanted_names()

    assert first_signature != second_signature
    assert second_signature[0][-1] == (9999, 5678)


def test_network_file_signature_uses_mtime_ns_then_size(lora_networks, tmp_path):
    networks = lora_networks
    lora_file = tmp_path / "same-name.safetensors"
    lora_file.write_bytes(b"abcd")
    os.utime(lora_file, ns=(111_000_000_001, 222_000_000_002))
    assert networks.network_file_signature(lora_file) == (222_000_000_002, 4)

    first = networks.network_file_signature(lora_file)
    lora_file.write_bytes(b"wxyz")
    os.utime(lora_file, ns=(333_000_000_003, 444_000_000_004))
    second = networks.network_file_signature(lora_file)

    assert first != second
    assert second == (444_000_000_004, 4)


def test_duplicate_aliases_to_same_lora_keep_independent_multiplier_owners(lora_networks, monkeypatch):
    networks = lora_networks
    base, _module, _payload = _base_network(networks)
    calls = []
    monkeypatch.setattr(networks, "network_file_signature", lambda filename: base.source_signature)
    monkeypatch.setattr(networks, "load_network", lambda name, on_disk: calls.append(name) or base)

    networks.load_networks(["alpha", "alpha-alias"], [0.2, 0.8], [0.3, 1.3], [3, 9])

    first, second = networks.loaded_networks
    assert calls == ["alpha"]
    assert first is not second
    assert first.mentioned_name == "alpha"
    assert second.mentioned_name == "alpha-alias"
    assert first.modules["layer"].network is first
    assert second.modules["layer"].network is second
    assert first.te_multiplier == 0.2
    assert second.te_multiplier == 0.8
    assert first.unet_multiplier == 0.3
    assert second.unet_multiplier == 1.3
