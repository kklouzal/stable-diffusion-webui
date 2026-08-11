import os
import sys
from pathlib import Path
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
    monkeypatch.setattr(networks, "available_network_aliases", {"alpha-alias": on_disk, "alpha": on_disk}, raising=False)
    monkeypatch.setattr(networks, "forbidden_network_aliases", {}, raising=False)
    monkeypatch.setattr(networks, "networks_in_memory", {}, raising=False)
    networks.loaded_networks.clear()
    monkeypatch.setattr(networks, "_applied_state_key", None, raising=False)
    networks.openclaw_cache_epochs.reset_for_tests()
    yield networks
    networks.loaded_networks.clear()
    networks.openclaw_cache_epochs.reset_for_tests()
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
    assert second_signature[0][0][1] == (9999, 5678)


def test_same_size_restored_mtime_rewrite_misses(lora_networks, tmp_path):
    networks = lora_networks
    lora_file = tmp_path / "same-name.safetensors"
    lora_file.write_bytes(b"abcd")
    os.utime(lora_file, ns=(111_000_000_001, 222_000_000_002))
    first = networks.network_file_signature(lora_file)
    lora_file.write_bytes(b"wxyz")
    os.utime(lora_file, ns=(111_000_000_001, 222_000_000_002))
    second = networks.network_file_signature(lora_file)
    assert first != second
    assert first[0] == second[0] == "sha256"

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


def test_repeated_requested_name_reuses_cache_despite_alias_insertion_order(lora_networks, monkeypatch):
    networks = lora_networks
    base, _module, tensor_payload = _base_network(networks)
    calls = []
    monkeypatch.setattr(networks, "network_file_signature", lambda filename: base.source_signature)
    monkeypatch.setattr(networks, "load_network", lambda name, on_disk: calls.append(name) or base)

    networks.load_networks(["alpha"], [0.2], [0.3], [3])
    first = networks.loaded_networks[0]
    networks.load_networks(["alpha"], [0.8], [1.3], [9])
    second = networks.loaded_networks[0]

    assert calls == ["alpha"]
    assert second is not first
    assert second.modules["layer"].tensor_payload is tensor_payload
    assert second.modules["layer"].network is second
    assert second.te_multiplier == 0.8
    assert second.unet_multiplier == 1.3
    assert second.dyn_dim == 9


def test_alias_switch_reuses_source_tensors_and_multiplier_owner(lora_networks, monkeypatch):
    networks = lora_networks
    base, _module, tensor_payload = _base_network(networks)
    calls = []
    monkeypatch.setattr(networks, "network_file_signature", lambda filename: base.source_signature)
    monkeypatch.setattr(networks, "load_network", lambda name, on_disk: calls.append(name) or base)

    networks.load_networks(["alpha"], [0.2], [0.3], [3])
    networks.load_networks(["alpha-alias"], [0.8], [1.3], [9])
    switched = networks.loaded_networks[0]

    assert calls == ["alpha"]
    assert switched.modules["layer"].tensor_payload is tensor_payload
    assert switched.modules["layer"].network is switched
    assert switched.mentioned_name == "alpha-alias"
    assert switched.te_multiplier == 0.8
    assert switched.unet_multiplier == 1.3
    assert switched.dyn_dim == 9


def test_changed_source_signature_forces_reload(lora_networks, monkeypatch):
    networks = lora_networks
    signature = [(20, 10)]
    calls = []

    def load_network(name, on_disk):
        net, _module, _payload = _base_network(networks, name=name)
        net.source_signature = signature[0]
        calls.append((name, net))
        return net

    monkeypatch.setattr(networks, "network_file_signature", lambda filename: signature[0])
    monkeypatch.setattr(networks, "load_network", load_network)

    networks.load_networks(["alpha-alias"])
    first = networks.loaded_networks[0]
    signature[0] = (21, 10)
    networks.load_networks(["alpha-alias"])
    second = networks.loaded_networks[0]

    assert len(calls) == 2
    assert second is not first
    assert second.source_signature == (21, 10)


def _epoch(networks, dimension):
    return dict(networks.openclaw_cache_epochs.epoch_subset((dimension,)))[dimension]


def test_applied_identity_change_only_bumps(lora_networks, monkeypatch):
    networks = lora_networks
    base, _module, _payload = _base_network(networks)
    base.source_key = ("opaque", ("sha256", "a"), networks.LORA_SOURCE_SCHEMA_REVISION, ())
    monkeypatch.setattr(networks, "network_file_signature", lambda _filename: ("sha256", "a"))
    monkeypatch.setattr(networks, "network_source_key", lambda *_args: base.source_key)
    monkeypatch.setattr(networks, "load_network", lambda *_args: base)

    networks.load_networks(["alpha"], [0.5], [0.5], [4])
    first = _epoch(networks, "lora_applied_epoch")
    networks.load_networks(["alpha"], [0.5], [0.5], [4])
    assert _epoch(networks, "lora_applied_epoch") == first
    networks.load_networks(["alpha"], [0.75], [0.5], [4])
    assert _epoch(networks, "lora_applied_epoch") == first + 1
    assert networks.unload_networks()
    assert _epoch(networks, "lora_applied_epoch") == first + 2
    assert not networks.unload_networks()


def test_reorder_dyn_dim_checkpoint_precision_device_change_identity(lora_networks, monkeypatch):
    networks = lora_networks
    a, _, _ = _base_network(networks, "a")
    b, _, _ = _base_network(networks, "b")
    a.source_key = ("a",); b.source_key = ("b",)
    a.te_multiplier = a.unet_multiplier = b.te_multiplier = b.unet_multiplier = 1.0
    a.dyn_dim = b.dyn_dim = 4
    monkeypatch.setattr(networks, "_execution_identity", lambda: ("cpu", "float32", (("checkpoint_object_epoch", 0),)))
    assert networks.network_applied_state_key([a, b]) != networks.network_applied_state_key([b, a])
    before = networks.network_applied_state_key([a])
    a.dyn_dim = 8
    assert before != networks.network_applied_state_key([a])
    before = networks.network_applied_state_key([a])
    monkeypatch.setattr(networks, "_execution_identity", lambda: ("cuda:0", "float16", (("checkpoint_object_epoch", 1),)))
    assert before != networks.network_applied_state_key([a])


def test_failed_apply_restores_without_epoch_or_stale_publish(lora_networks, monkeypatch):
    networks = lora_networks
    old, _, _ = _base_network(networks, "old")
    old.source_key = ("old",); old.te_multiplier = old.unet_multiplier = 1.0; old.dyn_dim = None
    networks.loaded_networks[:] = [old]
    networks._applied_state_key = networks.network_applied_state_key([old])
    new, _, _ = _base_network(networks, "new")
    new.source_key = ("new",); new.te_multiplier = new.unet_multiplier = 1.0; new.dyn_dim = None
    calls = []
    def apply():
        calls.append(tuple(x.source_key for x in networks.loaded_networks))
        if networks.loaded_networks and networks.loaded_networks[0] is new:
            raise RuntimeError("injected")
    monkeypatch.setattr(networks, "_apply_loaded_state_to_model", apply)
    before = _epoch(networks, "lora_applied_epoch")
    with pytest.raises(RuntimeError, match="injected"):
        networks._publish_applied_state([new])
    assert networks.loaded_networks == [old]
    assert networks._applied_state_key == networks.network_applied_state_key([old])
    assert _epoch(networks, "lora_applied_epoch") == before
    assert calls == [(("new",),), (("old",),)]


def test_irrelevant_generation_inputs_not_in_applied_key(lora_networks):
    source = Path("extensions-builtin/Lora/networks.py").read_text()
    key_block = source[source.index("def network_applied_state_key"):source.index("def _apply_loaded_state_to_model")]
    for forbidden in ("prompt", "seed", "cfg", "sampler"):
        assert forbidden not in key_block.lower()


def test_s05_lock_order_telemetry_and_dependency_consumers_are_static_contracts():
    source = Path("extensions-builtin/Lora/networks.py").read_text()
    publish = source[source.index("def _publish_applied_state"):source.index("def unload_networks")]
    assert publish.index("epoch_transaction") < publish.index("_network_application_lock")
    assert publish.count('bump_epoch("lora_applied_epoch"') == 1
    assert 'observe("E12", "reject"' in publish
    assert "semantic_key=wanted_key" in publish
    assert "lora_applied_epoch" in Path("modules/processing.py").read_text()
    assert "lora_applied_epoch" in Path("modules/openclaw_cuda_graphs.py").read_text()
    assert "lora_applied_epoch" in Path("modules/mxfp8_diagnostics.py").read_text()


def test_epoch_transaction_prevents_lifecycle_interleave_with_apply_publication(lora_networks):
    import threading
    networks = lora_networks
    entered = threading.Event(); release = threading.Event(); lifecycle_done = threading.Event()
    def apply_publish():
        with networks.openclaw_cache_epochs.epoch_transaction():
            entered.set()
            assert release.wait(2)
            networks.openclaw_cache_epochs.bump_epoch("lora_applied_epoch", reason="published")
    def lifecycle():
        assert entered.wait(2)
        networks.openclaw_cache_epochs.bump_epoch("checkpoint_object_epoch", reason="checkpoint_commit")
        lifecycle_done.set()
    first = threading.Thread(target=apply_publish); second = threading.Thread(target=lifecycle)
    first.start(); second.start()
    assert entered.wait(2) and not lifecycle_done.wait(0.05)
    release.set(); first.join(2); second.join(2)
    assert lifecycle_done.is_set()


def test_generation_owner_rejects_cross_request_overlap_and_cleans_exception(lora_networks):
    import threading
    networks = lora_networks
    entered = threading.Event()
    release = threading.Event()
    rejected = []

    def first_request():
        with networks.openclaw_cache_epochs.generation_owner():
            entered.set()
            assert release.wait(2)

    def second_request():
        assert entered.wait(2)
        try:
            with networks.openclaw_cache_epochs.generation_owner():
                pass
        except networks.openclaw_cache_epochs.GenerationOwnerError:
            rejected.append(True)

    first = threading.Thread(target=first_request)
    second = threading.Thread(target=second_request)
    first.start()
    second.start()
    second.join(2)
    assert rejected == [True]
    release.set()
    first.join(2)
    assert not networks.openclaw_cache_epochs.generation_owner_public_summary()["active"]

    with pytest.raises(RuntimeError, match="injected"):
        with networks.openclaw_cache_epochs.generation_owner():
            raise RuntimeError("injected")
    assert not networks.openclaw_cache_epochs.generation_owner_public_summary()["active"]
