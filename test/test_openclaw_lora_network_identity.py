import os
import sys
import threading
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

    class FakeEmbeddingDB:
        def __init__(self):
            self.word_embeddings = {}
            self.ids_lookup = {}
            self.expected_shape = -1
            self.skipped_embeddings = {}
            self.register_calls = []
            self._publication_lock = threading.RLock()
            self.fail_name = None

        def register_embedding_by_name(self, embedding, _model, name):
            self.register_calls.append((name, embedding))
            if embedding is not None and name == self.fail_name:
                raise RuntimeError("injected register failure")
            token = sum(name.encode())
            entries = [entry for entry in self.ids_lookup.get(token, []) if entry[1].name != name]
            if embedding is None:
                self.word_embeddings.pop(name, None)
            else:
                entries.append(([token], embedding))
                self.word_embeddings[name] = embedding
            if entries:
                self.ids_lookup[token] = entries
            else:
                self.ids_lookup.pop(token, None)
            return embedding

        def register_embedding(self, embedding, model):
            return self.register_embedding_by_name(embedding, model, embedding.name)

    embedding_db = FakeEmbeddingDB()
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
    monkeypatch.setattr(networks, "loaded_bundle_embeddings", {}, raising=False)
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


def test_restore_weights_backup_uses_no_grad_for_parameter_weights(lora_networks):
    networks = lora_networks
    module = networks.torch.nn.Linear(1, 1)
    backup = module.weight.detach().clone()
    module.network_weights_backup = backup
    module.network_bias_backup = None

    with networks.torch.no_grad():
        module.weight.add_(1)

    networks.network_restore_weights_from_backup(module)

    assert networks.torch.equal(module.weight, backup)
    assert module.network_weights_backup is not None


def test_apply_loaded_state_skips_conditioner_wrappers_without_weights(lora_networks, monkeypatch):
    networks = lora_networks
    leaf = SimpleNamespace(weight=object())
    wrapper = SimpleNamespace()
    networks.shared.sd_model.network_layer_mapping = {
        "0": wrapper,
        "0_transformer_text_model_encoder_layers_0": leaf,
    }
    applied = []
    monkeypatch.setattr(networks, "network_apply_weights", applied.append)

    networks._apply_loaded_state_to_model()

    assert applied == [leaf]


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
    assert publish.count('bump_epoch("lora_applied_epoch"') == 2
    assert 'observe("E12", "reject"' in publish
    assert "semantic_key=wanted_key" in publish
    assert "current_network_state_identity" in Path("modules/processing.py").read_text()
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




def test_quant_unload_restores_managed_base_and_invalidates_stale_active_config(lora_networks, monkeypatch):
    import torch
    networks = lora_networks
    linear = torch.nn.Linear(2, 2, bias=True, dtype=torch.bfloat16)
    base_weight = linear.weight.detach().clone()
    base_bias = linear.bias.detach().clone()
    with torch.no_grad():
        linear.weight.add_(torch.ones_like(linear.weight))
        linear.bias.add_(torch.ones_like(linear.bias))
    linear.network_layer_name = "layer"
    linear.network_mxfp8_base_weight = base_weight.detach().cpu().clone()
    linear.network_mxfp8_base_bias = base_bias.detach().cpu().clone()
    linear.network_current_names = (("alpha",),)
    linear.network_mxfp8_merged_lora_applied = True
    model = SimpleNamespace(
        network_layer_mapping={"layer": linear},
        network_mxfp8_managed_modules=[("layer", linear)],
        network_mxfp8_active_config_ready=True,
        network_mxfp8_active_config_signature=("stale",),
    )
    monkeypatch.setattr(networks.shared, "sd_model", model, raising=False)
    monkeypatch.setattr(networks.devices, "mxfp8", True, raising=False)
    monkeypatch.setattr(networks.devices, "nvfp4", False, raising=False)
    monkeypatch.setattr(networks.devices, "device", torch.device("cpu"), raising=False)

    assert networks.unload_networks()

    assert torch.equal(linear.weight, base_weight)
    assert torch.equal(linear.bias, base_bias)
    assert linear.network_current_names == ()
    assert linear.network_mxfp8_merged_lora_applied is False
    assert not getattr(model, "network_mxfp8_active_config_ready", False)
    assert not networks.network_mxfp8_is_model_prepared(model)


def test_quant_prepared_check_rejects_same_signature_module_marker_mismatch(lora_networks, monkeypatch):
    import torch
    networks = lora_networks
    linear = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
    linear.network_mxfp8_base_weight = linear.weight.detach().cpu().clone()
    linear.network_mxfp8_base_bias = linear.bias.detach().cpu().clone()
    linear.network_current_names = ()
    model = SimpleNamespace(
        network_mxfp8_managed_modules=[("layer", linear)],
        network_mxfp8_active_config_ready=True,
    )
    net = SimpleNamespace(source_key=("alpha",), te_multiplier=1.0, unet_multiplier=1.0, dyn_dim=None)
    networks.loaded_networks[:] = [net]

    assert networks.network_quant_capture_managed_base(model, "network_mxfp8_base_weight", "network_mxfp8_base_bias") == 0
    networks.network_quant_restore_managed_base(model, "network_mxfp8_base_weight", "network_mxfp8_base_bias", "network_mxfp8_merged_lora_applied")
    assert networks.network_quant_capture_managed_base(model, "network_mxfp8_base_weight", "network_mxfp8_base_bias") == 0
    assert torch.equal(linear.weight, linear.network_mxfp8_base_weight)
    assert not networks.network_mxfp8_is_model_prepared(model)
    linear.network_current_names = networks.network_mxfp8_wanted_names()
    assert networks.network_mxfp8_is_model_prepared(model)


def test_nvfp4_unload_restores_managed_base_and_invalidates_stale_active_config(lora_networks, monkeypatch):
    import torch
    networks = lora_networks
    linear = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    base_weight = linear.weight.detach().clone()
    with torch.no_grad():
        linear.weight.mul_(2)
    linear.network_layer_name = "layer"
    linear.network_nvfp4_base_weight = base_weight.detach().cpu().clone()
    linear.network_nvfp4_base_bias = None
    linear.network_current_names = (("alpha",),)
    linear.network_nvfp4_merged_lora_applied = True
    model = SimpleNamespace(
        network_layer_mapping={"layer": linear},
        network_nvfp4_managed_modules=[("layer", linear)],
        network_nvfp4_active_config_ready=True,
        network_nvfp4_active_config_signature=("stale",),
    )
    monkeypatch.setattr(networks.shared, "sd_model", model, raising=False)
    monkeypatch.setattr(networks.devices, "mxfp8", False, raising=False)
    monkeypatch.setattr(networks.devices, "nvfp4", True, raising=False)
    monkeypatch.setattr(networks.devices, "device", torch.device("cpu"), raising=False)

    assert networks.unload_networks()

    assert torch.equal(linear.weight, base_weight)
    assert linear.bias is None
    assert linear.network_current_names == ()
    assert linear.network_nvfp4_merged_lora_applied is False
    assert not getattr(model, "network_nvfp4_active_config_ready", False)
    assert not networks.network_nvfp4_is_model_prepared(model)

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


def _embedding(name, shape=4):
    return SimpleNamespace(name=name, shape=shape, loaded=None)


def _applied_network(networks, key, bundles=None):
    net, _, _ = _base_network(networks, key)
    net.source_key = (key,)
    net.te_multiplier = net.unet_multiplier = 1.0
    net.dyn_dim = None
    net.bundle_embeddings = bundles or {}
    return net


def _epochs(networks):
    return {name: _epoch(networks, name) for name in (
        "lora_applied_epoch", "textual_inversion_epoch", "tokenizer_epoch",
    )}


def test_bundled_ti_load_noop_and_unload_are_atomic(lora_networks, monkeypatch):
    networks = lora_networks
    db = networks.sd_hijack.model_hijack.embedding_db
    emb = _embedding("hero")
    net = _applied_network(networks, "a", {"hero": emb})
    graph_observations = []
    monkeypatch.setattr(networks, "_apply_loaded_state_to_model", lambda: None)
    monkeypatch.setattr(networks.openclaw_cuda_graphs, "note_lora_loaded", lambda _reason: graph_observations.append((dict(db.word_embeddings), _epochs(networks))))

    before = _epochs(networks)
    assert networks._publish_applied_state([net], db)
    loaded = _epochs(networks)
    assert db.word_embeddings == {"hero": emb}
    assert all(loaded[name] == before[name] + 1 for name in loaded)
    assert not networks._publish_applied_state([net], db)
    assert _epochs(networks) == loaded
    assert len(db.register_calls) == 1
    assert networks.unload_networks()
    unloaded = _epochs(networks)
    assert db.word_embeddings == {}
    assert all(unloaded[name] == loaded[name] + 1 for name in unloaded)
    assert len(graph_observations) == 2
    assert graph_observations[0][0] == {"hero": emb}
    assert graph_observations[0][1] == loaded
    assert graph_observations[1][0] == {}
    assert graph_observations[1][1] == unloaded


def test_bundled_ti_switch_deduplicates_names_and_preserves_folder_collision(lora_networks, monkeypatch):
    networks = lora_networks
    db = networks.sd_hijack.model_hijack.embedding_db
    monkeypatch.setattr(networks, "_apply_loaded_state_to_model", lambda: None)
    old = _embedding("old")
    first_shared = _embedding("shared")
    duplicate_shared = _embedding("shared")
    folder = _embedding("folder")
    db.register_embedding_by_name(folder, networks.shared.sd_model, "folder")
    first = _applied_network(networks, "a", {"old": old})
    second = _applied_network(networks, "b", {"shared": first_shared, "folder": _embedding("folder")})
    duplicate = _applied_network(networks, "c", {"shared": duplicate_shared})

    networks._publish_applied_state([first], db)
    before = _epochs(networks)
    networks._publish_applied_state([second, duplicate], db)
    assert db.word_embeddings == {"folder": folder, "shared": first_shared}
    assert "old" not in db.word_embeddings
    assert _epochs(networks) == {name: value + 1 for name, value in before.items()}
    assert sum(name == "shared" and embedding is not None for name, embedding in db.register_calls) == 1


def test_multi_name_bundled_ti_switch_publishes_complete_maps_atomically(lora_networks, monkeypatch):
    networks = lora_networks
    db = networks.sd_hijack.model_hijack.embedding_db
    old_a = _embedding("old-a")
    old_b = _embedding("old-b")
    new_a = _embedding("new-a")
    new_b = _embedding("new-b")
    previous = {"old-a": old_a, "old-b": old_b}
    planned = {"new-a": new_a, "new-b": new_b}
    for name, embedding in previous.items():
        db.register_embedding_by_name(embedding, networks.shared.sd_model, name)

    reached_intermediate = threading.Event()
    allow_completion = threading.Event()
    original_register = type(db).register_embedding_by_name

    def pausing_register(staged_db, embedding, model, name):
        original_register(staged_db, embedding, model, name)
        if name == "new-a":
            reached_intermediate.set()
            assert allow_completion.wait(timeout=5)

    monkeypatch.setattr(type(db), "register_embedding_by_name", pausing_register)
    worker = threading.Thread(target=networks._replace_bundled_embeddings, args=(db, previous, planned))
    worker.start()
    assert reached_intermediate.wait(timeout=5)

    # The intermediate staged maps must not leak to raw or lock-following readers.
    assert db.word_embeddings == previous
    with db._publication_lock:
        assert db.word_embeddings == previous
        assert {entry[1].name for entries in db.ids_lookup.values() for entry in entries} == set(previous)

    allow_completion.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert db.word_embeddings == planned
    assert {entry[1].name for entries in db.ids_lookup.values() for entry in entries} == set(planned)


def test_register_failure_restores_exact_db_weights_state_and_epochs(lora_networks, monkeypatch):
    networks = lora_networks
    db = networks.sd_hijack.model_hijack.embedding_db
    old_emb = _embedding("old")
    old = _applied_network(networks, "old", {"old": old_emb})
    new = _applied_network(networks, "new", {"new": _embedding("new")})
    monkeypatch.setattr(networks, "_apply_loaded_state_to_model", lambda: None)
    networks._publish_applied_state([old], db)
    snapshot = (dict(db.word_embeddings), {key: list(value) for key, value in db.ids_lookup.items()})
    before = _epochs(networks)
    graphs = []
    monkeypatch.setattr(networks.openclaw_cuda_graphs, "note_lora_loaded", lambda reason: graphs.append(reason))
    db.fail_name = "new"

    with pytest.raises(RuntimeError, match="register failure"):
        networks._publish_applied_state([new], db)
    assert networks.loaded_networks == [old]
    assert networks.loaded_bundle_embeddings == {"old": old_emb}
    assert db.word_embeddings == snapshot[0]
    assert db.ids_lookup == snapshot[1]
    assert _epochs(networks) == before
    assert graphs == []


def test_weight_failure_restores_bundles_weights_and_no_publication(lora_networks, monkeypatch):
    networks = lora_networks
    db = networks.sd_hijack.model_hijack.embedding_db
    old = _applied_network(networks, "old", {"old": _embedding("old")})
    new = _applied_network(networks, "new", {"new": _embedding("new")})
    monkeypatch.setattr(networks, "_apply_loaded_state_to_model", lambda: None)
    networks._publish_applied_state([old], db)
    before = _epochs(networks)
    calls = []
    def apply():
        calls.append(list(networks.loaded_networks))
        if networks.loaded_networks == [new]:
            raise RuntimeError("injected weight failure")
    monkeypatch.setattr(networks, "_apply_loaded_state_to_model", apply)

    with pytest.raises(RuntimeError, match="weight failure"):
        networks._publish_applied_state([new], db)
    assert calls == [[new], [old]]
    assert networks.loaded_networks == [old]
    assert set(db.word_embeddings) == {"old"}
    assert _epochs(networks) == before


def test_rollback_failure_publishes_explicit_empty_fail_closed_state(lora_networks, monkeypatch):
    networks = lora_networks
    db = networks.sd_hijack.model_hijack.embedding_db
    old = _applied_network(networks, "old", {"old": _embedding("old")})
    new = _applied_network(networks, "new", {"new": _embedding("new")})
    monkeypatch.setattr(networks, "_apply_loaded_state_to_model", lambda: None)
    networks._publish_applied_state([old], db)
    before = _epochs(networks)
    graphs = []
    attempts = []
    def apply():
        attempts.append(list(networks.loaded_networks))
        if networks.loaded_networks:
            raise RuntimeError("injected apply/rollback failure")
    monkeypatch.setattr(networks, "_apply_loaded_state_to_model", apply)
    monkeypatch.setattr(networks.openclaw_cuda_graphs, "note_lora_loaded", lambda reason: graphs.append(reason))

    with pytest.raises(RuntimeError, match="rollback failed closed"):
        networks._publish_applied_state([new], db)
    assert attempts == [[new], [old], []]
    assert networks.loaded_networks == []
    assert networks.loaded_bundle_embeddings == {}
    assert networks._applied_state_key is None
    assert db.word_embeddings == {}
    assert _epochs(networks) == {name: value + 1 for name, value in before.items()}
    assert graphs == ["lora_changed"]


def test_current_network_state_identity_reuses_semantically_identical_lifecycle(lora_networks, monkeypatch):
    networks = lora_networks
    base, _module, _payload = _base_network(networks)
    base.source_key = ("opaque", ("sha256", "same"), networks.LORA_SOURCE_SCHEMA_REVISION, ())
    monkeypatch.setattr(networks, "network_file_signature", lambda _filename: ("sha256", "same"))
    monkeypatch.setattr(networks, "network_source_key", lambda *_args: base.source_key)
    monkeypatch.setattr(networks, "load_network", lambda *_args: base)

    networks.load_networks(["alpha"], [0.75], [1.25], [4])
    first = networks.current_network_state_identity()
    networks.load_networks([], [], [], [])
    unloaded = networks.current_network_state_identity()
    networks.load_networks(["alpha"], [0.75], [1.25], [4])
    reapplied = networks.current_network_state_identity()

    assert first == reapplied
    assert first != unloaded
    assert networks.loaded_networks[0].te_multiplier == 0.75
    assert networks.loaded_networks[0].unet_multiplier == 1.25
    assert networks.loaded_networks[0].dyn_dim == 4


def test_current_network_state_identity_changes_for_effective_inputs(lora_networks):
    networks = lora_networks
    first, _module, _payload = _base_network(networks)
    first.source_key = ("opaque", ("sha256", "a"), networks.LORA_SOURCE_SCHEMA_REVISION, ())
    baseline = networks.network_applied_state_key([first])

    changed_source, _module, _payload = _base_network(networks)
    changed_source.source_key = ("opaque", ("sha256", "b"), networks.LORA_SOURCE_SCHEMA_REVISION, ())
    changed_te, _module, _payload = _base_network(networks)
    changed_te.source_key = first.source_key
    changed_te.te_multiplier = 0.5
    changed_unet, _module, _payload = _base_network(networks)
    changed_unet.source_key = first.source_key
    changed_unet.unet_multiplier = 0.5
    changed_dyn, _module, _payload = _base_network(networks)
    changed_dyn.source_key = first.source_key
    changed_dyn.dyn_dim = 8

    assert baseline != networks.network_applied_state_key([changed_source])
    assert baseline != networks.network_applied_state_key([changed_te])
    assert baseline != networks.network_applied_state_key([changed_unet])
    assert baseline != networks.network_applied_state_key([changed_dyn])

def test_generic_apply_skips_nvfp4_managed_layer(lora_networks, monkeypatch):
    import torch
    networks = lora_networks
    layer = torch.nn.Linear(2, 2, bias=False)
    layer.network_layer_name = "managed"
    layer.network_nvfp4_base_weight = layer.weight.detach().cpu().clone()
    layer.network_current_names = ()
    monkeypatch.setattr(networks.devices, "nvfp4", True)
    monkeypatch.setattr(networks, "loaded_networks", [object()])

    networks.network_apply_weights(layer)

    assert layer.network_current_names == ()
    assert not hasattr(layer, "network_weights_backup")


def _mha_delta_module(torch, value):
    class Delta:
        def calc_updown(self, weight):
            return torch.full_like(weight, value), None
    return Delta()


def test_mha_lifecycle_restores_qkv_and_applies_out_proj_exactly_once(lora_networks, monkeypatch):
    torch = pytest.importorskip("torch")
    networks = lora_networks
    mha = torch.nn.MultiheadAttention(4, 1, bias=False, batch_first=True)
    mha.network_layer_name = "1_model_transformer_resblocks_0_attn"
    mha.out_proj.network_layer_name = mha.network_layer_name + "_out_proj"
    base_qkv = mha.in_proj_weight.detach().clone()
    base_out = mha.out_proj.weight.detach().clone()
    net = SimpleNamespace(name="alpha", te_multiplier=1.0, unet_multiplier=1.0, dyn_dim=None, modules={
        mha.network_layer_name + "_q_proj": _mha_delta_module(torch, 1.0),
        mha.network_layer_name + "_k_proj": _mha_delta_module(torch, 2.0),
        mha.network_layer_name + "_v_proj": _mha_delta_module(torch, 3.0),
        mha.network_layer_name + "_out_proj": _mha_delta_module(torch, 4.0),
    })
    monkeypatch.setattr(networks, "loaded_networks", [net])
    monkeypatch.setattr(networks, "network_wanted_names", lambda: (("alpha", 1.0, 1.0, None),))

    networks.network_apply_weights(mha)
    networks.network_apply_weights(mha.out_proj)
    expected_qkv = base_qkv + torch.cat([torch.ones_like(base_out), torch.full_like(base_out, 2), torch.full_like(base_out, 3)])
    assert torch.equal(mha.in_proj_weight, expected_qkv)
    assert torch.equal(mha.out_proj.weight, base_out + 4)

    # Same signature is a no-op; unload and identical reactivation are exact.
    networks.network_apply_weights(mha)
    networks.network_apply_weights(mha.out_proj)
    assert torch.equal(mha.in_proj_weight, expected_qkv)
    assert torch.equal(mha.out_proj.weight, base_out + 4)
    monkeypatch.setattr(networks, "loaded_networks", [])
    monkeypatch.setattr(networks, "network_wanted_names", lambda: ())
    networks.network_apply_weights(mha)
    networks.network_apply_weights(mha.out_proj)
    assert torch.equal(mha.in_proj_weight, base_qkv)
    assert torch.equal(mha.out_proj.weight, base_out)
    monkeypatch.setattr(networks, "loaded_networks", [net])
    monkeypatch.setattr(networks, "network_wanted_names", lambda: (("alpha", 1.0, 1.0, None),))
    networks.network_apply_weights(mha)
    networks.network_apply_weights(mha.out_proj)
    assert torch.equal(mha.in_proj_weight, expected_qkv)
    assert torch.equal(mha.out_proj.weight, base_out + 4)



def test_mha_failed_reactivation_restores_exact_base(lora_networks, monkeypatch):
    torch = pytest.importorskip("torch")
    networks = lora_networks
    monkeypatch.setattr(networks, "extra_network_lora", SimpleNamespace(errors={}))
    mha = torch.nn.MultiheadAttention(4, 1, bias=False, batch_first=True)
    mha.network_layer_name = "1_model_transformer_resblocks_0_attn"
    base_qkv = mha.in_proj_weight.detach().clone()
    class Failing:
        def calc_updown(self, weight):
            raise RuntimeError("injected failure")
    net = SimpleNamespace(name="broken", modules={
        mha.network_layer_name + "_q_proj": _mha_delta_module(torch, 1.0),
        mha.network_layer_name + "_k_proj": Failing(),
        mha.network_layer_name + "_v_proj": _mha_delta_module(torch, 3.0),
    })
    monkeypatch.setattr(networks, "loaded_networks", [net])
    monkeypatch.setattr(networks, "network_wanted_names", lambda: (("broken", 1.0, 1.0, None),))
    networks.network_apply_weights(mha)
    assert torch.equal(mha.in_proj_weight, base_qkv)
    assert mha.network_current_names == (("broken", 1.0, 1.0, None),)


def test_model_level_apply_includes_mha_and_deduplicates_out_proj(lora_networks, monkeypatch):
    torch = pytest.importorskip("torch")
    networks = lora_networks
    mha = torch.nn.MultiheadAttention(4, 1, bias=False, batch_first=True)
    model = SimpleNamespace(network_layer_mapping={"attn": mha, "attn_out_proj": mha.out_proj})
    monkeypatch.setattr(networks.shared, "sd_model", model)
    applied = []
    monkeypatch.setattr(networks, "network_apply_weights", applied.append)
    networks._apply_loaded_state_to_model()
    assert applied == [mha, mha.out_proj]

def test_identical_load_is_physical_noop_and_semantic_changes_invalidate(lora_networks, monkeypatch):
    networks = lora_networks
    on_disk = SimpleNamespace(filename="alpha.safetensors", shorthash="abc", read_hash=lambda: None)
    monkeypatch.setattr(networks, "available_networks", {"alpha": on_disk}, raising=False)
    monkeypatch.setattr(networks, "available_network_aliases", {"alpha": on_disk}, raising=False)
    monkeypatch.setattr(networks, "forbidden_network_aliases", {}, raising=False)
    monkeypatch.setattr(networks, "network_file_signature", lambda _filename: (10, 20, "digest"))
    parsed = SimpleNamespace(network_on_disk=on_disk, modules={}, bundle_embeddings={})
    monkeypatch.setattr(networks, "load_network", lambda *_args: parsed)
    physical = []
    monkeypatch.setattr(networks, "_apply_loaded_state_to_model", lambda: physical.append("apply"))

    assert networks.load_networks(["alpha"], [0.5], [0.5], [None])
    first_physical = len(physical)
    assert not networks.load_networks(["alpha"], [0.5], [0.5], [None])
    assert len(physical) == first_physical
    telemetry = networks.lora_steady_state_telemetry()
    assert telemetry["hits"] >= 1
    assert all(count >= 1 for count in telemetry["avoided"].values())

    assert networks.load_networks(["alpha"], [0.6], [0.5], [None])
    assert len(physical) > first_physical
    assert networks.load_networks(["alpha", "alpha"], [0.6, 0.5], [0.5, 0.5], [None, None])
    assert networks.load_networks(["alpha"], [0.6], [0.5], [4])
    assert networks.unload_networks()
    assert networks.load_networks(["alpha"], [0.5], [0.5], [None])
