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
