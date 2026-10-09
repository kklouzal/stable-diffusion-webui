import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from modules.torchao_weight_quant import MXFP8, NVFP4


def _base_network(networks, name="alpha"):
    base = networks.network.Network(name, networks.available_networks["alpha"])
    base.mtime = 1
    base.source_signature = (20, 10)
    tensor_payload = object()
    module = SimpleNamespace(network=base, tensor_payload=tensor_payload, marker="shared immutable weights")
    base.modules = {"layer": module}
    return base, module, tensor_payload


def test_lora_off_on_change_off_crosses_each_applied_boundary_once(lora_networks, monkeypatch):
    """LoRA off -> 0.5 -> 0.8 -> off crosses each applied boundary once."""
    networks = lora_networks
    base, _module, _payload = _base_network(networks)
    base.source_key = ("opaque", ("sha256", "a"), networks.LORA_SOURCE_SCHEMA_REVISION, ())
    monkeypatch.setattr(networks, "network_file_signature", lambda _filename: ("sha256", "a"))
    monkeypatch.setattr(networks, "network_source_key", lambda *_args: base.source_key)
    monkeypatch.setattr(networks, "load_network", lambda *_args: base)

    before = dict(networks.openclaw_cache_epochs.epoch_subset(("lora_applied_epoch",)))["lora_applied_epoch"]
    networks.load_networks(["alpha"], [0.5], [0.5], [None])
    networks.load_networks(["alpha"], [0.8], [0.8], [None])
    networks.load_networks([])
    after = dict(networks.openclaw_cache_epochs.epoch_subset(("lora_applied_epoch",)))["lora_applied_epoch"]
    assert after == before + 3


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
    # A changed source is reloaded into a new Network object (published networks are immutable).
    reloaded = networks.network.Network("alpha", on_disk)
    reloaded.__dict__.update(net.__dict__)
    reloaded.source_signature = (9999, 5678)
    networks.loaded_networks[:] = [reloaded]
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

def test_file_signature_is_hashed_once_per_file_revision(lora_networks, tmp_path, monkeypatch):
    networks = lora_networks
    hashed = []
    real_sha256 = networks.hashlib.sha256
    monkeypatch.setattr(networks.hashlib, "sha256", lambda: hashed.append(1) or real_sha256())
    lora_file = tmp_path / "memo.safetensors"
    lora_file.write_bytes(b"abcd")

    first = networks.network_file_signature(lora_file)
    assert networks.network_file_signature(str(lora_file)) == first
    assert len(hashed) == 1  # unchanged revision: no second read of the file

    stat = os.stat(lora_file)
    lora_file.write_bytes(b"wxyz")  # same size, mtime restored: ctime still moves
    os.utime(lora_file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    second = networks.network_file_signature(lora_file)
    assert len(hashed) == 2 and second != first
    assert second == ("sha256", real_sha256(b"wxyz").hexdigest())
    assert networks.network_file_signature(tmp_path / "missing.safetensors") is None


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


def test_in_memory_cache_evicts_least_recently_requested(lora_networks, monkeypatch):
    networks = lora_networks
    disks = {name: SimpleNamespace(filename=f"{name}.safetensors", shorthash="", read_hash=lambda: None) for name in "abc"}
    monkeypatch.setattr(networks, "available_networks", disks, raising=False)
    monkeypatch.setattr(networks, "available_network_aliases", disks, raising=False)
    monkeypatch.setattr(networks, "network_file_signature", lambda filename: ("sha256", os.fspath(filename)))
    parsed = []
    monkeypatch.setattr(networks, "load_network", lambda name, on_disk: parsed.append(name) or SimpleNamespace(network_on_disk=on_disk, modules={}, bundle_embeddings={}))
    monkeypatch.setattr(networks, "_apply_loaded_state_to_model", lambda: None)
    monkeypatch.setattr(networks.shared.opts, "lora_in_memory_limit", 2, raising=False)
    gcs = []
    monkeypatch.setattr(networks.devices, "torch_gc", lambda: gcs.append(len(networks.networks_in_memory)))

    for names in (["a"], ["b"], ["a"], ["c"], ["a"]):  # "a" stays in use; "b" is the stale entry
        networks.load_networks(names)

    assert parsed == ["a", "b", "c"]
    assert gcs == [2]  # the device cache is released only after the eviction of "b", not on every activation
    assert [key[0].rsplit("/", 1)[-1] for key in networks.networks_in_memory] == ["c.safetensors", "a.safetensors"]


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
    assert networks.load_networks([])
    assert _epoch(networks, "lora_applied_epoch") == first + 2
    assert not networks.load_networks([])


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


def test_irrelevant_generation_inputs_not_in_applied_key():
    source = Path("extensions-builtin/Lora/networks.py").read_text()
    key_block = source[source.index("def network_applied_state_key"):source.index("def _apply_loaded_state_to_model")]
    for forbidden in ("prompt", "seed", "cfg", "sampler"):
        assert forbidden not in key_block.lower()


def test_applied_state_publication_takes_the_epoch_transaction_before_the_application_lock(lora_networks, monkeypatch):
    import contextlib
    networks = lora_networks
    epochs = networks.openclaw_cache_epochs
    monkeypatch.setattr(networks, "_apply_loaded_state_to_model", lambda: None)
    entered = threading.Event()
    bumped = threading.Event()
    real_transaction = epochs.epoch_transaction

    @contextlib.contextmanager
    def transaction():
        with real_transaction():
            entered.set()
            yield

    monkeypatch.setattr(epochs, "epoch_transaction", transaction)
    net = _applied_network(networks, "alpha")
    publisher = threading.Thread(target=lambda: networks._publish_applied_state([net]), daemon=True)
    bumper = threading.Thread(target=lambda: (epochs.bump_epoch("checkpoint_object_epoch", reason="checkpoint_commit"), bumped.set()), daemon=True)
    networks._network_application_lock.acquire()
    try:
        publisher.start()
        assert entered.wait(5), "the epoch transaction must be taken before the LoRA application lock"
        bumper.start()
        assert not bumped.wait(0.1)  # a lifecycle bump cannot interleave with the publication
    finally:
        networks._network_application_lock.release()
        publisher.join(5)
        if bumper.ident is not None:
            bumper.join(5)
    assert not publisher.is_alive() and bumped.is_set() and networks.loaded_networks == [net]


def test_applied_state_publication_reports_opaque_outcomes_and_drops_captured_graphs(lora_networks, monkeypatch):
    networks = lora_networks
    observed, graphs = [], []
    monkeypatch.setattr(networks.openclaw_cache_epochs, "observe", lambda family, event, *, reason, semantic_key=None, count=1: observed.append((family, event, reason, semantic_key)))
    monkeypatch.setattr(networks.openclaw_cuda_graphs, "note_lora_loaded", lambda: graphs.append("lora_changed"))
    monkeypatch.setattr(networks, "_apply_loaded_state_to_model", lambda: None)
    old = _applied_network(networks, "old")
    new = _applied_network(networks, "new")
    old_key, new_key, empty_key = (networks.network_applied_state_key(nets) for nets in ([old], [new], []))

    assert networks._publish_applied_state([old])
    assert not networks._publish_applied_state([old])
    assert graphs == ["lora_changed"]  # CUDA graphs drop captured LoRA state on a change, not on a hit
    assert observed == [("E12", "publish", "published", old_key), ("E12", "hit", "cache_hit", old_key)]

    def apply():
        if networks.loaded_networks == [new]:
            raise RuntimeError("injected")

    observed.clear()
    monkeypatch.setattr(networks, "_apply_loaded_state_to_model", apply)
    before = _epochs(networks)
    with pytest.raises(RuntimeError, match="injected"):
        networks._publish_applied_state([new])
    assert observed == [("E12", "reject", "rejected", new_key)]
    assert graphs == ["lora_changed"] and _epochs(networks) == before

    observed.clear()
    monkeypatch.setattr(networks, "_apply_loaded_state_to_model", lambda: None)
    assert networks._publish_applied_state([])  # unloading is a change too
    assert graphs == ["lora_changed", "lora_changed"]
    assert observed == [("E12", "publish", "published", empty_key)]
    assert _epoch(networks, "lora_applied_epoch") == before["lora_applied_epoch"] + 1


def test_execution_identity_is_device_and_precision_not_epoch_history(lora_networks, monkeypatch):
    import torch
    networks = lora_networks
    monkeypatch.setattr(networks.devices, "device", torch.device("cpu"), raising=False)
    monkeypatch.setattr(networks.devices, "dtype", torch.float16, raising=False)
    monkeypatch.setattr(networks.devices, "dtype_unet", torch.float16, raising=False)
    identity = networks._execution_identity()
    for dimension in ("checkpoint_object_epoch", "model_movement_epoch", "lora_applied_epoch", "device_epoch", "precision_epoch"):
        networks.openclaw_cache_epochs.bump_epoch(dimension, reason="other")
    assert networks._execution_identity() == identity  # a clean unload/reactivation keeps the identity

    seen = {identity}
    for name, value in (("device", torch.device("meta")), ("dtype", torch.bfloat16), ("dtype_unet", torch.bfloat16)):
        monkeypatch.setattr(networks.devices, name, value, raising=False)
        assert networks._execution_identity() not in seen, name
        seen.add(networks._execution_identity())


def test_source_key_is_the_resolved_source_bytes_and_parser_revision(lora_networks, tmp_path, monkeypatch):
    networks = lora_networks
    source = tmp_path / "alpha.safetensors"
    source.write_bytes(b"weights")
    link = tmp_path / "alias.safetensors"
    link.symlink_to(source)
    signature = networks.network_file_signature(source)

    key = networks.network_source_key(SimpleNamespace(filename=str(link)))
    assert key == (str(source.resolve()), signature, networks.LORA_SOURCE_SCHEMA_REVISION)
    for dimension in ("lora_applied_epoch", "checkpoint_object_epoch", "textual_inversion_epoch"):
        networks.openclaw_cache_epochs.bump_epoch(dimension, reason="other")
    assert networks.network_source_key(SimpleNamespace(filename=str(source))) == key  # lifecycle epochs are not source identity
    assert networks.network_source_key(SimpleNamespace(filename=str(source)), ("sha256", "other")) != key
    monkeypatch.setattr(networks, "LORA_SOURCE_SCHEMA_REVISION", "lora-source-next")
    assert networks.network_source_key(SimpleNamespace(filename=str(source))) != key
    assert networks.network_source_key(None) is None


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




@pytest.mark.parametrize("backend, with_bias", [(MXFP8, True), (NVFP4, False)], ids=["mxfp8", "nvfp4-no-bias"])
def test_quant_lora_clear_rebuilds_managed_base_and_replaces_stale_active_config(lora_networks, monkeypatch, backend, with_bias):
    """Clearing LoRAs is load_networks([]) followed by the activation's prepare_quant_active_config: a stale prepared
    config stays unusable until the managed modules are rebuilt from their BF16 base."""
    import dataclasses
    import torch
    networks = lora_networks
    name = backend.name
    linear = torch.nn.Linear(2, 2, bias=with_bias, dtype=torch.bfloat16)
    base_weight = linear.weight.detach().clone()
    base_bias = linear.bias.detach().clone() if with_bias else None
    with torch.no_grad():
        linear.weight.add_(torch.ones_like(linear.weight))
        if with_bias:
            linear.bias.add_(torch.ones_like(linear.bias))
    linear.network_layer_name = "layer"
    setattr(linear, f"network_{name}_base_weight", base_weight.detach().cpu().clone())
    setattr(linear, f"network_{name}_base_bias", base_bias.detach().cpu().clone() if with_bias else None)
    linear.network_current_names = (("alpha",),)
    setattr(linear, f"network_{name}_merged_lora_applied", True)
    model = SimpleNamespace(**{
        "network_layer_mapping": {"layer": linear},
        f"network_{name}_managed_modules": [("layer", linear)],
        f"network_{name}_active_config_ready": True,
        f"network_{name}_active_config_signature": ("stale",),
    })
    monkeypatch.setattr(networks.shared, "sd_model", model, raising=False)
    monkeypatch.setattr(networks.shared.opts, f"{name}_linear_coverage", [], raising=False)
    for each in (MXFP8, NVFP4):
        monkeypatch.setattr(networks.devices, each.name, each is backend, raising=False)
    monkeypatch.setattr(networks.devices, "device", torch.device("cpu"), raising=False)
    quantized = []
    monkeypatch.setitem(sys.modules, "torchao.quantization", SimpleNamespace(quantize_=lambda module, config, filter_fn, device: quantized.append(module)))
    backend = dataclasses.replace(backend, make_config=lambda: "config", validate_config=lambda _config: None, tensor_type=lambda: torch.nn.Parameter)

    assert networks.load_networks([])
    assert not networks.network_quant_is_model_prepared(backend, model)

    assert networks.prepare_quant_active_config(backend)
    assert quantized == [linear]
    assert torch.equal(linear.weight, base_weight)
    if with_bias:
        assert torch.equal(linear.bias, base_bias)
    else:
        assert linear.bias is None
    assert linear.network_current_names == ()
    assert getattr(linear, f"network_{name}_merged_lora_applied") is False
    assert getattr(model, f"network_{name}_active_config_signature") != ("stale",)
    assert networks.network_quant_is_model_prepared(backend, model)


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
    net = SimpleNamespace(source_key=("alpha",), te_multiplier=1.0, unet_multiplier=1.0, dyn_dim=None, modules={})
    networks.loaded_networks[:] = [net]

    assert networks.network_quant_capture_managed_base(MXFP8, model) == 0
    assert torch.equal(linear.weight, linear.network_mxfp8_base_weight)
    assert not networks.network_quant_is_model_prepared(MXFP8, model)
    linear.network_current_names = networks.network_wanted_names()
    assert networks.network_quant_is_model_prepared(MXFP8, model)


def test_quant_prepare_is_one_transaction_and_rolls_back_on_failure(lora_networks, monkeypatch):
    import dataclasses
    import torch
    networks = lora_networks
    linear = torch.nn.Linear(2, 2, bias=True, dtype=torch.bfloat16)
    linear.network_layer_name = "layer"
    linear.network_nvfp4_base_weight = linear.weight.detach().cpu().clone()
    linear.network_nvfp4_base_bias = linear.bias.detach().cpu().clone()
    model = SimpleNamespace(network_nvfp4_managed_modules=[("layer", linear)], sd_checkpoint_info=SimpleNamespace(filename="ckpt", hash="h", sha256="s"))
    monkeypatch.setattr(networks.shared, "sd_model", model, raising=False)
    monkeypatch.setattr(networks.shared.opts, "nvfp4_linear_coverage", ["unet_other"], raising=False)
    monkeypatch.setattr(networks.devices, "nvfp4", True, raising=False)
    monkeypatch.setattr(networks.devices, "device", torch.device("cpu"), raising=False)
    quantized = []

    def quantize_(module, config, filter_fn, device):
        assert config == "nvfp4-config" and filter_fn(module, "layer")
        quantized.append(module)

    monkeypatch.setitem(sys.modules, "torchao.quantization", SimpleNamespace(quantize_=quantize_))
    backend = dataclasses.replace(NVFP4, make_config=lambda: "nvfp4-config", validate_config=lambda _config: None, tensor_type=lambda: torch.nn.Parameter)

    assert networks.prepare_quant_active_config(backend)
    assert quantized == [linear]
    stats = model.network_nvfp4_prepare_stats
    assert (stats["prepared_linear"], stats["quantized_linear"], stats["failed_linear"], stats["nvfp4_linear_coverage"]) == (1, 1, 0, ["unet_other"])
    assert model.network_nvfp4_active_config_ready is True
    assert networks.network_quant_is_model_prepared(backend, model)
    assert networks.prepare_quant_active_config(backend)  # same active signature: nothing is re-quantized
    assert quantized == [linear]

    def failing_quantize_(module, config, filter_fn, device):
        raise RuntimeError("injected quantize failure")

    monkeypatch.setitem(sys.modules, "torchao.quantization", SimpleNamespace(quantize_=failing_quantize_))
    networks.network_quant_mark_model_unprepared(backend, model)
    weight_before, bias_before = linear.weight, linear.bias
    assert not networks.prepare_quant_active_config(backend)
    assert linear.weight is weight_before and linear.bias is bias_before
    assert model.network_nvfp4_prepare_error == "failed to prepare active NVFP4 LoRA config for 1 Linear modules: ['layer']"
    assert not getattr(model, "network_nvfp4_active_config_ready", False)


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
    monkeypatch.setattr(networks.openclaw_cuda_graphs, "note_lora_loaded", lambda: graph_observations.append((dict(db.word_embeddings), _epochs(networks))))

    before = _epochs(networks)
    assert networks._publish_applied_state([net], db)
    loaded = _epochs(networks)
    assert db.word_embeddings == {"hero": emb}
    assert all(loaded[name] == before[name] + 1 for name in loaded)
    assert not networks._publish_applied_state([net], db)
    assert _epochs(networks) == loaded
    assert len(db.register_calls) == 1
    assert networks._publish_applied_state([], db)
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
    monkeypatch.setattr(networks.openclaw_cuda_graphs, "note_lora_loaded", lambda: graphs.append("lora_changed"))
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
    monkeypatch.setattr(networks.openclaw_cuda_graphs, "note_lora_loaded", lambda: graphs.append("lora_changed"))

    with pytest.raises(RuntimeError, match="rollback failed closed"):
        networks._publish_applied_state([new], db)
    assert attempts == [[new], [old], []]
    assert networks.loaded_networks == []
    assert networks.loaded_bundle_embeddings == {}
    assert networks._applied_state_key is None
    assert db.word_embeddings == {}
    assert _epochs(networks) == {name: value + 1 for name, value in before.items()}
    assert graphs == ["lora_changed"]


def test_current_text_encoder_state_identity_reuses_semantically_identical_lifecycle(lora_networks, monkeypatch):
    networks = lora_networks
    base, module, _payload = _base_network(networks)
    base.modules = {"0_transformer_text_model_encoder_layers_0_mlp_fc1": module}
    base.source_key = ("opaque", ("sha256", "same"), networks.LORA_SOURCE_SCHEMA_REVISION, ())
    monkeypatch.setattr(networks, "network_file_signature", lambda _filename: ("sha256", "same"))
    monkeypatch.setattr(networks, "network_source_key", lambda *_args: base.source_key)
    monkeypatch.setattr(networks, "load_network", lambda *_args: base)

    networks.load_networks(["alpha"], [0.75], [1.25], [4])
    first = networks.current_text_encoder_state_identity()
    networks.load_networks([], [], [], [])
    unloaded = networks.current_text_encoder_state_identity()
    networks.load_networks(["alpha"], [0.75], [1.25], [4])
    reapplied = networks.current_text_encoder_state_identity()

    assert first == reapplied
    assert first != unloaded
    assert networks.loaded_networks[0].te_multiplier == 0.75
    assert networks.loaded_networks[0].unet_multiplier == 1.25
    assert networks.loaded_networks[0].dyn_dim == 4


def test_network_applied_state_key_changes_for_effective_inputs(lora_networks):
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
    networks.network_apply_weights(mha)
    networks.network_apply_weights(mha.out_proj)
    assert torch.equal(mha.in_proj_weight, base_qkv)
    assert torch.equal(mha.out_proj.weight, base_out)
    monkeypatch.setattr(networks, "loaded_networks", [net])
    networks.network_apply_weights(mha)
    networks.network_apply_weights(mha.out_proj)
    assert torch.equal(mha.in_proj_weight, expected_qkv)
    assert torch.equal(mha.out_proj.weight, base_out + 4)



def test_mha_failed_reactivation_restores_exact_base(lora_networks, monkeypatch):
    torch = pytest.importorskip("torch")
    networks = lora_networks
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
    with pytest.raises(RuntimeError, match="LoRA broken cannot be applied to layer 1_model_transformer_resblocks_0_attn_k_proj: injected failure"):
        networks.network_apply_weights(mha)
    assert torch.equal(mha.in_proj_weight, base_qkv)
    assert mha.network_current_names == ()  # the layer holds its base weights, which the next apply sees


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
    assert networks.load_networks([])
    assert networks.load_networks(["alpha"], [0.5], [0.5], [None])


def test_quant_active_config_signature_keys_on_loaded_source_key_without_file_reads(lora_networks, monkeypatch):
    networks = lora_networks

    def no_file_reads(_filename):
        raise AssertionError("LoRA files must not be re-hashed per request")

    monkeypatch.setattr(networks, "network_file_signature", no_file_reads)
    monkeypatch.setattr(networks.shared, "sd_model", SimpleNamespace(sd_checkpoint_info=None), raising=False)
    source_key = ("alpha.safetensors", ("sha256", "a"), networks.LORA_SOURCE_SCHEMA_REVISION)
    net = SimpleNamespace(name="alpha", te_multiplier=1.0, unet_multiplier=0.5, dyn_dim=None, source_key=source_key, network_on_disk=SimpleNamespace(filename="alpha.safetensors"))
    networks.loaded_networks[:] = [net]

    signature = networks.network_quant_active_config_signature(MXFP8)

    assert signature[-1] == (("alpha", 1.0, 0.5, None, source_key),)
    assert not hasattr(networks, "network_lora_source_signature")


def _published_net(name, updown=None):
    calls = []

    def calc_updown(weight):
        calls.append(name)
        return updown, None

    net = SimpleNamespace(
        name=name,
        source_key=(name,),
        te_multiplier=1.0,
        unet_multiplier=1.0,
        dyn_dim=None,
        modules={"layer": SimpleNamespace(calc_updown=calc_updown)},
    )
    return net, calls


def test_wanted_names_are_built_once_per_published_set(lora_networks, monkeypatch):
    networks = lora_networks
    built = []
    signature = networks.network_loaded_weight_signature
    monkeypatch.setattr(networks, "network_loaded_weight_signature", lambda net, *used: built.append(net.name) or signature(net, *used))
    alpha, _ = _published_net("alpha")
    beta, _ = _published_net("beta")

    networks._set_loaded_networks([alpha])
    names = networks.network_wanted_names()
    assert built == ["alpha", "alpha"]  # the set's and its U-Net layer's, built at publish, reused by every later forward
    assert all(networks.network_wanted_names() is names for _ in range(3))
    assert names == (signature(alpha),)
    assert networks.network_layer_wanted_names("layer") == (signature(alpha, False, True),)

    networks._set_loaded_networks([alpha, beta])
    assert built == ["alpha", "alpha", "alpha", "beta", "alpha", "beta"]
    networks.loaded_networks[:] = [beta]  # a direct list replacement is still observed
    assert networks.network_wanted_names() == (signature(beta),)
    networks._set_loaded_networks([])
    assert networks.network_wanted_names() == ()
    assert networks._wanted_names_memo == ((), (), {})  # no reference to the unloaded set is kept


def test_apply_weights_returns_early_without_loras(lora_networks, monkeypatch):
    import torch
    networks = lora_networks
    linear = torch.nn.Linear(2, 2)
    linear.network_layer_name = "layer"
    weight = linear.weight.detach().clone()
    monkeypatch.setattr(networks, "_wanted_names_state", lambda: (_ for _ in ()).throw(AssertionError("no-LoRA forward must not build names")))

    networks.network_apply_weights(linear)

    assert torch.equal(linear.weight, weight)
    assert not hasattr(linear, "network_weights_backup")
    assert getattr(linear, "network_current_names", ()) == ()


def test_published_names_merge_and_restore_like_per_call_names(lora_networks):
    import torch
    networks = lora_networks
    torch.manual_seed(0)
    linear = torch.nn.Linear(4, 4)
    linear.network_layer_name = "layer"
    base_weight, base_bias = linear.weight.detach().clone(), linear.bias.detach().clone()
    alpha_updown, beta_updown = torch.randn(4, 4), torch.randn(4, 4)
    alpha, alpha_calls = _published_net("alpha", alpha_updown)
    beta, beta_calls = _published_net("beta", beta_updown)

    networks._set_loaded_networks([alpha])
    for _ in range(3):  # per-forward calls after the first merge are no-ops
        networks.network_apply_weights(linear)
    assert alpha_calls == ["alpha"]
    assert torch.equal(linear.weight, base_weight + alpha_updown)

    networks._set_loaded_networks([beta])
    networks.network_apply_weights(linear)
    assert beta_calls == ["beta"]
    assert torch.equal(linear.weight, base_weight + beta_updown)

    networks._set_loaded_networks([])
    networks.network_apply_weights(linear)
    assert torch.equal(linear.weight, base_weight) and torch.equal(linear.bias, base_bias)
    assert linear.network_current_names == ()
    networks.network_apply_weights(linear)  # now takes the early return
    assert torch.equal(linear.weight, base_weight)


def test_bundled_ti_dropped_by_ti_reload_is_registered_again(lora_networks, monkeypatch):
    """A TI reload publishes a folder-only database; the unchanged LoRA set must register its bundles again."""
    networks = lora_networks
    db = networks.sd_hijack.model_hijack.embedding_db
    on_disk = SimpleNamespace(filename="alpha.safetensors", shorthash="abc", read_hash=lambda: None)
    monkeypatch.setattr(networks, "available_networks", {"alpha": on_disk}, raising=False)
    monkeypatch.setattr(networks, "available_network_aliases", {"alpha": on_disk}, raising=False)
    monkeypatch.setattr(networks, "network_file_signature", lambda _filename: (10, 20, "digest"))
    hero = _embedding("hero")
    parsed = SimpleNamespace(network_on_disk=on_disk, modules={}, bundle_embeddings={"hero": hero})
    monkeypatch.setattr(networks, "load_network", lambda *_args: parsed)
    monkeypatch.setattr(networks, "_apply_loaded_state_to_model", lambda: None)

    assert networks.load_networks(["alpha"], [1.0], [1.0], [None])
    assert db.word_embeddings == {"hero": hero}
    assert not networks.load_networks(["alpha"], [1.0], [1.0], [None])

    with db._publication_lock:  # what load_textual_inversion_embeddings publishes for an empty folder
        db.ids_lookup, db.word_embeddings = {}, {}
    before = _epochs(networks)

    assert networks.load_networks(["alpha"], [1.0], [1.0], [None])
    assert db.word_embeddings == {"hero": hero}
    assert hero.loaded is True
    after = _epochs(networks)
    assert after["textual_inversion_epoch"] > before["textual_inversion_epoch"]
    assert after["tokenizer_epoch"] > before["tokenizer_epoch"]
    assert not networks.load_networks(["alpha"], [1.0], [1.0], [None])


def test_lora_activation_errors_stop_generation_instead_of_dropping_every_lora(lora_networks, monkeypatch):
    """A LoRA that fails to prepare stops the request: generating with an empty LoRA set would drop every LoRA."""
    import extra_networks_lora
    from modules import extra_networks

    networks = lora_networks
    lora = extra_networks_lora.ExtraNetworkLora()
    p = SimpleNamespace(all_prompts=["x"], comment=lambda _text: None, extra_generation_params={}, scripts=None)
    loaded = []

    def missing(*_args):
        raise RuntimeError("one or more requested LoRA sources are unavailable")

    monkeypatch.setattr(networks, "load_networks", missing)
    monkeypatch.setattr(extra_networks, "extra_network_registry", {"lora": lora})
    with pytest.raises(extra_networks_lora.FatalLoraPreparationError, match="unavailable"):
        extra_networks.activate(p, {"lora": [extra_networks.ExtraNetworkParams(items=["missing", "0.8"])]})

    monkeypatch.setattr(networks, "load_networks", lambda *args: loaded.append(args))
    for items in (["alpha", "nan"], ["alpha", "1", "inf"], ["alpha", "te=-inf"], ["alpha", "abc"], ["alpha", "1", "1", "8.5"], ["alpha", "1", "1", "0"], ["alpha", "dyn=-2"]):
        with pytest.raises(extra_networks_lora.FatalLoraPreparationError):
            lora.activate(p, [extra_networks.ExtraNetworkParams(items=items)])
    assert loaded == []


def _activation(networks, monkeypatch, parsed):
    import extra_networks_lora
    from modules import extra_networks

    monkeypatch.setattr(networks, "network_file_signature", lambda _filename: ("sha256", "a"))
    monkeypatch.setattr(networks, "load_network", lambda *_args: parsed)
    monkeypatch.setattr(networks, "_apply_loaded_state_to_model", lambda: None)
    monkeypatch.setattr(networks.shared.opts, "lora_add_hashes_to_infotext", True, raising=False)
    lora = extra_networks_lora.ExtraNetworkLora()

    def activate(name, p=None, is_hr_pass=False):
        p = p or SimpleNamespace(all_prompts=["x"], extra_generation_params={})
        p.is_hr_pass = is_hr_pass
        lora.activate(p, [extra_networks.ExtraNetworkParams(items=[name, "0.5"])])
        return p
    return activate


def test_unmatched_lora_keys_are_reported_on_every_activation(lora_networks, monkeypatch):
    """Keys that name no model layer cannot be applied; the request reports them ("Lora errors"), also when the
    applied state is reused (no parse, no merge) and when the hires pass activates again."""
    networks = lora_networks
    parsed = networks.network.Network("alpha", networks.available_networks["alpha"])
    parsed.unmatched_keys = ("lora_te2_text_projection.alpha", "lora_te2_text_projection.lora_down.weight", "lora_te2_text_projection.lora_up.weight")
    activate = _activation(networks, monkeypatch, parsed)

    first = activate("alpha")
    assert first.extra_generation_params["Lora errors"] == "alpha: 3 unmatched keys"

    loads = []
    real_load_networks = networks.load_networks
    monkeypatch.setattr(networks, "load_networks", lambda *args: loads.append(real_load_networks(*args)))
    second = activate("alpha")
    assert loads == [False]  # the applied state was reused
    assert second.extra_generation_params["Lora errors"] == "alpha: 3 unmatched keys"
    assert activate("alpha", p=second, is_hr_pass=True).extra_generation_params["Lora errors"] == "alpha: 3 unmatched keys"

    parsed.unmatched_keys = ()
    networks.loaded_networks.clear()
    monkeypatch.setattr(networks, "_applied_state_key", None)
    assert "Lora errors" not in activate("alpha").extra_generation_params


def test_lora_hashes_name_the_alias_this_request_used(lora_networks, monkeypatch):
    """Switching between two names of one file with equal multipliers reuses the applied state, whose networks
    carry the previous request's mentioned_name; the infotext used that stale alias."""
    networks = lora_networks
    parsed = networks.network.Network("alpha", networks.available_networks["alpha"])
    activate = _activation(networks, monkeypatch, parsed)

    assert activate("alpha-alias").extra_generation_params["Lora hashes"] == "alpha-alias: abc"
    second = activate("alpha")
    assert networks.loaded_networks[0].mentioned_name == "alpha-alias"  # the applied state was reused
    assert second.extra_generation_params["Lora hashes"] == "alpha: abc"


@pytest.mark.parametrize("enabled", [False, True])
def test_bundled_ti_hash_follows_the_infotext_option(lora_networks, monkeypatch, enabled):
    """sd_hijack_clip skips embeddings whose shorthash is false and writes f"{name}: {shorthash}"; with the option off
    a bundled embedding wrote "<embedding>: " (the hash str is the LoRA name, so it was always true)."""
    networks = lora_networks
    monkeypatch.setattr(networks.shared.opts, "lora_bundled_ti_to_infotext", enabled)
    shorthash = networks.BundledTIHash("my_lora")

    entries = [f"emb: {h}" for h in (shorthash,) if h]

    assert str(shorthash) == ("my_lora" if enabled else "")
    assert entries == (["emb: my_lora"] if enabled else [])


def _sd1_like_model(torch):
    model = torch.nn.Module()
    model.is_sdxl = False
    model.cond_stage_model = torch.nn.Linear(2, 2)
    model.model = torch.nn.Module()
    model.model.diffusion_model = torch.nn.Linear(2, 2)
    return model


def test_parsed_networks_and_applied_state_belong_to_one_model(lora_networks, monkeypatch):
    """Parsed networks hold the layers they were matched against and the applied key describes weights merged into
    one model: a replaced model (load callback) or a switch to another cached model (no callback) drops both."""
    import torch
    networks = lora_networks
    parses = []

    def load_network(name, _on_disk):
        net = networks.network.Network(name, networks.available_networks["alpha"])
        parses.append(networks.shared.sd_model)
        return net

    monkeypatch.setattr(networks, "network_file_signature", lambda _filename: ("sha256", "a"))
    monkeypatch.setattr(networks, "load_network", load_network)
    monkeypatch.setattr(networks, "_apply_loaded_state_to_model", lambda: None)
    first, second = _sd1_like_model(torch), _sd1_like_model(torch)

    monkeypatch.setattr(networks.shared, "sd_model", first)
    networks.assign_network_names_to_compvis_modules(first)
    assert networks.load_networks(["alpha"], [1.0], [1.0], [None])
    published_key = networks._applied_state_key

    networks.assign_network_names_to_compvis_modules(first)  # e.g. a VAE reload: same model, same layers
    assert len(networks.networks_in_memory) == 1 and networks._applied_state_key == published_key
    assert not networks.load_networks(["alpha"], [1.0], [1.0], [None]) and parses == [first]

    monkeypatch.setattr(networks.shared, "sd_model", second)
    networks.assign_network_names_to_compvis_modules(second)
    assert networks.networks_in_memory == {} and networks.loaded_networks == [] and networks._applied_state_key is None
    assert networks.load_networks(["alpha"], [1.0], [1.0], [None]) and parses == [first, second]

    monkeypatch.setattr(networks.shared, "sd_model", first)  # a cached model made current again: no callback runs
    assert networks.load_networks(["alpha"], [1.0], [1.0], [None]) and parses == [first, second, first]


def test_trashed_model_drops_lora_weight_backups(lora_networks):
    import torch
    from modules import sd_models
    model = _sd1_like_model(torch)
    layer = model.model.diffusion_model
    layer.network_weights_backup = layer.weight.detach().clone()
    layer.network_bias_backup = layer.bias.detach().clone()

    sd_models.send_model_to_trash(model)

    assert not hasattr(layer, "network_weights_backup") and not hasattr(layer, "network_bias_backup")


def test_network_listing_publishes_new_registries_under_iterating_readers(lora_networks, monkeypatch, tmp_path):
    """GET /sdapi/v1/loras iterates available_networks on the event loop while a refresh or a generation's lookup of
    a new name rebuilt it in place ("dictionary changed size during iteration", or a half-built list)."""
    import safetensors.torch
    import torch
    networks = lora_networks
    lora_dir = tmp_path / "Lora"
    lora_dir.mkdir()
    for name in ("a", "b"):
        safetensors.torch.save_file({"x": torch.zeros(1)}, str(lora_dir / f"{name}.safetensors"))
    monkeypatch.setattr(networks.shared.cmd_opts, "lora_dir", str(lora_dir), raising=False)
    monkeypatch.setattr(networks.shared.cmd_opts, "lyco_dir_backcompat", str(tmp_path / "LyCORIS"), raising=False)
    networks.list_available_networks()
    listing = networks.available_networks
    reader = iter(listing.values())
    next(reader)

    safetensors.torch.save_file({"x": torch.zeros(1)}, str(lora_dir / "c.safetensors"))
    networks.list_available_networks()
    safetensors.torch.save_file({"x": torch.zeros(1)}, str(lora_dir / "d.safetensors"))
    networks.update_available_networks_by_names(["d"])

    assert [entry.name for entry in reader] == ["b"]  # the reader finishes the registry it started on
    assert sorted(listing) == ["a", "b"]
    assert sorted(networks.available_networks) == ["a", "b", "c", "d"]
    assert networks.forbidden_network_aliases == {"none": 1, "Addams": 1}
