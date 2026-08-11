"""Cross-slice cache matrix nodes declared by the consolidated validation plan."""

pytest_plugins = ["test.test_openclaw_lora_network_identity"]


def test_core_08(lora_networks, monkeypatch):
    """LoRA off -> 0.5 -> 0.8 -> off crosses each applied boundary once."""
    networks = lora_networks
    base, _module, _payload = __import__(
        "test.test_openclaw_lora_network_identity", fromlist=["_base_network"]
    )._base_network(networks)
    base.source_key = ("opaque", ("sha256", "a"), networks.LORA_SOURCE_SCHEMA_REVISION, ())
    monkeypatch.setattr(networks, "network_file_signature", lambda _filename: ("sha256", "a"))
    monkeypatch.setattr(networks, "network_source_key", lambda *_args: base.source_key)
    monkeypatch.setattr(networks, "load_network", lambda *_args: base)

    before = dict(networks.openclaw_cache_epochs.epoch_subset(("lora_applied_epoch",)))["lora_applied_epoch"]
    networks.load_networks(["alpha"], [0.5], [0.5], [None])
    networks.load_networks(["alpha"], [0.8], [0.8], [None])
    networks.unload_networks()
    after = dict(networks.openclaw_cache_epochs.epoch_subset(("lora_applied_epoch",)))["lora_applied_epoch"]
    assert after == before + 3
