import os
import threading
from types import SimpleNamespace

import pytest

from test.helpers import ROOT, init_shared


def pytest_configure(config):
    # We don't want to fail on Py.test command line arguments being
    # parsed by webui:
    os.environ.setdefault("IGNORE_CMD_ARGS_ERRORS", "1")


@pytest.fixture(scope="session")
def initialize() -> None:
    """The real, initialized modules.shared (test.helpers.init_shared) for tests that import webui modules lazily."""
    init_shared()


@pytest.fixture
def lora_networks(monkeypatch):
    """extensions-builtin/Lora's `networks` module, imported as the top-level `networks` the way the webui loads it, on
    the real initialized shared, with its network registries, loaded networks and cache epochs reset and the
    side-effecting hooks (embedding DB, torch_gc, CUDA-graph notification) replaced for the test."""
    shared = init_shared()
    monkeypatch.setattr(shared.cmd_opts, "use_ipex", False, raising=False)
    monkeypatch.setattr(shared.cmd_opts, "use_cpu", [], raising=False)
    monkeypatch.setattr(shared.opts, "hide_samplers", [], raising=False)
    monkeypatch.setattr(shared.opts, "samples_format", "png", raising=False)
    monkeypatch.setattr(shared.opts, "lora_in_memory_limit", 10, raising=False)
    # ExtraNetworkLora.activate reads sd_lora before its FatalLoraPreparationError wrapping: shared.opts holds no Lora
    # extension options unless the extension's scripts were loaded, so reading it would raise AttributeError instead.
    monkeypatch.setattr(shared.opts, "sd_lora", "None", raising=False)
    monkeypatch.setattr(shared.opts, "lora_bundled_ti_to_infotext", False, raising=False)
    monkeypatch.setattr(shared.opts, "lora_not_found_warning_console", False, raising=False)
    monkeypatch.setattr(shared.opts, "lora_not_found_gradio_warning", False, raising=False)

    monkeypatch.syspath_prepend(str(ROOT / "extensions-builtin" / "Lora"))
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
    monkeypatch.setattr(networks.openclaw_cuda_graphs, "note_lora_loaded", lambda: None)

    on_disk = SimpleNamespace(filename="alpha.safetensors", shorthash="abc", read_hash=lambda: None)
    monkeypatch.setattr(networks, "available_networks", {"alpha": on_disk}, raising=False)
    monkeypatch.setattr(networks, "available_network_aliases", {"alpha-alias": on_disk, "alpha": on_disk}, raising=False)
    monkeypatch.setattr(networks, "forbidden_network_aliases", {}, raising=False)
    monkeypatch.setattr(networks, "networks_in_memory", {}, raising=False)
    networks.loaded_networks.clear()
    monkeypatch.setattr(networks, "loaded_bundle_embeddings", {}, raising=False)
    monkeypatch.setattr(networks, "_applied_state_key", None, raising=False)
    monkeypatch.setattr(networks, "_model_token", None, raising=False)
    networks._adopt_model(networks.shared.sd_model)  # the networks state below belongs to this model
    networks.openclaw_cache_epochs.reset_for_tests()
    yield networks
    networks.loaded_networks.clear()
    networks.openclaw_cache_epochs.reset_for_tests()


@pytest.fixture
def default_runtime(monkeypatch):
    """The production state sd_hijack_unet's bf16-native path expects: no upcast, no quantized storage, no functional
    LoRA, no hypernetworks, switch on. Its users put repositories/ on sys.path before importing sd_hijack_unet."""
    from modules import devices, sd_hijack_unet

    monkeypatch.setattr(sd_hijack_unet, "UNET_BF16_NATIVE_NORMS", True)
    monkeypatch.setattr(sd_hijack_unet, "shared", SimpleNamespace(opts=SimpleNamespace(lora_functional=False), loaded_hypernetworks=[]))
    for flag in ("unet_needs_upcast", "fp8", "mxfp8", "nvfp4"):
        monkeypatch.setattr(devices, flag, False)
