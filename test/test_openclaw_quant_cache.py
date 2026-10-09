import os

import torch

from modules import torchao_model_cache


def test_cache_bias_validation_accepts_legacy_parameter_metadata():
    linear = torch.nn.Linear(4, 4, bias=True, dtype=torch.bfloat16)
    bias = linear.bias.detach()
    metadata = torchao_model_cache.tensor_meta(bias)
    metadata["tensor_type"] = "torch.nn.parameter.Parameter"

    assert torchao_model_cache.cached_bias_matches(bias, metadata, linear, "cpu")

    parameter = torchao_model_cache.parameter_on_device(bias, "cpu")
    assert isinstance(parameter, torch.nn.Parameter)
    assert parameter.data_ptr() == bias.data_ptr()
    assert parameter.requires_grad is False


def test_torchao_cache_metadata_uses_identity_fast_path_without_rehash(monkeypatch, tmp_path):
    from modules import torchao_model_cache

    source = tmp_path / "model.safetensors"
    source.write_bytes(b"a" * 1024)
    metadata = torchao_model_cache.stat_source(str(source))

    def fail_hash(_filename):
        raise AssertionError("fast path should not rehash unchanged identity")

    monkeypatch.setattr(torchao_model_cache, "sha256_file", fail_hash)

    assert torchao_model_cache.file_metadata_matches(str(source), metadata)


def test_torchao_cache_metadata_rehashes_when_mtime_changes(monkeypatch, tmp_path):
    from modules import torchao_model_cache

    source = tmp_path / "model.safetensors"
    source.write_bytes(b"a" * 1024)
    metadata = torchao_model_cache.stat_source(str(source))
    changed_mtime = metadata["mtime_ns"] + 1_000_000_000
    os.utime(source, ns=(changed_mtime, changed_mtime))
    calls = []

    def tracked_hash(filename):
        calls.append(filename)
        return metadata["sha256"]

    monkeypatch.setattr(torchao_model_cache, "sha256_file", tracked_hash)

    assert torchao_model_cache.file_metadata_matches(str(source), metadata)
    assert calls == [str(source)]
    assert metadata["identity_trusted"] is True
    calls.clear()
    assert torchao_model_cache.file_metadata_matches(str(source), metadata)
    assert calls == []


def test_torchao_cache_metadata_rejects_truncated_or_corrupt_changed_file(tmp_path):
    from modules import torchao_model_cache

    source = tmp_path / "model.safetensors"
    source.write_bytes(b"a" * 4096)
    metadata = torchao_model_cache.stat_source(str(source))

    source.write_bytes(b"a" * 1024)
    assert not torchao_model_cache.file_metadata_matches(str(source), metadata)

    source.write_bytes(b"b" * 4096)
    corrupt_metadata = dict(metadata)
    current = torchao_model_cache.file_identity(str(source))
    corrupt_metadata.update(current)
    corrupt_metadata["identity_trusted"] = False
    assert not torchao_model_cache.file_metadata_matches(str(source), corrupt_metadata)


def test_torchao_contract_rejects_runtime_and_scheme_changes(monkeypatch):
    from modules import torchao_model_cache

    monkeypatch.setattr(torchao_model_cache, "runtime_compatibility", lambda: {"torch": "one", "sm": [12, 1]})
    expected = torchao_model_cache.artifact_contract("mxfp8", ["diffusion_model"])
    assert torchao_model_cache.contract_matches(expected, expected)
    assert not torchao_model_cache.contract_matches({**expected, "runtime": {"torch": "two", "sm": [12, 1]}}, expected)
    changed_scheme = {**expected, "quantization": {**expected["quantization"], "implementation": "nvfp4"}}
    assert not torchao_model_cache.contract_matches(changed_scheme, expected)


def test_torchao_runtime_contract_records_only_probed_facts_as_plain_json():
    # Sidecars store the contract as JSON and payloads reload it through the weights_only unpickler, so every
    # value must survive a JSON round trip. Fields are only what this torch build can actually probe.
    import json

    runtime = torchao_model_cache.runtime_compatibility()
    assert set(runtime) == {"python", "platform", "torch", "torchao", "cuda_runtime", "device"}
    assert json.loads(json.dumps(runtime)) == runtime


def test_torchao_sidecar_corruption_or_missing_contract_forces_regeneration(monkeypatch, tmp_path):
    from modules import torchao_model_cache

    source = tmp_path / "model.safetensors"
    cache = tmp_path / "model.pt"
    source.write_bytes(b"source")
    cache.write_bytes(b"cache")
    suffix = ".json"
    sidecar = {
        "cache_version": 9,
        "config": "mxfp8",
        "source": torchao_model_cache.stat_source(str(source)),
        "cache": torchao_model_cache.stat_source(str(cache)),
        "coverage": None,
    }
    Path = __import__("pathlib").Path
    Path(str(cache) + suffix).write_text(__import__("json").dumps(sidecar))
    monkeypatch.setattr(torchao_model_cache, "runtime_compatibility", lambda: {"torch": "test"})
    assert torchao_model_cache.verified_sidecar(str(source), str(cache), 9, "mxfp8", suffix) is None
    sidecar["contract"] = torchao_model_cache.artifact_contract("mxfp8")
    Path(str(cache) + suffix).write_text(__import__("json").dumps(sidecar))
    assert torchao_model_cache.verified_sidecar(str(source), str(cache), 9, "mxfp8", suffix) is not None
    cache.write_bytes(b"broken")
    assert torchao_model_cache.verified_sidecar(str(source), str(cache), 9, "mxfp8", suffix) is None


def test_artifact_contract_survives_weights_only_load(tmp_path):
    # Cache payloads embed the contract and are loaded with weights_only=True; a torch.torch_version.TorchVersion
    # (what torch.__version__ is) is not an allowed global there, so every cache would be rejected and rebuilt.
    import torch

    from modules import torchao_model_cache

    path = tmp_path / "artifact.pt"
    torch.save({"contract": torchao_model_cache.artifact_contract("cfg", ["unet_other"])}, path)
    loaded = torchao_model_cache.torch_load_cache(str(path), "cpu", lambda: None)  # the production weights_only loader
    assert loaded["contract"]["runtime"]["torch"] == str(torch.__version__)


def test_torchao_cache_rehash_records_the_identity_seen_before_hashing(monkeypatch, tmp_path):
    # A file replaced (same size, other bytes) while it is being re-hashed must not become trusted under its new identity.
    source = tmp_path / "model.safetensors"
    source.write_bytes(b"aaaa")
    metadata = torchao_model_cache.stat_source(str(source))
    os.utime(source, ns=(metadata["mtime_ns"] + 10**9,) * 2)  # same bytes, new identity: the next check re-hashes
    real_hash = torchao_model_cache.sha256_file

    def hash_then_replace(filename):
        digest = real_hash(filename)
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"bbbb")
        os.replace(replacement, filename)
        return digest

    monkeypatch.setattr(torchao_model_cache, "sha256_file", hash_then_replace)
    assert torchao_model_cache.file_metadata_matches(str(source), metadata)  # the bytes it hashed were the recorded ones
    monkeypatch.setattr(torchao_model_cache, "sha256_file", real_hash)
    assert not torchao_model_cache.file_metadata_matches(str(source), metadata)


def _linear_model():
    torch.manual_seed(0)
    model = torch.nn.Module()
    model.linear = torch.nn.Linear(64, 64, dtype=torch.bfloat16)
    return model


def _saved_mxfp8_cache(tmp_path, coverage):
    from torchao.quantization import quantize_

    from modules import torchao_weight_quant

    backend = torchao_weight_quant.MXFP8
    root = tmp_path / "Stable-diffusion"
    root.mkdir()
    source = root / "model.safetensors"
    source.write_bytes(b"checkpoint bytes")
    fresh = _linear_model()
    quantize_(fresh, backend.make_config(), filter_fn=_only_linear)
    cache_path = torchao_model_cache.save_from_model(backend, fresh, str(source), _only_linear, 1, 0, {}, coverage)
    assert cache_path
    return backend, source, fresh, cache_path


def _only_linear(module, fqn):
    return fqn == "linear"


def test_torchao_cached_load_matches_quantize_and_hashes_a_touched_source_once(monkeypatch, tmp_path):
    coverage = ["unet_other"]
    backend, source, fresh, _ = _saved_mxfp8_cache(tmp_path, coverage)
    stat = os.stat(source)
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))  # metadata change only: the bytes are the same
    hashed = []
    real_hash = torchao_model_cache.sha256_file
    monkeypatch.setattr(torchao_model_cache, "sha256_file", lambda filename: hashed.append(filename) or real_hash(filename))

    loads = []
    for _ in range(2):
        model = _linear_model()
        assert torchao_model_cache.load_into_model(backend, model, str(source), _only_linear, "cpu", coverage)
        loads.append((model, list(hashed)))
        hashed.clear()

    # One re-hash proves the touched source still has the recorded bytes; the payload's stale identity adds none.
    assert [calls for _, calls in loads] == [[str(source)], []]
    expected = fresh.linear.weight
    for model, _ in loads:
        weight = model.linear.weight
        assert type(weight) is type(expected)
        assert isinstance(weight, torch.nn.Parameter) and isinstance(expected, torch.nn.Parameter)
        assert weight.requires_grad is False
        assert torch.equal(weight.qdata.view(torch.uint8), expected.qdata.view(torch.uint8))
        assert torch.equal(weight.scale.view(torch.uint8), expected.scale.view(torch.uint8))
        assert weight.act_quant_kwargs == expected.act_quant_kwargs and weight.is_swizzled_scales == expected.is_swizzled_scales
        assert torch.equal(model.linear.bias.detach(), fresh.linear.bias.detach())


def test_torchao_cache_rejects_a_payload_built_from_other_bytes(tmp_path):
    import json

    backend, source, _, cache_path = _saved_mxfp8_cache(tmp_path, None)

    # Rewrite the payload as if it had been built from other bytes, and re-seal the sidecar around the new artifact.
    payload = torchao_model_cache.torch_load_cache(cache_path, "cpu", backend.register_safe_globals)
    payload["source"] = {**payload["source"], "sha256": "0" * 64}
    torch.save(payload, cache_path)
    sidecar = torchao_model_cache.load_sidecar(cache_path, backend.sidecar_suffix)
    sidecar["cache"] = torchao_model_cache.stat_source(cache_path)
    with open(torchao_model_cache.sidecar_path(cache_path, backend.sidecar_suffix), "w", encoding="utf8") as f:
        json.dump(sidecar, f)

    assert torchao_model_cache.verified_sidecar(str(source), cache_path, backend.cache_version, backend.config_name, backend.sidecar_suffix) is not None
    assert not torchao_model_cache.load_into_model(backend, _linear_model(), str(source), _only_linear, "cpu", None)


def test_torchao_cache_quota_covers_every_artifact_of_the_backend(monkeypatch, tmp_path):
    # A checkpoint in a subdirectory must be budgeted with the rest of the backend's artifacts, not on its own.
    from types import SimpleNamespace

    from modules import persistent_artifact_cache

    roots = []

    def record_quota(root, **_kwargs):
        roots.append(os.fspath(root))
        return {"evicted": [], "within_quota": True}

    monkeypatch.setattr(persistent_artifact_cache, "enforce_directory_quota", record_quota)
    backend = SimpleNamespace(
        cache_dir_name="mxfp8", cache_version=1, config_name="cfg", sidecar_suffix=".mxfp8-cache.json", label="test",
        is_quant_tensor=torch.is_tensor,
    )
    checkpoints = tmp_path / "Stable-diffusion"
    for source in (checkpoints / "top.safetensors", checkpoints / "sub" / "dir" / "nested.safetensors"):
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"checkpoint bytes")
        cache_path = torchao_model_cache.save_from_model(backend, _linear_model(), str(source), _only_linear, 1, 0, {})
        assert cache_path and os.path.isfile(cache_path)
    assert roots == [str(checkpoints / "mxfp8")] * 2
