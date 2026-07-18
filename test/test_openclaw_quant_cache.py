import os

import torch

from modules import mxfp8_model_cache, nvfp4_model_cache


def _assert_bias_helpers_accept_legacy_tensor_type(cache_module):
    linear = torch.nn.Linear(4, 4, bias=True, dtype=torch.bfloat16)
    bias = linear.bias.detach()
    metadata = cache_module._tensor_meta(bias)
    metadata["tensor_type"] = "torch.nn.parameter.Parameter"

    assert cache_module._cached_bias_matches(bias, metadata, linear, "cpu")

    parameter = cache_module._parameter_on_device(bias, "cpu")
    assert isinstance(parameter, torch.nn.Parameter)
    assert parameter.data_ptr() == bias.data_ptr()
    assert parameter.requires_grad is False


def test_mxfp8_cache_bias_validation_accepts_legacy_parameter_metadata():
    _assert_bias_helpers_accept_legacy_tensor_type(mxfp8_model_cache)


def test_nvfp4_cache_bias_validation_accepts_legacy_parameter_metadata():
    _assert_bias_helpers_accept_legacy_tensor_type(nvfp4_model_cache)


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
