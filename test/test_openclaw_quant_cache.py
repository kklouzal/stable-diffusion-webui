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
