import pytest
import torch

from modules import devices


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TF32 applies to CUDA")
def test_enable_tf32_matches_the_legacy_flag_state():
    backends = torch.backends
    saved = (backends.cuda.matmul.fp32_precision, backends.cudnn.conv.fp32_precision, backends.cudnn.rnn.fp32_precision)
    try:
        backends.cuda.matmul.fp32_precision = "ieee"
        backends.cudnn.conv.fp32_precision = "ieee"
        backends.cudnn.rnn.fp32_precision = "ieee"

        devices.enable_tf32()

        # The state the legacy allow_tf32 = True (cuBLAS and cuDNN) plus set_float32_matmul_precision("high") produced;
        # reading the legacy getters must also not raise the legacy/new API mixing error.
        assert backends.cuda.matmul.allow_tf32
        assert backends.cudnn.allow_tf32
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        backends.cuda.matmul.fp32_precision, backends.cudnn.conv.fp32_precision, backends.cudnn.rnn.fp32_precision = saved
