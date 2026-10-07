import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from modules import call_queue, mxfp8_diagnostics


def test_probe_never_touches_global_rng_or_global_sdpa(monkeypatch, tmp_path):
    monkeypatch.setattr(mxfp8_diagnostics, "_data_dir", lambda: tmp_path)
    # The audit only reads the loaded model's attributes; importing modules.shared here would parse pytest's argv.
    monkeypatch.setattr(mxfp8_diagnostics, "a1111_integration_audit", lambda: {"ok": False, "error": "sd_model is not loaded"})
    original_sdpa = F.scaled_dot_product_attention
    torch.manual_seed(99)
    rng_state = torch.get_rng_state()

    result = mxfp8_diagnostics.run_probe(include_benchmarks=True, save=False)

    assert torch.equal(rng_state, torch.get_rng_state())
    assert F.scaled_dot_product_attention is original_sdpa
    assert result["sdpa_coverage_check"]["ok"] is True
    assert result["sdpa_coverage_check"]["call_count"] == 1
    assert result["native_vs_emulated_detection"]["bf16"]["method"] in {"wall", "cuda_event"}
    assert "F.scaled_dot_product_attention =" not in Path("modules/mxfp8_diagnostics.py").read_text(encoding="utf8")


def test_seeded_linear_matches_nn_linear_default_init_from_the_same_seed():
    expected = torch.Generator().manual_seed(1234)
    reference = torch.empty(16, 64, dtype=torch.bfloat16)
    torch.nn.init.kaiming_uniform_(reference, a=5 ** 0.5, generator=expected)

    linear = mxfp8_diagnostics._seeded_linear(64, 16, "cpu", mxfp8_diagnostics._probe_generator("cpu", 1234))

    assert torch.equal(linear.weight, reference)
    assert linear.bias is None and linear.weight.dtype == torch.bfloat16


def test_sdpa_recorder_only_sees_calls_from_its_own_thread():
    q = torch.randn(1, 2, 4, 8)
    recorder = mxfp8_diagnostics._SdpaCallRecorder()

    with recorder:
        other = threading.Thread(target=lambda: F.scaled_dot_product_attention(q, q, q))
        other.start()
        other.join()
        F.scaled_dot_product_attention(query=q, key=q, value=q)

    assert [call["q_shape"] for call in recorder.calls] == [(1, 2, 4, 8)]


def test_background_probe_waits_for_the_generation_queue_lock(monkeypatch, tmp_path):
    monkeypatch.setattr(mxfp8_diagnostics, "_data_dir", lambda: tmp_path)
    started = threading.Event()

    def fake_run_probe(include_benchmarks=True, save=True):
        started.set()
        assert not call_queue.queue_lock.acquire(blocking=False)
        return {"ok": True}

    monkeypatch.setattr(mxfp8_diagnostics, "run_probe", fake_run_probe)

    with call_queue.queue_lock:
        assert mxfp8_diagnostics.run_probe_background(include_benchmarks=False) is True
        assert not started.wait(0.3), "probe ran while a generation held queue_lock"

    assert started.wait(10)
    deadline = time.monotonic() + 10
    while mxfp8_diagnostics.get_last_result()["running"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not mxfp8_diagnostics.get_last_result()["running"]
    assert mxfp8_diagnostics.get_last_result()["result"] == {"ok": True}
