"""run_modelmerger's result contract with its only caller, /sdapi/v1/openclaw/model-merge (openclaw-clear-cond-cache):
a list whose last element is the status message, "Failed: ..." for a rejected request."""

from types import SimpleNamespace

import safetensors.torch
import torch

from modules import shared, shared_init

if getattr(shared, "opts", None) is None:
    shared_init.initialize()

from modules import extras, sd_models  # noqa: E402


class RecordingState(SimpleNamespace):
    def __init__(self, events):
        super().__init__(events=events, textinfo=None, job_count=0, sampling_step=0, sampling_steps=0)

    def begin(self, job):
        self.events.append(("begin", job))

    def end(self):
        self.events.append(("end",))

    def nextjob(self):
        pass


def merge(*, primary, secondary="", interp_method="Weighted sum", multiplier=0.25):
    return extras.run_modelmerger(None, primary, secondary, "", interp_method, multiplier, False, "", "safetensors", 0, "None", "", False, False, False, "{}")


def test_rejected_merge_returns_the_failed_message(monkeypatch):
    events = []
    monkeypatch.setattr(shared, "state", RecordingState(events))

    assert merge(primary="") == ["Failed: Merging requires a primary model."]
    assert events == [("begin", "model-merge"), ("end",)]


def test_weighted_sum_saves_the_merge_rescans_checkpoints_and_returns_the_saved_message(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(shared, "state", RecordingState(events))
    monkeypatch.setattr(shared.cmd_opts, "ckpt_dir", str(tmp_path / "out"))
    (tmp_path / "out").mkdir()

    a = {"model.diffusion_model.w": torch.tensor([1.0, 2.0]), "alphas": torch.tensor([5.0])}
    b = {"model.diffusion_model.w": torch.tensor([3.0, 6.0]), "alphas": torch.tensor([7.0])}
    checkpoints = {}
    for name, tensors in (("a", a), ("b", b)):
        filename = str(tmp_path / f"{name}.safetensors")
        safetensors.torch.save_file(tensors, filename)
        checkpoints[name] = SimpleNamespace(name=f"{name}.safetensors", model_name=name, filename=filename, metadata={})
    monkeypatch.setattr(sd_models, "checkpoints_list", checkpoints)
    rescans = []
    monkeypatch.setattr(sd_models, "list_models", lambda: rescans.append(True))

    output = str(tmp_path / "out" / "0.75(a) + 0.25(b).safetensors")
    assert merge(primary="a", secondary="b") == ["Checkpoint saved to " + output]

    merged = safetensors.torch.load_file(output)
    assert torch.equal(merged["model.diffusion_model.w"], 0.75 * a["model.diffusion_model.w"] + 0.25 * b["model.diffusion_model.w"])
    assert torch.equal(merged["alphas"], a["alphas"])  # only keys containing "model" are interpolated
    assert rescans == [True]
    assert events == [("begin", "model-merge"), ("end",)]
