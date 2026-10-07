"""Alwayson script hooks and generation callbacks fail the request instead of being logged, and a failed
generation still gives every script that ran a hook its postprocess_batch/postprocess cleanup."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from modules import script_callbacks, scripts

ROOT = Path(__file__).resolve().parents[1]


class Recorder(scripts.Script):
    """An alwayson script that records its hooks into a shared event list and raises where told to."""

    def __init__(self, name, events, raise_in=()):
        self.name = name
        self.filename = f"{name}.py"
        self.args_from = self.args_to = 0
        self.events = events
        self.raise_in = set(raise_in)

    def _hook(self, hook, detail=None):
        self.events.append((self.name, hook) if detail is None else (self.name, hook, detail))
        if hook in self.raise_in:
            raise RuntimeError(f"{self.name} failed in {hook}")

    def before_process(self, p, *args):
        self._hook("before_process")

    def process(self, p, *args):
        self._hook("process")

    def before_process_batch(self, p, *args, **kwargs):
        self._hook("before_process_batch")

    def process_batch(self, p, *args, **kwargs):
        self._hook("process_batch")

    def postprocess_batch(self, p, *args, **kwargs):
        self._hook("postprocess_batch", (tuple(kwargs["images"].shape), kwargs["batch_number"]))

    def postprocess_image(self, p, pp, *args):
        self._hook("postprocess_image")

    def postprocess(self, p, processed, *args):
        self._hook("postprocess", len(processed.images))


def make_runner(*script_list):
    runner = scripts.ScriptRunner()
    runner.scripts = list(script_list)
    runner.alwayson_scripts = list(script_list)
    return runner


def make_p(runner):
    return SimpleNamespace(scripts=runner, script_args=(), iteration=0, height=64, width=96)


def test_setup_hook_failure_fails_the_request_and_stops_later_scripts():
    # Before: Dynamic Thresholding's "Cannot use sampler UniPC" (and SEG's missing-attention error) were only
    # logged; the image rendered without the requested extension.
    events = []
    first, second = Recorder("dt", events, raise_in={"process_batch"}), Recorder("cn", events)
    runner = make_runner(first, second)

    with pytest.raises(RuntimeError, match="dt failed in process_batch") as raised:
        runner.process_batch(make_p(runner), batch_number=0, prompts=[], seeds=[], subseeds=[])

    assert events == [("dt", "process_batch")]
    assert any("process_batch of script dt.py" in note for note in raised.value.__notes__)


@pytest.mark.parametrize("hook, call", [
    ("process", lambda runner, p: runner.process(p)),
    ("before_process", lambda runner, p: runner.before_process(p)),
    ("postprocess_image", lambda runner, p: runner.postprocess_image(p, scripts.PostprocessImageArgs(None))),
])
def test_every_setup_and_output_hook_family_propagates(hook, call):
    events = []
    runner = make_runner(Recorder("a", events, raise_in={hook}))
    with pytest.raises(RuntimeError, match=f"a failed in {hook}"):
        call(runner, make_p(runner))


def test_cleanup_hooks_run_for_every_script_and_then_raise():
    events = []
    runner = make_runner(Recorder("a", events, raise_in={"postprocess"}), Recorder("b", events))
    with pytest.raises(RuntimeError, match="a failed in postprocess"):
        runner.postprocess(make_p(runner), SimpleNamespace(images=[1]))
    assert events == [("a", "postprocess", 1), ("b", "postprocess", 1)]

    events.clear()
    runner = make_runner(Recorder("a", events, raise_in={"postprocess"}), Recorder("b", events, raise_in={"postprocess"}))
    with pytest.raises(ExceptionGroup) as raised:
        runner.postprocess(make_p(runner), SimpleNamespace(images=[]))
    assert [str(e) for e in raised.value.exceptions] == ["a failed in postprocess", "b failed in postprocess"]


def _load_process_images(**namespace):
    tree = ast.parse((ROOT / "modules/processing.py").read_text(encoding="utf8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ("process_images", "_failed_batch_images")]
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    namespace |= {
        "StableDiffusionProcessing": object,
        "Processed": lambda p, images: SimpleNamespace(images=images),
        "torch": torch,
        "store_processing_override_settings": lambda p: {"stored": True},
        "apply_processing_override_settings": lambda p: None,
        "sd_models": SimpleNamespace(apply_token_merging=lambda model, ratio: None),
        "sd_samplers": SimpleNamespace(fix_p_invalid_sampler_and_scheduler=lambda p: None),
        "profiling": SimpleNamespace(Profiler=nullcontext),
    }
    exec(compile(module, "modules/processing.py", "exec"), namespace)
    return namespace["process_images"]


def _processing(runner, events, inner):
    p = SimpleNamespace(
        scripts=runner, script_args=(), iteration=0, height=64, width=96, sd_model=object(),
        override_settings_restore_afterwards=True, get_token_merging_ratio=lambda: 0.0,
    )
    process_images = _load_process_images(
        process_images_inner=inner,
        extra_networks=SimpleNamespace(deactivate=lambda p, data: events.append(("core", "deactivate"))),
        restore_processing_override_settings=lambda stored: events.append(("core", "restore-overrides")),
    )
    return p, process_images


class SamplingFailure(RuntimeError):
    pass


def test_failed_generation_runs_batch_and_request_cleanup_before_core_restore():
    events = []
    runner = make_runner(Recorder("tiled_vae", events), Recorder("detail_daemon", events))

    def inner(p):
        p._active_extra_network_data = {"lora": []}
        runner.process(p)
        runner.before_process_batch(p, batch_number=0, prompts=[], seeds=[], subseeds=[])
        runner.process_batch(p, batch_number=0, prompts=[], seeds=[], subseeds=[])
        raise SamplingFailure("NaN in unet")

    p, process_images = _processing(runner, events, inner)
    with pytest.raises(SamplingFailure):
        process_images(p)

    cleanup = [e for e in events if e[1] in ("postprocess_batch", "postprocess", "deactivate", "restore-overrides")]
    assert cleanup == [
        ("tiled_vae", "postprocess_batch", ((0, 3, 64, 96), 0)),
        ("detail_daemon", "postprocess_batch", ((0, 3, 64, 96), 0)),
        ("tiled_vae", "postprocess", 0),
        ("detail_daemon", "postprocess", 0),
        ("core", "deactivate"),
        ("core", "restore-overrides"),
    ]
    assert p._script_lifecycle is None


def test_cleanup_only_reaches_scripts_that_ran_and_never_repeats_started_cleanup():
    events = []
    failing = Recorder("controlnet", events, raise_in={"before_process"})
    never_ran = Recorder("teacache", events)
    runner = make_runner(failing, never_ran)

    p, process_images = _processing(runner, events, lambda p: pytest.fail("generation must not start"))
    with pytest.raises(RuntimeError, match="controlnet failed in before_process"):
        process_images(p)
    # No hook of teacache ran, so it gets no cleanup; no batch was open, so no postprocess_batch.
    assert [e for e in events if e[0] != "core"] == [("controlnet", "before_process"), ("controlnet", "postprocess", 0)]

    events.clear()
    runner = make_runner(Recorder("a", events, raise_in={"postprocess"}), Recorder("b", events))

    def inner(p):
        runner.process(p)
        runner.before_process_batch(p, batch_number=0, prompts=[], seeds=[], subseeds=[])
        runner.postprocess_batch(p, torch.zeros(1, 3, 64, 96), batch_number=0)
        runner.postprocess(p, SimpleNamespace(images=[object()]))

    p, process_images = _processing(runner, events, inner)
    with pytest.raises(RuntimeError, match="a failed in postprocess"):
        process_images(p)
    assert [e[1] for e in events if e[0] == "a"] == ["before_process", "process", "before_process_batch", "postprocess_batch", "postprocess"]
    assert [e[1] for e in events if e[0] == "b"] == ["before_process", "process", "before_process_batch", "postprocess_batch", "postprocess"]


def test_cleanup_failure_is_attached_to_the_primary_error_not_raised_instead():
    events = []
    runner = make_runner(Recorder("a", events, raise_in={"postprocess"}), Recorder("b", events))

    def inner(p):
        runner.process(p)
        raise SamplingFailure("out of memory")

    p, process_images = _processing(runner, events, inner)
    with pytest.raises(SamplingFailure) as raised:
        process_images(p)
    assert ("b", "postprocess", 0) in events
    assert any("a failed in postprocess" in note for note in raised.value.__notes__)


def test_nested_generation_on_the_same_p_restores_the_outer_tracker():
    runner = make_runner()
    p = make_p(runner)
    outer = runner.begin_generation(p)
    tracker = p._script_lifecycle
    inner = runner.begin_generation(p)
    assert inner is tracker and p._script_lifecycle is not tracker
    runner.end_generation(p, inner)
    assert p._script_lifecycle is tracker
    runner.end_generation(p, outer)
    assert p._script_lifecycle is None


@pytest.mark.parametrize("register, dispatch, params", [
    (script_callbacks.on_cfg_denoiser, script_callbacks.cfg_denoiser_callback, "denoiser"),
    (script_callbacks.on_cfg_denoised, script_callbacks.cfg_denoised_callback, "denoised"),
    (script_callbacks.on_cfg_after_cfg, script_callbacks.cfg_after_cfg_callback, "after_cfg"),
    (script_callbacks.on_extra_noise, script_callbacks.extra_noise_callback, "noise"),
    (script_callbacks.on_before_image_saved, script_callbacks.before_image_saved_callback, "save"),
])
def test_output_callbacks_fail_the_generation(monkeypatch, register, dispatch, params):
    # Before: a PAG/SEG/ControlNet/TeaCache per-step callback that raised was logged every step and the
    # image rendered without its guidance while the infotext still claimed it.
    monkeypatch.setattr(script_callbacks, "callback_map", {key: [] for key in script_callbacks.callback_map})
    monkeypatch.setattr(script_callbacks, "ordered_callbacks_map", {})
    seen = []

    def broken(p):
        raise ValueError(f"broken {p}")

    register(broken)
    register(lambda p: seen.append(p))
    with pytest.raises(ValueError, match=f"broken {params}") as raised:
        dispatch(params)
    assert seen == []
    assert any("callback" in note for note in raised.value.__notes__)


def test_notification_callbacks_keep_the_upstream_log_only_contract(monkeypatch, capsys):
    monkeypatch.setattr(script_callbacks, "callback_map", {key: [] for key in script_callbacks.callback_map})
    monkeypatch.setattr(script_callbacks, "ordered_callbacks_map", {})
    seen = []
    script_callbacks.on_image_saved(lambda params: 1 / 0)
    script_callbacks.on_image_saved(lambda params: seen.append(params))
    script_callbacks.image_saved_callback("saved")
    assert seen == ["saved"]
    assert "ZeroDivisionError" in capsys.readouterr().err
