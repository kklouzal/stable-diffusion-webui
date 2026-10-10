"""modules/openclaw_warmup.py: the startup warm-up replays the last generation once, under queue_lock, without saving
anything, without persisted option changes and without recording itself in generation_last."""
from __future__ import annotations

import base64
import copy
import io
import threading
from types import SimpleNamespace

import pytest
from PIL import Image

from modules import openclaw_warmup
from modules.fifo_lock import FIFOLock


def _png(width, height):
    with io.BytesIO() as output:
        Image.new("RGB", (width, height), (40, 80, 120)).save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("ascii")


def _snapshot(generation_type="img2img", **parameters):
    base = {
        "sampler_name": "Euler", "scheduler": "Automatic", "steps": 3, "cfg_scale": 5.0, "seed": 1234, "subseed": 5,
        "batch_size": 1, "n_iter": 1, "denoising_strength": 0.5, "styles": ["old style"],
        "send_images": True, "save_images": False,
        "override_settings": {"CLIP_stop_at_last_layers": 2}, "override_settings_restore_afterwards": False,
    }
    if generation_type == "img2img":
        base["init_images"] = [_png(96, 64)]
    base.update(parameters)
    return {
        "schema_version": 3, "completed_at": "2026-10-09T00:00:00Z", "generation_type": generation_type,
        "replayable": True, "limitations": [], "checkpoint": {}, "parameters": base,
        "settings_parameters": {}, "settings_replayable": True, "settings_limitations": [],
        "lora_tags": ["<lora:alpha:0.5>", "<lora:beta>"],
    }


def _opts(labels=("save_init_img", "control_net_detectmap_autosaving"), data=None):
    return SimpleNamespace(data_labels={key: None for key in labels}, data=dict(data or {}))


def test_mode_is_off_unless_generation_last_and_rejects_other_values():
    assert openclaw_warmup.mode_from_env({}) == "off"
    assert openclaw_warmup.mode_from_env({"OPENCLAW_WARMUP": " "}) == "off"
    assert openclaw_warmup.mode_from_env({"OPENCLAW_WARMUP": "off"}) == "off"
    assert openclaw_warmup.mode_from_env({"OPENCLAW_WARMUP": "generation-last"}) == "generation-last"
    for value in ("on", "1", "Generation-Last"):
        with pytest.raises(ValueError, match="OPENCLAW_WARMUP"):
            openclaw_warmup.mode_from_env({"OPENCLAW_WARMUP": value})


def test_img2img_request_replays_the_snapshot_without_outputs_and_restores_settings():
    multi_args = [True, "run-1", 1, 0, "Approx cheap"]
    snapshot = _snapshot(alwayson_scripts={
        "OpenClaw Multi-Sampler": {"args": list(multi_args)},
        "ControlNet": {"args": [{"enabled": True, "image": "unit-image"}]},
    })
    original = copy.deepcopy(snapshot)

    tabname, request = openclaw_warmup.build_request(snapshot, _opts())

    assert snapshot == original  # the snapshot is not modified
    assert tabname == "img2img"
    assert request["prompt"] == "warm-up <lora:alpha:0.5> <lora:beta>"
    assert request["negative_prompt"] == "" and request["styles"] == []
    assert (request["width"], request["height"]) == (96, 64)  # the init image's size
    assert request["save_images"] is False and request["send_images"] is False and request["include_init_images"] is False
    assert request["override_settings_restore_afterwards"] is True
    assert request["override_settings"] == {"CLIP_stop_at_last_layers": 2, "save_init_img": False, "control_net_detectmap_autosaving": False}
    assert request["force_task_id"] == openclaw_warmup.TASK_ID
    assert request["alwayson_scripts"]["OpenClaw Multi-Sampler"]["args"] == [False, *multi_args[1:]]
    assert request["alwayson_scripts"]["ControlNet"] == {"args": [{"enabled": True, "image": "unit-image"}]}
    for key in ("sampler_name", "steps", "cfg_scale", "seed", "denoising_strength", "init_images"):
        assert request[key] == snapshot["parameters"][key]


def test_txt2img_request_uses_the_fixed_size_without_hires():
    tabname, request = openclaw_warmup.build_request(_snapshot("txt2img", enable_hr=True), _opts(labels=("save_init_img",)))
    assert tabname == "txt2img"
    assert (request["width"], request["height"], request["enable_hr"]) == (1024, 1024, False)
    assert request["override_settings"] == {"CLIP_stop_at_last_layers": 2, "save_init_img": False}


@pytest.mark.parametrize(("snapshot", "opts", "reason"), [
    (None, _opts(), "no last-generation snapshot"),
    ({**_snapshot(), "schema_version": 2}, _opts(), "schema version 2"),
    ({**_snapshot(), "generation_type": "extras"}, _opts(), "not txt2img or img2img"),
    ({**_snapshot(), "replayable": False, "limitations": ["init image too large"]}, _opts(), "init image too large"),
    (_snapshot(script_name="Ultimate SD upscale"), _opts(), "Ultimate SD upscale"),
    (_snapshot(), _opts(labels=("save_init_img",), data={"control_net_detectmap_autosaving": True}), "control_net_detectmap_autosaving"),
])
def test_unreplayable_snapshots_skip(snapshot, opts, reason):
    with pytest.raises(openclaw_warmup.WarmupSkipped, match=reason):
        openclaw_warmup.build_request(snapshot, opts)


@pytest.fixture
def fresh_status(initialize, monkeypatch):
    """A private warm-up status, with the API module the warm-up runs requests through importable (modules.processing
    first, as at startup: importing modules.api.api on its own enters an sd_hijack import cycle)."""
    import modules.processing  # noqa: F401
    import modules.api.api  # noqa: F401

    monkeypatch.setattr(openclaw_warmup, "_status", dict(openclaw_warmup._status))


def _api(events, *, fail=None, images=("image",)):
    """An Api stand-in: the real FIFOLock, and generation steps that record what the warm-up hands them."""
    lock = FIFOLock()

    def prepare(model, tabname, script_runner, default_script_args, update=None, extra_pop_fields=()):
        events.append(("prepare", tabname, model, default_script_args, update, extra_pop_fields))
        return {"args": True}, False, None, ["script-args"], {}

    def run_generation_task(task_id, tabname, args, script_runner, selectable_scripts, script_args, script_arg_ranges, *, configure=None):
        assert lock._owner == threading.get_ident()  # queue_lock is held by the warm-up thread
        assert openclaw_warmup.status()["state"] == "running"
        p = SimpleNamespace()
        configure(p)
        events.append(("task", task_id, tabname, p))
        if fail:
            raise fail
        return SimpleNamespace(images=list(images))

    return SimpleNamespace(
        queue_lock=lock, default_script_arg_img2img=["img2img-default"], default_script_arg_txt2img=["txt2img-default"],
        _prepare_generation_api_request=prepare, _run_generation_task=run_generation_task,
    )


def test_run_holds_queue_lock_and_marks_the_processing_object_captured(fresh_status):
    from modules import generation_last

    events = []
    api = _api(events)
    openclaw_warmup.run(api, load_snapshot=_snapshot)

    (_, tabname, model, default_args, update, extra_pop), (_, task_id, _, p) = events
    assert tabname == "img2img" and task_id == openclaw_warmup.TASK_ID
    assert model.save_images is False and model.send_images is False and model.override_settings_restore_afterwards is True
    assert model.prompt == "warm-up <lora:alpha:0.5> <lora:beta>" and (model.width, model.height) == (96, 64)
    assert default_args == api.default_script_arg_img2img and default_args is not api.default_script_arg_img2img
    assert update == {"mask": None} and extra_pop == ("include_init_images",)
    assert [image.size for image in p.init_images] == [(96, 64)]
    # generation_last records nothing for this processing object, built or persisted.
    assert p._generation_last_captured is True
    assert generation_last.snapshot_or_report(p, SimpleNamespace(images=["image"])) is None
    status = openclaw_warmup.status()
    assert status["state"] == "succeeded" and status["error"] is None
    assert status["started_at"] and status["finished_at"] and status["seconds"] >= 0
    assert status["request"]["generation_type"] == "img2img" and status["request"]["lora_tags"] == 2
    assert api.queue_lock._owner is None  # released


def test_a_waiting_request_runs_after_the_warmup(fresh_status):
    """A request arriving during the warm-up waits on queue_lock and runs once it is released."""
    events = []
    api = _api(events)
    entered = threading.Event()
    release = threading.Event()
    original = api._run_generation_task

    def slow_task(*args, **kwargs):
        entered.set()
        release.wait(10)
        return original(*args, **kwargs)

    api._run_generation_task = slow_task
    warmup = threading.Thread(target=openclaw_warmup.run, args=(api, _snapshot))
    warmup.start()
    assert entered.wait(10)

    def request():
        with api.queue_lock:
            events.append(("request",))

    waiting = threading.Thread(target=request)
    waiting.start()
    waiting.join(0.2)
    assert waiting.is_alive()  # blocked behind the warm-up
    release.set()
    warmup.join(10)
    waiting.join(10)
    assert [event[0] for event in events] == ["prepare", "task", "request"]
    assert openclaw_warmup.status()["state"] == "succeeded"


@pytest.mark.parametrize(("fail", "images", "error"), [
    (RuntimeError("CUDA error: out of memory"), ("image",), "RuntimeError: CUDA error: out of memory"),
    (None, (), "returned no images"),
])
def test_a_failed_warmup_is_reported_with_its_traceback_and_releases_the_lock(fresh_status, capsys, fail, images, error):
    events = []
    api = _api(events, fail=fail, images=images)
    openclaw_warmup.run(api, load_snapshot=_snapshot)

    status = openclaw_warmup.status()
    assert status["state"] == "failed" and error in status["error"]
    assert status["finished_at"] and status["seconds"] is not None
    stderr = capsys.readouterr().err
    assert "*** OpenClaw warm-up failed after" in stderr and "Traceback" in stderr and error.split(": ", 1)[-1] in stderr
    assert api.queue_lock._owner is None
    with api.queue_lock:  # a later request is not blocked
        pass


@pytest.mark.parametrize("phase", ["request", "generation"])
def test_a_base_exception_is_recorded_as_failed_then_propagates(fresh_status, capsys, phase):
    """A BaseException must not leave the status `running`/`pending`: the deploy smoke test waits for a final state."""
    events = []
    api = _api(events, fail=KeyboardInterrupt("stop") if phase == "generation" else None)

    def interrupted_snapshot():
        raise KeyboardInterrupt("stop")

    with pytest.raises(KeyboardInterrupt):
        openclaw_warmup.run(api, load_snapshot=_snapshot if phase == "generation" else interrupted_snapshot)

    status = openclaw_warmup.status()
    assert status["state"] == "failed" and status["error"] == "KeyboardInterrupt: stop" and status["finished_at"]
    assert (status["seconds"] is not None) == (phase == "generation")
    assert "OpenClaw warm-up failed" in capsys.readouterr().err
    assert api.queue_lock._owner is None


def test_a_missing_snapshot_skips_without_taking_the_lock(fresh_status, capsys):
    events = []
    api = _api(events)
    openclaw_warmup.run(api, load_snapshot=lambda: None)
    assert events == []
    status = openclaw_warmup.status()
    assert status["state"] == "skipped" and status["error"] == "no last-generation snapshot"
    assert "OpenClaw warm-up skipped: no last-generation snapshot" in capsys.readouterr().out


def test_install_registers_the_status_route_and_starts_only_when_enabled(fresh_status, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from modules import script_callbacks

    registered = []
    monkeypatch.setattr(script_callbacks, "on_app_started", lambda callback, name=None: registered.append((name, callback)))
    app = FastAPI()
    api = SimpleNamespace(add_api_route=app.add_api_route)

    monkeypatch.setenv("OPENCLAW_WARMUP", "off")
    openclaw_warmup.install(api)
    assert registered == []
    body = TestClient(app).get("/sdapi/v1/openclaw/warmup").json()
    assert body["mode"] == "off" and body["state"] == "off"

    monkeypatch.setenv("OPENCLAW_WARMUP", "generation-last")
    started = []
    monkeypatch.setattr(openclaw_warmup, "start", started.append)
    app = FastAPI()
    api = SimpleNamespace(add_api_route=app.add_api_route)
    openclaw_warmup.install(api)
    ((name, callback),) = registered
    assert name == "openclaw_warmup" and started == []
    callback(None, app)
    assert started == [api]
    openclaw_warmup._set_status(state="failed", error="RuntimeError: boom", seconds=1.5)
    body = TestClient(app).get("/sdapi/v1/openclaw/warmup").json()
    assert body["mode"] == "generation-last" and body["state"] == "failed" and body["error"] == "RuntimeError: boom" and body["seconds"] == 1.5

    monkeypatch.setenv("OPENCLAW_WARMUP", "yes")
    with pytest.raises(ValueError):
        openclaw_warmup.install(SimpleNamespace(add_api_route=FastAPI().add_api_route))


def test_the_real_api_generation_path_gets_a_saveless_restoring_uncaptured_request(fresh_status, monkeypatch):
    """The real Api request preparation and generation task (process_images stubbed on CPU): the processing object
    saves nothing, restores its overrides, is never captured by generation_last, runs as TASK_ID under queue_lock, and
    the denoise-ramp default persistence does not reach the Api's defaults."""
    from modules import generation_last, progress
    from modules.api import api as api_module

    class Script:
        alwayson = True

        def __init__(self, title, args_from, args_to):
            self._title, self.args_from, self.args_to = title, args_from, args_to

        def title(self):
            return self._title

    multi = Script("OpenClaw Multi-Sampler", 1, 6)
    ramp = Script("OpenClaw Denoise Ramp", 6, 8)
    setups = []
    runner = SimpleNamespace(scripts=[multi, ramp], selectable_scripts=[], alwayson_scripts=[multi, ramp],
                             setup_scrips=lambda p, *, is_ui: setups.append(is_ui))
    monkeypatch.setattr(api_module.scripts, "scripts_img2img", runner)
    api = api_module.Api.__new__(api_module.Api)
    api.queue_lock = FIFOLock()
    api.default_script_arg_img2img = [None, False, "", 1, 0, "Approx cheap", False, 0.0]
    defaults = list(api.default_script_arg_img2img)

    seen = {}

    def process_images(p):
        assert api.queue_lock._owner == threading.get_ident()
        seen.update(task=progress.current_task, p=p)
        return SimpleNamespace(images=["image"])

    monkeypatch.setattr(api_module, "process_images", process_images)
    snapshot = _snapshot(alwayson_scripts={
        "OpenClaw Multi-Sampler": {"args": [True, "run-1", 2, 4, "TAESD"]},
        "OpenClaw Denoise Ramp": {"args": [True, 0.25]},
    })
    openclaw_warmup.run(api, load_snapshot=lambda: snapshot)

    assert openclaw_warmup.status()["state"] == "succeeded", openclaw_warmup.status()
    p = seen["p"]
    assert seen["task"] == openclaw_warmup.TASK_ID and setups == [False]  # set up as an API request
    assert p.do_not_save_samples and p.do_not_save_grid
    assert p.override_settings_restore_afterwards is True
    assert p.override_settings["save_init_img"] is False and p.override_settings["CLIP_stop_at_last_layers"] == 2
    assert (p.width, p.height, p.prompt, p.negative_prompt, p.seed) == (96, 64, "warm-up <lora:alpha:0.5> <lora:beta>", "", 1234)
    assert [image.size for image in p.init_images] == [(96, 64)]
    assert list(p.script_args[1:8]) == [False, "run-1", 2, 4, "TAESD", True, 0.25]
    assert p._generation_last_captured is True
    assert generation_last.snapshot_or_report(p, SimpleNamespace(images=["image"])) is None
    assert api.default_script_arg_img2img == defaults
