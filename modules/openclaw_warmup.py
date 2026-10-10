"""Opt-in startup warm-up: replay the last completed generation once, in-process, before production traffic.

After a (re)start the first production request pays one-time costs (cuDNN/cuBLAS first use and autotuning per shape,
ControlNet model and annotator loads, LoRA loads and merges, CUDA graph captures). With OPENCLAW_WARMUP set to
`generation-last`, a background thread started from the app_started callbacks takes queue_lock (the FIFO lock every
generation takes) and runs the request recorded by modules/generation_last.py once; requests arriving meanwhile wait on
queue_lock like behind any other generation. The API itself, /sdapi/v1/progress included, serves during the warm-up.

What is replayed (build_request): the snapshot's prior-task `parameters` (sampler, scheduler, steps, CFG, denoising,
seed, init image, ControlNet units with their images, every always-on script's arguments, override_settings), with:
- prompt: NEUTRAL_PROMPT followed by the snapshot's `lora_tags` (the snapshot keeps no prompt text); negative prompt
  empty, styles cleared. A real prompt longer than one 75-token chunk has a longer conditioning, so shapes keyed on
  the context length (UNet CUDA graphs) can still be captured by the first real request.
- size: img2img uses the first init image's size; txt2img uses 1024x1024 with the hires pass off (the snapshot keeps
  no size, see docs/gb10/generation-last-api.md).
- no outputs: save_images/send_images off, the init-image and ControlNet detected-map autosave options overridden
  off, OpenClaw Multi-Sampler snapshots off, override_settings restored afterwards, and the processing object marked
  as already captured so generation_last never records the warm-up. The request runs through the API path, which
  builds its script arguments from a copy of the Api's defaults and leaves those unchanged. A selectable script
  (script_name) is never replayed: such scripts save images on their own (Ultimate SD Upscale).
- labelled: progress task id TASK_ID (`current_task` of /sdapi/v1/progress while it runs).

A snapshot that is missing or not replayable skips the warm-up (logged, state `skipped`). A failing warm-up is not a
production failure: it is reported with its traceback (errors.report), recorded in the status, and later requests run
normally. GET /sdapi/v1/openclaw/warmup returns status()."""

from __future__ import annotations

import base64
import copy
import datetime as dt
import io
import os
import threading
import time
from typing import Any

ENV = "OPENCLAW_WARMUP"
MODES = ("off", "generation-last")
TASK_ID = "openclaw-warmup"
NEUTRAL_PROMPT = "warm-up"
TXT2IMG_SIZE = 1024
MULTI_SAMPLER_TITLE = "openclaw multi-sampler"
# Options that write files even when a request does not save images; the warm-up overrides each one off.
OUTPUT_OPTIONS = ("save_init_img", "control_net_detectmap_autosaving")

_status_lock = threading.Lock()
_status: dict[str, Any] = {"mode": "off", "state": "off", "started_at": None, "finished_at": None, "seconds": None, "error": None, "request": None}


class WarmupSkipped(Exception):
    """The snapshot cannot be replayed as a warm-up; not a failure."""


def mode_from_env(environ=os.environ) -> str:
    """The OPENCLAW_WARMUP mode: unset or empty -> "off"; any value outside MODES raises ValueError (read at startup)."""
    value = environ.get(ENV, "").strip()
    if not value:
        return "off"
    if value not in MODES:
        raise ValueError(f"{ENV}={value!r} is not one of {', '.join(MODES)}")
    return value


def status() -> dict[str, Any]:
    with _status_lock:
        return copy.deepcopy(_status)


def _set_status(**fields) -> None:
    with _status_lock:
        _status.update(fields)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _image_size(encoded: str) -> tuple[int, int]:
    from PIL import Image

    data = encoded.split(",", 1)[1] if encoded.startswith("data:") else encoded
    with Image.open(io.BytesIO(base64.b64decode(data, validate=True))) as image:
        return image.size


def build_request(snapshot: dict[str, Any] | None, opts) -> tuple[str, dict[str, Any]]:
    """(tabname, API request fields) replaying `snapshot` as described in the module docstring. `opts` is the live
    shared.opts: an output option that is on but unregistered (so it cannot be overridden) skips the warm-up."""
    if snapshot is None:
        raise WarmupSkipped("no last-generation snapshot")
    if snapshot.get("schema_version") != 3:
        raise WarmupSkipped(f"last-generation snapshot has schema version {snapshot.get('schema_version')!r}, not 3")
    tabname = snapshot.get("generation_type")
    if tabname not in ("txt2img", "img2img"):
        raise WarmupSkipped(f"last generation type {tabname!r} is not txt2img or img2img")
    if not snapshot.get("replayable"):
        raise WarmupSkipped("last generation is not replayable: " + "; ".join(snapshot.get("limitations") or ["no reason recorded"]))
    request = copy.deepcopy(snapshot.get("parameters") or {})
    if request.get("script_name"):
        raise WarmupSkipped(f"last generation ran the script {request['script_name']!r}, which can save images on its own")

    request["prompt"] = " ".join([NEUTRAL_PROMPT, *snapshot.get("lora_tags", [])])
    request["negative_prompt"] = ""
    request["styles"] = []
    if tabname == "img2img":
        request["width"], request["height"] = _image_size(request["init_images"][0])
        request["include_init_images"] = False
    else:
        request["width"] = request["height"] = TXT2IMG_SIZE
        request["enable_hr"] = False
    request["save_images"] = False
    request["send_images"] = False
    request["force_task_id"] = TASK_ID
    request["override_settings_restore_afterwards"] = True

    override_settings = dict(request.get("override_settings") or {})
    for key in OUTPUT_OPTIONS:
        if key in opts.data_labels:
            override_settings[key] = False
        elif opts.data.get(key):
            raise WarmupSkipped(f"option {key} is on but not registered, so the warm-up cannot turn it off")
    request["override_settings"] = override_settings

    for title, entry in (request.get("alwayson_scripts") or {}).items():
        if title.casefold() == MULTI_SAMPLER_TITLE and isinstance(entry, dict) and entry.get("args"):
            entry["args"][0] = False  # snapshots enabled
    return tabname, request


def _request_summary(tabname: str, request: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "generation_type": tabname,
        "snapshot_completed_at": snapshot.get("completed_at"),
        "width": request["width"],
        "height": request["height"],
        "steps": request.get("steps"),
        "sampler_name": request.get("sampler_name"),
        "lora_tags": len(snapshot.get("lora_tags", [])),
        "alwayson_scripts": sorted(request.get("alwayson_scripts") or {}),
    }


def _generate(api, tabname: str, request: dict[str, Any]):
    """Run `request` through the API's generation path (the caller holds queue_lock). Returns the Processed."""
    from modules import scripts
    from modules.api import api as api_module, models

    img2img = tabname == "img2img"
    model = (models.StableDiffusionImg2ImgProcessingAPI if img2img else models.StableDiffusionTxt2ImgProcessingAPI)(**request)
    script_runner = scripts.scripts_img2img if img2img else scripts.scripts_txt2img
    default_script_args = api.default_script_arg_img2img if img2img else api.default_script_arg_txt2img

    @api_module.decode_inline_images_once
    def run():
        update, extra_pop_fields, init_images = None, (), None
        if img2img:
            update = {"mask": api_module.decode_base64_to_image(model.mask) if model.mask else None}
            extra_pop_fields = ("include_init_images",)
            init_images = [api_module.decode_base64_to_image(image) for image in model.init_images]
        args, _send_images, selectable_scripts, script_args, script_arg_ranges = api._prepare_generation_api_request(
            model, tabname, script_runner, default_script_args, update=update, extra_pop_fields=extra_pop_fields)

        def configure(p):
            if init_images is not None:
                p.init_images = init_images
            p._generation_last_captured = True  # generation_last never records the warm-up

        return api._run_generation_task(TASK_ID, tabname, args, script_runner, selectable_scripts, script_args, script_arg_ranges, configure=configure)

    return run()


def run(api, load_snapshot=None) -> None:
    """One warm-up, on the calling thread: build the request, then hold queue_lock while it runs. Always leaves a final
    state (succeeded, skipped or failed); raises only a BaseException that is not an Exception, after recording it."""
    from modules import errors, generation_last, shared

    tabname = request = None
    try:
        snapshot = (load_snapshot or generation_last.get_last_snapshot)()
        tabname, request = build_request(snapshot, shared.opts)
        _set_status(request=_request_summary(tabname, request, snapshot))
    except WarmupSkipped as reason:
        print(f"OpenClaw warm-up skipped: {reason}")
        _set_status(state="skipped", error=str(reason), finished_at=_utc_now())
        return
    except BaseException as error:
        errors.report("OpenClaw warm-up failed: the last-generation snapshot could not be turned into a request", exc_info=True)
        _set_status(state="failed", error=f"{type(error).__name__}: {error}", finished_at=_utc_now())
        if not isinstance(error, Exception):
            raise
        return

    with api.queue_lock:
        started = time.perf_counter()
        _set_status(state="running", started_at=_utc_now())
        print(f"OpenClaw warm-up: replaying the last {tabname} generation ({request['width']}x{request['height']}, {request.get('steps')} steps)")
        try:
            processed = _generate(api, tabname, request)
            if getattr(shared.state, "interrupted", False) or getattr(shared.state, "stopping_generation", False):
                raise RuntimeError("the warm-up generation was interrupted")
            if not getattr(processed, "images", None):
                raise RuntimeError("the warm-up generation returned no images")
        except BaseException as error:
            # Every failure ends the `running` state (GET /sdapi/v1/openclaw/warmup and the deploy smoke test wait
            # for a final one); only an Exception is absorbed, a BaseException (KeyboardInterrupt, SystemExit, a
            # thread-killing exception) propagates once it is recorded.
            seconds = round(time.perf_counter() - started, 3)
            errors.report(f"OpenClaw warm-up failed after {seconds} s replaying the last {tabname} generation (request: {status()['request']})", exc_info=True)
            _set_status(state="failed", error=f"{type(error).__name__}: {error}", finished_at=_utc_now(), seconds=seconds)
            if not isinstance(error, Exception):
                raise
            return
        seconds = round(time.perf_counter() - started, 3)
        _set_status(state="succeeded", finished_at=_utc_now(), seconds=seconds)
        print(f"OpenClaw warm-up finished in {seconds} s")


def start(api) -> threading.Thread:
    """Start run(api) on a daemon thread; the state is `pending` until it holds queue_lock."""
    _set_status(state="pending", started_at=None, finished_at=None, seconds=None, error=None, request=None)
    thread = threading.Thread(target=run, args=(api,), name="openclaw-warmup", daemon=True)
    thread.start()
    return thread


def install(api) -> None:
    """At startup, once the Api exists: read OPENCLAW_WARMUP (a bad value fails the start), register
    GET /sdapi/v1/openclaw/warmup, and when enabled start the warm-up from an app_started callback, so it runs after
    the extensions' app_started callbacks."""
    from modules import script_callbacks

    mode = mode_from_env()
    _set_status(mode=mode, state="off" if mode == "off" else "pending")
    api.add_api_route("/sdapi/v1/openclaw/warmup", status, methods=["GET"])
    if mode == "generation-last":
        script_callbacks.on_app_started(lambda _demo, _app: start(api), name="openclaw_warmup")
