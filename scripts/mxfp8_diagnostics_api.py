
from __future__ import annotations

import os
import threading

import gradio as gr
from fastapi import FastAPI

from modules import script_callbacks


def api_mxfp8_diagnostics(_: gr.Blocks, app: FastAPI):
    from modules import mxfp8_diagnostics

    @app.get("/sdapi/v1/mxfp8-diagnostics")
    async def get_mxfp8_diagnostics():
        return mxfp8_diagnostics.get_last_result()

    @app.post("/sdapi/v1/mxfp8-diagnostics/run")
    async def run_mxfp8_diagnostics(include_benchmarks: bool = True):
        started = mxfp8_diagnostics.run_probe_background(include_benchmarks=include_benchmarks)
        return {"ok": True, "started": started, "running": mxfp8_diagnostics.get_last_result().get("running"), "path": str(mxfp8_diagnostics.last_result_path())}

    delay = float(os.environ.get("A1111_MXFP8_STARTUP_PROBE_DELAY", "20"))
    if os.environ.get("A1111_MXFP8_STARTUP_PROBE", "1") not in ("0", "false", "False"):
        threading.Timer(delay, lambda: mxfp8_diagnostics.run_probe_background(include_benchmarks=True)).start()


script_callbacks.on_app_started(api_mxfp8_diagnostics)
