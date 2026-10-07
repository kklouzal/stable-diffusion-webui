from __future__ import annotations

import importlib.util
import inspect
import sys
import types
import unittest
import uuid
from pathlib import Path
from unittest import mock

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "openclaw_conditioning_probe.py"


class RecordingLock:
    def __init__(self):
        self.held = False

    def __enter__(self):
        self.held = True

    def __exit__(self, exc_type, exc, tb):
        self.held = False


class FakeApp:
    def __init__(self):
        self.routes = {}

    def get(self, path):
        def decorator(fn):
            self.routes[path] = fn
            return fn

        return decorator


def load_probe(lock):
    stubs = {name: types.ModuleType(f"modules.{name}") for name in ("call_queue", "script_callbacks", "shared", "openclaw_cache_epochs", "sd_hijack")}
    stubs["call_queue"].queue_lock = lock
    stubs["script_callbacks"].on_app_started = lambda callback: None
    modules_pkg = types.ModuleType("modules")
    for name, module in stubs.items():
        setattr(modules_pkg, name, module)

    # Stub A1111 only while the script imports, so other tests in the session keep the real `modules`.
    with mock.patch.dict(sys.modules, {"modules": modules_pkg, **{f"modules.{name}": module for name, module in stubs.items()}}):
        spec = importlib.util.spec_from_file_location(f"openclaw_conditioning_probe_test_{uuid.uuid4().hex}", SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class SnapshotEndpointTests(unittest.TestCase):
    def test_snapshot_is_a_threadpool_endpoint_that_holds_queue_lock(self):
        lock = RecordingLock()
        probe = load_probe(lock)
        app = FakeApp()
        probe.on_app_started(None, app)
        seen = []

        def snapshot(label):
            seen.append((label, lock.held))
            return {"label": label}

        probe.snapshot = snapshot
        handler = app.routes["/sdapi/v1/openclaw/conditioning-probe/snapshot"]

        self.assertFalse(inspect.iscoroutinefunction(handler))
        self.assertEqual(handler(label="before"), {"label": "before"})
        self.assertEqual(seen, [("before", True)])
        self.assertFalse(lock.held)


if __name__ == "__main__":
    unittest.main()
