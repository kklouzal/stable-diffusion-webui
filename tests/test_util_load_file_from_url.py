"""modules.util.load_file_from_url: model downloads are bounded by a (connect, read) timeout, close their response, and
are saved under the URL's basename unless a file name is given."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_STUBBED = ("modules", "modules.shared", "modules.paths_internal", "modules.util", "requests")


class _Response:
    def __init__(self, chunks, error=None):
        self.headers = {"content-length": str(sum(map(len, chunks)))}
        self.chunks = chunks
        self.error = error
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield from self.chunks
        if self.error is not None:
            raise self.error


@pytest.fixture()
def util():
    previous = {name: sys.modules.get(name) for name in _STUBBED}
    modules_pkg = types.ModuleType("modules")
    modules_pkg.__path__ = []
    paths_internal = types.ModuleType("modules.paths_internal")
    paths_internal.cwd = str(ROOT)
    requests = types.ModuleType("requests")
    requests.calls = []

    def get(url, **kwargs):
        requests.calls.append((url, kwargs))
        return requests.response

    requests.get = get
    sys.modules.update({
        "modules": modules_pkg,
        "modules.shared": types.ModuleType("modules.shared"),
        "modules.paths_internal": paths_internal,
        "requests": requests,
    })
    try:
        spec = importlib.util.spec_from_file_location("modules.util", ROOT / "modules" / "util.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["modules.util"] = module
        spec.loader.exec_module(module)
        yield types.SimpleNamespace(module=module, requests=requests)
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def test_download_has_connect_and_read_timeouts(util, tmp_path):
    util.requests.response = _Response([b"weights"])
    url = "https://example.invalid/releases/model_x4.pth"

    path = util.module.load_file_from_url(url, model_dir=str(tmp_path), progress=False)

    assert path == str(tmp_path / "model_x4.pth")
    assert Path(path).read_bytes() == b"weights"
    assert util.requests.calls == [(url, {"stream": True, "timeout": (10, 60)})]
    assert util.requests.response.closed


def test_stalled_download_fails_and_closes_the_response(util, tmp_path):
    util.requests.response = _Response([b"part"], error=TimeoutError("read timed out"))

    with pytest.raises(TimeoutError):
        util.module.load_file_from_url("https://example.invalid/m.pth", model_dir=str(tmp_path), progress=False)

    assert util.requests.response.closed
    assert not (tmp_path / "m.pth").exists()
