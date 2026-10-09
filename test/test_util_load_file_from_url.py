"""modules.util.load_file_from_url: model downloads are bounded by a (connect, read) timeout, close their response, and
are saved under the URL's basename unless a file name is given."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from modules import persistent_artifact_cache
from test.helpers import ROOT, load_source, module, stub_modules


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
    requests = module("requests", calls=[])

    def get(url, **kwargs):
        requests.calls.append((url, kwargs))
        return requests.response

    requests.get = get
    # load_file_from_url imports requests when called: the stubs stay installed for the test.
    with stub_modules({
        "modules": module("modules", package=True),
        "modules.shared": module("modules.shared"),
        "modules.paths_internal": module("modules.paths_internal", cwd=str(ROOT)),
        "modules.persistent_artifact_cache": persistent_artifact_cache,
        "requests": requests,
    }):
        yield SimpleNamespace(module=load_source("modules.util", "modules/util.py"), requests=requests)


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
