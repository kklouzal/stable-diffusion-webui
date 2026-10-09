import asyncio
import gzip
import json
import os

import pytest
from fastapi import FastAPI
from starlette.middleware.gzip import GZipMiddleware

from modules import initialize_util

os.environ.setdefault("IGNORE_CMD_ARGS_ERRORS", "1")  # setup_middleware reads cmd_opts for CORS

BODY = json.dumps({"images": [os.urandom(3000).hex()]}).encode()


async def _json_app(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(BODY)).encode())]})
    await send({"type": "http.response.body", "body": BODY})


def _request(middleware_cls, client, accept_encoding="gzip, deflate"):
    middleware = middleware_cls(_json_app, minimum_size=1000)
    scope = {"type": "http", "method": "POST", "path": "/sdapi/v1/txt2img", "headers": [(b"accept-encoding", accept_encoding.encode())], "client": client}
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    asyncio.run(middleware(scope, receive, send))
    start, body = messages
    headers = {k.decode().lower(): v.decode() for k, v in start["headers"]}
    return headers, body["body"]


@pytest.mark.parametrize("client", [("127.0.0.1", 50000), ("127.8.9.10", 1), ("::1", 50000)])
def test_loopback_clients_get_the_identity_response_starlette_sends_without_gzip(client):
    headers, body = _request(initialize_util.LoopbackIdentityGZipMiddleware, client)
    expected_headers, expected_body = _request(GZipMiddleware, client, accept_encoding="identity")

    assert "content-encoding" not in headers
    assert body == BODY
    assert (headers, body) == (expected_headers, expected_body)


@pytest.mark.parametrize("client", [("192.168.1.20", 50000), ("10.0.0.5", 1), ("fe80::1", 1), None, ("testclient", 50000)])
def test_non_loopback_clients_keep_the_stock_gzip_response(client):
    headers, body = _request(initialize_util.LoopbackIdentityGZipMiddleware, client)
    expected_headers, expected_body = _request(GZipMiddleware, client)

    assert headers["content-encoding"] == "gzip"
    assert gzip.decompress(body) == BODY
    assert headers == expected_headers
    assert gzip.decompress(expected_body) == BODY


def test_setup_middleware_installs_the_loopback_aware_gzip_layer():
    app = FastAPI()

    initialize_util.setup_middleware(app)

    gzip_layers = [m for m in app.user_middleware if issubclass(m.cls, GZipMiddleware)]
    assert [(m.cls, m.kwargs) for m in gzip_layers] == [(initialize_util.LoopbackIdentityGZipMiddleware, {"minimum_size": 1000})]
