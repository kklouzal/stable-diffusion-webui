"""API image input decoding: data URLs, URL fetches with api_forbid_local_requests (every redirect hop, IPv6, ports)."""

import ast
import base64
import http.server
import io
import ipaddress
import socket
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from PIL import Image

API_PATH = Path(__file__).resolve().parents[1] / "modules" / "api" / "api.py"


class HTTPException(Exception):
    def __init__(self, status_code, detail):
        super().__init__(status_code, detail)
        self.status_code = status_code
        self.detail = detail


def load_api_functions(*, forbid_local=True, requests_module=requests):
    tree = ast.parse(API_PATH.read_text(encoding="utf8"))
    wanted = {"verify_url", "_get_image_url", "decode_base64_to_image"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert {f.name for f in functions} == wanted
    namespace = {
        "base64": base64,
        "BytesIO": io.BytesIO,
        "ipaddress": ipaddress,
        "requests": requests_module,
        "HTTPException": HTTPException,
        "opts": SimpleNamespace(api_enable_requests=True, api_forbid_local_requests=forbid_local, api_useragent=""),
        "images": SimpleNamespace(read=lambda fp: Image.open(fp)),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(API_PATH), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in wanted}, namespace=namespace)


def png_base64():
    data = io.BytesIO()
    Image.new("RGB", (3, 2), (1, 2, 3)).save(data, format="PNG")
    return base64.b64encode(data.getvalue()).decode()


@pytest.mark.parametrize("prefix", ["", "data:image/png;base64,", "data:image/png;name=init.png;base64,", "data:image/png;BASE64,"])
def test_base64_and_data_urls_decode(prefix):
    api = load_api_functions()

    image = api.decode_base64_to_image(prefix + png_base64())

    assert image.size == (3, 2) and image.convert("RGB").getpixel((0, 0)) == (1, 2, 3)


@pytest.mark.parametrize("value", ["data:image/png,not-base64", "data:image/png;base64", "data:image/png;base64,!!!!"])
def test_malformed_data_urls_are_invalid_encoded_images(value):
    api = load_api_functions()

    with pytest.raises(HTTPException) as raised:
        api.decode_base64_to_image(value)

    assert raised.value.detail == "Invalid encoded image"


def fake_getaddrinfo(table, calls):
    def getaddrinfo(host, port, *args, **kwargs):
        calls.append((host, port))
        return [(family, socket.SOCK_STREAM, 6, "", (address, port or 0) if family == socket.AF_INET else (address, port or 0, 0, 0)) for family, address in table[host]]
    return getaddrinfo


def test_verify_url_checks_ipv6_addresses_and_hostname(monkeypatch):
    api = load_api_functions()
    calls = []
    table = {
        "dual.example": [(socket.AF_INET, "93.184.216.34"), (socket.AF_INET6, "::1")],
        "global.example": [(socket.AF_INET, "93.184.216.34"), (socket.AF_INET6, "2606:2800:220:1:248:1893:25c8:1946")],
        "mapped.example": [(socket.AF_INET6, "::ffff:127.0.0.1")],
    }
    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo(table, calls))

    assert not api.verify_url("https://dual.example/a.png")  # gethostbyname_ex only saw the global IPv4 address
    assert not api.verify_url("https://mapped.example/a.png")
    assert api.verify_url("https://global.example/a.png")
    assert api.verify_url("https://user:pw@global.example:8443/a.png")  # netloc lookup failed on port/userinfo
    assert calls[-1] == ("global.example", 8443)
    assert not api.verify_url("https:///no-host.png")
    assert not api.verify_url("https://global.example:notaport/a.png")


class FakeResponse:
    def __init__(self, content=b"", location=None):
        self.content = content
        self.next = SimpleNamespace(url=location) if location else None


class FakeSession:
    def __init__(self, routes, visited):
        self.routes = routes
        self.visited = visited

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, timeout, headers, allow_redirects):
        assert allow_redirects is False and timeout == 30
        self.visited.append(url)
        return self.routes[url]


def fake_requests(routes, visited):
    return SimpleNamespace(Session=lambda: FakeSession(routes, visited), models=requests.models)


def test_redirect_to_a_local_resource_is_refused():
    visited = []
    routes = {"https://global.example/a.png": FakeResponse(location="http://169.254.169.254/latest/meta-data")}
    api = load_api_functions(requests_module=fake_requests(routes, visited))
    api.namespace["verify_url"] = lambda url: "169.254" not in url

    with pytest.raises(HTTPException) as raised:
        api.decode_base64_to_image("https://global.example/a.png")

    assert raised.value.detail == "Request to local resource not allowed"
    assert visited == ["https://global.example/a.png"]  # the metadata endpoint was never requested


def test_global_redirects_are_followed_and_bounded():
    visited = []
    png = base64.b64decode(png_base64())
    routes = {"https://a.example/x": FakeResponse(location="https://b.example/y"), "https://b.example/y": FakeResponse(content=png)}
    api = load_api_functions(requests_module=fake_requests(routes, visited))
    api.namespace["verify_url"] = lambda url: True

    assert api.decode_base64_to_image("https://a.example/x").size == (3, 2)
    assert visited == ["https://a.example/x", "https://b.example/y"]

    loop = {"https://loop.example/": FakeResponse(location="https://loop.example/")}
    api = load_api_functions(requests_module=fake_requests(loop, []))
    api.namespace["verify_url"] = lambda url: True
    with pytest.raises(HTTPException) as raised:
        api.decode_base64_to_image("https://loop.example/")
    assert raised.value.detail == "Invalid image url"


def test_real_requests_resolves_relative_redirects_hop_by_hop():
    """Against a loopback server with real requests: response.next.url resolves a relative Location."""
    png = base64.b64decode(png_base64())

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/start":
                self.send_response(302)
                self.send_header("Location", "../img/final.png")
                self.end_headers()
            else:
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(png)))
                self.end_headers()
                self.wfile.write(png)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        checked = []
        api = load_api_functions()
        api.namespace["verify_url"] = lambda url: checked.append(url) or True

        assert api.decode_base64_to_image(base + "/start").size == (3, 2)
        assert checked == [base + "/start", base + "/img/final.png"]

        api = load_api_functions()
        api.namespace["verify_url"] = lambda url: url.endswith("/start")
        with pytest.raises(HTTPException) as raised:
            api.decode_base64_to_image(base + "/start")
        assert raised.value.detail == "Request to local resource not allowed"
    finally:
        server.shutdown()
        server.server_close()
