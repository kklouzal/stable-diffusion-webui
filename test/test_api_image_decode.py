"""API image input decoding: data URLs, URL fetches with api_forbid_local_requests (every redirect hop, IPv6, ports,
connections pinned to the checked address), and the byte/pixel budgets derived from img_max_size_mp."""

import ast
import base64
import contextvars
import functools
import gzip
import http.server
import io
import ipaddress
import socket
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
import urllib3.util.connection
from PIL import Image

API_PATH = Path(__file__).resolve().parents[1] / "modules" / "api" / "api.py"


class HTTPException(Exception):
    def __init__(self, status_code, detail):
        super().__init__(status_code, detail)
        self.status_code = status_code
        self.detail = detail


class UnsupportedImageError(ValueError):
    """Stands in for modules.images.UnsupportedImageError."""


def read_image(fp, *, max_pixels=None):
    """images.read's contract (test/test_images_boundary.py tests the real one): the pixel budget is checked from
    the header, before decoding."""
    image = Image.open(fp)
    if max_pixels is not None and image.width * image.height > max_pixels:
        raise Image.DecompressionBombError(f"{image.size} exceeds {max_pixels} pixels")
    image.load()
    return image


def load_api_functions(*, forbid_local=True, img_max_size_mp=200):
    tree = ast.parse(API_PATH.read_text(encoding="utf8"))
    wanted = {"verify_url", "_PinnedAddressAdapter", "_ImageUrlSession", "_get_image_url", "decode_inline_images_once", "decode_base64_to_image"}
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted]
    nodes += [node for node in tree.body if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "_request_inline_images"]
    namespace = {
        "contextvars": contextvars,
        "functools": functools,
        "base64": base64,
        "BytesIO": io.BytesIO,
        "Image": Image,
        "ipaddress": ipaddress,
        "requests": requests,
        "HTTPException": HTTPException,
        "opts": SimpleNamespace(api_enable_requests=True, api_forbid_local_requests=forbid_local, api_useragent="", img_max_size_mp=img_max_size_mp),
        "images": SimpleNamespace(read=read_image, UnsupportedImageError=UnsupportedImageError),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(API_PATH), "exec"), namespace)
    return SimpleNamespace(**{node.name: namespace[node.name] for node in nodes if hasattr(node, "name")}, namespace=namespace)


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


def test_images_without_an_srgb_mapping_answer_422():
    """images.read raises UnsupportedImageError for 32-bit/float modes and unusable ICC profiles: a client error."""
    api = load_api_functions()

    def read(fp, *, max_pixels=None):
        raise UnsupportedImageError("Unsupported image mode F: the range of its values is undefined")

    api.namespace["images"] = SimpleNamespace(read=read, UnsupportedImageError=UnsupportedImageError)
    with pytest.raises(HTTPException) as raised:
        api.decode_base64_to_image(png_base64())
    assert (raised.value.status_code, raised.value.detail) == (422, "Unsupported image mode F: the range of its values is undefined")

    api = fake_session(api, {"https://global.example/a.png": FakeResponse(content=base64.b64decode(png_base64()))}, [])
    api.namespace["verify_url"] = lambda url: ["93.184.216.34"]
    with pytest.raises(HTTPException) as raised:
        api.decode_base64_to_image("https://global.example/a.png")
    assert raised.value.status_code == 422


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
    assert api.verify_url("https://global.example/a.png") == ["93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"]
    assert api.verify_url("https://user:pw@global.example:8443/a.png")  # netloc lookup failed on port/userinfo
    assert calls[-1] == ("global.example", 8443)
    assert not api.verify_url("https:///no-host.png")
    assert not api.verify_url("https://global.example:notaport/a.png")


class FakeResponse:
    def __init__(self, content=b"", location=None):
        self.content = content
        self.location = location
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_content(self, chunk_size):
        yield self.content


class FakeSession:
    def __init__(self, routes, visited):
        self.routes = routes
        self.visited = visited

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def mount(self, prefix, adapter):
        pass

    def get(self, url, timeout, headers, allow_redirects, stream):
        assert allow_redirects is False and timeout == 30 and stream is True
        self.visited.append(url)
        response = self.routes[url]
        response.url = url
        return response

    def get_redirect_target(self, response):
        return response.location


def fake_session(api, routes, visited):
    api.namespace["_ImageUrlSession"] = lambda: FakeSession(routes, visited)
    return api


def test_redirect_to_a_local_resource_is_refused():
    visited = []
    routes = {"https://global.example/a.png": FakeResponse(location="http://169.254.169.254/latest/meta-data")}
    api = fake_session(load_api_functions(), routes, visited)
    api.namespace["verify_url"] = lambda url: [] if "169.254" in url else ["93.184.216.34"]

    with pytest.raises(HTTPException) as raised:
        api.decode_base64_to_image("https://global.example/a.png")

    assert raised.value.detail == "Request to local resource not allowed"
    assert visited == ["https://global.example/a.png"]  # the metadata endpoint was never requested


def test_global_redirects_are_followed_and_bounded():
    visited = []
    png = base64.b64decode(png_base64())
    routes = {"https://a.example/x": FakeResponse(location="https://b.example/y"), "https://b.example/y": FakeResponse(content=png)}
    api = fake_session(load_api_functions(), routes, visited)
    api.namespace["verify_url"] = lambda url: ["93.184.216.34"]

    assert api.decode_base64_to_image("https://a.example/x").size == (3, 2)
    assert visited == ["https://a.example/x", "https://b.example/y"]

    loop = {"https://loop.example/": FakeResponse(location="https://loop.example/")}
    api = fake_session(load_api_functions(), loop, [])
    api.namespace["verify_url"] = lambda url: ["93.184.216.34"]
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
        api.namespace["verify_url"] = lambda url: checked.append(url) or ["127.0.0.1"]

        assert api.decode_base64_to_image(base + "/start").size == (3, 2)
        assert checked == [base + "/start", base + "/img/final.png"]

        api = load_api_functions()
        api.namespace["verify_url"] = lambda url: ["127.0.0.1"] if url.endswith("/start") else []
        with pytest.raises(HTTPException) as raised:
            api.decode_base64_to_image(base + "/start")
        assert raised.value.detail == "Request to local resource not allowed"
    finally:
        server.shutdown()
        server.server_close()


# --- budgets and DNS rebinding (R3) ------------------------------------------------------------------------------------
# img_max_size_mp=0.001 is a 1000-pixel budget and therefore a 4000-byte download budget.

class Server:
    """A loopback HTTP server whose GET handler is `respond(handler, sent)`; `sent` counts the body bytes the socket
    accepted per path, until the client stopped reading. `hosts` lists the Host headers received."""

    def __init__(self, respond):
        sent = self.sent = {}
        hosts = self.hosts = []

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                hosts.append(self.headers["Host"])
                try:
                    respond(self, sent)
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def send_body(handler, sent, total, *, chunked=False):
    """Writes `total` zero bytes in 64 KiB pieces, counting what the socket accepted."""
    piece = bytes(65536)
    sent[handler.path] = 0
    while sent[handler.path] < total:
        n = min(len(piece), total - sent[handler.path])
        handler.wfile.write(b"%x\r\n%s\r\n" % (n, piece[:n]) if chunked else piece[:n])
        handler.wfile.flush()
        sent[handler.path] += n
    if chunked:
        handler.wfile.write(b"0\r\n\r\n")


def png_bytes(size):
    data = io.BytesIO()
    Image.new("RGB", size, (1, 2, 3)).save(data, format="PNG")
    return data.getvalue()


def serve_png(png):
    def respond(handler, sent):
        handler.send_response(200)
        handler.send_header("Content-Length", str(len(png)))
        handler.end_headers()
        handler.wfile.write(png)
    return respond


def test_chunked_body_over_the_byte_budget_is_refused_while_streaming():
    def respond(handler, sent):
        handler.send_response(200)
        handler.send_header("Transfer-Encoding", "chunked")
        handler.end_headers()
        send_body(handler, sent, 64 << 20, chunked=True)

    with Server(respond) as server:
        api = load_api_functions(forbid_local=False, img_max_size_mp=0.001)
        with pytest.raises(HTTPException) as raised:
            api.decode_base64_to_image(f"http://127.0.0.1:{server.port}/big")

    assert raised.value.status_code == 413
    assert server.sent["/big"] < 64 << 20  # the client stopped reading instead of buffering the whole body


def test_declared_length_over_the_byte_budget_is_refused_before_reading():
    def respond(handler, sent):
        handler.send_response(200)
        handler.send_header("Content-Length", str(10 ** 12))
        handler.end_headers()
        handler.wfile.write(b"x" * 10)
        handler.close_connection = True

    with Server(respond) as server:
        api = load_api_functions(forbid_local=False, img_max_size_mp=0.001)
        with pytest.raises(HTTPException) as raised:
            api.decode_base64_to_image(f"http://127.0.0.1:{server.port}/declared")

    assert raised.value.status_code == 413


def test_compressed_body_is_budgeted_by_its_decoded_length():
    """Content-Length is the compressed size (about 1 KiB) while the decoded body is 1 MiB."""
    compressed = gzip.compress(bytes(1 << 20))
    assert len(compressed) < 4000

    def respond(handler, sent):
        handler.send_response(200)
        handler.send_header("Content-Encoding", "gzip")
        handler.send_header("Content-Length", str(len(compressed)))
        handler.end_headers()
        handler.wfile.write(compressed)

    with Server(respond) as server:
        api = load_api_functions(forbid_local=False, img_max_size_mp=0.001)
        with pytest.raises(HTTPException) as raised:
            api.decode_base64_to_image(f"http://127.0.0.1:{server.port}/gzip")

    assert raised.value.status_code == 413


def test_redirect_body_is_not_read():
    """requests' redirect resolver read a redirect response's whole body, without a limit, even for a streamed
    request."""
    png = png_bytes((3, 2))

    def respond(handler, sent):
        if handler.path == "/start":
            handler.send_response(302)
            handler.send_header("Location", "/final.png")
            handler.send_header("Content-Length", str(64 << 20))
            handler.end_headers()
            send_body(handler, sent, 64 << 20)
        else:
            serve_png(png)(handler, sent)

    with Server(respond) as server:
        api = load_api_functions(forbid_local=False)
        assert api.decode_base64_to_image(f"http://127.0.0.1:{server.port}/start").size == (3, 2)

    assert server.sent["/start"] < 64 << 20


@pytest.mark.parametrize("size,refused", [((40, 25), False), ((40, 26), True)])
def test_images_over_the_pixel_budget_are_refused(size, refused):
    png = png_bytes(size)

    with Server(serve_png(png)) as server:
        api = load_api_functions(forbid_local=False, img_max_size_mp=0.001)
        for encoding in (base64.b64encode(png).decode(), f"http://127.0.0.1:{server.port}/a.png"):
            if refused:
                with pytest.raises(HTTPException) as raised:
                    api.decode_base64_to_image(encoding)
                assert raised.value.status_code == 413
            else:
                assert api.decode_base64_to_image(encoding).size == size


def fake_network(monkeypatch, dns, routes):
    """Stubs DNS and the connect step urllib3 uses. `dns` maps a host name to the addresses its successive lookups
    return (the last one repeats); `routes` maps an address to the loopback port it reaches; every other address
    refuses. Returns the list of addresses connected to, in order."""
    real_create_connection = urllib3.util.connection.create_connection
    lookups = {host: list(answers) for host, answers in dns.items()}
    connected = []

    def getaddrinfo(host, port, *args, **kwargs):
        try:
            address = str(ipaddress.ip_address(host))
        except ValueError:
            if host not in lookups:
                raise socket.gaierror(socket.EAI_NONAME, "not in the test's DNS") from None
            address = lookups[host].pop(0) if len(lookups[host]) > 1 else lookups[host][0]
        if ":" in address:
            return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", (address, port or 0, 0, 0))]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port or 0))]

    def create_connection(address, *args, **kwargs):
        host, port = address
        ip = getaddrinfo(host, port)[0][4][0]
        connected.append(ip)
        if ip in routes:
            return real_create_connection(("127.0.0.1", routes[ip]), *args, **kwargs)
        raise ConnectionRefusedError(f"test network: {ip} refused")

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(urllib3.util.connection, "create_connection", create_connection)
    return connected


def test_connection_goes_to_the_address_that_was_checked(monkeypatch):
    """DNS rebinding: the host resolves to a global address for the check and to a local one afterwards."""
    connected = fake_network(monkeypatch, {"rebind.example": ["93.184.216.34", "127.0.0.1"]}, routes={})
    api = load_api_functions()

    with pytest.raises(requests.exceptions.ConnectionError):
        api.decode_base64_to_image("http://rebind.example:8080/a.png")

    assert connected == ["93.184.216.34"]


def test_pinned_connection_tries_each_checked_address_and_names_the_url_host(monkeypatch):
    with Server(serve_png(png_bytes((3, 2)))) as server:
        # The first checked address refuses; the second reaches the loopback server.
        connected = fake_network(monkeypatch, {"img.example": ["93.184.216.34"]}, routes={"93.184.216.35": server.port})
        api = load_api_functions()
        api.namespace["verify_url"] = lambda url: ["93.184.216.34", "93.184.216.35"]

        assert api.decode_base64_to_image(f"http://Img.Example:{server.port}/a.png").size == (3, 2)

    assert connected == ["93.184.216.34", "93.184.216.35"]
    assert server.hosts == [f"img.example:{server.port}"]


def test_pinned_https_connection_verifies_the_certificate_for_the_url_host():
    api = load_api_functions()
    adapter = api._PinnedAddressAdapter()
    adapter.address = "2606:2800:220:1:248:1893:25c8:1946"
    request = requests.Request("GET", "https://img.example/a.png").prepare()

    host_params, pool_kwargs = adapter.build_connection_pool_key_attributes(request, True)

    assert host_params == {"scheme": "https", "host": "2606:2800:220:1:248:1893:25c8:1946", "port": None}
    assert pool_kwargs["server_hostname"] == "img.example" and pool_kwargs["cert_reqs"] == "CERT_REQUIRED"
    assert "assert_hostname" not in pool_kwargs  # ssl checks the certificate against the server name (check_hostname)


# --- request-scoped decode memo (decode_inline_images_once) -----------------------------------------------------------

def counting_api(calls, **kwargs):
    """API functions whose images.read is modules.images.read's contract (eager load, then fix_image: EXIF transpose,
    which returns a plain Image copy, and RGB/P bytes transparency -> RGBA), counting decodes."""
    from PIL import ImageOps

    api = load_api_functions(**kwargs)

    def read(fp, *, max_pixels=None):
        calls.append(max_pixels)
        image = ImageOps.exif_transpose(read_image(fp, max_pixels=max_pixels))
        if image.mode in ("RGB", "P") and isinstance(image.info.get("transparency"), bytes):
            image = image.convert("RGBA")
        return image

    api.namespace["images"] = SimpleNamespace(read=read, UnsupportedImageError=UnsupportedImageError)
    api.reference_read = read
    return api


def memo_test_images():
    import random
    rng = random.Random(7)
    rgb = Image.frombytes("RGB", (40, 24), rng.randbytes(40 * 24 * 3))
    palette = Image.frombytes("P", (16, 16), bytes(range(256)))
    palette.putpalette(bytes(range(256)) * 3)
    exif = Image.Exif()
    exif[0x0112] = 6
    out = {}
    for name, image, options in (("rgb", rgb, {}), ("rgba", rgb.convert("RGBA"), {}), ("l", rgb.convert("L"), {}),
                                 ("p", palette, {}), ("p-trns", palette, {"transparency": bytes([0, 255] * 128)}),
                                 ("rgb-key", rgb, {"transparency": (1, 2, 3)}), ("oriented", rgb, {"exif": exif.tobytes()})):
        data = io.BytesIO()
        image.save(data, format="PNG", **options)
        out[name] = base64.b64encode(data.getvalue()).decode()
    return out


def assert_same_image(actual, expected):
    assert type(actual) is type(expected)
    assert (actual.mode, actual.size, actual.info) == (expected.mode, expected.size, expected.info)
    assert actual.tobytes() == expected.tobytes()
    assert actual.getexif() == expected.getexif()
    assert (actual.getpalette() if actual.mode in ("P", "PA") else None) == (expected.getpalette() if expected.mode in ("P", "PA") else None)


def test_a_request_decodes_each_inline_image_once_into_independent_copies():
    calls = []
    api = counting_api(calls)
    for name, encoding in memo_test_images().items():
        for value in (encoding, "data:image/png;base64," + encoding):
            expected = api.reference_read(io.BytesIO(base64.b64decode(encoding)))  # oracle: a fresh decode
            calls.clear()

            @api.decode_inline_images_once
            def request(value=value):
                first, second = api.decode_base64_to_image(value), api.decode_base64_to_image(value)
                second.paste(9 if second.mode in ("P", "L") else (9,) * len(second.getbands()), (0, 0, 4, 4))
                second.info["parameters"] = "mutated"
                first.paste(7 if first.mode in ("P", "L") else (7,) * len(first.getbands()), (0, 0, 4, 4))
                return first, second, api.decode_base64_to_image(value)

            first, second, third = request()
            assert len(calls) == 1, name
            assert first is not third and second is not third
            assert_same_image(third, expected)  # callers' changes to earlier results never reach later ones


def test_the_memo_never_outlives_or_crosses_requests():
    calls = []
    api = counting_api(calls)
    encoding = png_base64()
    for _ in range(2):  # outside a request every call decodes
        api.decode_base64_to_image(encoding)
    assert len(calls) == 2

    seen = []
    request = api.decode_inline_images_once(lambda: seen.append(api.namespace["_request_inline_images"].get()) or api.decode_base64_to_image(encoding))
    request()
    request()
    assert len(calls) == 4  # a later request decodes again
    assert seen[0] is not seen[1] and api.namespace["_request_inline_images"].get() is None

    @api.decode_inline_images_once
    def request_with_thread():
        api.decode_base64_to_image(encoding)
        thread = threading.Thread(target=api.decode_base64_to_image, args=(encoding,))  # new thread: no request context
        thread.start()
        thread.join()
        api.decode_base64_to_image(encoding)

    request_with_thread()
    assert len(calls) == 6

    with pytest.raises(RuntimeError):
        api.decode_inline_images_once(lambda: (_ for _ in ()).throw(RuntimeError("request failed")))()
    assert api.namespace["_request_inline_images"].get() is None


def test_memoized_decodes_keep_the_current_pixel_budget():
    calls = []
    encoding = base64.b64encode(png_bytes((40, 25))).decode()
    api = counting_api(calls, img_max_size_mp=0.001)

    @api.decode_inline_images_once
    def request():
        assert api.decode_base64_to_image(encoding).size == (40, 25)
        assert api.decode_base64_to_image(encoding).size == (40, 25)
        api.namespace["opts"].img_max_size_mp = 0.0009
        with pytest.raises(HTTPException) as raised:
            api.decode_base64_to_image(encoding)
        assert raised.value.status_code == 413

    request()
    assert len(calls) == 2 and calls[0] == 1000 and calls[1] < 1000


def test_failures_urls_and_image_files_are_not_memoized():
    calls = []
    api = counting_api(calls, forbid_local=False)

    def file_read(fp, max_pixels=None):
        calls.append(max_pixels)
        return read_image(fp, max_pixels=max_pixels)

    @api.decode_inline_images_once
    def request(server_port):
        for _ in range(2):
            with pytest.raises(HTTPException):
                api.decode_base64_to_image("data:image/png;base64,AAAA")
        assert len(calls) == 2
        for _ in range(2):
            assert api.decode_base64_to_image(f"http://127.0.0.1:{server_port}/a.png").size == (3, 2)
        assert len(calls) == 4
        api.namespace["images"] = SimpleNamespace(read=file_read)
        for _ in range(2):
            assert api.decode_base64_to_image(png_base64()).format == "PNG"  # an ImageFile result stays as decoded
        assert len(calls) == 6

    with Server(serve_png(png_bytes((3, 2)))) as server:
        request(server.port)
