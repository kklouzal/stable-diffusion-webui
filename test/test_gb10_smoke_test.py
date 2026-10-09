"""gb10/smoke-test.sh, run for real against a stand-in API server and stand-in sudo/docker: the in-container check runs
as the app's UID/GID 2323, so what it compiles into the mounted compile cache stays writable by the app."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ["controlnet", "tiled diffusion"]
RESPONSES = {
    "/sdapi/v1/progress": {"progress": 0},
    "/sdapi/v1/sd-models": [],
    "/sdapi/v1/scripts": {"txt2img": EXPECTED, "img2img": []},
    "/sdapi/v1/openclaw/precision-map": {"ok": True, "checkpoint": {"title": "m"}, "summary": {"precision": "bf16"}},
    "/sdapi/v1/openclaw/vae-decode-graphs": {"enabled": True, "cache_size": 0},
}


class _Api(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps(RESPONSES[self.path.split("?")[0]]).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def _script(path: Path, body: str):
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def test_the_container_check_runs_as_the_app_user(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Api)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _script(bin_dir / "sudo", '#!/bin/sh\nexec "$@"\n')
    # Records its arguments and the script it is handed on stdin.
    _script(bin_dir / "docker", f'#!{sys.executable}\nimport json, sys\njson.dump([sys.argv[1:], sys.stdin.read()], open({str(tmp_path / "exec.json")!r}, "w"))\n')
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "DOCKER_BIN": str(bin_dir / "docker"),
           "BASE_URL": f"http://127.0.0.1:{server.server_address[1]}", "EXPECTED_SCRIPTS": ",".join(EXPECTED), "CONTAINER_NAME": "c"}
    try:
        result = subprocess.run(["bash", str(ROOT / "gb10" / "smoke-test.sh")], env=env, capture_output=True, text=True, timeout=60)
    finally:
        server.shutdown()

    assert result.returncode == 0, result.stderr
    assert "/sdapi/v1/scripts: ok (2 expected scripts loaded)" in result.stdout
    argv, script = json.loads((tmp_path / "exec.json").read_text(encoding="utf-8"))
    assert argv == ["exec", "-i", "-u", "2323:2323", "c", "python", "-"]
    assert "container dependencies: ok" in script
