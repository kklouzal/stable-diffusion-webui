#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-gb10-a1111-latest}"
PORT="${PORT:-7860}"
DOCKER_BIN="${DOCKER_BIN:-/usr/bin/docker}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}}"

python3 - "${BASE_URL}" <<'PY'
import json
import sys
import urllib.request

base_url = sys.argv[1].rstrip('/')

for path in ('/sdapi/v1/progress?skip_current_image=true', '/sdapi/v1/sd-models'):
    url = base_url + path
    with urllib.request.urlopen(url, timeout=10) as response:
        body = response.read()
        content_type = response.headers.get('content-type', '')
        if response.status != 200:
            raise SystemExit(f'{path}: HTTP {response.status}')
        if 'application/json' not in content_type:
            raise SystemExit(f'{path}: expected JSON, got {content_type!r}')
        payload = json.loads(body)
    if path.startswith('/sdapi/v1/sd-models'):
        if not isinstance(payload, list):
            raise SystemExit(f'{path}: expected model list')
        print(f'{path}: ok ({len(payload)} models)')
    else:
        print(f'{path}: ok (progress={payload.get("progress")})')
PY

sudo "${DOCKER_BIN}" exec -i "${CONTAINER_NAME}" python - <<'PY'
import importlib.util
import sys

import torch

print(f'python: {sys.version.split()[0]}')
print(f'torch: {torch.__version__} cuda={torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'cuda device: {torch.cuda.get_device_name(0)}')

required = ['sageattention', 'triton', 'gradio', 'transformers']
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f'missing required modules: {", ".join(missing)}')

optional_absent = [name for name in ['xformers'] if importlib.util.find_spec(name) is None]
if optional_absent:
    print(f'optional absent: {", ".join(optional_absent)}')

print('container imports: ok')
PY
