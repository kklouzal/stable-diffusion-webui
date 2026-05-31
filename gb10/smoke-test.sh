#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-gb10-a1111-latest-mxfp8}"
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

required = ['triton', 'transformers', 'torchao', 'mslk']
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f'missing required modules: {", ".join(missing)}')

optional_absent = [name for name in ['xformers'] if importlib.util.find_spec(name) is None]
if optional_absent:
    print(f'optional absent: {", ".join(optional_absent)}')


from torchao.prototype.mx_formats.inference_workflow import MXDynamicActivationMXWeightConfig, NVFP4DynamicActivationNVFP4WeightConfig
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
from torchao.quantization import quantize_

if torch.cuda.is_available():
    layer = torch.nn.Linear(1024, 1024, bias=False).cuda().bfloat16().eval()
    quantize_(
        layer,
        config=MXDynamicActivationMXWeightConfig(
            activation_dtype=torch.float8_e4m3fn,
            weight_dtype=torch.float8_e4m3fn,
            kernel_preference=KernelPreference.AUTO,
        ),
        filter_fn=lambda mod, fqn: isinstance(mod, torch.nn.Linear),
    )
    sample = torch.randn(1, 1024, device='cuda', dtype=torch.bfloat16)
    out = layer(sample)
    torch.cuda.synchronize()
    if out.dtype != torch.bfloat16 or not torch.isfinite(out).all():
        raise SystemExit('MXFP8 TorchAO/MSLK smoke failed')
    print('mxfp8 torchao/mslk: ok')

    nvfp4_layer = torch.nn.Linear(1024, 1024, bias=False).cuda().bfloat16().eval()
    quantize_(
        nvfp4_layer,
        config=NVFP4DynamicActivationNVFP4WeightConfig(
            use_dynamic_per_tensor_scale=True,
            use_triton_kernel=True,
        ),
        filter_fn=lambda mod, fqn: isinstance(mod, torch.nn.Linear),
        device=torch.device('cuda'),
    )
    nvfp4_out = nvfp4_layer(sample)
    torch.cuda.synchronize()
    if nvfp4_out.dtype != torch.bfloat16 or not torch.isfinite(nvfp4_out).all():
        raise SystemExit('NVFP4 TorchAO/MSLK smoke failed')
    print('nvfp4 torchao/mslk: ok')

print('container imports: ok')
PY
