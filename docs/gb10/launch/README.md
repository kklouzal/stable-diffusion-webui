# Launch / Runtime Notes

## Canonical runtime baseline

Current canonical container defaults:

- image tag: `local/gb10-a1111:latest-mxfp8-dev`
- container name: `gb10-a1111-latest-mxfp8`
- host root: `/opt/gb10/stable-diffusion`
- port: `7860`
- network mode: `host`
- GPU flag: `--gpus all`
- IPC mode: `host`
- restart policy: `unless-stopped`

Current expected URL:

- `http://<GB10-LAN-IP>:7860`

## Helper scripts

### Build

```bash
./gb10/build.sh
```

Supported environment overrides:

- `DOCKERFILE`
- `BASE_IMAGE`
- `PYTORCH_NIGHTLY_CUDA_TAG`
- `IMAGE_TAG`
- `DOCKER_BUILDKIT`
- `BUILDKIT_PROGRESS`

### Run

```bash
./gb10/run.sh
```

Supported environment overrides:

- `IMAGE_TAG`
- `CONTAINER_NAME`
- `HOST_ROOT`
- `PORT`
- `COMMANDLINE_ARGS`

Default `COMMANDLINE_ARGS` baseline:

- `--listen --port 7860 --no-hashing --disable-console-progressbars --api --opt-sdp-attention --opt-channelslast --dtype bfloat16 --precision autocast --enable-insecure-extension-access`

### Smoke test

```bash
./gb10/smoke-test.sh
```

The smoke test is intentionally non-generative. It checks A1111 API health, model listing, CUDA/PyTorch visibility, and required runtime imports against the live container.

Supported environment overrides:

- `CONTAINER_NAME`
- `PORT`
- `DOCKER_BIN`
- `BASE_URL`

### Stop / remove

```bash
./gb10/stop.sh
```

## Container-owned launch path

The image does not use upstream `webui.sh` as the runtime authority.

Current runtime flow:

1. `docker/entrypoint.sh`
   - performs only minimal runtime prep
   - repairs missing/zero-byte JSON config files to valid `{}` defaults
   - ensures `/opt/stable-diffusion-webui/tmp/` exists
   - drops to user `a1111`
2. `docker/launch-a1111.sh`
   - runs A1111 directly with the image-owned Python environment
   - skips upstream environment preparation
   - skips upstream Python-version enforcement
   - relies on A1111's normal `COMMANDLINE_ARGS` parsing in `modules/paths_internal.py` instead of passing those flags a second time on `python launch.py`

`gb10/run.sh` is the canonical relaunch path for this appliance. It owns the default image/container names, persistent mounts, extension synchronization, and runtime `COMMANDLINE_ARGS`.

xformers is intentionally not part of the GB10 runtime path. The launcher uses PyTorch SDPA for attention.

Actual launch command shape inside the container:

```bash
COMMANDLINE_ARGS="--listen --port 7860 --no-hashing --disable-console-progressbars --api --opt-sdp-attention --opt-channelslast --dtype bfloat16 --precision autocast --enable-insecure-extension-access" \
python launch.py \
  --skip-prepare-environment \
  --skip-python-version-check
```

## Persistent mapping baseline

Current persistent host root:

- `/opt/gb10/stable-diffusion`

Current mappings:

- `/opt/gb10/stable-diffusion/BLIP` -> `/opt/stable-diffusion-webui/models/BLIP`
- `/opt/gb10/stable-diffusion/CLIP` -> `/opt/stable-diffusion-webui/models/CLIP`
- `/opt/gb10/stable-diffusion/Codeformer` -> `/opt/stable-diffusion-webui/models/Codeformer`
- `/opt/gb10/stable-diffusion/deepbooru` -> `/opt/stable-diffusion-webui/models/deepbooru`
- `/opt/gb10/stable-diffusion/GFPGAN` -> `/opt/stable-diffusion-webui/models/GFPGAN`
- `/opt/gb10/stable-diffusion/Hypernetworks` -> `/opt/stable-diffusion-webui/models/hypernetworks`
- `/opt/gb10/stable-diffusion/karlo` -> `/opt/stable-diffusion-webui/models/karlo`
- `/opt/gb10/stable-diffusion/Lora` -> `/opt/stable-diffusion-webui/models/Lora`
- `/opt/gb10/stable-diffusion/RealESGRAN` -> `/opt/stable-diffusion-webui/models/ESRGAN`
- `/opt/gb10/stable-diffusion/torch_deepdanbooru` -> `/opt/stable-diffusion-webui/models/torch_deepdanbooru`
- `/opt/gb10/stable-diffusion/VAE` -> `/opt/stable-diffusion-webui/models/VAE`
- `/opt/gb10/stable-diffusion/VAE-approx` -> `/opt/stable-diffusion-webui/models/VAE-approx`
- `/opt/gb10/stable-diffusion/Embeddings` -> `/opt/stable-diffusion-webui/embeddings`
- `/opt/gb10/stable-diffusion/Extensions` -> `/opt/stable-diffusion-webui/extensions`
- `/opt/gb10/stable-diffusion/Models` -> `/opt/stable-diffusion-webui/models/Stable-diffusion`
- `/opt/gb10/stable-diffusion/Outputs` -> `/opt/stable-diffusion-webui/outputs`
- `/opt/gb10/stable-diffusion/config/config.json` -> `/opt/stable-diffusion-webui/config.json`
- `/opt/gb10/stable-diffusion/config/ui-config.json` -> `/opt/stable-diffusion-webui/ui-config.json`
- `/opt/gb10/stable-diffusion/config/styles.csv` -> `/opt/stable-diffusion-webui/styles.csv`

This keeps user-owned runtime state on the host while leaving upstream code and baked-in companion repos image-owned, without a `/data` indirection layer.

## Current runtime observations

Observed working baseline behavior:

- A1111 binds to `0.0.0.0:7860`
- model loading succeeds
- UI loads in browser
- image generation succeeds

Observed non-blocking runtime caveats:

- no `xformers`
- some upstream warning noise in logs
- default auto-download behavior for baseline model/helper artifacts when absent

## Current baked upstream companion repos

The runtime image includes these pinned repos because upstream A1111 expects them to exist:

- `repositories/stable-diffusion-stability-ai`
- `repositories/generative-models`
- `repositories/k-diffusion`
- `repositories/BLIP`
- `repositories/stable-diffusion-webui-assets`

Sibling compatibility links are also provided for:

- `../generative-models`
- `../k-diffusion`
- `../BLIP`
