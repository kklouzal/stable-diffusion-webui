# Launch / runtime operations

The scripts are the authority for everything below. This page lists what they read and do as of the cleanup pass of
2026-10-09.

## Container defaults

`gb10/run.sh` starts the container with these settings:

- image `local/gb10-a1111:latest` and container name `gb10-a1111-latest`
- host data root `/opt/gb10/stable-diffusion`
- `--network host`, so the API is at `http://<GB10-LAN-IP>:7860/sdapi/v1/...`
- `--gpus all`, `--ipc host` and `--init`
- `--cpuset-cpus 5-9,15-19`, the performance cores
- `--restart unless-stopped`

## Build: `gb10/build.sh`

```bash
gb10/build.sh
```

The script reads these environment overrides:

- `DOCKERFILE`
- `BASE_IMAGE`, default `nvcr.io/nvidia/pytorch:26.08-py3`. The `Dockerfile` has the same default.
- `MSLK_REPO` and `MSLK_COMMIT`
- `IMAGE_TAG`, default `local/gb10-a1111:latest`
- `BUILDKIT_PROGRESS`, default `plain`
- `CACHE_FROM`, default `IMAGE_TAG`. When that image exists, it is passed as `--cache-from`.
- `BUILD_CGROUP_PARENT`, default `gb10build.slice`, the systemd unit in `gb10/gb10build.slice`.
  - BuildKit steps run in that slice unless its CPUs are boot-isolated (`isolcpus=domain`).
  - If the slice is not active, the build fails. Set it empty to build with the default CPU placement.

BuildKit is always on: a `DOCKER_BUILDKIT` other than `1` is overridden with a warning. The build always runs with
`--pull` and runs through `sudo docker`.

## Deploy: `gb10/run.sh`

```bash
gb10/run.sh
IMAGE_TAG=local/gb10-a1111:<tag> gb10/run.sh   # deploy or roll back to another image
```

Environment overrides:

| Variable | Default | Effect |
|---|---|---|
| `IMAGE_TAG` | `local/gb10-a1111:latest` | image to run |
| `CONTAINER_NAME` | `gb10-a1111-latest` | container to replace |
| `HOST_ROOT` | `/opt/gb10/stable-diffusion` | host data root |
| `PORT` | `7860` | passed as `A1111_PORT` to the launcher |
| `OUTPUTS_TARGET` | `/mnt/nas-warehouse/StableDiffusion/Outputs` | target of the `${HOST_ROOT}/Outputs` symlink. run.sh creates the link and fails if it points elsewhere |
| `DOCKER_BIN` | `/usr/bin/docker` | docker binary |
| `CPUSET_CPUS` | `5-9,15-19` | `--cpuset-cpus` |
| `COMMANDLINE_ARGS` | empty | replaces the launcher's default flags (see [Launch flags](#launch-flags)) |
| `OPENCLAW_SDPA_BACKEND` | `cudnn,flash,efficient,math` | SDPA backend priority |
| `OPENCLAW_CUDA_GRAPHS` | `1` | UNet CUDA graphs |
| `OPENCLAW_CUDA_GRAPH_CACHE_MAX` | `8` | UNet graph cache size |
| `OPENCLAW_VAE_DECODE_GRAPHS` | `1` | VAE decode CUDA graphs |
| `OPENCLAW_VAE_DECODE_GRAPH_CACHE_MAX` | `4` | VAE decode graph cache size |
| `OPENCLAW_COMPILE_CACHE_ROOT` | `${HOST_ROOT}/Caches/compile` | host root of the Inductor, Triton and CUDA kernel caches |
| `OPENCLAW_COMPILE_CACHE_NAMESPACE` | image torch/Triton/CUDA versions + host driver version | cache namespace. App-only deploys reuse warm caches |
| `A1111_COMMIT_HASH` | `git rev-parse HEAD` of the checkout | reported commit |
| `A1111_VERSION_TAG` | `git describe --tags` of the checkout | the infotext `Version` (the image has no `.git`) |

Order of operations:

1. Create the host directories, the `Outputs` symlink and the `config/` files.
2. Discover the owned extensions under `extensions/`.
3. Run the checks that can fail without touching production:
   - resolve the compile-cache namespace (this reads the host driver version and runs the image once with
     `--network none`)
   - create the namespace directories and check that UID 2323 can write them
   - resolve the image ID
   - rehearse the third-party extension patchers on a scratch copy

   A failure here leaves the running container untouched.
4. Remove the old container.
5. Mirror each owned extension into `${HOST_ROOT}/Extensions/<name>` with `rsync --delete`. These subtrees survive the
   mirror (`P` filters):
   - `data/` (openclaw-multi-sampler's saved chains)
   - `annotator/downloads/` (ControlNet preprocessor weights)
   - `models/` (ControlNet models)
6. Patch the host-installed third-party extensions in place: `gb10/patch-*.py`, built on the `gb10/patchlib.py`
   contract. Each patcher accepts a pristine upstream file or an already-patched one, and fails on anything else.
7. `chown -R 2323:2323` the mounted host directories, then start the new container.

run.sh never deletes an owned extension that was removed from the checkout. Remove its host copy by hand.

`gb10/smoke-test.sh` (overrides `CONTAINER_NAME`, `PORT`, `DOCKER_BIN`, `BASE_URL`) checks a running container in two
parts:
- GET requests to `/sdapi/v1/progress`, `/sd-models`, `/openclaw/precision-map` and `/openclaw/vae-decode-graphs`
- a `docker exec` import check

The import check also runs one small TorchAO MXFP8 and one NVFP4 `Linear` on the GPU. The script generates no images.

`gb10/stop.sh` removes the container.

## Launch flags

Inside the image, the entrypoint drops to user `a1111` and runs the launcher (`docker/launch-a1111.sh`). The launcher
runs `python launch.py --skip-prepare-environment --skip-python-version-check`. Those two flags are accepted no-ops now
that the bootstrap is gone. A1111 itself appends `COMMANDLINE_ARGS` to its arguments (`modules/paths_internal.py`).

When `COMMANDLINE_ARGS` is empty (run.sh's default), the launcher uses:

```
--listen --port ${A1111_PORT} --no-hashing --disable-console-progressbars --api --nowebui --opt-sdp-attention --opt-channelslast --dtype bfloat16 --precision autocast --enable-insecure-extension-access
```

A non-empty `COMMANDLINE_ARGS` replaces this whole list. It must keep `--nowebui --api`: without `--nowebui`,
`webui.py` exits ("The browser UI has been removed ..."), and `--restart unless-stopped` then restarts the container in
a loop.

Keep `--opt-channelslast`. Dropping it (all-NCHW) measured 8.5-16% faster, but it hard-locked the host twice under
sustained load on 2026-10-07 ([performance pass 2](../notes/performance-pass-2-2026-10-07.md)).

## Host mounts

Every mount is a direct bind mount from `${HOST_ROOT}` (default `/opt/gb10/stable-diffusion`) into the app directory
`/opt/stable-diffusion-webui`. The mounts below come from run.sh's `HOST_DIR_MOUNTS` table and the extra mounts after
it.

| Host (`${HOST_ROOT}/...`) | Container (`/opt/stable-diffusion-webui/...`) |
|---|---|
| `BLIP` | `models/BLIP` |
| `CLIP` | `models/CLIP` |
| `Codeformer` | `models/Codeformer` |
| `GFPGAN` | `models/GFPGAN` |
| `Hypernetworks` | `models/hypernetworks` |
| `karlo` | `models/karlo` |
| `Lora` | `models/Lora` |
| `RealESGRAN` | `models/ESRGAN` |
| `torch_deepdanbooru` | `models/torch_deepdanbooru` |
| `VAE` | `models/VAE` |
| `VAE-approx` | `models/VAE-approx` |
| `Embeddings` | `embeddings` |
| `Extensions` | `extensions` |
| `Models` | `models/Stable-diffusion` |
| `Outputs` | `outputs` |
| `Caches/app` | `cache` (app caches; the image sets `TORCH_HOME=cache/torch` and `HF_HOME=cache/huggingface`) |
| `${OPENCLAW_COMPILE_CACHE_ROOT}` (`Caches/compile`) | `cache/compile` (nested inside the `cache` mount) |
| `config/config.json` | `config.json` |
| `config/styles.csv` | `styles.csv` |
| `config/generation-last` | `generation-last` (`GENERATION_LAST_DIR`, see [generation-last-api.md](../generation-last-api.md)) |

On the host, `Outputs`, `Embeddings`, `Hypernetworks` and `Lora` are symlinks into `/mnt/nas-warehouse/StableDiffusion/`.

Anything written outside these mounts is lost when the container is replaced. That includes `tmp/` and any weights
downloaded into a model directory that has no mount. torch.hub and Hugging Face downloads land under the `cache` mount.
