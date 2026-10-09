# Stable Diffusion web UI: GB10 API fork

This is a fork of [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui),
branched from upstream `dev` at `1937682a` (2026-03-02). It runs only as an API server in a Docker image on an NVIDIA GB10
(aarch64, NVIDIA NGC PyTorch base). The HTTP API (`/sdapi/v1/*`, plus this fork's `/sdapi/v1/openclaw/*` routes) listens
on port 7860. There is no browser UI.

## Build

```bash
gb10/build.sh
```

This builds `local/gb10-a1111:latest` from this checkout with the [Dockerfile](Dockerfile), using `sudo docker build`
with BuildKit. The base image is `nvcr.io/nvidia/pytorch:26.08-py3`. BuildKit steps run in the `gb10build.slice`
systemd unit. If that unit is not installed, set `BUILD_CGROUP_PARENT=`.

The build context is an allowlist ([.dockerignore](.dockerignore)): only the runtime source goes into the image. Tests,
docs, `gb10/` and the owned extensions under `extensions/` do not.

## Deploy

```bash
gb10/run.sh                                                  # deploy local/gb10-a1111:latest
IMAGE_TAG=local/gb10-a1111:deploy8-2b5e4039 gb10/run.sh      # deploy (or roll back to) another tag
```

`run.sh` does all of its checks before it removes the running container, so a failed check leaves production running.
The checks cover the image, the host driver, the compile-cache namespace and a rehearsal of the extension patchers.
After the checks it:

- replaces the container `gb10-a1111-latest`
- bind-mounts the host data under `/opt/gb10/stable-diffusion`
- mirrors the owned extensions (`extensions/*`) into the host `Extensions/` directory
- patches the host-installed third-party extensions with `gb10/patch-*.py`
- starts the image's API-only launcher with host networking and `--gpus all`

`gb10/smoke-test.sh` checks a running container without generating anything. `gb10/stop.sh` removes the container.

## Test

The tests need the image's Python environment (torch and the companion repositories), but no GPU. Run them in a
throwaway container from the image:

1. Replace the app tree `/opt/stable-diffusion-webui` with this checkout. Keep the image's `repositories/` and
   `models/`.
2. Run `pip install pytest pytest-base-url`, because the image does not ship pytest.
3. Run the suites:

```bash
python -m pytest test                                       # core CPU suite
python -m pytest extensions/<name>/tests                    # each owned extension, in its own pytest process
python -m pytest test/live --base-url http://<host>:<port>  # live-server API tests
```

The live tests are skipped without `--base-url`. They POST generations and settings changes, so point them at a test
server, not production.

## Documentation

- [docs/gb10/README.md](docs/gb10/README.md): architecture, image layout and dependency policy
- [docs/gb10/launch/README.md](docs/gb10/launch/README.md): operations (script overrides, launch flags, host mounts)
- [docs/gb10/STATUS.md](docs/gb10/STATUS.md): what is deployed, rollback images and parked work
- [docs/gb10/EXTENSIONS.md](docs/gb10/EXTENSIONS.md): owned and third-party extensions
- [docs/gb10/notes/](docs/gb10/notes/README.md): dated records of audits, performance passes and cleanups

## Differences from upstream

Removed:

- **The browser UI.** Gradio and the front end are gone. `webui.py` exits unless it is started with `--nowebui`; the
  image launcher's default flags include `--nowebui --api`.
- **The upstream bootstrap.** That covers `prepare_environment`, `webui.sh`, `webui-user.*` and the `.bat` launchers.
  The image owns its Python environment. Bootstrap flags such as `--skip-prepare-environment` are still accepted and
  do nothing.
- **The LDSR upscaler.** Its `ldsr_*` options and `--ldsr-models-path` are still registered and do nothing.
- **Alt-Diffusion checkpoints**, **macOS/MPS and Ascend NPU support**, and **ngrok**.

Added:

- GB10 runtime work: CUDA graphs, TorchAO weight quantization, conditioning and artifact caches, and diagnostics routes.
- Owned extensions: vendored ControlNet, Incantations and TeaCache, plus the `openclaw-*` extensions.

## Credits

Licenses for borrowed code are in [html/licenses.html](html/licenses.html). The full license of this project is in
[LICENSE.txt](LICENSE.txt).

- Stable Diffusion - https://github.com/Stability-AI/stablediffusion, https://github.com/CompVis/taming-transformers, https://github.com/mcmonkey4eva/sd3-ref
- k-diffusion - https://github.com/crowsonkb/k-diffusion.git
- Spandrel - https://github.com/chaiNNer-org/spandrel implementing
  - GFPGAN - https://github.com/TencentARC/GFPGAN.git
  - CodeFormer - https://github.com/sczhou/CodeFormer
  - ESRGAN - https://github.com/xinntao/ESRGAN
  - SwinIR - https://github.com/JingyunLiang/SwinIR
  - Swin2SR - https://github.com/mv-lab/swin2sr
- MiDaS - https://github.com/isl-org/MiDaS
- Ideas for optimizations - https://github.com/basujindal/stable-diffusion
- Cross Attention layer optimization - Doggettx - https://github.com/Doggettx/stable-diffusion, original idea for prompt editing.
- Cross Attention layer optimization - InvokeAI, lstein - https://github.com/invoke-ai/InvokeAI (originally http://github.com/lstein/stable-diffusion)
- Sub-quadratic Cross Attention layer optimization - Alex Birch (https://github.com/Birch-san/diffusers/pull/1), Amin Rezaei (https://github.com/AminRezaei0x443/memory-efficient-attention)
- Textual Inversion - Rinon Gal - https://github.com/rinongal/textual_inversion (we're not using his code, but we are using his ideas).
- Idea for SD upscale - https://github.com/jquesnelle/txt2imghd
- Noise generation for outpainting mk2 - https://github.com/parlance-zz/g-diffuser-bot
- CLIP interrogator idea and borrowing some code - https://github.com/pharmapsychotic/clip-interrogator
- Idea for Composable Diffusion - https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch
- DeepDanbooru - interrogator for anime diffusers https://github.com/KichangKim/DeepDanbooru
- Sampling in float32 precision from a float16 UNet - marunine for the idea, Birch-san for the example Diffusers implementation (https://github.com/Birch-san/diffusers-play/tree/92feee6)
- Instruct pix2pix - Tim Brooks (star), Aleksander Holynski (star), Alexei A. Efros (no star) - https://github.com/timothybrooks/instruct-pix2pix
- Security advice - RyotaK
- UniPC sampler - Wenliang Zhao - https://github.com/wl-zhao/UniPC
- TAESD - Ollin Boer Bohan - https://github.com/madebyollin/taesd
- LyCORIS - KohakuBlueleaf
- Restart sampling - lambertae - https://github.com/Newbeeer/diffusion_restart_sampling
- Hypertile - tfernd - https://github.com/tfernd/HyperTile
- Initial Gradio script - posted on 4chan by an Anonymous user. Thank you Anonymous user.
- (You)
