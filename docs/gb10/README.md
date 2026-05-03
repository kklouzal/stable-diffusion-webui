# GB10 A1111 fork runtime

GB10-native **AUTOMATIC1111 Stable Diffusion Web UI** container project for Schwi's NVIDIA GB10 host.

## Current direction

This repository is being refactored in place from the older NVIDIA NGC PyTorch-base path to the newer CUDA-base doctrine.

Current repo defaults now target:

- builds on the GB10 host itself
- uses official **NVIDIA NGC CUDA** as the base image family
- installs **PyTorch nightly** explicitly from the `cu132` wheel lane
- builds torch-extension CUDA code with `12.1a` as the practical GB10 Blackwell target; `12.1f` is not directly accepted by the current PyTorch extension build path and should not be treated as the preferred repo target
- freezes/protects the resulting base Python package set so later app dependency installs cannot overwrite or shadow it
- uses official upstream **AUTOMATIC1111/stable-diffusion-webui** source
- uses a **multi-stage Dockerfile**
- keeps the runtime Python environment **image-owned**
- persists user-owned models, outputs, configs, embeddings, and extensions on the host
- vendors the GB10-owned `sd-webui-incantations` guidance extension source for PAG, SEG, CFG-combiner, and CFG-Fix behavior
- keeps upstream `webui.sh` out of authority for runtime bootstrap
- emits a build-time package manifest that inventories base-layer, direct, and indirect Python packages and compares each against the latest visible version with source/reason tags

The older PyTorch-base baseline was previously validated. This new CUDA-base refactor is the current active direction and should be treated as the canonical repo intent.

Default URL posture:

- `http://<GB10-LAN-IP>:7860`

## What this project is

This repo is a reproducible, reviewable A1111 appliance build for the GB10.

It is intentionally **not** a wrapper around a random third-party community image, and it is intentionally **not** a normal upstream local-install flow.

The purpose is to keep the system understandable:

- base CUDA/runtime stack comes from NVIDIA
- PyTorch is installed explicitly and deliberately in repo-controlled build steps
- A1111 application code comes from upstream
- large user data surfaces live on the host
- launch/bootstrap behavior is owned here in repo-visible files
- quality-critical extension behavior such as PAG/SEG/CFG-combiner/CFG-Fix is owned here in repo-visible files

## Container posture

### Upstream `webui.sh` is not authoritative here

This project does **not** use upstream `webui.sh` as the container authority.

The container owns launch/bootstrap behavior directly so that:

- startup does not create or manage a separate runtime venv
- startup does not reinstall or replace the protected CUDA/PyTorch stack
- dependency behavior is reviewable at build time instead of hidden in runtime installer logic

Canonical launch path inside the image:

- entrypoint: `docker/entrypoint.sh`
- launcher: `docker/launch-a1111.sh`

That launch path runs:

- `python launch.py --skip-prepare-environment --skip-python-version-check --listen --port 7860 --no-hashing --disable-console-progressbars --api --opt-sdp-attention --opt-channelslast --enable-insecure-extension-access`

## Base image and upstream target

Current defaults:

- base image: `nvcr.io/nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04`
- PyTorch nightly lane: `https://download.pytorch.org/whl/nightly/cu132`
- upstream repo: `https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`
- upstream ref: `dev`
- image tag: `local/gb10-a1111:base-protected-app-latest`
- container name: `gb10-a1111-latest`

## Owned A1111 extensions

The repo vendors `extensions/sd-webui-incantations` as first-class GB10 source. This replaces dependence on abandoned external checkouts for PAG, SEG, CFG-combiner, and Dynamic Thresholding / CFG-Fix behavior.

The Docker image bakes this extension into `/opt/stable-diffusion-webui/extensions/sd-webui-incantations`. Because the normal run path bind-mounts the host `Extensions/` directory over A1111's extension directory, `gb10/run.sh` also syncs the repo-owned extension into `${HOST_ROOT}/Extensions/sd-webui-incantations` before starting the container.

Treat this extension as owned code: preserve GPL-3.0 provenance, keep changes reviewable here, and patch it conservatively because guidance math and hook cleanup materially affect generated image quality.

Mounted external extensions are tracked separately in `docs/gb10/EXTENSIONS.md`. Now that A1111-Controller is canonical for Schwi's frontend/workflow direction, UI-only external extensions should be purged or migrated into Controller-owned data rather than adopted as A1111 extensions. Generation-affecting extensions that remain should become first-class repo-owned source or be replaced by source-level modernization.

## Persistent storage layout

Default host root:

- `/opt/gb10/stable-diffusion`

Current intended GB10-local persistent directories:

- `BLIP/`
- `CLIP/`
- `Codeformer/`
- `deepbooru/`
- `GFPGAN/`
- `Hypernetworks/`
- `karlo/`
- `Lora/`
- `RealESGRAN/`
- `torch_deepdanbooru/`
- `VAE/`
- `VAE-approx/`
- `Embeddings/`
- `Extensions/`
- `config/`

Current special host-side paths:

- `Outputs/` -> direct host-side symlink to `SERVER-002/StableDiffusion/Outputs`
- `Models/ook/` -> local GB10 directory for directly stored copied models
- `Models/SDXL` -> direct host-side symlink to `/mnt/nas-warehouse/StableDiffusion/models/sdxl`

Older lowercase transition-era paths (for example `models/`, `outputs/`, `embeddings/`, `extensions/`) should be treated as stale migration residue rather than part of the intended final layout.

Current runtime binds mount these A1111 surfaces **directly** onto persistent host storage with no `/data` indirection and no container-side symlink remap layer:

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

## Config bootstrap behavior

The entrypoint is now intentionally minimal.

Current policy:

- create `/opt/stable-diffusion-webui/tmp/`
- create or repair `config.json` as valid `{}` JSON when absent or zero-byte
- create or repair `ui-config.json` as valid `{}` JSON when absent or zero-byte
- allow `styles.csv` to exist as an empty file

This avoids the bad upstream path where zero-byte placeholder config files are treated as corrupted JSON, while leaving persistence itself to direct Docker bind mounts rather than container-side path rewriting.

## Entrenched upstream repository layout

This image intentionally bakes in the upstream companion repositories that current A1111 expects.

Pinned repos baked into the image:

- `repositories/stable-diffusion-stability-ai`
- `repositories/generative-models`
- `repositories/k-diffusion`
- `repositories/BLIP`
- `repositories/stable-diffusion-webui-assets`

The image also provides sibling-path compatibility for:

- `../generative-models`
- `../k-diffusion`
- `../BLIP`

This is deliberate. For this dedicated container, matching upstream filesystem expectations is the correct posture.

## Python dependency posture

### Framework ownership

The CUDA base image plus the explicit PyTorch nightly install own the core framework layer.

This repo intentionally avoids handing framework ownership back to later dependency resolution.

Current policy:

- install `torch` / `torchvision` / `torchaudio` explicitly from the chosen nightly lane
- freeze/protect the resulting base Python package set after torch install
- resolve the non-framework Python dependency closure in the builder stage
- filter out packages already protected in the base layer
- prebuild wheels in the throwaway builder stage
- install the resulting non-framework runtime set with `--no-deps` under the protected constraints file

### Current curated additions worth knowing about

Some upstream runtime expectations are handled explicitly here because they are required in this containerized baseline/refactor path.

#### Lightning runtime cluster

Explicitly included:

- `pytorch_lightning==2.6.1`
- `torchmetrics==1.9.0`
- `lightning-utilities==0.15.3`

Reason:

- A1111 imports `pytorch_lightning` during startup
- the protected CUDA/torch base layer does not provide it by itself

#### OpenAI CLIP module

This stack includes both:

- `open-clip-torch`
- the original OpenAI `clip` Python module that `k-diffusion` still imports directly

The OpenAI CLIP package is built from the upstream-pinned source archive:

- `https://github.com/openai/CLIP/archive/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6.zip`

Because that package has old packaging behavior, it is built separately with `--no-build-isolation`, verified as a `clip-*.whl` artifact, and then installed by explicit wheel path.

#### Tokenizers resolver guard

This project intentionally follows the latest compatible Transformers/tokenizers lane instead of keeping the old A1111-era `transformers==4.30.2` / `tokenizers==0.13.3` pair.

Current validated runtime baseline:

- `transformers==5.7.0`
- `tokenizers==0.22.2`
- `huggingface-hub==1.13.0`

Current policy:

- keep `tokenizers` direct and unpinned in `requirements_versions.txt` so the resolver selects the newest version allowed by current Transformers metadata
- fail the build if Transformers resolves below `5.7.0`
- fail the build if tokenizers resolves below `0.22.2`
- fail the build if tokenizers resolves to an sdist/source artifact instead of a prebuilt wheel
- fail the build if Hugging Face Hub resolves below `1.13.0`

This removes the old builder-stage `RUSTFLAGS="-A invalid_reference_casting"` workaround. If tokenizers ever lacks a compatible aarch64 wheel again, the Docker build should fail clearly instead of silently falling back into a Rust compatibility lane.
