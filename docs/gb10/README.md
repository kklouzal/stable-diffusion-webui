# GB10 A1111 fork runtime

GB10-native **AUTOMATIC1111 Stable Diffusion Web UI** container project for Schwi's NVIDIA GB10 host.

## Current direction

Current repo defaults target:

- builds on the GB10 host itself
- uses the official **NVIDIA NGC PyTorch** image as the base (CUDA, cuDNN, TensorRT, and NVIDIA's PyTorch/torchvision/Triton builds)
- builds MSLK from source against the inherited CUDA/PyTorch stack
- builds torch-extension CUDA code with `12.1a` as the practical GB10 Blackwell target; `12.1f` is not directly accepted by the current PyTorch extension build path and should not be treated as the preferred repo target
- protects the inherited base Python package set so later app dependency installs cannot overwrite or shadow it, except a reviewed list of stock PyPI wheels that the A1111 resolver may move forward (see "Python dependency posture")
- uses this fork checkout, which is upstream-derived from **AUTOMATIC1111/stable-diffusion-webui**
- uses a **multi-stage Dockerfile**
- keeps the runtime Python environment **image-owned**
- persists user-owned models, outputs, configs, embeddings, and extensions on the host
- vendors the GB10-owned `sd-webui-incantations` guidance extension source for PAG, SEG, CFG-combiner, and CFG-Fix behavior
- keeps upstream `webui.sh` out of authority for runtime bootstrap
- emits a build-time package manifest that inventories base-layer, direct, and indirect Python packages and compares each against the latest visible version with source/reason tags

Default URL posture:

- `http://<GB10-LAN-IP>:7860`

## What this project is

This repo is a reproducible, reviewable A1111 appliance build for the GB10.

It is intentionally **not** a wrapper around a random third-party community image, and it is intentionally **not** a normal upstream local-install flow.

The purpose is to keep the system understandable:

- base CUDA/runtime stack and the PyTorch framework builds come from NVIDIA's NGC PyTorch image
- A1111 application code comes from this fork checkout, with upstream AUTOMATIC1111 provenance
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

- `COMMANDLINE_ARGS="--listen --port 7860 --no-hashing --disable-console-progressbars --api --opt-sdp-attention --opt-channelslast --dtype bfloat16 --precision autocast --enable-insecure-extension-access" python launch.py --skip-prepare-environment --skip-python-version-check`

## Base image and upstream target

Current defaults:

- base image: `nvcr.io/nvidia/pytorch:26.08-py3` (CUDA 13.4.1, cuDNN 9.25, NVIDIA PyTorch `2.14.0a0+4fdf77b` built for CUDA 13.4, Triton 3.8.0). NGC publishes monthly `YY.MM-py3` tags; `gb10/build.sh --pull` refreshes the tag, and moving to a newer month means changing `BASE_IMAGE` in `Dockerfile` and `gb10/build.sh`.
- PyTorch nightly CUDA tag (`cu134`): used only by the build manifest's latest-version lookups. The image runs NVIDIA's torch, not an upstream nightly wheel; replacing it would also require rebuilding the NVIDIA packages compiled against it (torchvision, TransformerEngine, flash-attn, apex, torchao, MSLK).
- host driver: the GB10 R580 driver reports CUDA 13.0; NGC's bundled forward-compatibility driver libraries (`/usr/local/cuda/compat`) run the CUDA 13.4 user space on it
- A1111 source: this fork checkout (`https://github.com/kklouzal/stable-diffusion-webui.git`), upstream-derived from AUTOMATIC1111
- companion Stable Diffusion repo: `https://github.com/w-e-w/stablediffusion.git` at the Dockerfile-pinned ref
- image tag: `local/gb10-a1111:latest`
- container name: `gb10-a1111-latest`

## Owned A1111 extensions

The repo vendors `extensions/sd-webui-incantations` as first-class GB10 source. This replaces dependence on abandoned external checkouts for PAG, SEG, CFG-combiner, and Dynamic Thresholding / CFG-Fix behavior.

The repo also vendors `extensions/sd-webui-teacache` as a first-class GB10-maintained derivative of `feffy380/sd-webui-teacache` for SDXL TeaCache experimentation. It remains disabled by default until runtime benchmarking validates thresholds and quality impact.

The normal run path bind-mounts the host `Extensions/` directory over A1111's extension directory, so `gb10/run.sh` syncs the repo-owned extensions into `${HOST_ROOT}/Extensions/` before starting the container. The image does not need duplicate baked extension copies.

Treat these extensions as owned code: preserve GPL-3.0 provenance for Incantations and MIT provenance for TeaCache, keep changes reviewable here, and patch generation math/cache behavior conservatively because it materially affects image quality.

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

The NGC PyTorch base image owns the core framework layer. This repo intentionally avoids handing framework ownership back to later dependency resolution.

Current policy:

- inherit `torch` / `torchvision` / Triton and the rest of the NVIDIA stack from NGC; `torchaudio` is optional-absent
- record the pristine NGC package set as the first step of the base stage. The snapshot fails if any later build step removed or changed an inherited package. MSLK is the only intentional rebuild, and its wheel install uses `--no-deps` so it cannot pull a newer numpy.
- snapshot the effective base Python package set (`docker/snapshot-base-packages.py`) and pin every package at its exact NGC version, except the released list below
- resolve the A1111 dependency closure once in the builder stage against resolver stubs of the protected packages, then prebuild wheels in that throwaway stage
- install the resolved application set with `--no-deps`, then fail the build if any protected package changed (`docker/check-protected-stack.py`)

### Released NGC packages

`docker/base-released-packages.txt` lists NGC-inherited packages that the A1111 resolver owns instead of NGC. They are resolved to the newest release that satisfies A1111, the protected pins, and every installed package's declared requirements, and never below the NGC version. For example, `huggingface-hub` stays below 2.0 because Transformers requires it, and `protobuf` stays below 7 because NVIDIA's cutlass-dsl requires it.

The build enforces the release rule for every listed package:

- the NGC copy is a stock PyPI wheel (pip-installed from an index, with no local version label and no direct URL). NVIDIA did not build or tune it.
- it is outside the declared requirement closure of the NVIDIA-built runtime stack A1111 imports (`torch`, `torchvision`, `triton`, `torchao`, `mslk`). That keeps `numpy`, `pillow`, `sympy`, `networkx`, `filelock`, `fsspec`, `jinja2`, `setuptools`, and `typing-extensions` protected.
- it has no CUDA/NVIDIA package prefix

The list holds the stock wheels in the A1111 runtime dependency closure. dpkg-owned copies (`PyYAML`, `Pygments`) are not pip-replaceable and stay protected. `setuptools` must stay below 82 anyway, because ControlNet's OneFormer annotator imports `pkg_resources`.

Mechanics: resolver stubs of protected packages declare only their requirements on released packages. The dry-run requests every stub that declares one, with the protected pins and the NGC floors as constraints. After installation, `check-protected-stack.py --released-floors` verifies the floors and all declared requirements on released packages. `PROTECTED_PACKAGES.json` and the build manifest (`Released-From-NGC:<floor>`) record the result.

### Current curated additions worth knowing about

Some upstream runtime expectations are handled explicitly here because they are required in this containerized baseline/refactor path.

#### Lightning runtime cluster

Listed unpinned in `requirements_versions.txt` (resolved 2026-09-25: `pytorch-lightning 2.6.6`, `torchmetrics 1.9.0`, `lightning-utilities 0.15.3`):

- `pytorch_lightning`
- `torchmetrics`
- `lightning-utilities`

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

Minimum validated floors (enforced by the build): `transformers 5.7.0`, `tokenizers 0.22.2`, `huggingface-hub 1.13.0`. Resolved 2026-09-25: `transformers 5.17.0`, `tokenizers 0.23.2`, `huggingface-hub 1.33.0`. `tokenizers` and `huggingface-hub` are released NGC packages, so they follow the resolver rather than the NGC versions.

Current policy:

- keep `tokenizers` direct and unpinned in `requirements_versions.txt` so the resolver selects the newest version allowed by current Transformers metadata
- fail the build if Transformers resolves below `5.7.0`
- fail the build if tokenizers resolves below `0.22.2`
- fail the build if tokenizers resolves to an sdist/source artifact instead of a prebuilt wheel
- fail the build if Hugging Face Hub resolves below `1.13.0`

This removes the old builder-stage `RUSTFLAGS="-A invalid_reference_casting"` workaround. If tokenizers ever lacks a compatible aarch64 wheel again, the Docker build should fail clearly instead of silently falling back into a Rust compatibility lane.
