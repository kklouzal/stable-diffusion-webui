# STATUS.md

## Mission

Refactor the GB10-native AUTOMATIC1111 container from the older NVIDIA PyTorch-base approach to the newer NVIDIA CUDA-base + explicit PyTorch nightly `cu130` approach, while keeping the repo reviewable and the host-mounted user-data layout intact.

## Current status

**Full image build now succeeds on the GB10 with the CUDA-base refactor.**

The repo currently:

- still targets the same upstream A1111 source and persistent host-mounted runtime surfaces
- now defaults the Dockerfile/build flow to NVIDIA CUDA NGC + explicit PyTorch nightly `cu130`
- now freezes/protects the base Python package set after torch install so later app deps cannot overwrite or shadow it
- now keeps `wheelbuilder` on a Rust + OpenSSL-capable path that can build `tokenizers==0.13.3`
- now routes heavy builder compiles through `ccache` via BuildKit cache mounts
- still keeps upstream `webui.sh` out of authority for runtime bootstrap
- still keeps user-owned runtime surfaces under `/opt/gb10/stable-diffusion`
- now vendors `sd-webui-incantations` as GB10-owned first-class extension source for PAG, SEG, CFG-combiner, and CFG-Fix behavior
- has completed a real successful full-image build on the GB10 host
- now emits `/opt/stable-diffusion-webui/BUILD_MANIFEST.txt` and `.json` during image build, classifying installed Python packages into base-layer-provided vs A1111 direct vs A1111 indirect and annotating latest-visible-version drift reasons

## Current chosen defaults

- base image family: `nvcr.io/nvidia/cuda`
- base image tag: `nvcr.io/nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04`
- PyTorch nightly CUDA lane: `cu130`
- torch-extension arch policy in repo build path: `12.1a 12.1+PTX 12.0` (current best practical mapping of requested `12.1f > 12.1a > 12.0` for the PyTorch/xformers toolchain)
- upstream repo: `https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`
- upstream ref: `dev`
- host storage root: `/opt/gb10/stable-diffusion`
- default port: `7860`

## Persistent host surfaces

Default host root:

- `/opt/gb10/stable-diffusion`

Host-owned persistent surfaces:

- `config/`
- `Embeddings/`
- `Extensions/`
- `Models/`
- `Outputs/` (special host-side symlink to SERVER-002)
- `Models/` (special mixed local+symlink model root)

## Current owned extension posture

- `extensions/sd-webui-incantations` is vendored in this repository under GPL-3.0
- upstream provenance is preserved in the extension README and upstream README copy
- the image contains the owned extension source directly
- the repo run script syncs the owned extension into the host-mounted `Extensions/` surface so the bind mount does not hide the image copy
- future PAG/SEG/CFG-combiner/CFG-Fix fixes should be made in this repo, not in an untracked external extension checkout

## Current image/runtime doctrine

- upstream `webui.sh` is not the container authority
- runtime Python environment is image-owned
- NVIDIA CUDA base image owns the CUDA/runtime substrate
- PyTorch is installed explicitly from the selected nightly lane during build
- the resulting base Python package set is frozen/protected before later app dependency installs
- builder stage resolves and prebuilds the non-framework Python dependency closure
- runtime installs the curated non-framework runtime set with `--no-deps` under the protected constraints file
- upstream companion repos required by A1111 are baked into the image

## Current baked upstream companion repos

- `stable-diffusion-stability-ai`
- `generative-models`
- `k-diffusion`
- `BLIP`
- `stable-diffusion-webui-assets`

## Current explicit compatibility handling

- explicit `pytorch-lightning` / `torchmetrics` / `lightning-utilities` runtime cluster
- explicit OpenAI CLIP wheel build/install path
- config bootstrap repair for missing/zero-byte config files
- temporary builder-stage `RUSTFLAGS="-A invalid_reference_casting"` concession for `tokenizers 0.13.x`
- explicit `libssl-dev` + `pkg-config` support in `wheelbuilder`

## Latest build evidence

Successful full-image probe on GB10:

- image tag: `local/gb10-a1111:full-probe`
- image digest: `sha256:205e443219a72e9e8792ca31046638fd0dc88c16f570d4314f0835a7c3157d99`
- torch after runtime install: `2.13.0.dev20260422+cu130`
- torchvision after runtime install: `0.27.0.dev20260423+cu130`
- torchaudio after runtime install: `2.11.0.dev20260423+cu130`
- wheelbuilder also succeeded separately as `local/gb10-a1111:wheelbuilder-probe`
- follow-up arch-priority probe: `local/gb10-a1111:arch-priority-probe` (`xformers/_C.so` contains `sm_121a`, `sm_121`, and `sm_120` cubins plus `sm_121` PTX; torch/vision/audio remained on `+cu130`)
- package-manifest probe: `local/gb10-a1111:manifest-probe` (`BUILD_MANIFEST.json` summary: `base=35`, `direct=45`, `indirect=81`; torch/xformers import cleanly with CUDA available)

## Immediate next validation work

1. relaunch/validate the container against the existing host-mounted runtime surfaces
2. confirm the repo-visible run path still behaves correctly with the refactored image
3. decide whether any remaining docs cleanup beyond `README.md` / `STATUS.md` is worth doing
