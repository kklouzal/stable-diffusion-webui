# GB10 historical dependency baseline

This file records the older pre-MXFP8 GB10 A1111 runtime baseline. It remains useful as a package/runtime hygiene reference, but the current operator state lives in `docs/gb10/STATUS.md` and `docs/gb10/launch/README.md`.

Use `docs/gb10/enhanced_requirements.md` for historical widening experiments and `BUILD_MANIFEST.txt` in the live image for the full package inventory.

## Validation snapshot

Historical validated runtime:

- repo branch: `latest`
- image tag: `local/gb10-a1111:base-protected-app-latest`
- image ID: `sha256:85c902073586364c4406d36604050c6ab7ccc531b1e7fa4449d26089d923b3c8`
- image created: `2026-05-03T15:47:10.304319823-07:00`
- live container at the time: `gb10-a1111-latest`
- host root: `/opt/gb10/stable-diffusion`
- API: `http://127.0.0.1:7860`

Smoke evidence from the current image:

- `/sdapi/v1/progress?skip_current_image=true`: OK, `progress=0.0`
- `/sdapi/v1/sd-models`: OK, `10` models
- `/sdapi/v1/options`: OK, checkpoint `test2.safetensors`
- CUDA visible in-container on `NVIDIA GB10`
- required imports pass for `sageattention`, `triton`, `gradio`, and `transformers`
- `xformers` is intentionally absent

The only current startup warning we accept as non-actionable is the unauthenticated Hugging Face Hub rate-limit warning. Do not add an HF token just to silence it.

## Current platform baseline

### Base image and framework lane

- base image: `nvcr.io/nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04`
- Python: `3.12.3`
- PyTorch source: explicit nightly install from `https://download.pytorch.org/whl/nightly/cu132`
- torch: `2.13.0.dev20260502+cu132`
- torchvision: `0.27.0.dev20260502+cu132`
- torchaudio: `2.11.0.dev20260502+cu132`
- triton: `3.7.0+git88b227e2`
- CUDA device: `NVIDIA GB10`

### Framework ownership doctrine

These rules are part of the working baseline and should not be broken casually:

- the CUDA/base/PyTorch stack is established before app dependency resolution
- the base package set is frozen/protected after PyTorch install
- app dependency resolution must not replace or shadow torch/torchvision/torchaudio/triton/CUDA packages
- runtime app package install uses the resolved wheel set with `--no-deps`
- `xformers` remains absent on GB10; use PyTorch SDPA and SageAttention paths instead

## Current proven runtime apt packages

Runtime apt additions owned by this image include:

- `bc`
- `ca-certificates`
- `git`
- `gosu`
- `libgl1`
- `libglib2.0-0`

Builder-only packages/toolchains do not define runtime package policy.

## Current direct Python package baseline

The current direct package set is intentionally more modern than upstream A1111's older pins while preserving the protected CUDA/PyTorch stack.

High-signal direct versions:

- `GitPython==3.1.49`
- `Pillow==12.2.0`
- `accelerate==1.13.0`
- `blendmodes==2025`
- `clean-fid==0.1.35`
- `diskcache==5.6.3`
- `einops==0.8.2`
- `facexlib==0.3.0`
- `fastapi==0.94.0`
- `gradio==3.41.2`
- `httpcore==1.0.9`
- `httpx==0.28.1`
- `jsonmerge==1.9.2`
- `kornia==0.8.2`
- `lark==1.3.1`
- `lightning-utilities==0.15.3`
- `numpy==2.4.4`
- `omegaconf==2.3.0`
- `open-clip-torch==3.3.0`
- `pillow-avif-plugin==1.5.5`
- `protobuf==7.34.1`
- `psutil==7.2.2`
- `pydantic==1.10.26`
- `pytorch-lightning==2.6.1`
- `safetensors==0.7.0`
- `scikit-image==0.26.0`
- `spandrel==0.4.2`
- `spandrel-extra-arches==0.2.0`
- `tokenizers==0.22.2`
- `tomesd==0.1.3`
- `torchdiffeq==0.2.5`
- `torchmetrics==1.9.0`
- `torchsde==0.2.6`
- `transformers==5.7.0`
- `huggingface-hub==1.13.0`

The full inventory is emitted at build time into `/opt/stable-diffusion-webui/BUILD_MANIFEST.txt`.

## Special package policy

### Gradio

`gradio==3.41.2` remains intentionally pinned. A prior `3.50.2` widening was rejected because the A1111 UI surface became materially broken. Do not casually widen Gradio as part of routine dependency cleanup.

The long-term direction is to reduce/replace Gradio dependency through the A1111-Controller lane, not to chase newer Gradio releases inside the current UI.

### Tokenizers / Transformers

Current validated versions:

- `transformers==5.7.0`
- `tokenizers==0.22.2`
- `huggingface-hub==1.13.0`

Build guard policy:

- fail if Transformers resolves below `5.7.0`
- fail if tokenizers resolves below `0.22.2`
- fail if Hugging Face Hub resolves below `1.13.0`
- fail if tokenizers resolves to anything other than a `.whl` artifact

This intentionally prevents returning to the old `tokenizers==0.13.3` Rust/source-build lane.

### OpenAI CLIP module

The original OpenAI `clip` module is still required by `k-diffusion` and is built from:

- `https://github.com/openai/CLIP/archive/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6.zip`

It is built as a wheel separately with `--no-build-isolation`, verified as `clip-*.whl`, and installed by explicit wheel path.

## Current launch baseline

Canonical launch flags:

- `--listen --port 7860 --no-hashing --disable-console-progressbars --api --opt-sdp-attention --opt-channelslast --enable-insecure-extension-access`

Canonical host relaunch path:

- `gb10/run.sh`

`gb10/run.sh` owns runtime mounts, container replacement, owned-extension sync, and `COMMANDLINE_ARGS` defaults.

## Persistent surfaces

Host-owned persistent surfaces:

- `/opt/gb10/stable-diffusion/BLIP`
- `/opt/gb10/stable-diffusion/CLIP`
- `/opt/gb10/stable-diffusion/Codeformer`
- `/opt/gb10/stable-diffusion/deepbooru`
- `/opt/gb10/stable-diffusion/GFPGAN`
- `/opt/gb10/stable-diffusion/Hypernetworks`
- `/opt/gb10/stable-diffusion/karlo`
- `/opt/gb10/stable-diffusion/Lora`
- `/opt/gb10/stable-diffusion/RealESGRAN`
- `/opt/gb10/stable-diffusion/torch_deepdanbooru`
- `/opt/gb10/stable-diffusion/VAE`
- `/opt/gb10/stable-diffusion/VAE-approx`
- `/opt/gb10/stable-diffusion/Embeddings`
- `/opt/gb10/stable-diffusion/Extensions`
- `/opt/gb10/stable-diffusion/Models`
- `/opt/gb10/stable-diffusion/Outputs`
- `/opt/gb10/stable-diffusion/config`

Special host-side paths:

- `Outputs` is a host-side symlink to `/mnt/nas-warehouse/StableDiffusion/Outputs`
- `Models/ook` is local on GB10
- `Models/SDXL` is a host-side symlink to `/mnt/nas-warehouse/StableDiffusion/models/sdxl`

## Companion repos baked into the image

Pinned upstream companion repositories remain baked into the image:

- `repositories/stable-diffusion-stability-ai`
- `repositories/generative-models`
- `repositories/k-diffusion`
- `repositories/BLIP`
- `repositories/stable-diffusion-webui-assets`

Sibling-path compatibility links remain intentional:

- `../generative-models`
- `../k-diffusion`
- `../BLIP`

## Cleanup/widening rules

1. Do not downgrade currently validated packages during cleanup.
2. Do not widen Gradio casually; that is a dedicated UI/controller migration lane.
3. Do not install or enable `xformers` on GB10 unless a separate compatibility lane proves it.
4. Keep the CUDA/PyTorch layer protected from app dependency churn.
5. Treat external mounted extensions as separate from the image baseline unless they are adopted into repo-owned first-class source.
6. Record meaningful widening experiments in `enhanced_requirements.md`; keep this file as the current validated baseline.
