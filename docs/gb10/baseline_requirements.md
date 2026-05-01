# baseline_requirements.md

This file records the **currently proven working baseline** for the GB10-A1111 container.

Use it as the ground truth before any package widening work.

## Purpose

This is **not** a generic list of everything A1111 might ever want.
It is the exact stack we have currently validated as working for this project’s present baseline:

- image builds successfully
- container starts successfully
- A1111 serves UI/API successfully
- local model loads successfully
- image generation succeeds
- persistent host storage wiring works through direct bind mounts to the real A1111 paths
- output and model-share host-side pathing can be expressed cleanly without requiring container-side path rewrites

Any widening experiments should be judged against this baseline.

---

## Current proven platform baseline

### Base image ownership

The following framework components are confirmed to still come from the NVIDIA base image, not from our own pip overlay:

- base image: `nvcr.io/nvidia/pytorch:25.11-py3`
- Python: `3.12.3`
- torch: `2.10.0a0+b558c986e8.nv25.11`
- torchvision: `0.25.0a0+7a13ad0f`
- CUDA runtime reported by torch: `13.0`

Also confirmed in both the NVIDIA base image and our runtime image:

- `torchaudio`: not installed
- `triton`: not installed as a Python package

### Framework ownership doctrine

These rules are part of the working baseline and should not be broken casually:

- `torch` stays owned by the NVIDIA base image
- `torchvision` stays owned by the NVIDIA base image
- upstream A1111 bootstrap must **not** be allowed to replace the base-image framework stack
- runtime package installation stays filtered and `--no-deps`
- `TORCH_COMMAND=true` remains part of the anti-override posture

---

## Current proven runtime apt packages

These are the apt packages explicitly installed in the runtime image layer:

- `bc`
- `ca-certificates`
- `git`
- `gosu`
- `libgl1`
- `libglib2.0-0`

These are not the full contents of the base image; they are the repo-owned runtime additions on top of the NVIDIA base.

---

## Current proven Python package baseline

### Upstream-pinned / aligned core set

These are the currently installed and working package versions in the live container baseline, and they align with the current baked upstream `requirements_versions.txt` posture unless otherwise noted.

- `GitPython==3.1.32`
- `Pillow==9.5.0`
- `accelerate==0.21.0`
- `blendmodes==2022`
- `clean-fid==0.1.35`
- `diskcache==5.6.3`
- `einops==0.4.1`
- `facexlib==0.3.0`
- `fastapi==0.94.0`
- `gradio==3.41.2`
- `httpcore==0.15.0`
- `httpx==0.24.1`
- `inflection==0.5.1`
- `jsonmerge==1.8.0`
- `kornia==0.6.7`
- `lark==1.1.2`
- `numpy==1.26.2`
- `omegaconf==2.2.3`
- `open-clip-torch==2.20.0`
- `piexif==1.1.3`
- `protobuf==3.20.0`
- `psutil==5.9.5`
- `pytorch-lightning==1.9.4`
- `resize-right==0.0.2`
- `safetensors==0.4.5`
- `scikit-image==0.21.0`
- `spandrel==0.3.4`
- `spandrel-extra-arches==0.1.1`
- `tomesd==0.1.3`
- `torchdiffeq==0.2.3`
- `torchsde==0.2.6`
- `transformers==4.30.2`
- `pillow-avif-plugin==1.4.3`

### Supplemental / compatibility cluster we currently rely on

These are part of the currently working baseline even though they are not simply “copy the exact upstream pinned line and stop thinking.”

- `tokenizers==0.13.3`
  - compatible with the current `transformers==4.30.2` posture
  - currently needs the builder-stage Rust concession on this Python 3.12 / arm64 path
- `torchmetrics==1.9.0`
- `lightning-utilities==0.15.3`
  - part of the explicit Lightning runtime cluster needed for this containerized baseline
- `clip==1.0`
  - built from the pinned OpenAI CLIP source archive expected by the current stack

### Framework packages intentionally not repo-owned

These are intentionally **not** supplied by our curated requirements overlay:

- `torch`
- `torchvision`
- `torchaudio`
- `triton`
- `nvidia-*`
- `cuda-*`

---

## Current proven upstream companion repos baked into the image

These are required by the current working A1111 image layout:

- `repositories/stable-diffusion-stability-ai`
- `repositories/generative-models`
- `repositories/k-diffusion`
- `repositories/BLIP`
- `repositories/stable-diffusion-webui-assets`

Sibling-path compatibility links are also intentionally provided for:

- `../generative-models`
- `../k-diffusion`
- `../BLIP`

---

## Current proven launch baseline

### Launch flags

Current winning launch baseline:

- `--listen --port 7860 --no-hashing --disable-console-progressbars --api --opt-sdp-attention --opt-channelslast --enable-insecure-extension-access`

### Current UI/runtime settings confirmed on the live container

- `hypertile_enable_unet = true`
- `hypertile_enable_vae = true`

### Current persistent surfaces

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

### Current host-path doctrine

The container binds those host paths directly onto the real A1111 filesystem locations instead of mounting an intermediate `/data` tree and rebuilding the A1111 path map inside the container.

Special host-side exceptions remain deliberate and explicit:

- `Outputs` is a host-side symlink to the SERVER-002 output share
- `Models/ook` is local on GB10
- `Models/SDXL` is a host-side symlink to the UGREEN NAS SDXL subtree

Those choices are expressed on the host side; the container just sees the final mounted A1111 paths.

---

## Known current warning/noise that is not treated as a blocker

Current non-blocking log residue includes things like:

- `timm` deprecation warning noise
- PyTorch TF32 API deprecation warning noise
- some upstream Python `SyntaxWarning` noise
- Hugging Face `resume_download` future-warning noise
- `xformers` absence messages
- partial failure of `--disable-console-progressbars` to fully suppress all sampler output paths

These are baseline-quality annoyances, not known blockers.

---

## Widening rules

When testing newer package versions:

1. Start from this baseline only.
2. Widen one package, or one tightly related package cluster, at a time.
3. Preserve framework ownership by the NVIDIA base image.
4. Rebuild cleanly.
5. Re-test at least:
   - image build
   - container startup
   - model load
   - UI/API availability
   - actual image generation
6. If a widening passes, record it in `enhanced_requirements.md`.
7. If it fails, record the failure there too if the result is informative enough to avoid future re-learning.

Do **not** silently replace this file with speculative newer versions. This file is the known-good restore map.
