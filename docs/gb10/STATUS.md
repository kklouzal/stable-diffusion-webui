# STATUS.md

## Mission

Refactor the GB10-native AUTOMATIC1111 container from the older NVIDIA PyTorch-base approach to the newer NVIDIA CUDA-base + explicit PyTorch nightly `cu132` approach, while keeping the repo reviewable and the host-mounted user-data layout intact.

## Current status

**Full image build now succeeds on the GB10 with the CUDA-base refactor.**

The repo currently:

- still targets the same upstream A1111 source and persistent host-mounted runtime surfaces
- now defaults the Dockerfile/build flow to NVIDIA CUDA NGC + explicit PyTorch nightly `cu132`
- now freezes/protects the base Python package set after torch install so later app deps cannot overwrite or shadow it
- now keeps `wheelbuilder` on a Rust + OpenSSL-capable path for packages that still need native wheels, while `tokenizers` is required to resolve to a current compatible aarch64 wheel
- now routes heavy builder compiles through `ccache` via BuildKit cache mounts
- still keeps upstream `webui.sh` out of authority for runtime bootstrap
- still keeps user-owned runtime surfaces under `/opt/gb10/stable-diffusion`
- now vendors `sd-webui-incantations` as GB10-owned first-class extension source for PAG, SEG, CFG-combiner, and CFG-Fix behavior
- has completed a real successful full-image build on the GB10 host
- now emits `/opt/stable-diffusion-webui/BUILD_MANIFEST.txt` and `.json` during image build, classifying installed Python packages into base-layer-provided vs A1111 direct vs A1111 indirect and annotating latest-visible-version drift reasons

## Current chosen defaults

- base image family: `nvcr.io/nvidia/cuda`
- base image tag: `nvcr.io/nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04`
- PyTorch nightly CUDA lane: `cu132`
- torch-extension arch policy in repo build path: `12.1a` for GB10 Blackwell-targeted extension wheels; `12.1f` is not directly accepted by the current PyTorch extension build path and is no longer preferred over `12.1a`
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
- tokenizers resolver guard: `transformers>=5.7.0`, `tokenizers>=0.22.2`, `huggingface-hub>=1.13.0`, and tokenizers must resolve from a wheel rather than an sdist/Rust build
- explicit `libssl-dev` + `pkg-config` support in `wheelbuilder`

## Latest build/runtime evidence

Current validated GB10 runtime:

- image tag: `local/gb10-a1111:base-protected-app-latest`
- image ID: `sha256:85c902073586364c4406d36604050c6ab7ccc531b1e7fa4449d26089d923b3c8`
- image created: `2026-05-03T15:47:10.304319823-07:00`
- live container: `gb10-a1111-latest`
- torch after runtime install: `2.13.0.dev20260502+cu132`
- torchvision after runtime install: `0.27.0.dev20260502+cu132`
- torchaudio after runtime install: `2.11.0.dev20260502+cu132`
- Transformers/tokenizers/HF Hub runtime: `transformers==5.7.0`, `tokenizers==0.22.2`, `huggingface-hub==1.13.0`
- tokenizers build guard: Docker build fails if Transformers, tokenizers, or Hugging Face Hub resolve below the validated floors, and tokenizers must resolve from a `.whl` artifact rather than an sdist/Rust source build
- A1111 API health after extension quarantine: `GET /sdapi/v1/progress`, `GET /sdapi/v1/sd-models`, and `GET /sdapi/v1/options` return JSON on `127.0.0.1:7860`; latest smoke saw `10` models and checkpoint `test2.safetensors`
- `BUILD_MANIFEST.json` summary: `base=31`, `direct=52`, `indirect=87`
- `sageattention`, `triton`, `gradio`, and `transformers` import in the live container
- `xformers` is intentionally absent in the current CUDA 13 / GB10 aarch64 runtime; A1111 uses SDP/SageAttention paths instead
- repo smoke coverage now includes `gb10/smoke-test.sh` for API health, model listing, CUDA/PyTorch visibility, and required runtime imports without starting a generation job
- `gb10/run.sh` is the canonical relaunch path; it owns runtime mounts, first-class extension sync, container replacement, and `COMMANDLINE_ARGS`
- external mounted extension posture is documented in `docs/gb10/EXTENSIONS.md`; approved removals were purged from `/opt/gb10/stable-diffusion/Extensions` and the quarantine tree was deleted after validation

Older probe tags worth keeping as historical breadcrumbs:

- `local/gb10-a1111:full-probe` / digest `sha256:205e443219a72e9e8792ca31046638fd0dc88c16f570d4314f0835a7c3157d99` proved the earlier full-image bring-up
- `local/gb10-a1111:wheelbuilder-probe` proved the separate wheelbuilder path
- `local/gb10-a1111:arch-priority-probe` proved the earlier `sm_121a` extension-build direction
- `local/gb10-a1111:manifest-probe` proved the package-manifest path before the current `cu132` runtime refresh


## Latest MXFP8 img2img baseline

The current known-good MXFP8/img2img baseline is documented in `docs/gb10/notes/mxfp8-img2img-final-baseline-2026-05-06.md`.

Key validated defaults:

- image tag: `local/gb10-a1111:latest-mxfp8-dev`
- live container: `gb10-a1111-latest-mxfp8-dev`
- checkpoint: `test2.safetensors`
- VAE: `ftasticVAE_v10.safetensors`
- attention backend: `sdpa`
- MXFP8 storage: `Enable for SDXL`
- MXFP8 LoRA handling: `Merge LoRA then quantize to MXFP8`
- A1111 MXFP8 audit: `183/911` Linear modules quantized; attention and conditioner Linear modules intentionally skipped
- 9-case txt2img/img2img SDPA/Sage/SEG/PAG generation matrix completed successfully

## Latest MXFP8 LoRA final-merge fix

The MXFP8+LoRA repeat-step slowdown was traced to LoRA-count-sensitive MXFP8 preparation work remaining in the generation path. The 2026-05-07 refactor makes MXFP8+LoRA preparation a model-level active-config transaction: BF16 master weights + active LoRA deltas are merged once, selected final effective weights are quantized once, and `Linear.forward()` stays a fast path while the active signature matches. Details and validation are in `docs/gb10/notes/mxfp8-lora-final-merge-2026-05-07.md`.

Validated repeat timings after the fix at 832x832 / 4 Euler-a steps / `unet_other`: `0` LoRAs `0.516s/step`, `1` LoRA `0.516s/step`, `4` LoRAs `0.516s/step`, `13` LoRAs `0.519s/step`.

## Immediate next validation work

1. plan first-class adoption/replacement for retained external extensions, starting with `multidiffusion-upscaler-for-automatic1111` and `sd-webui-detail-daemon`
2. map exactly which A1111-Controller paths depend on `sd-webui-model-converter` and `ultimate-upscale-for-automatic1111`
3. continue modern Python/PyTorch/runtime cleanup only when new warnings/errors appear under real generation, model swap, or LoRA-swap workloads
