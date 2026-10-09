# GB10 A1111 runtime

This repo builds AUTOMATIC1111 as a GB10-native, API-only container appliance for Schwi's NVIDIA GB10 host.

- The image (`local/gb10-a1111:latest`) is built on the GB10 host from this checkout.
- It runs as the container `gb10-a1111-latest`, with its API at `http://<GB10-LAN-IP>:7860/sdapi/v1/...`.
- There is no browser UI.

Other docs:
- Operations (script overrides, launch flags, host mounts): [launch/README.md](launch/README.md).
- What is deployed now: [STATUS.md](STATUS.md).
- Extensions: [EXTENSIONS.md](EXTENSIONS.md).
- Dated records: [notes/](notes/README.md).

## Design

The design keeps the system reproducible and reviewable:

- **Framework stack.** The official NVIDIA NGC PyTorch image provides CUDA, cuDNN, TensorRT and NVIDIA's
  torch/torchvision/Triton/TransformerEngine builds. Later dependency installs cannot replace those packages (see
  "Python dependency posture").
- **MSLK.** It is built from source at a pinned commit against the inherited stack.
- **Torch-extension CUDA code.** It is built for `12.1a`, the practical GB10 Blackwell target. The PyTorch extension
  build path does not accept `12.1f`.
- **Python environment.** It is image-owned. Nothing is installed at startup: upstream's `webui.sh` and its
  `prepare_environment` bootstrap are gone.
- **Launch path.**
  - The entrypoint (`docker/entrypoint.sh`) creates `tmp/`, `models/ControlNet/` and `models/VAE-approx/`.
  - It writes `{}` to a missing `config.json` and creates a missing `styles.csv`. It never rewrites an existing
    `config.json`.
  - The app rewrites `config.json` in place and fsyncs it, because rename cannot replace a single-file bind mount
    (`modules/settings_file.py`). A file it cannot use (truncated, empty, not a JSON object) is reported, copied to
    `tmp/config.json.corrupt-<UTC time>` and reset to `{}`, so the settings revert to their defaults instead of the
    container crash-looping. `tmp/` survives restarts of the container but not its replacement.
  - It then drops to user `a1111` (UID/GID 2323) and runs the launcher (`docker/launch-a1111.sh`).
  - The launcher's default flags and the override rules are in [launch/README.md](launch/README.md#launch-flags).
- **User data** (models, outputs, config, embeddings, extensions) stays on the host. It reaches the container through
  direct bind mounts, with no `/data` indirection and no symlink remapping in the container.
- **Owned extensions.** Generation-affecting behavior such as ControlNet, PAG/SEG/CFG-combiner/Dynamic Thresholding and
  TeaCache lives as owned source under `extensions/`. `gb10/run.sh` mirrors that source into the host `Extensions/`
  mount, and the image does not contain it.
- **Build context.** It is an allowlist (`.dockerignore`): tests, docs, `gb10/` and untracked host files never reach
  the image.
- **Build records.** Every build writes `BUILD_MANIFEST.txt/.json` and `PROTECTED_PACKAGES.json` into
  `/opt/stable-diffusion-webui/`.
  - The manifest classifies every installed Python distribution as base, direct or indirect, with its installed
    version.
  - `PROTECTED_PACKAGES.json` records the protection check.

## Base image

- **Base image.** `nvcr.io/nvidia/pytorch:26.08-py3`: CUDA 13.4.1, cuDNN 9.25, NVIDIA PyTorch `2.14.0a0+4fdf77b` built
  for CUDA 13.4, and Triton 3.8.0.
  - NGC publishes monthly `YY.MM-py3` tags.
  - `gb10/build.sh` always builds with `--pull`.
  - To move to a newer month, change `BASE_IMAGE` in both the `Dockerfile` and `gb10/build.sh`.
- **NVIDIA's torch.** The image runs NVIDIA's torch, not an upstream wheel. Replacing it would also mean rebuilding
  every NVIDIA package compiled against it: torchvision, TransformerEngine, flash-attn, apex, torchao and MSLK.
- **Host driver.** The GB10's R580 driver reports CUDA 13.0. NGC's bundled forward-compatibility libraries
  (`/usr/local/cuda/compat`) run the CUDA 13.4 user space on it.
- **xformers** is not part of the GB10 runtime. Attention uses PyTorch SDPA.

## Companion repositories

The image clones the upstream companion repositories into `repositories/`. The repository URLs and commits are pinned
by `ARG`s in the `Dockerfile`:

- `repositories/stable-diffusion-stability-ai` (`w-e-w/stablediffusion`)
- `repositories/generative-models`
- `repositories/k-diffusion`
- `repositories/BLIP`

`modules/paths.py` finds the last three as siblings of `repositories/stable-diffusion-stability-ai`. The build applies
`patches/stable-diffusion-stability-ai/` and `patches/generative-models/`; see [patches/README.md](../../patches/README.md).

## Python dependency posture

### Framework ownership

The NGC PyTorch base owns the core framework layer, and later dependency resolution never takes it back:

- Inherit `torch`, `torchvision`, Triton and the rest of the NVIDIA stack from NGC. `torchaudio` is optional and absent.
- Record the pristine NGC package set as the first step of the base stage. The snapshot fails if any later build step
  removed or changed an inherited package. MSLK is the only intentional rebuild. Its wheel install uses `--no-deps`, so
  it cannot pull in a newer numpy.
- Snapshot the effective base Python package set (`docker/snapshot-base-packages.py`). Pin every package at its exact
  NGC version, except the released list below.
- Resolve the A1111 dependency closure once in the builder stage, against resolver stubs of the protected packages.
  Then prebuild the wheels in that throwaway stage.
- Drop every requirement on a protected package from the resolver input (`docker/prepare-resolver-input.py`). A
  version specifier there would never be applied, so the build fails when the protected version does not satisfy it.
- Install the resolved application set with `--no-deps`. Then fail the build if any protected package changed
  (`docker/check-protected-stack.py`).
- The build also asserts that `gradio`, `gradio-client`, `opencv-python` and `mediapipe` are absent.

### Released NGC packages

`docker/base-released-packages.txt` lists NGC-inherited packages that the A1111 resolver owns instead of NGC. They are
resolved to the newest release that satisfies all of these:

- A1111's requirements
- the protected pins
- every installed package's declared requirements
- the NGC version as a floor (a released package never goes below it)

For example, `huggingface-hub` stays below 2.0 because Transformers requires it. `protobuf` stays below 7 because
NVIDIA's cutlass-dsl requires it.

The build enforces the release rule for every listed package:

- The NGC copy is a stock PyPI wheel: pip-installed from an index, with no local version label and no direct URL.
  NVIDIA did not build or tune it.
- It is outside the declared requirement closure of the NVIDIA-built runtime stack that A1111 imports (`torch`,
  `torchvision`, `triton`, `torchao`, `mslk`). That rule keeps `numpy`, `pillow`, `sympy`, `networkx`, `filelock`,
  `fsspec`, `jinja2`, `setuptools` and `typing-extensions` protected.
- It has no CUDA/NVIDIA package prefix.

The list holds the stock wheels in the A1111 runtime dependency closure. dpkg-owned copies (`PyYAML`, `Pygments`) are
not pip-replaceable, so they stay protected.

How it works:

- Resolver stubs of the protected packages declare only their requirements on released packages.
- The dry run requests every stub that declares such a requirement. The protected pins and the NGC floors are its
  constraints.
- After installation, `check-protected-stack.py --released-floors` verifies the floors and every declared requirement
  on a released package.
- `PROTECTED_PACKAGES.json` and the build manifest (`Released-From-NGC:<floor>`) record the result.

### Explicitly handled packages

The installed versions are in the image's `BUILD_MANIFEST.txt`.

- **Lightning runtime cluster.** `pytorch_lightning`, `torchmetrics` and `lightning-utilities` are listed unpinned in
  `requirements_versions.txt`. A1111 imports `pytorch_lightning` during startup, and the NGC base does not provide it.
- **OpenAI CLIP module.** `k-diffusion` imports the original OpenAI `clip` module directly, so the stack has both it
  and `open-clip-torch`.
  - It is built from the pinned source archive
    `https://github.com/openai/CLIP/archive/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6.zip`.
  - The build uses `--no-build-isolation`, checks for a `clip-*.whl` artifact and installs that wheel by explicit path.
- **Tokenizers resolver guard.** This project follows the current compatible Transformers/tokenizers line, not the old
  A1111-era `transformers==4.30.2` / `tokenizers==0.13.3` pair.
  - `tokenizers` is direct and unpinned in `requirements_versions.txt`, so the resolver picks the newest version that
    Transformers allows.
  - The build fails if Transformers resolves below `5.7.0`, if `huggingface-hub` resolves below `1.13.0`, or if
    tokenizers resolves below `0.22.2` or to an sdist instead of a prebuilt wheel.
  - When no compatible aarch64 tokenizers wheel exists, the build fails rather than falling back to a Rust source
    build.
