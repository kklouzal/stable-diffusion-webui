# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=nvcr.io/nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04
ARG PYTHON_VERSION=3.12
ARG PYTORCH_NIGHTLY_CUDA_TAG=cu132
ARG STABLE_DIFFUSION_REPO=https://github.com/w-e-w/stablediffusion.git
ARG STABLE_DIFFUSION_COMMIT=cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf
ARG GENERATIVE_MODELS_REPO=https://github.com/Stability-AI/generative-models.git
ARG GENERATIVE_MODELS_COMMIT=45c443b316737a4ab6e40413d7794a7f5657c19f
ARG K_DIFFUSION_REPO=https://github.com/crowsonkb/k-diffusion.git
ARG K_DIFFUSION_COMMIT=ab527a9a6d347f364e3d185ba6d714e22d80cb3c
ARG BLIP_REPO=https://github.com/salesforce/BLIP.git
ARG BLIP_COMMIT=48211a1594f1321b00f14c9f7a5b4813144b2fb9
ARG ASSETS_REPO=https://github.com/AUTOMATIC1111/stable-diffusion-webui-assets.git
ARG ASSETS_COMMIT=6f7db241d2f8ba7457bac5ca9753331f0c266917
ARG CLIP_PACKAGE_URL=https://github.com/openai/CLIP/archive/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6.zip
ARG SAGEATTENTION_REPO=https://github.com/thu-ml/SageAttention.git
ARG SAGEATTENTION_COMMIT=d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5
ARG CUTLASS_REPO=https://github.com/NVIDIA/cutlass.git
ARG CUTLASS_COMMIT=7a9fe055cb69ab2de605a0cf7dbb33f27833f7f3

FROM ${BASE_IMAGE} AS torch-base

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION
ARG PYTORCH_NIGHTLY_CUDA_TAG

SHELL ["/bin/bash", "-lc"]
WORKDIR /opt/build

ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_ROOT_USER_ACTION=ignore

RUN apt-get update && apt-get install -y --no-install-recommends \
    bc \
    ca-certificates \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python3 \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python

RUN python -m pip install --break-system-packages --upgrade setuptools==69.5.1

# CUDA-base doctrine:
# - start from the NVIDIA CUDA image, not the NVIDIA PyTorch image
# - install the PyTorch nightly lane explicitly from the selected CUDA wheel index
# - freeze the resulting system-Python package set so later app deps cannot overwrite it
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --break-system-packages --no-cache-dir --pre \
      torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/nightly/${PYTORCH_NIGHTLY_CUDA_TAG}

RUN python - <<'PY'
import importlib.metadata as md
import json
import re
from pathlib import Path

def normalize(name: str) -> str:
    return re.sub(r'[-_.]+', '-', name.strip().lower())

pins = []
seen = set()
for dist in sorted(md.distributions(), key=lambda d: normalize(d.metadata.get('Name', ''))):
    name = dist.metadata.get('Name')
    if not name:
        continue
    norm = normalize(name)
    if norm in seen:
        continue
    seen.add(norm)
    pins.append(f'{norm}=={dist.version}')

Path('/opt/build/base-python-protected-constraints.txt').write_text('\n'.join(pins) + '\n')
Path('/opt/build/base-python-protected-names.txt').write_text('\n'.join(x.split('==', 1)[0] for x in pins) + '\n')
print(json.dumps({
    'protected_count': len(pins),
    'torch': md.version('torch'),
    'torchvision': md.version('torchvision'),
    'torchaudio': md.version('torchaudio'),
}, indent=2))
PY

FROM torch-base AS source

ARG DEBIAN_FRONTEND=noninteractive
ARG STABLE_DIFFUSION_REPO
ARG STABLE_DIFFUSION_COMMIT
ARG GENERATIVE_MODELS_REPO
ARG GENERATIVE_MODELS_COMMIT
ARG K_DIFFUSION_REPO
ARG K_DIFFUSION_COMMIT
ARG BLIP_REPO
ARG BLIP_COMMIT
ARG ASSETS_REPO
ARG ASSETS_COMMIT
ARG CLIP_PACKAGE_URL

COPY patches /opt/build/patches
COPY docker/apply-local-patches.py /opt/build/apply-local-patches.py

SHELL ["/bin/bash", "-lc"]
WORKDIR /opt/build

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgoogle-perftools-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . /opt/build/stable-diffusion-webui

RUN cd stable-diffusion-webui \
    && mkdir -p repositories \
    && git clone --filter=blob:none "${STABLE_DIFFUSION_REPO}" repositories/stable-diffusion-stability-ai \
    && git -c advice.detachedHead=false -C repositories/stable-diffusion-stability-ai checkout "${STABLE_DIFFUSION_COMMIT}" \
    && git clone --filter=blob:none "${GENERATIVE_MODELS_REPO}" repositories/generative-models \
    && git -c advice.detachedHead=false -C repositories/generative-models checkout "${GENERATIVE_MODELS_COMMIT}" \
    && git clone --filter=blob:none "${K_DIFFUSION_REPO}" repositories/k-diffusion \
    && git -c advice.detachedHead=false -C repositories/k-diffusion checkout "${K_DIFFUSION_COMMIT}" \
    && git clone --filter=blob:none "${BLIP_REPO}" repositories/BLIP \
    && git -c advice.detachedHead=false -C repositories/BLIP checkout "${BLIP_COMMIT}" \
    && git clone --filter=blob:none "${ASSETS_REPO}" repositories/stable-diffusion-webui-assets \
    && git -c advice.detachedHead=false -C repositories/stable-diffusion-webui-assets checkout "${ASSETS_COMMIT}" \
    && ln -sfn repositories/generative-models ../generative-models \
    && ln -sfn repositories/k-diffusion ../k-diffusion \
    && ln -sfn repositories/BLIP ../BLIP \
    && python /opt/build/apply-local-patches.py

FROM torch-base AS wheelbuilder

ARG DEBIAN_FRONTEND=noninteractive
ARG CLIP_PACKAGE_URL
ARG SAGEATTENTION_REPO
ARG SAGEATTENTION_COMMIT
ARG CUTLASS_REPO
ARG CUTLASS_COMMIT

SHELL ["/bin/bash", "-lc"]
WORKDIR /opt/build/stable-diffusion-webui

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ccache \
    libssl-dev \
    ninja-build \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/opt/rustup
ENV CARGO_HOME=/opt/cargo
ENV CCACHE_DIR=/root/.cache/ccache
ENV CARGO_TARGET_DIR=/root/.cache/cargo-target
ENV CC="ccache gcc"
ENV CXX="ccache g++"
ENV PATH=/usr/lib/ccache:/opt/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y --profile minimal --default-toolchain stable

COPY --from=source /opt/build/stable-diffusion-webui /opt/build/stable-diffusion-webui
COPY --from=torch-base /opt/build/base-python-protected-constraints.txt /opt/build/base-python-protected-constraints.txt
COPY --from=torch-base /opt/build/base-python-protected-names.txt /opt/build/base-python-protected-names.txt
COPY requirements_versions.txt /opt/build/requirements-image.txt
COPY docker/render-resolved-requirements.py /opt/build/render-resolved-requirements.py
COPY docker/filter-resolved-requirements.py /opt/build/filter-resolved-requirements.py
COPY docker/prepare-resolver-input.py /opt/build/prepare-resolver-input.py
COPY docker/patch-sageattention.py /opt/build/patch-sageattention.py

# Builder-stage wheel doctrine:
# - use a modern rustup-managed Rust toolchain here for packages like tokenizers
# - resolve the full dependency closure once against the CUDA-base + explicit torch lane
# - prebuild wheels for the full resolved closure in this throwaway stage
# - temporary compatibility concession: tokenizers 0.13.x fails on current Rust with
#   `invalid_reference_casting`; relax that single lint in this builder stage only
# - local resolver compatibility concession: Gradio 3.41.2 still advertises stale
#   `numpy~=1.0` wheel metadata even though this stack runs on numpy 2.x, so the resolver
#   input patches that metadata to `numpy>=1.0` before generating the dry-run report used
#   to build the wheel closure. This should disappear when the UI stack can move past
#   the Gradio 3.x compatibility lane.
RUN rustc --version \
    && cargo --version \
    && python -m pip install --break-system-packages --upgrade setuptools==69.5.1 \
    && python /opt/build/prepare-resolver-input.py --source /opt/build/requirements-image.txt --target /opt/build/requirements-resolver.txt --wheel-dir /opt/build/resolve-wheel-overrides \
    && python -m pip install --break-system-packages --dry-run --report /opt/build/report.json -r /opt/build/requirements-resolver.txt \
    && python /opt/build/render-resolved-requirements.py
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/opt/cargo/registry \
    --mount=type=cache,target=/opt/cargo/git \
    --mount=type=cache,target=/root/.cache/cargo-target \
    export RUSTFLAGS="-A invalid_reference_casting" \
    && export CCACHE_DIR=/root/.cache/ccache \
    && export MAX_JOBS="$(nproc)" \
    && ccache --zero-stats \
    && python -m pip wheel --no-deps --wheel-dir /opt/wheels -r /opt/build/requirements-resolved.txt \
    && test -n "${CLIP_PACKAGE_URL}" \
    && python -m pip wheel --no-deps --no-build-isolation --wheel-dir /opt/wheels "${CLIP_PACKAGE_URL}" \
    && ls -1 /opt/wheels/clip-*.whl \
    && ccache --show-stats

# SageAttention build doctrine:
# - build both SageAttention2++ and SageAttention3 Blackwell wheels into the image so
#   attention backends can be toggled at runtime without reinstalling packages
# - target the GB10 Blackwell lane explicitly as sm_121a for the current CUDA/PyTorch stack
# - keep this isolated in the wheelbuilder stage; runtime installs the prebuilt wheels only
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    export CCACHE_DIR=/root/.cache/ccache \
    && export EXT_PARALLEL=2 \
    && export MAX_JOBS=8 \
    && export TORCH_CUDA_ARCH_LIST="12.1a" \
    && export CC=gcc \
    && export CXX=g++ \
    && python -m pip install --break-system-packages --no-cache-dir packaging \
    && rm -rf /opt/build/SageAttention \
    && git clone --filter=blob:none "${SAGEATTENTION_REPO}" /opt/build/SageAttention \
    && git -c advice.detachedHead=false -C /opt/build/SageAttention checkout "${SAGEATTENTION_COMMIT}" \
    && git clone --filter=blob:none "${CUTLASS_REPO}" /opt/build/SageAttention/sageattention3_blackwell/csrc/cutlass \
    && git -c advice.detachedHead=false -C /opt/build/SageAttention/sageattention3_blackwell/csrc/cutlass checkout "${CUTLASS_COMMIT}" \
    && python /opt/build/patch-sageattention.py \
    && export SAGEATTN3_CUDA_ARCH="12.1" \
    && python -m pip wheel --no-deps --no-build-isolation --wheel-dir /opt/wheels /opt/build/SageAttention \
    && python -m pip wheel --no-deps --no-build-isolation --wheel-dir /opt/wheels /opt/build/SageAttention/sageattention3_blackwell \
    && ls -1 /opt/wheels/sageattention-*.whl /opt/wheels/sageattn3-*.whl


FROM torch-base AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG A1111_UID=2323
ARG A1111_GID=2323
ARG PYTORCH_NIGHTLY_CUDA_TAG

SHELL ["/bin/bash", "-lc"]
WORKDIR /opt/stable-diffusion-webui

RUN apt-get update && apt-get install -y --no-install-recommends \
    gosu \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${A1111_GID} a1111 \
    && useradd --uid ${A1111_UID} --gid ${A1111_GID} --create-home --shell /bin/bash a1111

COPY --from=source /opt/build/stable-diffusion-webui /opt/stable-diffusion-webui
COPY extensions/sd-webui-incantations /opt/stable-diffusion-webui/extensions/sd-webui-incantations
COPY --from=wheelbuilder /opt/wheels /opt/wheels
COPY --from=wheelbuilder /opt/build/requirements-resolved.txt /opt/requirements-resolved.txt
COPY --from=torch-base /opt/build/base-python-protected-constraints.txt /opt/base-python-protected-constraints.txt
COPY --from=torch-base /opt/build/base-python-protected-names.txt /opt/base-python-protected-names.txt
COPY requirements_versions.txt /opt/requirements-image.txt
COPY docker/filter-resolved-requirements.py /usr/local/bin/gb10-a1111-filter-requirements
COPY docker/render-build-manifest.py /usr/local/bin/gb10-a1111-render-build-manifest
COPY docker/entrypoint.sh /usr/local/bin/gb10-a1111-entrypoint
COPY docker/launch-a1111.sh /usr/local/bin/gb10-a1111-launch

# Container-owned environment doctrine:
# - do not let upstream webui.sh create/manage its own venv here
# - do not let upstream launch bootstrap replace the CUDA-base + PyTorch package set
# - do aggressively protect all packages present in the torch-base layer so later
#   A1111 installs cannot upgrade or shadow CUDA/PyTorch/base-image packages
# - do install the repo-owned A1111 dependency closure from requirements_versions.txt
#   as normal application dependencies, filtered only against the protected base set
RUN python - <<'PY'
import importlib.metadata as md
import json

def version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None

print(json.dumps({
    'before_runtime_install': {
        'torch': version('torch'),
        'torchvision': version('torchvision'),
        'torchaudio': version('torchaudio'),
        'gradio': version('gradio'),
        'transformers': version('transformers'),
        'clip': version('clip'),
    }
}, indent=2))
PY
RUN chmod +x /usr/local/bin/gb10-a1111-filter-requirements \
    && SOURCE=/opt/requirements-resolved.txt TARGET=/opt/requirements-runtime.txt BASE_PROTECTED_NAMES_FILE=/opt/base-python-protected-names.txt /usr/local/bin/gb10-a1111-filter-requirements \
    && python -m pip install --break-system-packages --upgrade -c /opt/base-python-protected-constraints.txt setuptools==69.5.1 \
    && python -m pip install --break-system-packages --no-deps --no-index --find-links=/opt/wheels -c /opt/base-python-protected-constraints.txt -r /opt/requirements-runtime.txt \
    && python -m pip install --break-system-packages --no-deps --no-index --find-links=/opt/wheels -c /opt/base-python-protected-constraints.txt /opt/wheels/clip-*.whl sageattention sageattn3 \
    && python - <<'PY'
import importlib.metadata as md
import json
print(json.dumps({
    'after_runtime_install': {
        'torch': md.version('torch'),
        'torchvision': md.version('torchvision'),
        'torchaudio': md.version('torchaudio'),
        'gradio': md.version('gradio'),
        'transformers': md.version('transformers'),
        'clip': md.version('clip'),
        'sageattention': md.version('sageattention'),
        'sageattn3': md.version('sageattn3'),
    }
}, indent=2))
PY
RUN chmod +x /usr/local/bin/gb10-a1111-render-build-manifest \
    && PYTORCH_NIGHTLY_INDEX_URL="https://download.pytorch.org/whl/nightly/${PYTORCH_NIGHTLY_CUDA_TAG}" /usr/local/bin/gb10-a1111-render-build-manifest
RUN rm -rf /opt/wheels /opt/requirements-resolved.txt /opt/requirements-runtime.txt /root/.cache/pip \
    && chmod +x /usr/local/bin/gb10-a1111-entrypoint /usr/local/bin/gb10-a1111-launch \
    && chown -R a1111:a1111 /opt/stable-diffusion-webui /home/a1111 \
    && mkdir -p /opt/stable-diffusion-webui/tmp

ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV A1111_HOME=/opt/stable-diffusion-webui
ENV A1111_RUN_AS_USER=a1111
ENV COMMANDLINE_ARGS=
ENV TORCH_COMMAND=true
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

EXPOSE 7860
ENTRYPOINT ["/usr/local/bin/gb10-a1111-entrypoint"]
