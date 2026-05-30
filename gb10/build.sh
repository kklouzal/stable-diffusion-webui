#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKERFILE="${DOCKERFILE:-${PROJECT_ROOT}/Dockerfile}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04}"
PYTORCH_NIGHTLY_CUDA_TAG="${PYTORCH_NIGHTLY_CUDA_TAG:-cu132}"
MSLK_REPO="${MSLK_REPO:-https://github.com/meta-pytorch/MSLK.git}"
MSLK_COMMIT="${MSLK_COMMIT:-e54ee82d57492dfc08d89df65c3898d767ad8b24}"
IMAGE_TAG="${IMAGE_TAG:-local/gb10-a1111:latest-mxfp8-dev}"
DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"
BUILDKIT_PROGRESS="${BUILDKIT_PROGRESS:-plain}"

cat <<EOM
[build.sh]
Project root:              ${PROJECT_ROOT}
Dockerfile:                ${DOCKERFILE}
Base image:                ${BASE_IMAGE}
PyTorch nightly CUDA tag:  ${PYTORCH_NIGHTLY_CUDA_TAG}
MSLK source repo:          ${MSLK_REPO}
MSLK source commit:        ${MSLK_COMMIT}
Image tag:                 ${IMAGE_TAG}
A1111 source:              local fork checkout (${PROJECT_ROOT})
DOCKER_BUILDKIT:           ${DOCKER_BUILDKIT}
BUILDKIT_PROGRESS:         ${BUILDKIT_PROGRESS}
EOM

sudo env DOCKER_BUILDKIT="${DOCKER_BUILDKIT}" BUILDKIT_PROGRESS="${BUILDKIT_PROGRESS}" docker build \
  -f "${DOCKERFILE}" \
  -t "${IMAGE_TAG}" \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg PYTORCH_NIGHTLY_CUDA_TAG="${PYTORCH_NIGHTLY_CUDA_TAG}" \
  --build-arg MSLK_REPO="${MSLK_REPO}" \
  --build-arg MSLK_COMMIT="${MSLK_COMMIT}" \
  "${PROJECT_ROOT}"
