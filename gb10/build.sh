#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKERFILE="${DOCKERFILE:-${PROJECT_ROOT}/Dockerfile}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.08-py3}"
PYTORCH_NIGHTLY_CUDA_TAG="${PYTORCH_NIGHTLY_CUDA_TAG:-cu134}"
MSLK_REPO="${MSLK_REPO:-https://github.com/meta-pytorch/MSLK.git}"
MSLK_COMMIT="${MSLK_COMMIT:-88d06bc2784f3b550d7ec851d4ca67a16a844fe2}"
IMAGE_TAG="${IMAGE_TAG:-local/gb10-a1111:latest}"
BUILDKIT_PROGRESS="${BUILDKIT_PROGRESS:-plain}"
CACHE_FROM="${CACHE_FROM:-${IMAGE_TAG}}"

if [[ "${DOCKER_BUILDKIT:-1}" != "1" ]]; then
  echo "[build.sh] DOCKER_BUILDKIT=${DOCKER_BUILDKIT} requested, but GB10 builds require BuildKit cache; forcing DOCKER_BUILDKIT=1." >&2
fi
DOCKER_BUILDKIT=1

# BuildKit RUN steps run in cgroups that dockerd creates outside the docker-*.scope
# policy that gives containers the Cortex-X925 performance cores, so they inherit the
# efficiency-core placement of the control plane. --cgroup-parent moves them into
# gb10build.slice (systemd unit, AllowedCPUs=5-9 15-19). While those CPUs are also
# boot-isolated with isolcpus=domain the kernel does not load-balance between them and
# every parallel compile job would share one core, so the slice is used only when none
# of its CPUs are domain-isolated.
BUILD_CGROUP_PARENT="${BUILD_CGROUP_PARENT:-gb10build.slice}"
expand_cpus() {  # "5-9 15-19" or "5-9,15-19" -> one CPU number per line, sorted as text for comm
  local part
  for part in ${1//,/ }; do seq "${part%-*}" "${part#*-}"; done | sort
}
BUILD_PLACEMENT="default (efficiency cores)"
PLACEMENT_ARGS=()
if [[ -n "${BUILD_CGROUP_PARENT}" ]]; then
  slice_cpus="$(systemctl show "${BUILD_CGROUP_PARENT}" -p EffectiveCPUs --value 2>/dev/null || true)"
  isolated_cpus="$(cat /sys/devices/system/cpu/isolated)"
  if [[ "$(systemctl is-active "${BUILD_CGROUP_PARENT}" 2>/dev/null || true)" != "active" || -z "${slice_cpus}" ]]; then
    echo "[build.sh] ERROR: ${BUILD_CGROUP_PARENT} is not active with a CPU mask; install and enable it, or set BUILD_CGROUP_PARENT= to build with default placement." >&2
    exit 1
  elif [[ -n "$(comm -12 <(expand_cpus "${slice_cpus}") <(expand_cpus "${isolated_cpus}"))" ]]; then
    echo "[build.sh] NOTE: ${BUILD_CGROUP_PARENT} CPUs (${slice_cpus}) are boot-isolated (isolcpus=domain: ${isolated_cpus}); using default placement until that isolation is removed." >&2
  else
    PLACEMENT_ARGS=(--cgroup-parent "${BUILD_CGROUP_PARENT}")
    BUILD_PLACEMENT="${BUILD_CGROUP_PARENT} (CPUs ${slice_cpus})"
  fi
fi

CACHE_ARGS=(--build-arg BUILDKIT_INLINE_CACHE=1)
CACHE_FROM_STATUS="not found"
if sudo docker image inspect "${CACHE_FROM}" >/dev/null 2>&1; then
  CACHE_ARGS+=(--cache-from "${CACHE_FROM}")
  CACHE_FROM_STATUS="enabled"
fi

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
Docker build cache:        enabled
Compiler cache:            enabled (BuildKit cache mount gb10-global-ccache)
Cache-from image:          ${CACHE_FROM} (${CACHE_FROM_STATUS})
Build CPU placement:       ${BUILD_PLACEMENT}
EOM

sudo env DOCKER_BUILDKIT="${DOCKER_BUILDKIT}" BUILDKIT_PROGRESS="${BUILDKIT_PROGRESS}" docker build \
  --pull \
  "${PLACEMENT_ARGS[@]}" \
  "${CACHE_ARGS[@]}" \
  -f "${DOCKERFILE}" \
  -t "${IMAGE_TAG}" \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg PYTORCH_NIGHTLY_CUDA_TAG="${PYTORCH_NIGHTLY_CUDA_TAG}" \
  --build-arg MSLK_REPO="${MSLK_REPO}" \
  --build-arg MSLK_COMMIT="${MSLK_COMMIT}" \
  "${PROJECT_ROOT}"
