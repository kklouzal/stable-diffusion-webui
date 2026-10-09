#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKERFILE="${DOCKERFILE:-${PROJECT_ROOT}/Dockerfile}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.08-py3}"
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
BUILD_CGROUP_PARENT="${BUILD_CGROUP_PARENT-gb10build.slice}"  # set and empty: default placement
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

# Provenance, recorded as OCI labels on the image (gb10/run.sh reads them, so every image reports its own version):
# - revision: the checkout's commit, with -dirty when what the build reads has uncommitted or untracked (not ignored)
#   changes: the build context, which is .dockerignore's allowlist (its "!path" lines, the one list of what is sent),
#   .dockerignore itself and the Dockerfile. A Dockerfile outside the checkout is not described by the commit at all.
# - version: `git describe --tags`, the infotext Version (the image has no .git)
# - base.name/base.digest: the base image, built by that digest so the label names exactly the base used
SOURCE_REVISION="$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
BUILD_INPUT_PATHSPECS=(":(literal).dockerignore")
while IFS= read -r context_path; do
  if [[ "${context_path}" == *[][*?\\]* ]]; then
    echo "[build.sh] ERROR: .dockerignore allows '${context_path}', a pattern; the provenance check reads the allowlist as literal paths." >&2
    exit 1
  fi
  BUILD_INPUT_PATHSPECS+=(":(literal)${context_path}")
done < <(sed -n 's/^!//p' "${PROJECT_ROOT}/.dockerignore")
SOURCE_STATUS=""
if [[ "${DOCKERFILE}" == "${PROJECT_ROOT}/"* ]]; then
  BUILD_INPUT_PATHSPECS+=(":(literal)${DOCKERFILE#"${PROJECT_ROOT}/"}")
else
  SOURCE_STATUS="Dockerfile outside the checkout: ${DOCKERFILE}"
fi
SOURCE_STATUS+="$(git -C "${PROJECT_ROOT}" status --porcelain -- "${BUILD_INPUT_PATHSPECS[@]}")"
if [[ -n "${SOURCE_STATUS}" ]]; then
  SOURCE_REVISION="${SOURCE_REVISION}-dirty"
fi
SOURCE_VERSION="$(git -C "${PROJECT_ROOT}" describe --tags)"
BASE_IMAGE_NAME="${BASE_IMAGE%@*}"
BASE_IMAGE_DIGEST="$(sudo docker buildx imagetools inspect "${BASE_IMAGE}" --format '{{json .Manifest.Digest}}')"
BASE_IMAGE_DIGEST="${BASE_IMAGE_DIGEST//\"/}"
if [[ ! "${BASE_IMAGE_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "[build.sh] ERROR: could not resolve the digest of ${BASE_IMAGE} (got '${BASE_IMAGE_DIGEST}')" >&2
  exit 1
fi
LABEL_ARGS=(
  --label "org.opencontainers.image.revision=${SOURCE_REVISION}"
  --label "org.opencontainers.image.version=${SOURCE_VERSION}"
  --label "org.opencontainers.image.base.name=${BASE_IMAGE_NAME}"
  --label "org.opencontainers.image.base.digest=${BASE_IMAGE_DIGEST}"
)

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
Base image:                ${BASE_IMAGE_NAME}@${BASE_IMAGE_DIGEST}
MSLK source repo:          ${MSLK_REPO}
MSLK source commit:        ${MSLK_COMMIT}
Image tag:                 ${IMAGE_TAG}
A1111 source:              local fork checkout (${PROJECT_ROOT})
Source revision:           ${SOURCE_REVISION}
Source version:            ${SOURCE_VERSION}
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
  "${LABEL_ARGS[@]}" \
  -f "${DOCKERFILE}" \
  -t "${IMAGE_TAG}" \
  --build-arg BASE_IMAGE="${BASE_IMAGE_NAME}@${BASE_IMAGE_DIGEST}" \
  --build-arg MSLK_REPO="${MSLK_REPO}" \
  --build-arg MSLK_COMMIT="${MSLK_COMMIT}" \
  "${PROJECT_ROOT}"
