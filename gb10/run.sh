#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-local/gb10-a1111:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-gb10-a1111-latest}"
HOST_ROOT="${HOST_ROOT:-/opt/gb10/stable-diffusion}"
PORT="${PORT:-7860}"
OUTPUTS_TARGET="${OUTPUTS_TARGET:-/mnt/nas-warehouse/StableDiffusion/Outputs}"
DOCKER_BIN="${DOCKER_BIN:-/usr/bin/docker}"
CPUSET_CPUS="${CPUSET_CPUS:-5-9,15-19}"
OPENCLAW_SDPA_BACKEND="${OPENCLAW_SDPA_BACKEND:-cudnn,flash,efficient,math}"
OPENCLAW_CUDA_GRAPHS="${OPENCLAW_CUDA_GRAPHS:-1}"
OPENCLAW_CUDA_GRAPH_CACHE_MAX="${OPENCLAW_CUDA_GRAPH_CACHE_MAX:-8}"
OPENCLAW_VAE_DECODE_GRAPHS="${OPENCLAW_VAE_DECODE_GRAPHS:-1}"
OPENCLAW_VAE_DECODE_GRAPH_CACHE_MAX="${OPENCLAW_VAE_DECODE_GRAPH_CACHE_MAX:-4}"
OPENCLAW_COMPILE_CACHE_ROOT="${OPENCLAW_COMPILE_CACHE_ROOT:-${HOST_ROOT}/Caches/compile}"

# Host directory under HOST_ROOT -> path under the container's app directory. One table drives the host mkdir,
# the ownership repair and the bind mounts, in this order. Caches/app holds the app's cache/ (hash and metadata
# caches, and TORCH_HOME/HF_HOME downloads, which the image places under cache/); the compile cache mount below nests
# inside it at cache/compile.
HOST_DIR_MOUNTS=(
  BLIP:models/BLIP
  CLIP:models/CLIP
  Codeformer:models/Codeformer
  GFPGAN:models/GFPGAN
  Hypernetworks:models/hypernetworks
  karlo:models/karlo
  Lora:models/Lora
  RealESGRAN:models/ESRGAN
  torch_deepdanbooru:models/torch_deepdanbooru
  VAE:models/VAE
  VAE-approx:models/VAE-approx
  Embeddings:embeddings
  Extensions:extensions
  Models:models/Stable-diffusion
  Caches/app:cache
)
HOST_DIRS=("${HOST_DIR_MOUNTS[@]%%:*}" config)

for d in "${HOST_DIRS[@]}"; do
  sudo mkdir -p "${HOST_ROOT}/${d}"
done

if [[ ! -e "${HOST_ROOT}/Outputs" ]]; then
  sudo ln -s "${OUTPUTS_TARGET}" "${HOST_ROOT}/Outputs"
elif [[ -L "${HOST_ROOT}/Outputs" ]]; then
  current_target="$(readlink "${HOST_ROOT}/Outputs")"
  if [[ "${current_target}" != "${OUTPUTS_TARGET}" ]]; then
    echo "ERROR: ${HOST_ROOT}/Outputs points to ${current_target}, expected ${OUTPUTS_TARGET}" >&2
    exit 1
  fi
else
  echo "ERROR: ${HOST_ROOT}/Outputs exists but is not a symlink to ${OUTPUTS_TARGET}" >&2
  exit 1
fi

sudo mkdir -p "${HOST_ROOT}/config/generation-last"
# A new settings file starts as {}: the app treats an empty one as damaged (modules/settings_file.py).
if [[ ! -e "${HOST_ROOT}/config/config.json" ]]; then
  printf '{}\n' | sudo tee "${HOST_ROOT}/config/config.json" >/dev/null
fi
sudo touch "${HOST_ROOT}/config/styles.csv"

OWNED_EXTENSIONS=()
while IFS= read -r -d "" extension_path; do
  OWNED_EXTENSIONS+=("$(basename "${extension_path}")")
done < <(find "${PROJECT_ROOT}/extensions" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

if [[ ${#OWNED_EXTENSIONS[@]} -eq 0 ]]; then
  echo "ERROR: no owned extensions discovered under ${PROJECT_ROOT}/extensions" >&2
  exit 1
fi

printf "Discovered owned extensions:"
printf " %s" "${OWNED_EXTENSIONS[@]}"
printf "\n"

# Everything that can fail without touching the live container runs before it is removed, so a bad IMAGE_TAG, an
# unreadable driver version, an unwritable cache namespace or an extension source a patcher rejects leaves
# production running.

A1111_COMMIT_HASH="${A1111_COMMIT_HASH:-$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || true)}"
# Inductor/Triton/driver-JIT output depends on the image's compiler stack and the host driver, not on the
# A1111 commit, so key the namespace by those: app-only deploys then start with warm caches. Each cache also
# hashes its own inputs, so a shared namespace never serves a stale kernel.
if [[ -z "${OPENCLAW_COMPILE_CACHE_NAMESPACE:-}" ]]; then
  if [[ ! -r /sys/module/nvidia/version ]]; then
    echo "ERROR: cannot read the host NVIDIA driver version from /sys/module/nvidia/version" >&2
    exit 1
  fi
  IMAGE_COMPILE_STACK="$(sudo "$DOCKER_BIN" run --rm --network none --entrypoint python "${IMAGE_TAG}" -c 'import importlib.metadata as m, os; print("torch-" + m.version("torch") + "-triton-" + m.version("triton") + "-cuda-" + os.environ["CUDA_VERSION"])')"
  OPENCLAW_COMPILE_CACHE_NAMESPACE="${IMAGE_COMPILE_STACK}-driver-$(cat /sys/module/nvidia/version)"
  OPENCLAW_COMPILE_CACHE_NAMESPACE="${OPENCLAW_COMPILE_CACHE_NAMESPACE//[^a-zA-Z0-9_.-]/_}"
fi
A1111_VERSION_TAG="${A1111_VERSION_TAG:-$(git -C "${PROJECT_ROOT}" describe --tags 2>/dev/null || true)}"

# Namespace creation is a host-side ownership boundary. install -d is idempotent,
# preserves namespace contents, repairs only directory metadata, and prevents a
# root-created first launch from leaving runtime UID/GID 2323 unable to compile.
COMPILE_CACHE_NAMESPACE_PATHS=(
  "${OPENCLAW_COMPILE_CACHE_ROOT}/torchinductor/${OPENCLAW_COMPILE_CACHE_NAMESPACE}"
  "${OPENCLAW_COMPILE_CACHE_ROOT}/triton/${OPENCLAW_COMPILE_CACHE_NAMESPACE}"
  "${OPENCLAW_COMPILE_CACHE_ROOT}/cuda/${OPENCLAW_COMPILE_CACHE_NAMESPACE}"
)
sudo install -d -o 2323 -g 2323 -m 0750 "${OPENCLAW_COMPILE_CACHE_ROOT}"
for cache_namespace_path in "${COMPILE_CACHE_NAMESPACE_PATHS[@]}"; do
  sudo install -d -o 2323 -g 2323 -m 0750 "${cache_namespace_path}"
  if ! sudo setpriv --reuid=2323 --regid=2323 --clear-groups test -w "${cache_namespace_path}"; then
    echo "ERROR: compile cache namespace is not writable by runtime UID/GID 2323: ${cache_namespace_path}" >&2
    exit 1
  fi
done
# Namespaces for other stacks are kept, never pruned here; report what they hold so they can be removed by hand.
mapfile -d '' -t OTHER_COMPILE_CACHE_NAMESPACES < <(sudo find "${OPENCLAW_COMPILE_CACHE_ROOT}/torchinductor" "${OPENCLAW_COMPILE_CACHE_ROOT}/triton" "${OPENCLAW_COMPILE_CACHE_ROOT}/cuda" -mindepth 1 -maxdepth 1 -type d ! -name "${OPENCLAW_COMPILE_CACHE_NAMESPACE}" -print0)
if (( ${#OTHER_COMPILE_CACHE_NAMESPACES[@]} )); then
  echo "Compile cache: ${#OTHER_COMPILE_CACHE_NAMESPACES[@]} other namespace dirs hold $(sudo du -csh "${OTHER_COMPILE_CACHE_NAMESPACES[@]}" | tail -n 1 | cut -f 1) under ${OPENCLAW_COMPILE_CACHE_ROOT} (kept)"
fi

TARGET_IMAGE_ID="$(sudo "$DOCKER_BIN" image inspect "${IMAGE_TAG}" --format '{{.Id}}')"

# The owned extensions are mirrored from the tracked source below. The host-installed third-party extensions are
# patched in place: each patcher (gb10/patchlib.py contract) patches upstream text or verifies already-patched text,
# and fails on anything else, including a missing file. $1 is the Extensions directory to patch.
patch_third_party_extensions() {
  local extensions_root="$1"
  if [[ -d "${extensions_root}/multidiffusion-upscaler-for-automatic1111" ]]; then
    sudo python3 "${PROJECT_ROOT}/gb10/patch-multidiffusion-performance.py" "${extensions_root}/multidiffusion-upscaler-for-automatic1111"
  fi
  sudo python3 "${PROJECT_ROOT}/gb10/patch-ultimate-upscale-state-lifecycle.py" "${extensions_root}/ultimate-upscale-for-automatic1111"
  sudo python3 "${PROJECT_ROOT}/gb10/patch-ultimate-upscale-subcanvas.py" "${extensions_root}/ultimate-upscale-for-automatic1111"
}

# Rehearse the patchers on a scratch copy of the third-party checkouts while production still runs: nothing under
# Extensions changes before the live container is gone, and a source a patcher rejects fails the deploy here.
PATCH_REHEARSAL_ROOT="$(mktemp -d -t gb10-patch-rehearsal.XXXXXX)"
trap 'sudo rm -rf -- "${PATCH_REHEARSAL_ROOT}"' EXIT
echo "Rehearsing the third-party extension patchers on a scratch copy: ${PATCH_REHEARSAL_ROOT}"
for third_party_extension in multidiffusion-upscaler-for-automatic1111 ultimate-upscale-for-automatic1111; do
  if [[ -d "${HOST_ROOT}/Extensions/${third_party_extension}" ]]; then
    sudo rsync -a --exclude '.git/' --exclude '__pycache__/' \
      "${HOST_ROOT}/Extensions/${third_party_extension}" "${PATCH_REHEARSAL_ROOT}/"
  fi
done
patch_third_party_extensions "${PATCH_REHEARSAL_ROOT}" >/dev/null

# Stop the bind-mounted live container before mutating Extensions underneath it.
sudo "${DOCKER_BIN}" rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

for extension_name in "${OWNED_EXTENSIONS[@]}"; do
  owned_extension_source="${PROJECT_ROOT}/extensions/${extension_name}"
  owned_extension_target="${HOST_ROOT}/Extensions/${extension_name}"
  sudo mkdir -p "${owned_extension_target}"
  # Mirror the repo source, but keep runtime data and model weights that live inside an extension's own
  # directory: openclaw-multi-sampler's saved chains (data/), ControlNet's downloaded annotator weights
  # (annotator/downloads/) and ControlNet models (models/; the checkout's git-ignored copies are mirrored, a model
  # placed only on the host survives). The /*** form protects the directory AND its contents: 'P /data/' alone
  # protects only the directory entry, so when the checkout has its own (git-ignored, empty) data/ directory rsync
  # descends into it and deletes the saved files. Tool caches are excluded and so removed from the target.
  # rsync's size+mtime check decides what to copy: -a keeps the checkout's mtimes on the target, so a changed source
  # file differs in one of them.
  sudo rsync -a --delete --delete-excluded \
    --filter 'P /data/***' \
    --filter 'P /annotator/downloads/***' \
    --filter 'P /models/***' \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    --exclude '.ruff_cache/' \
    --exclude '.pytest_cache/' \
    "${owned_extension_source}/" "${owned_extension_target}/"
done

patch_third_party_extensions "${HOST_ROOT}/Extensions"

sudo chown -R 2323:2323 "${HOST_DIRS[@]/#/${HOST_ROOT}/}"

DOCKER_ARGS=(
  -d
  --init
  --name "${CONTAINER_NAME}"
  --cpuset-cpus "${CPUSET_CPUS}"
  --restart unless-stopped
  --gpus all
  --network host
  --ipc host
  -e A1111_PORT="${PORT}"
  -e A1111_COMMIT_HASH="${A1111_COMMIT_HASH}"
  -e A1111_VERSION_TAG="${A1111_VERSION_TAG}"
  # Empty selects the image launcher's API-only default flags (gb10-a1111-launch), with --port A1111_PORT.
  -e COMMANDLINE_ARGS="${COMMANDLINE_ARGS:-}"
  -e OPENCLAW_SDPA_BACKEND="${OPENCLAW_SDPA_BACKEND}"
  -e OPENCLAW_CUDA_GRAPHS="${OPENCLAW_CUDA_GRAPHS}"
  -e OPENCLAW_CUDA_GRAPH_CACHE_MAX="${OPENCLAW_CUDA_GRAPH_CACHE_MAX}"
  -e OPENCLAW_VAE_DECODE_GRAPHS="${OPENCLAW_VAE_DECODE_GRAPHS}"
  -e OPENCLAW_VAE_DECODE_GRAPH_CACHE_MAX="${OPENCLAW_VAE_DECODE_GRAPH_CACHE_MAX}"
  -e TORCHINDUCTOR_CACHE_DIR="/opt/stable-diffusion-webui/cache/compile/torchinductor/${OPENCLAW_COMPILE_CACHE_NAMESPACE}"
  -e TRITON_CACHE_DIR="/opt/stable-diffusion-webui/cache/compile/triton/${OPENCLAW_COMPILE_CACHE_NAMESPACE}"
  -e CUDA_CACHE_PATH="/opt/stable-diffusion-webui/cache/compile/cuda/${OPENCLAW_COMPILE_CACHE_NAMESPACE}"
  -e GENERATION_LAST_DIR="/opt/stable-diffusion-webui/generation-last"
)
for mount in "${HOST_DIR_MOUNTS[@]}"; do
  DOCKER_ARGS+=(-v "${HOST_ROOT}/${mount%%:*}:/opt/stable-diffusion-webui/${mount#*:}")
done
DOCKER_ARGS+=(
  -v "${HOST_ROOT}/Outputs:/opt/stable-diffusion-webui/outputs"
  -v "${OPENCLAW_COMPILE_CACHE_ROOT}:/opt/stable-diffusion-webui/cache/compile"
  -v "${HOST_ROOT}/config/config.json:/opt/stable-diffusion-webui/config.json"
  -v "${HOST_ROOT}/config/styles.csv:/opt/stable-diffusion-webui/styles.csv"
  -v "${HOST_ROOT}/config/generation-last:/opt/stable-diffusion-webui/generation-last"
)

if ! sudo "$DOCKER_BIN" run "${DOCKER_ARGS[@]}" \
  "${IMAGE_TAG}"; then
  observed_image_id="$(sudo "$DOCKER_BIN" inspect "${CONTAINER_NAME}" --format '{{.Image}}' 2>/dev/null || true)"
  observed_status="$(sudo "$DOCKER_BIN" inspect "${CONTAINER_NAME}" --format '{{.State.Status}}' 2>/dev/null || true)"
  if [[ "${observed_status}" == "running" && "${observed_image_id}" == "${TARGET_IMAGE_ID}" ]]; then
    echo "Docker run returned nonzero, but ${CONTAINER_NAME} is running target image ${TARGET_IMAGE_ID}; continuing." >&2
  else
    exit 1
  fi
fi

echo "Started ${CONTAINER_NAME} from ${IMAGE_TAG}"
echo "CPU set: ${CPUSET_CPUS}"
echo "Host data root: ${HOST_ROOT}"
echo "Outputs symlink target: ${OUTPUTS_TARGET}"
echo "OpenClaw SDPA backend: ${OPENCLAW_SDPA_BACKEND}"
echo "OpenClaw CUDA graphs: ${OPENCLAW_CUDA_GRAPHS} cache=${OPENCLAW_CUDA_GRAPH_CACHE_MAX}"
echo "OpenClaw VAE decode graphs: ${OPENCLAW_VAE_DECODE_GRAPHS} cache=${OPENCLAW_VAE_DECODE_GRAPH_CACHE_MAX}"
echo "Compile/kernel cache namespace: ${OPENCLAW_COMPILE_CACHE_NAMESPACE} root=${OPENCLAW_COMPILE_CACHE_ROOT}"
echo "API expectation: http://<GB10-LAN-IP>:${PORT}/sdapi/v1/progress (host networking)"
echo "Browser UI has been removed; this image is API/headless only."
