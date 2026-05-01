#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-local/gb10-a1111:sageattention-check}"
CONTAINER_NAME="${CONTAINER_NAME:-gb10-a1111}"
HOST_ROOT="${HOST_ROOT:-/opt/gb10/stable-diffusion}"
PORT="${PORT:-7860}"
OUTPUTS_TARGET="${OUTPUTS_TARGET:-/mnt/nas-warehouse/StableDiffusion/Outputs}"
DOCKER_BIN="${DOCKER_BIN:-/usr/bin/docker}"

LOCAL_DIRS=(
  BLIP
  CLIP
  Codeformer
  deepbooru
  GFPGAN
  Hypernetworks
  karlo
  Lora
  RealESGRAN
  torch_deepdanbooru
  VAE
  VAE-approx
  Embeddings
  Extensions
  Models
  config
)

for d in "${LOCAL_DIRS[@]}"; do
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

sudo touch "${HOST_ROOT}/config/config.json" \
           "${HOST_ROOT}/config/ui-config.json" \
           "${HOST_ROOT}/config/styles.csv"

OWNED_EXTENSION_SOURCE="${PROJECT_ROOT}/extensions/sd-webui-incantations"
OWNED_EXTENSION_TARGET="${HOST_ROOT}/Extensions/sd-webui-incantations"
SUPERSEDED_DYNTHRES_TARGET="${HOST_ROOT}/Extensions/sd-dynamic-thresholding"

if [[ ! -d "${OWNED_EXTENSION_SOURCE}" ]]; then
  echo "ERROR: owned extension source missing: ${OWNED_EXTENSION_SOURCE}" >&2
  exit 1
fi

# Stop the bind-mounted live container before mutating Extensions underneath it.
sudo "${DOCKER_BIN}" rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

sudo mkdir -p "${OWNED_EXTENSION_TARGET}"
sudo rsync -a --delete --delete-excluded \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  "${OWNED_EXTENSION_SOURCE}/" "${OWNED_EXTENSION_TARGET}/"
# Dynamic Thresholding / CFG-Fix is now vendored inside the owned Incantations extension.
# Remove the old standalone checkout so A1111 does not load duplicate CFG-Fix scripts.
sudo rm -rf "${SUPERSEDED_DYNTHRES_TARGET}"
if [[ -e "${SUPERSEDED_DYNTHRES_TARGET}" ]]; then
  echo "ERROR: failed to remove superseded extension: ${SUPERSEDED_DYNTHRES_TARGET}" >&2
  exit 1
fi

sudo chown -R 2323:2323 \
  "${HOST_ROOT}/BLIP" \
  "${HOST_ROOT}/CLIP" \
  "${HOST_ROOT}/Codeformer" \
  "${HOST_ROOT}/deepbooru" \
  "${HOST_ROOT}/GFPGAN" \
  "${HOST_ROOT}/Hypernetworks" \
  "${HOST_ROOT}/karlo" \
  "${HOST_ROOT}/Lora" \
  "${HOST_ROOT}/RealESGRAN" \
  "${HOST_ROOT}/torch_deepdanbooru" \
  "${HOST_ROOT}/VAE" \
  "${HOST_ROOT}/VAE-approx" \
  "${HOST_ROOT}/Embeddings" \
  "${HOST_ROOT}/Extensions" \
  "${HOST_ROOT}/Models" \
  "${HOST_ROOT}/config"

DOCKER_ARGS=(
  -d
  --init
  --name "${CONTAINER_NAME}"
  --restart unless-stopped
  --gpus all
  --network host
  --ipc host
  -e A1111_PORT="${PORT}"
  -e COMMANDLINE_ARGS="${COMMANDLINE_ARGS:---listen --port ${PORT} --no-hashing --disable-console-progressbars --api --opt-sdp-attention --opt-channelslast --enable-insecure-extension-access}"
  --tmpfs "/data/config:rw,noexec,nosuid,nodev,size=16m"
  --tmpfs "/data/embeddings:rw,noexec,nosuid,nodev,size=16m"
  --tmpfs "/data/extensions:rw,noexec,nosuid,nodev,size=16m"
  --tmpfs "/data/models:rw,noexec,nosuid,nodev,size=64m"
  --tmpfs "/data/outputs:rw,noexec,nosuid,nodev,size=16m"
  -v "${HOST_ROOT}/BLIP:/opt/stable-diffusion-webui/models/BLIP"
  -v "${HOST_ROOT}/CLIP:/opt/stable-diffusion-webui/models/CLIP"
  -v "${HOST_ROOT}/Codeformer:/opt/stable-diffusion-webui/models/Codeformer"
  -v "${HOST_ROOT}/deepbooru:/opt/stable-diffusion-webui/models/deepbooru"
  -v "${HOST_ROOT}/GFPGAN:/opt/stable-diffusion-webui/models/GFPGAN"
  -v "${HOST_ROOT}/Hypernetworks:/opt/stable-diffusion-webui/models/hypernetworks"
  -v "${HOST_ROOT}/karlo:/opt/stable-diffusion-webui/models/karlo"
  -v "${HOST_ROOT}/Lora:/opt/stable-diffusion-webui/models/Lora"
  -v "${HOST_ROOT}/RealESGRAN:/opt/stable-diffusion-webui/models/ESRGAN"
  -v "${HOST_ROOT}/torch_deepdanbooru:/opt/stable-diffusion-webui/models/torch_deepdanbooru"
  -v "${HOST_ROOT}/VAE:/opt/stable-diffusion-webui/models/VAE"
  -v "${HOST_ROOT}/VAE-approx:/opt/stable-diffusion-webui/models/VAE-approx"
  -v "${HOST_ROOT}/Embeddings:/opt/stable-diffusion-webui/embeddings"
  -v "${HOST_ROOT}/Extensions:/opt/stable-diffusion-webui/extensions"
  -v "${HOST_ROOT}/Models:/opt/stable-diffusion-webui/models/Stable-diffusion"
  -v "${HOST_ROOT}/Outputs:/opt/stable-diffusion-webui/outputs"
  -v "${HOST_ROOT}/config/config.json:/opt/stable-diffusion-webui/config.json"
  -v "${HOST_ROOT}/config/ui-config.json:/opt/stable-diffusion-webui/ui-config.json"
  -v "${HOST_ROOT}/config/styles.csv:/opt/stable-diffusion-webui/styles.csv"
)

sudo "$DOCKER_BIN" run "${DOCKER_ARGS[@]}" \
  "${IMAGE_TAG}"

echo "Started ${CONTAINER_NAME} from ${IMAGE_TAG}"
echo "Host data root: ${HOST_ROOT}"
echo "Outputs symlink target: ${OUTPUTS_TARGET}"
echo "Web UI expectation: http://<GB10-LAN-IP>:${PORT} (host networking)"
