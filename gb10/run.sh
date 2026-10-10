#!/usr/bin/env bash
set -euo pipefail
# A lost terminal or a closed output pipe (SSH drop, a caller killed on timeout) must not stop a deploy halfway: the
# deploy and its rollback carry on, and everything they print is kept in DEPLOY_LOG below.
trap '' HUP PIPE

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
# Startup warm-up: replay the last generation once before production traffic (modules/openclaw_warmup.py); "off" disables.
OPENCLAW_WARMUP="${OPENCLAW_WARMUP:-generation-last}"
OPENCLAW_COMPILE_CACHE_ROOT="${OPENCLAW_COMPILE_CACHE_ROOT:-${HOST_ROOT}/Caches/compile}"
# Expandable segments: same speed and pixel-identical output, 3.3 GB lower reserved peak on the img2img workload
# (docs/gb10/notes/correctness-quality-speed-pass-2026-10-09.md); reserved memory is host RAM on unified memory.
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
# Seconds docker stop waits after SIGTERM before it kills the container.
STOP_TIMEOUT="${STOP_TIMEOUT:-120}"
# Seconds a started container gets to answer /sdapi/v1/progress (the API serves only once the startup model loaded).
READY_TIMEOUT="${READY_TIMEOUT:-900}"
# The container being replaced keeps this name, stopped, until the new one passed its checks.
PREVIOUS_CONTAINER_NAME="${CONTAINER_NAME}-previous"

# All output (stdout and stderr) goes to the caller and to DEPLOY_LOG, owned by the invoking user. The tee ignores
# HUP, INT, TERM and a failing output pipe (GNU tee --output-error=warn-nopipe keeps writing the file), so the log
# holds the whole run, rollback included, even when the caller is gone; on_exit waits for it to drain.
DEPLOY_LOG_DIR="${HOST_ROOT}/deploy-logs"
sudo install -d -o "$(id -u)" -g "$(id -g)" -m 0755 "${DEPLOY_LOG_DIR}"
DEPLOY_LOG="${DEPLOY_LOG_DIR}/run-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
echo "Deploy log: ${DEPLOY_LOG}"
exec > >(trap '' INT TERM; exec tee --output-error=warn-nopipe -a "${DEPLOY_LOG}") 2>&1
DEPLOY_LOG_TEE_PID=$!
# Closing our ends of the pipe ends the tee; waiting for it means the caller and the log have every line at exit.
close_deploy_log() {
  exec >&- 2>&-
  # The exit status is the script's; a tee that lost its terminal (and only wrote the log) must not replace it.
  wait "${DEPLOY_LOG_TEE_PID}" || true
}
trap close_deploy_log EXIT

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

# Host paths are created and written as root, so they are also tested as root: the invoking user may not see them.
if ! sudo test -e "${HOST_ROOT}/Outputs" && ! sudo test -L "${HOST_ROOT}/Outputs"; then
  sudo ln -s "${OUTPUTS_TARGET}" "${HOST_ROOT}/Outputs"
elif sudo test -L "${HOST_ROOT}/Outputs"; then
  current_target="$(sudo readlink "${HOST_ROOT}/Outputs")"
  if [[ "${current_target}" != "${OUTPUTS_TARGET}" ]]; then
    echo "ERROR: ${HOST_ROOT}/Outputs points to ${current_target}, expected ${OUTPUTS_TARGET}" >&2
    exit 1
  fi
else
  echo "ERROR: ${HOST_ROOT}/Outputs exists but is not a symlink to ${OUTPUTS_TARGET}" >&2
  exit 1
fi
# Docker cannot bind-mount a dangling link, not even to restart the replaced container, so a missing target (the NAS
# not mounted) stops the deploy here, while production still runs.
if ! sudo test -d "${HOST_ROOT}/Outputs"; then
  echo "ERROR: ${HOST_ROOT}/Outputs -> ${OUTPUTS_TARGET} is not a reachable directory (NAS not mounted?)" >&2
  exit 1
fi

sudo mkdir -p "${HOST_ROOT}/config/generation-last"
# A new settings file starts as {}: the app treats an empty one as damaged (modules/settings_file.py).
if ! sudo test -e "${HOST_ROOT}/config/config.json"; then
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

# Everything that can fail without touching the live container runs before it is stopped, so a bad IMAGE_TAG, an
# image without provenance, an unreadable driver version, an unwritable cache namespace or an extension source a
# patcher rejects leaves production running.

container_exists() {  # $1 container name
  local names
  names="$(sudo "${DOCKER_BIN}" ps -a --filter "name=^/${1}\$" --format '{{.Names}}')"
  [[ "${names}" == "$1" ]]
}

if container_exists "${PREVIOUS_CONTAINER_NAME}"; then
  echo "ERROR: ${PREVIOUS_CONTAINER_NAME} exists: an earlier deploy did not finish. It is the container that deploy" \
    "replaced; restore it (docker rename ${PREVIOUS_CONTAINER_NAME} ${CONTAINER_NAME} && docker start ${CONTAINER_NAME})" \
    "or remove it, then deploy again." >&2
  exit 1
fi
for tool in curl "${PROJECT_ROOT}/gb10/smoke-test.sh"; do
  if ! command -v "${tool}" >/dev/null; then
    echo "ERROR: the post-start checks need ${tool}" >&2
    exit 1
  fi
done

# The tag is resolved once: every later step uses the image ID, so a tag moved meanwhile cannot change what is run.
TARGET_IMAGE_ID="$(sudo "$DOCKER_BIN" image inspect "${IMAGE_TAG}" --format '{{.Id}}')"
image_label() {  # $1 label; empty when the image does not carry it
  sudo "$DOCKER_BIN" image inspect "${TARGET_IMAGE_ID}" --format "{{index .Config.Labels \"$1\"}}"
}
# The image has no .git: the version it reports comes from the provenance labels gb10/build.sh records (the commit,
# and `git describe --tags` for the infotext Version), so a rollback image reports its own version. An image built
# before those labels needs both values set explicitly; its inherited org.opencontainers.image.version is NGC's.
IMAGE_REVISION="$(image_label org.opencontainers.image.revision)"
if [[ -n "${IMAGE_REVISION}" ]]; then
  A1111_COMMIT_HASH="${A1111_COMMIT_HASH:-${IMAGE_REVISION}}"
  A1111_VERSION_TAG="${A1111_VERSION_TAG:-$(image_label org.opencontainers.image.version)}"
fi
if [[ -z "${A1111_COMMIT_HASH:-}" || -z "${A1111_VERSION_TAG:-}" ]]; then
  echo "ERROR: ${IMAGE_TAG} (${TARGET_IMAGE_ID}) carries no provenance labels (built before gb10/build.sh recorded" \
    "them); set A1111_COMMIT_HASH and A1111_VERSION_TAG to the commit and \`git describe --tags\` it was built from." >&2
  exit 1
fi

# Inductor/Triton/driver-JIT output depends on the image's compiler stack and the host driver, not on the
# A1111 commit, so key the namespace by those: app-only deploys then start with warm caches. Each cache also
# hashes its own inputs, so a shared namespace never serves a stale kernel.
if [[ -z "${OPENCLAW_COMPILE_CACHE_NAMESPACE:-}" ]]; then
  if [[ ! -r /sys/module/nvidia/version ]]; then
    echo "ERROR: cannot read the host NVIDIA driver version from /sys/module/nvidia/version" >&2
    exit 1
  fi
  IMAGE_COMPILE_STACK="$(sudo "$DOCKER_BIN" run --rm --network none --entrypoint python "${TARGET_IMAGE_ID}" -c 'import importlib.metadata as m, os; print("torch-" + m.version("torch") + "-triton-" + m.version("triton") + "-cuda-" + os.environ["CUDA_VERSION"])')"
  OPENCLAW_COMPILE_CACHE_NAMESPACE="${IMAGE_COMPILE_STACK}-driver-$(cat /sys/module/nvidia/version)"
  OPENCLAW_COMPILE_CACHE_NAMESPACE="${OPENCLAW_COMPILE_CACHE_NAMESPACE//[^a-zA-Z0-9_.-]/_}"
fi

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

# The owned extensions are mirrored from the tracked source below. The host-installed third-party extensions are
# patched in place: each patcher (gb10/patchlib.py contract) patches upstream text, upgrades the previous release's
# text or verifies already-patched text, and fails on anything else, including a missing file. $1 is the Extensions
# directory to patch.
patch_third_party_extensions() {
  local extensions_root="$1"
  if sudo test -d "${extensions_root}/multidiffusion-upscaler-for-automatic1111"; then
    sudo python3 "${PROJECT_ROOT}/gb10/patch-multidiffusion-performance.py" "${extensions_root}/multidiffusion-upscaler-for-automatic1111"
  fi
  sudo python3 "${PROJECT_ROOT}/gb10/patch-ultimate-upscale-state-lifecycle.py" "${extensions_root}/ultimate-upscale-for-automatic1111"
  sudo python3 "${PROJECT_ROOT}/gb10/patch-ultimate-upscale-subcanvas.py" "${extensions_root}/ultimate-upscale-for-automatic1111"
  sudo python3 "${PROJECT_ROOT}/gb10/patch-detail-daemon.py" "${extensions_root}/sd-webui-detail-daemon"
}

# Mirrors an owned extension tree $1 onto $2, keeping the runtime data and model weights that live inside an
# extension's own directory: openclaw-multi-sampler's saved chains (data/), ControlNet's downloaded annotator weights
# (annotator/downloads/) and ControlNet models (models/; the checkout's git-ignored copies are mirrored, a model placed
# only on the host survives). The /*** form protects the directory AND its contents: 'P /data/' alone protects only
# the directory entry, so when the checkout has its own (git-ignored, empty) data/ directory rsync descends into it and
# deletes the saved files. Tool caches are excluded and so removed from the target. rsync's size+mtime check decides
# what to copy: -a keeps the checkout's mtimes on the target, so a changed source file differs in one of them.
mirror_owned_extension() {
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
    "$1/" "$2/"
}

SCRATCH_ROOT="$(mktemp -d -t gb10-deploy.XXXXXX)"
PATCH_REHEARSAL_ROOT="${SCRATCH_ROOT}/patch-rehearsal"
EXTENSION_BACKUP_ROOT="${SCRATCH_ROOT}/extension-backup"
mkdir "${PATCH_REHEARSAL_ROOT}" "${EXTENSION_BACKUP_ROOT}"

# Rollback: from the moment the live container is stopped until the new one passed its checks (DEPLOY_PHASE=replacing),
# any failure or interruption removes the new container, puts the owned extension sources back as they were and
# restarts the replaced container (same image ID, same arguments). The third-party patches stay applied: every
# patcher accepts its patched text. Images and tags are not touched.
DEPLOY_PHASE=checks
PREVIOUS_IMAGE_ID=""
PREVIOUS_PORT="${PORT}"
NEW_OWNED_EXTENSIONS=()
NEW_CONTAINER_CREATED=0

# Waits until container $1 answers /sdapi/v1/progress on port $2; fails when it stops, restarts or times out.
wait_ready() {
  local name="$1" port="$2" deadline=$((SECONDS + READY_TIMEOUT)) state initial_restarts
  initial_restarts="$(sudo "${DOCKER_BIN}" inspect --format '{{.RestartCount}}' "${name}")" || return 1
  while (( SECONDS < deadline )); do
    state="$(sudo "${DOCKER_BIN}" inspect --format '{{.State.Status}} {{.RestartCount}}' "${name}")" || return 1
    if [[ "${state}" != "running ${initial_restarts}" ]]; then
      echo "ERROR: ${name} stopped or restarted before it was ready (status, restarts: ${state})" >&2
      return 1
    fi
    if curl -fs --max-time 10 -o /dev/null "http://127.0.0.1:${port}/sdapi/v1/progress?skip_current_image=true"; then
      return 0
    fi
    sleep 5
  done
  echo "ERROR: ${name} did not answer /sdapi/v1/progress on port ${port} within ${READY_TIMEOUT} s" >&2
  return 1
}

# Marks the rollback incomplete (rollback's local ok) and says which step failed; the remaining steps still run.
rollback_step_failed() {  # $1 what failed
  echo "ERROR: rollback: ${1} failed; continuing with the remaining steps" >&2
  ok=0
}

rollback() {
  local ok=1 extension_name
  echo "ERROR: deploy of ${IMAGE_TAG} (${TARGET_IMAGE_ID}) failed; rolling back" >&2
  if (( NEW_CONTAINER_CREATED )) && container_exists "${CONTAINER_NAME}"; then
    echo "--- last 200 log lines of the failed ${CONTAINER_NAME}:" >&2
    sudo "${DOCKER_BIN}" logs --tail 200 "${CONTAINER_NAME}" >&2 || rollback_step_failed "reading the new container's log"
    echo "---" >&2
    { sudo "${DOCKER_BIN}" stop -t "${STOP_TIMEOUT}" "${CONTAINER_NAME}" >/dev/null && sudo "${DOCKER_BIN}" rm "${CONTAINER_NAME}" >/dev/null; } \
      || rollback_step_failed "removing the new container ${CONTAINER_NAME}"
  fi
  for extension_name in "${OWNED_EXTENSIONS[@]}"; do
    if sudo test -d "${EXTENSION_BACKUP_ROOT}/${extension_name}"; then
      mirror_owned_extension "${EXTENSION_BACKUP_ROOT}/${extension_name}" "${HOST_ROOT}/Extensions/${extension_name}" \
        || rollback_step_failed "restoring the owned extension ${extension_name}"
    fi
  done
  # An owned extension this deploy added did not exist before: the replaced image never loaded it.
  for extension_name in "${NEW_OWNED_EXTENSIONS[@]}"; do
    sudo rm -rf -- "${HOST_ROOT:?}/Extensions/${extension_name:?}" || rollback_step_failed "removing the added owned extension ${extension_name}"
  done
  if [[ -n "${PREVIOUS_IMAGE_ID}" ]]; then
    # Interrupted between the stop and the rename, the replaced container still has its name.
    if { ! container_exists "${PREVIOUS_CONTAINER_NAME}" || sudo "${DOCKER_BIN}" rename "${PREVIOUS_CONTAINER_NAME}" "${CONTAINER_NAME}"; } \
      && sudo "${DOCKER_BIN}" start "${CONTAINER_NAME}" >/dev/null \
      && wait_ready "${CONTAINER_NAME}" "${PREVIOUS_PORT}"; then
      echo "Rolled back: ${CONTAINER_NAME} runs the previous image ${PREVIOUS_IMAGE_ID} again." >&2
    else
      rollback_step_failed "restarting the replaced container"
    fi
  else
    echo "No container was running before this deploy; nothing to restart." >&2
  fi
  if (( ! ok )); then
    echo "ERROR: the rollback did not complete; see the errors above." >&2
  fi
  if [[ -n "${PREVIOUS_IMAGE_ID}" && "${PREVIOUS_IMAGE_ID}" != "${TARGET_IMAGE_ID}" ]]; then
    echo "NOTE: ${IMAGE_TAG} still names the failed image ${TARGET_IMAGE_ID}, and a later run.sh or recreate would" \
      "deploy it again. To point the tag back at the image that runs: docker tag ${PREVIOUS_IMAGE_ID} ${IMAGE_TAG}" >&2
  fi
}

# The rollback and the cleanup run to the end: errexit is off, so a failing step is reported and the next one runs,
# and INT/TERM are ignored (by the commands it runs too), so a second Ctrl-C cannot leave production down.
on_exit() {
  local status=$?
  set +e
  trap '' INT TERM
  trap - EXIT
  if [[ "${DEPLOY_PHASE}" == replacing ]]; then
    rollback
    (( status )) || status=1
  fi
  sudo rm -rf -- "${SCRATCH_ROOT}" || echo "ERROR: could not remove the scratch directory ${SCRATCH_ROOT}" >&2
  close_deploy_log
  exit "${status}"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Rehearse the patchers on a scratch copy of the third-party checkouts while production still runs: nothing under
# Extensions changes before the live container is stopped, and a source a patcher rejects fails the deploy here.
echo "Rehearsing the third-party extension patchers on a scratch copy: ${PATCH_REHEARSAL_ROOT}"
for third_party_extension in multidiffusion-upscaler-for-automatic1111 ultimate-upscale-for-automatic1111 sd-webui-detail-daemon; do
  if sudo test -d "${HOST_ROOT}/Extensions/${third_party_extension}"; then
    sudo rsync -a --exclude '.git/' --exclude '__pycache__/' \
      "${HOST_ROOT}/Extensions/${third_party_extension}" "${PATCH_REHEARSAL_ROOT}/"
  fi
done
patch_third_party_extensions "${PATCH_REHEARSAL_ROOT}" >/dev/null

# Back up what the mirror replaces (the owned extension sources, without the protected data and model subtrees), so
# a rollback can put it back.
for extension_name in "${OWNED_EXTENSIONS[@]}"; do
  owned_extension_target="${HOST_ROOT}/Extensions/${extension_name}"
  if sudo test -d "${owned_extension_target}"; then
    sudo rsync -a --exclude '/data/' --exclude '/annotator/downloads/' --exclude '/models/' \
      "${owned_extension_target}/" "${EXTENSION_BACKUP_ROOT}/${extension_name}/"
  else
    NEW_OWNED_EXTENSIONS+=("${extension_name}")
  fi
done

# Stop the bind-mounted live container before mutating Extensions underneath it, and keep it for the rollback.
if container_exists "${CONTAINER_NAME}"; then
  PREVIOUS_IMAGE_ID="$(sudo "${DOCKER_BIN}" inspect --format '{{.Image}}' "${CONTAINER_NAME}")"
  PREVIOUS_PORT="$(sudo "${DOCKER_BIN}" inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "${CONTAINER_NAME}" | sed -n 's/^A1111_PORT=//p')"
  PREVIOUS_PORT="${PREVIOUS_PORT:-${PORT}}"
fi
DEPLOY_PHASE=replacing
if [[ -n "${PREVIOUS_IMAGE_ID}" ]]; then
  sudo "${DOCKER_BIN}" stop -t "${STOP_TIMEOUT}" "${CONTAINER_NAME}" >/dev/null
  sudo "${DOCKER_BIN}" rename "${CONTAINER_NAME}" "${PREVIOUS_CONTAINER_NAME}"
fi

for extension_name in "${OWNED_EXTENSIONS[@]}"; do
  sudo mkdir -p "${HOST_ROOT}/Extensions/${extension_name}"
  mirror_owned_extension "${PROJECT_ROOT}/extensions/${extension_name}" "${HOST_ROOT}/Extensions/${extension_name}"
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
  -e OPENCLAW_WARMUP="${OPENCLAW_WARMUP}"
  -e PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}"
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

# From here a container named CONTAINER_NAME is the new one: the replaced one was renamed above.
NEW_CONTAINER_CREATED=1
if ! sudo "$DOCKER_BIN" run "${DOCKER_ARGS[@]}" "${TARGET_IMAGE_ID}" >/dev/null; then
  observed_image_id="$(sudo "$DOCKER_BIN" inspect "${CONTAINER_NAME}" --format '{{.Image}}' 2>/dev/null || true)"
  observed_status="$(sudo "$DOCKER_BIN" inspect "${CONTAINER_NAME}" --format '{{.State.Status}}' 2>/dev/null || true)"
  if [[ "${observed_status}" == "running" && "${observed_image_id}" == "${TARGET_IMAGE_ID}" ]]; then
    echo "Docker run returned nonzero, but ${CONTAINER_NAME} is running target image ${TARGET_IMAGE_ID}; continuing." >&2
  else
    exit 1
  fi
fi
echo "Started ${CONTAINER_NAME} from ${IMAGE_TAG} (${TARGET_IMAGE_ID}); waiting up to ${READY_TIMEOUT} s for the API"

wait_ready "${CONTAINER_NAME}" "${PORT}"
CONTAINER_NAME="${CONTAINER_NAME}" PORT="${PORT}" DOCKER_BIN="${DOCKER_BIN}" "${PROJECT_ROOT}/gb10/smoke-test.sh"

DEPLOY_PHASE=deployed
if [[ -n "${PREVIOUS_IMAGE_ID}" ]] && ! sudo "${DOCKER_BIN}" rm "${PREVIOUS_CONTAINER_NAME}" >/dev/null; then
  echo "ERROR: ${CONTAINER_NAME} is deployed and passed its checks, but the replaced container" \
    "${PREVIOUS_CONTAINER_NAME} could not be removed; remove it before the next deploy." >&2
  exit 1
fi

echo "Deployed ${CONTAINER_NAME} from ${IMAGE_TAG} (${TARGET_IMAGE_ID}); replaced image: ${PREVIOUS_IMAGE_ID:-none}"
echo "Version: ${A1111_VERSION_TAG} (commit ${A1111_COMMIT_HASH})"
echo "CPU set: ${CPUSET_CPUS}"
echo "Host data root: ${HOST_ROOT}"
echo "Outputs symlink target: ${OUTPUTS_TARGET}"
echo "OpenClaw SDPA backend: ${OPENCLAW_SDPA_BACKEND}"
echo "OpenClaw CUDA graphs: ${OPENCLAW_CUDA_GRAPHS} cache=${OPENCLAW_CUDA_GRAPH_CACHE_MAX}"
echo "OpenClaw VAE decode graphs: ${OPENCLAW_VAE_DECODE_GRAPHS} cache=${OPENCLAW_VAE_DECODE_GRAPH_CACHE_MAX}"
echo "OpenClaw startup warm-up: ${OPENCLAW_WARMUP}"
echo "CUDA allocator: PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF}"
echo "Compile/kernel cache namespace: ${OPENCLAW_COMPILE_CACHE_NAMESPACE} root=${OPENCLAW_COMPILE_CACHE_ROOT}"
echo "API: http://<GB10-LAN-IP>:${PORT}/sdapi/v1/progress (host networking)"
echo "Browser UI has been removed; this image is API/headless only."
