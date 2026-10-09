#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-gb10-a1111-latest}"
DOCKER_BIN="${DOCKER_BIN:-/usr/bin/docker}"
# Seconds docker stop waits after SIGTERM before it kills the container.
STOP_TIMEOUT="${STOP_TIMEOUT:-120}"

if [[ "$(sudo "${DOCKER_BIN}" ps -a --filter "name=^/${CONTAINER_NAME}\$" --format '{{.Names}}')" == "${CONTAINER_NAME}" ]]; then
  sudo "${DOCKER_BIN}" stop -t "${STOP_TIMEOUT}" "${CONTAINER_NAME}" >/dev/null
  sudo "${DOCKER_BIN}" rm "${CONTAINER_NAME}" >/dev/null
  echo "Stopped and removed ${CONTAINER_NAME}"
else
  echo "Container ${CONTAINER_NAME} not found"
fi
