#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-gb10-a1111-latest}"

if sudo docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  sudo docker rm -f "${CONTAINER_NAME}"
  echo "Removed ${CONTAINER_NAME}"
else
  echo "Container ${CONTAINER_NAME} not found"
fi
