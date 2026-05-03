#!/usr/bin/env bash
set -euo pipefail

A1111_HOME="${A1111_HOME:-/opt/stable-diffusion-webui}"
A1111_PORT="${A1111_PORT:-7860}"
COMMANDLINE_ARGS="${COMMANDLINE_ARGS:---listen --port ${A1111_PORT} --no-hashing --disable-console-progressbars --api --opt-sdp-attention --opt-channelslast --enable-insecure-extension-access}"

cd "$A1111_HOME"
export COMMANDLINE_ARGS

# Container-owned launch path:
# - do NOT use upstream webui.sh for env/bootstrap in this image
# - do NOT create/manage an upstream runtime venv here
# - do use the image-owned Python environment prepared during docker build
#
# A1111 appends COMMANDLINE_ARGS to sys.argv from modules/paths_internal.py.
# Do not also pass those flags here or launch logs and parsed options contain
# duplicate --listen/--port/API/attention flags.
exec python launch.py \
  --skip-prepare-environment \
  --skip-python-version-check
