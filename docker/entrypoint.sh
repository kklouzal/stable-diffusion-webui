#!/usr/bin/env bash
set -euo pipefail

A1111_HOME="${A1111_HOME:-/opt/stable-diffusion-webui}"
A1111_RUN_AS_USER="${A1111_RUN_AS_USER:-a1111}"

mkdir -p "$A1111_HOME/tmp" "$A1111_HOME/models/ControlNet" "$A1111_HOME/models/VAE-approx"

# Only a missing config.json starts as {}; an existing one is never rewritten here. gb10/run.sh seeds the host file
# with {}, so an empty file is a damaged one (a save cut short): the app reports it, keeps a copy under tmp/ and resets
# it to {} (modules/settings_file.py) instead of this script resetting the settings without a word.
if [[ ! -e "$A1111_HOME/config.json" ]]; then
  printf '{}\n' > "$A1111_HOME/config.json"
fi

if [[ ! -e "$A1111_HOME/styles.csv" ]]; then
  : > "$A1111_HOME/styles.csv"
fi

chown -R "$A1111_RUN_AS_USER:$A1111_RUN_AS_USER" \
  "$A1111_HOME/tmp" \
  "$A1111_HOME/models/ControlNet" \
  "$A1111_HOME/models/VAE-approx"
chown "$A1111_RUN_AS_USER:$A1111_RUN_AS_USER" \
  "$A1111_HOME/config.json" \
  "$A1111_HOME/styles.csv" || true

cd "$A1111_HOME"

if [[ $# -gt 0 ]]; then
  exec gosu "$A1111_RUN_AS_USER:$A1111_RUN_AS_USER" "$@"
fi

# An empty COMMANDLINE_ARGS selects gb10-a1111-launch's API-only default (--nowebui --api ...).
echo "Starting container-owned A1111 launch as ${A1111_RUN_AS_USER} with COMMANDLINE_ARGS=${COMMANDLINE_ARGS:-<launcher default>}"
exec gosu "$A1111_RUN_AS_USER:$A1111_RUN_AS_USER" /usr/local/bin/gb10-a1111-launch
