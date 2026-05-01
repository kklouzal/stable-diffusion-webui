#!/usr/bin/env bash
set -euo pipefail

A1111_HOME="${A1111_HOME:-/opt/stable-diffusion-webui}"
A1111_PORT="${A1111_PORT:-7860}"
A1111_RUN_AS_USER="${A1111_RUN_AS_USER:-a1111}"
COMMANDLINE_ARGS="${COMMANDLINE_ARGS:---listen --port ${A1111_PORT}}"

mkdir -p "$A1111_HOME/tmp"

for f in "$A1111_HOME/config.json" "$A1111_HOME/ui-config.json"; do
  if [[ ! -e "$f" || ! -s "$f" ]]; then
    printf '{}\n' > "$f"
  fi
done

if [[ ! -e "$A1111_HOME/styles.csv" ]]; then
  : > "$A1111_HOME/styles.csv"
fi

chown -R "$A1111_RUN_AS_USER:$A1111_RUN_AS_USER" "$A1111_HOME/tmp"
chown "$A1111_RUN_AS_USER:$A1111_RUN_AS_USER" \
  "$A1111_HOME/config.json" \
  "$A1111_HOME/ui-config.json" \
  "$A1111_HOME/styles.csv" || true

cd "$A1111_HOME"
export COMMANDLINE_ARGS

if [[ $# -gt 0 ]]; then
  exec gosu "$A1111_RUN_AS_USER:$A1111_RUN_AS_USER" "$@"
fi

echo "Starting container-owned A1111 launch as ${A1111_RUN_AS_USER} with COMMANDLINE_ARGS=${COMMANDLINE_ARGS}"
exec gosu "$A1111_RUN_AS_USER:$A1111_RUN_AS_USER" /usr/local/bin/gb10-a1111-launch
