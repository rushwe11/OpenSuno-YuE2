#!/bin/bash
set -euo pipefail

OPEN_SUNO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
EXISTING_YUE2="${OPENSUNO_YUE2_ROOT:-$(cd "$OPEN_SUNO_DIR/.." && pwd)}"
PORT="${YUE2_PORT:-7864}"

cd "$OPEN_SUNO_DIR"
if [[ ! -x .venv/bin/python ]]; then
  echo "OpenSuno's Studio environment is missing. Run scripts/Install OpenSuno.command once."
  exit 1
fi
if [[ ! -x "$EXISTING_YUE2/.venv/bin/python" \
   || ! -f "$EXISTING_YUE2/models/YuE2-3B/model.safetensors" \
   || ! -f "$EXISTING_YUE2/models/YuE2-Vae/model.safetensors" ]]; then
  echo "No complete YuE2 installation found at: $EXISTING_YUE2"
  echo "Set OPENSUNO_YUE2_ROOT to the project containing .venv and models/."
  exit 1
fi

echo "Refreshing imported YuE2 songs…"
.venv/bin/python app/import_yue2.py --source "$EXISTING_YUE2" --library library

export OPENSUNO_YUE2_ROOT="$EXISTING_YUE2"
export YUE2_PORT="$PORT"
echo "Opening OpenSuno with Existing YuE2 MPS at http://127.0.0.1:$PORT"
exec .venv/bin/python app/server.py
