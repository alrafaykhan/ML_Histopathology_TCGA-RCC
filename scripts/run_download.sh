#!/usr/bin/env bash
# Download one slide batch in the background.
# Usage: tmux new -d -s dl001 "bash scripts/run_download.sh 1"
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-$HOME/miniforge3/envs/rcc/bin/python}"
BATCH="$1"
LOG=$(printf "logs/download_batch_%03d.log" "$BATCH")
mkdir -p logs

exec > >(tee -a "$LOG") 2>&1
echo "== download batch $BATCH start $(date -Is) commit $(git rev-parse --short HEAD)"
"$PY" -u scripts/download_batch.py --batch "$BATCH"
echo "== download batch $BATCH done $(date -Is)"
