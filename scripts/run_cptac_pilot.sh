#!/usr/bin/env bash
# Download the CPTAC-CCRCC pilot sample in the background.
# Usage: tmux new -d -s cptac "bash scripts/run_cptac_pilot.sh"
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-$HOME/miniforge3/envs/rcc/bin/python}"
LOG=logs/cptac_pilot.log
mkdir -p logs

exec > >(tee -a "$LOG") 2>&1
echo "== cptac pilot start $(date -Is) commit $(git rev-parse --short HEAD)"
"$PY" -u scripts/cptac_pilot.py "$@"
echo "== cptac pilot done $(date -Is)"
