#!/usr/bin/env bash
# Phase 0b: prepare benchmark patches, then benchmark all open encoders.
# Usage: tmux new -d -s phase0b "bash scripts/run_phase0b.sh"
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-$HOME/miniforge3/envs/rcc/bin/python}"
LOG=logs/phase0b.log
mkdir -p logs

exec > >(tee -a "$LOG") 2>&1
echo "== phase0b start $(date -Is) commit $(git rev-parse --short HEAD)"

systemd-inhibit --what=idle:sleep --why="Phase 0b benchmark" \
  bash -c "\"$PY\" scripts/prepare_bench_patches.py && \"$PY\" scripts/benchmark_encoders.py"

echo "== phase0b done $(date -Is)"
