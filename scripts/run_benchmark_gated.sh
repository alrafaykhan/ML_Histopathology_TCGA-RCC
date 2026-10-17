#!/usr/bin/env bash
# Phase 0b follow-up: benchmark the gated encoders and the two that failed in the first run.
# Usage: tmux new -d -s bench "bash scripts/run_benchmark_gated.sh"
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-$HOME/miniforge3/envs/rcc/bin/python}"
LOG=logs/phase0b.log
mkdir -p logs
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"

exec > >(tee -a "$LOG") 2>&1
echo "== phase0b gated start $(date -Is) commit $(git rev-parse --short HEAD)"
"$PY" -u scripts/benchmark_encoders.py --encoders \
  ctranspath conch h0_mini uni virchow2 uni2_h h_optimus_1 prov_gigapath midnight12k
echo "== phase0b gated done $(date -Is)"
