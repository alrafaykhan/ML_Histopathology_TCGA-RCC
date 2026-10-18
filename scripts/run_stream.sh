#!/usr/bin/env bash
# Streaming pipeline: download batch b, then tile and export it while batch b+1 downloads.
# Raw WSIs of a batch are deleted once its shards are verified (process_batch.py).
# Usage: tmux new -d -s stream "bash scripts/run_stream.sh 2 19"
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-$HOME/miniforge3/envs/rcc/bin/python}"
FIRST="$1"
LAST="$2"
mkdir -p logs
exec > >(tee -a "logs/stream_${FIRST}_${LAST}.log") 2>&1
echo "== stream batches $FIRST-$LAST start $(date -Is) commit $(git rev-parse --short HEAD)"

proc_pid=""
for b in $(seq "$FIRST" "$LAST"); do
  tag=$(printf "%03d" "$b")
  echo "$(date -Is) download batch $b"
  # While the previous batch is still being tiled, wait for its raw slides to be deleted.
  wait_args=()
  [ -n "$proc_pid" ] && kill -0 "$proc_pid" 2>/dev/null && wait_args=(--wait-for-raw-h 6)
  if ! "$PY" -u scripts/download_batch.py --batch "$b" "${wait_args[@]}" >> "logs/download_batch_$tag.log" 2>&1; then
    echo "$(date -Is) download batch $b failed; see logs/download_batch_$tag.log"
    break
  fi
  if [ -n "$proc_pid" ] && ! wait "$proc_pid"; then
    echo "$(date -Is) processing of the previous batch failed; stopping"
    exit 1
  fi
  echo "$(date -Is) process batch $b"
  "$PY" -u scripts/process_batch.py --batch "$b" >> "logs/process_batch_$tag.log" 2>&1 &
  proc_pid=$!
done
[ -n "$proc_pid" ] && wait "$proc_pid"
"$PY" scripts/qc_report.py
echo "== stream done $(date -Is)"
