#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
mkdir -p logs

log_path="logs/e1-marian-small-16k-raw-identity15.log"
pid_path="logs/e1-marian-small-16k-raw-identity15.pid"

if [[ -f "$pid_path" ]] && kill -0 "$(cat "$pid_path")" 2>/dev/null; then
  echo "training is already running with PID $(cat "$pid_path")"
  exit 1
fi

nohup python3 -u scripts/train_marian.py \
  --config configs/train/pilot-e1.yaml \
  >"$log_path" 2>&1 &
training_pid=$!
echo "$training_pid" >"$pid_path"
echo "started PID $training_pid; log: $log_path"

