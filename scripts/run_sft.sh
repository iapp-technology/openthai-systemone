#!/usr/bin/env bash
# Launch Stage-2 decision-head SFT detached on one GPU.   scripts/run_sft.sh <gpu> [extra args...]
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${1:-0}; shift || true
source .venv/bin/activate
mkdir -p runs/logs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU setsid nohup python -u scripts/04_decision_train.py --config configs/sft.yaml "$@" \
  > runs/logs/sft_train.log 2>&1 < /dev/null &
sleep 2; pgrep -fa "04_decision_train" | cut -c1-120
