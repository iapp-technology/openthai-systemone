#!/usr/bin/env bash
# Launch the Stage-1 CPT run detached on one GPU.   scripts/run_cpt.sh <gpu> [extra args...]
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${1:-0}; shift || true
source .venv/bin/activate
mkdir -p runs/logs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU setsid nohup python -u scripts/02_cpt_train.py --config configs/cpt.yaml "$@" \
  > runs/logs/cpt_train.log 2>&1 < /dev/null &
sleep 2; pgrep -fa "02_cpt_train" | cut -c1-120
