#!/usr/bin/env bash
# Wait for the targeted generators and the order-invariant eval to finish, then SFT3 -> calib3 -> eval3 (plain + invariant).  <gpu>
set -uo pipefail
cd "$(dirname "$0")/.."
GPU=${1:-0}
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU
while pgrep -f "03e_targeted_synth.py" >/dev/null || ! grep -q "eval-inv\] done" runs/logs/eval_inv.log 2>/dev/null; do sleep 120; done
echo "[round3] start SFT3 at $(date)"
python -u scripts/04_decision_train.py --config configs/sft3.yaml --init runs/sft2/latest > runs/logs/sft3_train.log 2>&1
echo "[round3] SFT3 done at $(date); calibrating"
sed -e "s#^output_dir: runs/calib#output_dir: runs/calib3#" -e "s#^model_path: runs/sft/latest#model_path: runs/sft3/latest#" configs/calib.yaml > configs/calib3.yaml
python -u scripts/04_decision_train.py --config configs/calib3.yaml --init runs/sft3/latest > runs/logs/calib3_train.log 2>&1
echo "[round3] calib3 done at $(date); evaluating"
python scripts/06_eval.py --model runs/calib3/latest --data data/public_bench --out runs/eval3_public.json --batch-size 32 2>&1 | grep -vE "it/s\]|falling back"
python scripts/06_eval.py --model runs/calib3/latest --data data/decision --out runs/eval3_heldout.json --batch-size 32 2>&1 | grep -vE "it/s\]|falling back"
echo "[round3] all evals done at $(date)"
