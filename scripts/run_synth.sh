#!/usr/bin/env bash
# (Re)start the synthetic generator on the GPU server as a detached job.
#   scripts/run_synth.sh <seed> [n] [concurrency]
set -euo pipefail
cd "$(dirname "$0")/.."
SEED=${1:-2}; N=${2:-600000}; CONC=${3:-8}; shift 3 || true   # remaining args go to the generator (e.g. --lang en)
source .venv/bin/activate
export SYNTH_BASE_URL=${SYNTH_BASE_URL:?set SYNTH_BASE_URL}
export SYNTH_MODEL=${SYNTH_MODEL:?set SYNTH_MODEL}
export SYNTH_API_KEY=${SYNTH_API_KEY:?set SYNTH_API_KEY}
pkill -f "03b_synth_generate.py.*--seed ${SEED}\b" || true   # only this seed; other generators keep running
sleep 2
mkdir -p runs/logs
setsid nohup python -u scripts/03b_synth_generate.py --out data/synth --n "$N" --concurrency "$CONC" --seed "$SEED" "$@" \
  > "runs/logs/synth_${SEED}.log" 2>&1 < /dev/null &
sleep 3
pgrep -fa "03b_synth_generate.py" | cut -c1-120
