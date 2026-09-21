#!/usr/bin/env bash
# Launch the five targeted generators detached against the synthesis engine (concurrency sums to 96 of 128 engine slots).
#   SYNTH_BASE_URL=... SYNTH_MODEL=... SYNTH_API_KEY=... scripts/run_targeted.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
: "${SYNTH_BASE_URL:?}" "${SYNTH_MODEL:?}"; export SYNTH_API_KEY=${SYNTH_API_KEY:-none}
pkill -f "03e_targeted_synth.py" || true; sleep 1
mkdir -p runs/logs data/synth_targeted
launch() { setsid nohup python -u scripts/03e_targeted_synth.py --task "$1" --n "$2" --concurrency "$3" --out data/synth_targeted > "runs/logs/targeted_$1.log" 2>&1 < /dev/null & }
launch grounded_qa 30000 32
launch summary_rating 20000 24
launch finegrained_intent 12000 16
launch safety 8000 12
launch paraphrase 8000 12
sleep 2; pgrep -fa "03e_targeted_synth.py" | grep -oE "\-\-task [a-z_]+" | sort
