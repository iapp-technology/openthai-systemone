#!/usr/bin/env bash
# Record the marketing clip set sequentially on one GPU (detached). Output: runs/demo/clips/NN_<scenario>_<lang>.mp4 + .png
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=${GPU:-1}
MODEL=${MODEL:-release/OpenThai-SystemOne-v0.2}
OUT=runs/demo/clips; mkdir -p "$OUT"
i=0
clip() {  # scenario skill lang seconds seed
  i=$((i+1)); name=$(printf "%02d_%s_%s_s%s" $i "$1" "$3" "$5")
  echo "[$(date +%H:%M:%S)] $name"
  python demo/doom_demo.py --model "$MODEL" --scenario "$1" --skill "$2" --lang "$3" --seconds "$4" --seed "$5" \
    --out "$OUT/$name.mp4" --snapshot "$OUT/$name.png" > "runs/logs/clip_$name.log" 2>&1 || echo "  FAILED (see runs/logs/clip_$name.log)"
  grep -E "^(episode|done)" "runs/logs/clip_$name.log" | tail -2
}
clip deadly_corridor 1 en 75 1
clip deadly_corridor 1 th 75 2
clip defend_the_center 1 en 60 1
clip defend_the_center 1 th 60 2
clip health_gathering 1 en 60 1
clip health_gathering 1 th 60 2
clip deadly_corridor 2 en 60 3
clip defend_the_line 1 en 60 1
clip basic 1 th 45 1
clip my_way_home 1 en 60 1
clip take_cover 1 en 45 1
clip deadly_corridor 1 th 75 4
clip defend_the_center 2 en 60 3
echo "[$(date +%H:%M:%S)] ALL CLIPS DONE"; ls -la "$OUT"/*.mp4
