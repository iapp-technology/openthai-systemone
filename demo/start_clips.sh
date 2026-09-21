#!/usr/bin/env bash
# Start the clip batch detached (kept separate from stop_clips.sh so an ssh one-liner never matches its own kill pattern).
cd "$(dirname "$0")/.."
rm -rf runs/demo/clips; mkdir -p runs/logs
GPU=${GPU:-1} setsid nohup bash demo/run_clips.sh > runs/logs/doom_clips.log 2>&1 < /dev/null &
sleep 2; echo "running: $(pgrep -f 'demo/run_clips' | wc -l)"; TZ=Asia/Bangkok date "+%H:%M"
