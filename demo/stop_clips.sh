#!/usr/bin/env bash
pkill -f "demo/run_clips.sh" || true; pkill -f "demo/doom_demo.py" || true; sleep 1; echo "left: $(pgrep -f 'doom_demo|run_clips' | wc -l)"
