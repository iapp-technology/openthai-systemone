#!/usr/bin/env bash
# Stop the detached synthetic generators (safe to call from an ssh one-liner: the pattern is only inside this file).
pkill -f "03b_synth_generate.py" || true
pkill -f "03c_contrastive_th.py" || true
pkill -f "03d_thai_sentiment_synth.py" || true
pkill -f "03e_targeted_synth.py" || true
sleep 1; echo "generators left: $(pgrep -f '03[bcd]_' | wc -l)"
