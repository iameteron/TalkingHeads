#!/bin/sh
set -eu

DATA_DIR="${PLAY_WEB_LEADERBOARD_DIR:-/app/data}"
SEED_DIR="/app/play_web/leaderboard"

mkdir -p "$DATA_DIR"
for seed_file in active_agent_runs.jsonl companion_research.jsonl companion_test.jsonl; do
  if [ ! -f "$DATA_DIR/$seed_file" ] && [ -f "$SEED_DIR/$seed_file" ]; then
    cp "$SEED_DIR/$seed_file" "$DATA_DIR/$seed_file"
  fi
done

export PLAY_WEB_LEADERBOARD_DIR="$DATA_DIR"
export PLAY_WEB_PERSIST_KEYS_TO_ENV="${PLAY_WEB_PERSIST_KEYS_TO_ENV:-false}"
export PYTHONPATH="/app${PYTHONPATH:+:$PYTHONPATH}"

cd /app/play_web
exec python -m uvicorn server:app --host 0.0.0.0 --port 8000
