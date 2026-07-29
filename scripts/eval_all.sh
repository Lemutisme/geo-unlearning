#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${BASELINE_ROOT:-/root/baseline_reference_20260729}"
CODE_DIR="/workspace/re/GU/geo-unlearning/.worktrees/gu"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate unlearning
cd "$CODE_DIR"

python scripts/run_baseline_reference.py manifest --output-root "$ROOT_DIR"

setsid python scripts/run_baseline_reference.py worker \
  --manifest "$ROOT_DIR/manifest.json" --worker 0 --gpu 0 \
  >"$ROOT_DIR/worker0.log" 2>&1 &
worker0=$!

setsid python scripts/run_baseline_reference.py worker \
  --manifest "$ROOT_DIR/manifest.json" --worker 1 --gpu 1 \
  >"$ROOT_DIR/worker1.log" 2>&1 &
worker1=$!

stop_workers() {
  kill -TERM -- "-$worker0" "-$worker1" 2>/dev/null || true
}
trap stop_workers INT TERM

status=0
wait "$worker0" || status=1
wait "$worker1" || status=1
exit "$status"
