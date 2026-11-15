#!/usr/bin/env bash
# Train an SFT model with harness.sft_qwengem, then evaluate it on the held-out set (5 turns, thinking on).
#   scripts/sft_then_eval.sh <name> [extra sft args...]
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="$1"; shift
export PYTHONUNBUFFERED=1
uv run python -m harness.sft_qwengem --name "$NAME" "$@" 2>&1 | grep -v Warning | tee "runs/sft-$NAME.log"
P=$(python3 -c "import json;print(json.load(open('runs/sft/$NAME/checkpoint.json'))['sampler_path'])")
echo "EVAL $NAME $P"
HARNESS_BACKEND=modal uv run python -m harness.eval --problems data/synth_ioi/heldout.jsonl --samples 2 \
  --renderer qwen3_5 --model-path "$P" --max-tokens 32768 --max-context 64000 --max-turns 5 \
  --concurrency 68 --tag "eval-$NAME" 2>&1 | grep -v Warning | tail -25
