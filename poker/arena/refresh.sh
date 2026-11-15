#!/bin/sh
# Every INTERVAL seconds: pull gateway spans for our deployments and rebuild data/arena/joined.jsonl.
# The match collector (python -m poker.arena.collect --loop 600) runs separately.
# Usage: DEPLOYMENTS="infer_a infer_b" sh poker/arena/refresh.sh
INTERVAL="${INTERVAL:-600}"
cd "$(dirname "$0")/../.." || exit 1
while true; do
  for d in $DEPLOYMENTS; do
    uv run python -m poker.arena.pull_gateway --deployment "$d" --since 48h --out "data/arena/gateway/$d.jsonl"
  done
  cat data/arena/gateway/*.jsonl > data/arena/gateway_all.jsonl
  uv run python -m poker.arena.join_traces --gateway data/arena/gateway_all.jsonl --out data/arena/joined.jsonl
  date
  sleep "$INTERVAL"
done
