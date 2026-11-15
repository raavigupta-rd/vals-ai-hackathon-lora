"""Stop an RL run once the held-out score is flat (or the budget is spent) and record the best checkpoint.

Reads <log_dir>/metrics.jsonl after every eval:
  * unhinted held-out score  test/env/heldout/score  (eval_group_size rollouts per problem)
  * hinted held-out score    test/env/hinted/score   (1 truncation-hinted rollout per problem)
Checkpoints are ranked by the combined z-score from Hinted Scaling Laws (arXiv 2611.02270):
  combined = a * z(unhinted) + (1 - a) * z(hinted),  a = (k-1)/(k-1+lambda), lambda = 8,
  k = rollouts per problem.
The run is stopped (SIGTERM to the matching train_rl process) when the combined score hasn't
set a new best for `patience` consecutive evals, or the cumulative cost exceeds `budget`.
Writes <log_dir>/best.json with the best eval step and its sampler checkpoint.

    uv run python scripts/rl_watchdog.py runs/rl/<name> --patience 3 --budget 60
"""

import argparse
import json
import os
import signal
import statistics
import subprocess
import time
from pathlib import Path


def rows(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.setdefault(int(m.get("step", 0)), {}).update(m)
    return out


def find(m: dict, suffix: str) -> float | None:
    for k, v in m.items():
        if k.startswith("test") and k.endswith(suffix) and isinstance(v, (int, float)):
            return float(v)
    return None


def z(xs: list[float]) -> list[float]:
    if len(xs) < 2:
        return [0.0] * len(xs)
    sd = statistics.pstdev(xs) or 1.0
    mu = statistics.mean(xs)
    return [(x - mu) / sd for x in xs]


def sampler_for(log_dir: Path, step: int) -> str | None:
    """Sampler checkpoint saved at exactly the evaluated step (train_rl saves every eval_every steps)."""
    ck = log_dir / "checkpoints.jsonl"
    if not ck.exists():
        return None
    for line in ck.read_text().splitlines():
        c = json.loads(line)
        if c.get("sampler_path") and int(c.get("batch", -1)) == step:
            return c["sampler_path"]
    print(f"WARNING: no sampler checkpoint saved at evaluated step {step}", flush=True)
    return None


def train_pid(log_dir: Path) -> int | None:
    f = log_dir / "train.pid"
    return int(f.read_text().strip()) if f.exists() else None


def alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop_training(log_dir: Path) -> None:
    pid = train_pid(log_dir)
    if alive(pid):
        os.kill(pid, signal.SIGTERM)
        print(f"sent SIGTERM to {pid}", flush=True)
    else:
        print("WARNING: training process not found; nothing stopped", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min-evals", type=int, default=3)
    ap.add_argument("--budget", type=float, default=60.0, help="USD: training cost/cumulative_usd + evals")
    ap.add_argument("--eval-cost", type=float, default=6.0, help="USD per held-out eval (not in cost metrics)")
    ap.add_argument("--k", type=int, default=5, help="rollouts per held-out problem (unhinted + hinted)")
    ap.add_argument("--lam", type=float, default=8.0)
    ap.add_argument("--poll", type=int, default=30)
    a = ap.parse_args()
    log_dir = Path(a.log_dir)
    alpha = (a.k - 1) / (a.k - 1 + a.lam)
    seen = -1
    while True:
        data = rows(log_dir / "metrics.jsonl")
        evals = [(s, find(m, "env/heldout/score"), find(m, "env/hinted/score")) for s, m in sorted(data.items())]
        evals = [e for e in evals if e[1] is not None]
        cost = max((m.get("cost/cumulative_usd", 0.0) for m in data.values()), default=0.0)
        n_evals = sum(1 for m in data.values() if find(m, "env/heldout/score") is not None)
        cost += n_evals * a.eval_cost  # eval sampling isn't in cost/cumulative_usd
        if len(evals) != seen and evals:
            seen = len(evals)
            zu = z([e[1] for e in evals])
            zh = z([e[2] if e[2] is not None else e[1] for e in evals])
            comb = [alpha * u + (1 - alpha) * h for u, h in zip(zu, zh)]
            best_i = max(range(len(comb)), key=lambda i: comb[i])
            since_best = len(comb) - 1 - best_i
            best = {"step": evals[best_i][0], "heldout_score": evals[best_i][1], "hinted_score": evals[best_i][2],
                    "combined_z": comb[best_i], "sampler_path": sampler_for(log_dir, evals[best_i][0]),
                    "evals": [{"step": s, "heldout": u, "hinted": h, "combined_z": round(c, 3)}
                              for (s, u, h), c in zip(evals, comb)], "cost_usd": cost, "alpha": alpha}
            (log_dir / "best.json").write_text(json.dumps(best, indent=2))
            print(f"[{time.strftime('%H:%M')}] evals={len(evals)} last={evals[-1]} best_step={best['step']} "
                  f"since_best={since_best} cost=${cost:.2f}", flush=True)
            if len(evals) >= a.min_evals and since_best >= a.patience:
                print("held-out score is flat -> stopping", flush=True)
                stop_training(log_dir)
                return
        if cost > a.budget:
            print(f"budget ${a.budget} exceeded (${cost:.2f}) -> stopping", flush=True)
            stop_training(log_dir)
            return
        pid = train_pid(log_dir)
        if pid is not None and not alive(pid):
            print("training process ended", flush=True)
            return
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
