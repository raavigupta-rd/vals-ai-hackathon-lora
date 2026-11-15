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
    ck = log_dir / "checkpoints.jsonl"
    if not ck.exists():
        return None
    best = None
    for line in ck.read_text().splitlines():
        c = json.loads(line)
        if c.get("sampler_path") and int(c.get("batch", -1)) <= step:
            best = c["sampler_path"]  # latest checkpoint at or before the eval step
    return best


def stop_training(log_dir: Path) -> None:
    r = subprocess.run(["pgrep", "-f", f"harness.train_rl.*log_path={log_dir}"], capture_output=True, text=True)
    for pid in r.stdout.split():
        os.kill(int(pid), signal.SIGTERM)
        print(f"sent SIGTERM to {pid}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min-evals", type=int, default=3)
    ap.add_argument("--budget", type=float, default=60.0, help="USD (cost/cumulative_usd)")
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
        if (log_dir / "metrics.jsonl").exists() and not subprocess.run(
                ["pgrep", "-f", f"harness.train_rl.*log_path={log_dir}"], capture_output=True).stdout:
            print("training process ended", flush=True)
            return
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
