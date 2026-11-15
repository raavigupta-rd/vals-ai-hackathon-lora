"""Report zero-reward tasks and no-variance GRPO groups from managed score artifacts (no token logprobs)."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from ac2.sdk import Client


def summarize_rewards(rows, expected_samples):
    latest = {}
    for row in sorted(rows, key=lambda r: str(r.get("created_at", ""))):
        if row.get("parent_score_id") or not isinstance(row.get("score"), (int, float)):
            continue
        if not str(row.get("job_id", "")).startswith("train_"):
            continue
        if not row.get("agent_trace_id"):
            continue
        latest[(row["generated_step"], row["datapoint_id"], row["agent_trace_id"])] = row
    groups = defaultdict(list)
    for (step, task, sample), row in latest.items():
        groups[(step, task)].append(row["score"])
    complete = {key: values for key, values in groups.items() if len(values) == expected_samples}
    scores = [row["score"] for row in latest.values()]
    return {"scored_rollouts": len(scores), "zero_reward_rollouts": sum(v == 0 for v in scores),
            "zero_reward_fraction": sum(v == 0 for v in scores) / len(scores) if scores else None,
            "complete_groups": len(complete), "incomplete_groups": len(groups) - len(complete),
            "no_variance_groups": sum(min(v) == max(v) for v in complete.values()),
            "no_variance_fraction": sum(min(v) == max(v) for v in complete.values()) / len(complete) if complete else None,
            "all_zero_groups": sum(all(s == 0 for s in v) for v in complete.values()),
            "groups": [{"step": key[0], "task_id": key[1], "scores": values, "complete": len(values) == expected_samples} for key, values in groups.items()]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--expected-samples", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    client = Client(project="vals-hackathon")
    rows = client.analytics.scores(run_ids=[args.run]).select("job_id", "generated_step", "datapoint_id", "agent_trace_id", "score", "parent_score_id", "created_at").iter_rows()
    report = {"train_id": args.run, **summarize_rewards(rows, args.expected_samples)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "groups"}, indent=2))
