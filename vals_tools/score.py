"""Score a poker eval in big blinds per hand, paired per deal against a base eval on the same dataset.

    uv run python -m vals_tools.score --project <project> --eval eval_new --base eval_base
    uv run python -m vals_tools.score --project <project> --run train_xxx --step 20

Reads the sample grader's score artifacts (`ev_bb`, `realized_bb`, `policy_format_errors`), not the shifted reward.
Both evals must use the same dataset: the same deals, so the difference can be paired hand by hand.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from ac2.sdk import Client

COLUMNS = ("job_id", "generated_step", "datapoint_id", "artifacts", "parent_score_id", "created_at", "score")


def score_rows(client: Client, eval_id: str | None = None, run: str | None = None, step: int | None = None) -> dict:
    """Top-level harness scores keyed by task id; the latest score wins on retries."""
    rows: dict[str, dict] = {}
    query = client.analytics.scores(run_ids=[eval_id or run]).select(*COLUMNS)
    for row in sorted(query.iter_rows(), key=lambda r: str(r["created_at"])):
        if row["parent_score_id"]:
            continue
        job = str(row["job_id"])
        if eval_id and job != eval_id:
            continue
        if run and (not job.startswith("eval_") or str(row["generated_step"]) != str(step)):
            continue
        artifacts = row["artifacts"] if isinstance(row["artifacts"], dict) else json.loads(row["artifacts"] or "{}")
        if "ev_bb" in artifacts or artifacts.get("failure"):
            rows[str(row["datapoint_id"])] = artifacts | {"reward_score": row.get("score")}
    return rows


def _ci95(values: list[float]) -> float | None:
    return 1.96 * statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else None


def rounded(value):
    return round(value, 3) if value is not None else None


def summarize(rows: dict, base: dict | None = None) -> dict[str, Any]:
    if not rows:
        return {"games": 0, "hands": 0}
    played = {k: r for k, r in rows.items() if "ev_bb" in r}
    hands = sum(r.get("hands", 1) for r in played.values())
    total_ev = sum(r["ev_bb"] for r in played.values())
    per_hand = total_ev / hands if hands else None
    # Cluster uncertainty by task/game; thirty correlated hands are not thirty independent games.
    residuals = [r["ev_bb"] - per_hand * r.get("hands", 1) for r in played.values()] if per_hand is not None else []
    scale = statistics.fmean(r.get("hands", 1) for r in played.values()) if played else 1
    ci = _ci95(residuals)
    decisions = sum(r.get("policy_decisions", 0) for r in played.values())
    errors = sum(r.get("policy_format_errors", 0) for r in played.values())
    places = [r["place"] for r in played.values() if r.get("place") is not None]
    busts = [r["busted"] for r in played.values() if r.get("busted") is not None]
    failed = sum(bool(r.get("failure") or r.get("policy_format_errors", 0)) for r in rows.values())
    out: dict[str, Any] = {
        "games": len(rows), "games_with_chip_results": len(played), "hands": hands,
        "ev_bb_per_hand": rounded(per_hand), "ci95": rounded(ci / scale if ci is not None else None),
        "realized_bb_per_hand": rounded(sum(r.get("realized_bb", r["ev_bb"]) for r in played.values()) / hands) if hands else None,
        "failed_game_rate": round(failed / len(rows), 4),
        "policy_decisions": decisions, "policy_format_errors": errors,
        "decision_failure_rate": round(errors / decisions, 4) if decisions else None,
        "mean_final_place": rounded(statistics.fmean(places)) if places else None,
        "games_with_place": len(places),
        "bust_rate": round(statistics.fmean(busts), 4) if busts else None, "games_with_bust_status": len(busts),
    }
    if all(r.get("hands", 1) == 1 for r in played.values()):
        out["format_error_hands"] = out["failed_game_rate"]  # compatibility for single-hand datasets only
    if base:
        paired = [k for k in played if k in base and "ev_bb" in base[k]]
        diffs = [played[k]["ev_bb"] / played[k].get("hands", 1) - base[k]["ev_bb"] / base[k].get("hands", 1) for k in paired]
        if diffs:
            place_diffs = [played[k]["place"] - base[k]["place"] for k in paired if "place" in played[k] and "place" in base[k]]
            out["vs_base"] = {
                "paired_games": len(paired),
                "paired_hands": sum(played[k].get("hands", 1) for k in paired) if all(played[k].get("hands", 1) == base[k].get("hands", 1) for k in paired) else None,
                "diff": rounded(statistics.fmean(diffs)), "ci95": rounded(_ci95(diffs)),
                "diff_unit": "mean paired game difference in BB per played hand",
                "ev_bb_per_game_diff": rounded(statistics.fmean(played[k]["ev_bb"] - base[k]["ev_bb"] for k in paired)),
                "mean_final_place_diff": rounded(statistics.fmean(place_diffs)) if place_diffs else None,
                "place_diff_ci95": rounded(_ci95(place_diffs)), "paired_games_with_place": len(place_diffs),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="vals-hackathon")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--eval", help="a standalone eval")
    target.add_argument("--run", help="a training run; with --step, its sidecar eval at that step")
    parser.add_argument("--step", type=int)
    parser.add_argument("--base", help="the base model's eval on the same dataset")
    parser.add_argument("--traces", type=Path, help="completed task traces; recover missing bust status from final cleanup summaries")
    args = parser.parse_args()

    client = Client(project=args.project)
    source_info = client.eval.get(args.eval) if args.eval else None
    base_info = client.eval.get(args.base) if args.base else None
    if source_info and base_info and {t.dataset for t in source_info.tasks} != {t.dataset for t in base_info.tasks}:
        parser.error("paired comparison requires the same source dataset; task IDs are then paired by intersection")
    rows = score_rows(client, eval_id=args.eval, run=args.run, step=args.step)
    if args.traces:
        from poker.trace_quality import bust_status_by_task
        with args.traces.open(encoding="utf-8") as stream:
            busts = bust_status_by_task(json.loads(line) for line in stream)
        rows = {k: r | ({"busted": busts[k]} if "busted" not in r and k in busts else {}) for k, r in rows.items()}
    base = score_rows(client, eval_id=args.base) if args.base else None
    report = summarize(rows, base)
    if source_info:
        report["source"] = {"eval_id": source_info.id, "status": source_info.status,
                            "datasets": [t.dataset for t in source_info.tasks],
                            "partial": source_info.status != "succeeded"}
    if base_info:
        report["base"] = {"eval_id": base_info.id, "status": base_info.status,
                          "partial": base_info.status != "succeeded"}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
