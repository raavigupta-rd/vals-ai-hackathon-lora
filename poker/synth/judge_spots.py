"""Pull the gateway spans of synthetic games and judge the "facing a preflop all-in" decisions with the teacher.

Keeps action decisions where the deployment is heads-up against one preflop all-in with a price to pay, and
judges them with the teacher's allin_call_verdict against the FIELD shove range (every arena shove in the DB;
nothing is excluded because these hands are synthetic), with the evidence rule on. Output is joined.jsonl-shaped
(source "synth") plus `verdict`, for poker/synth/adapter.py.

    uv run python -m poker.synth.judge_spots --deployment-id infer_d3d2b6b92171 --manifest data/synth/run1.json \
        --out data/synth/spots.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from poker.hints import parse_prompt, text_content
from poker.teacher import hand_code

HAND = re.compile(r"^## Hand (\d+) of (\d+)", re.M)


def _submit(record: dict) -> dict | None:
    calls = [c for c in record.get("tool_calls") or [] if c.get("name") == "submit_action"]
    if len(calls) != 1:
        return None
    args = calls[0].get("arguments")
    try:
        args = json.loads(args) if isinstance(args, str) else args
    except json.JSONDecodeError:
        return None
    return args if isinstance(args, dict) else None


def hand_number(messages) -> int | None:
    for item in reversed(messages):
        found = HAND.findall(text_content(item))
        if found:
            return int(found[-1][0])
    return None


def facing_allin(state: dict) -> dict | None:
    """Heads-up against one preflop all-in: {shover, stack}, else None."""
    if not state or state["street"] != "preflop" or state["to_call"] <= 0:
        return None
    hero = next((s for s in state["seats"] if s["seat"] == state["seat"]), None)
    shoves = [s for s in state["seats"] if s["seat"] != state["seat"] and s["status"] == "all-in"]
    if hero is None or hero["status"] or len(shoves) != 1:
        return None
    shove = shoves[0]
    callers = [s for s in state["seats"] if s["seat"] not in (state["seat"], shove["seat"])
               and s["status"] == "" and s["bet"] >= shove["bet"]]
    return None if callers else {"shover": shove["seat"], "stack": hero["stack"]}


def judge_record(record: dict, codes: list[str], *, samples: int = 600) -> dict | None:
    """The synthetic record with a `verdict`, or None when the span is not a heads-up facing-all-in decision."""
    from poker import teacher

    if record.get("kind") != "action" or record.get("source") == "arena" or not record.get("messages"):
        return None
    args = _submit(record)
    state = parse_prompt(record["messages"])
    spot = facing_allin(state)
    number = hand_number(record["messages"])
    if args is None or spot is None or number is None:
        return None
    price = state["to_call"] / (state["pot"] + state["to_call"])
    risks_bust = state["to_call"] >= 0.9 * spot["stack"]
    previous, teacher.EVIDENCE = teacher.EVIDENCE, True  # the evidence rule, restored so other code is unaffected
    try:
        result = teacher.allin_call_verdict(state["hole"], codes, price=price, risks_bust=risks_bust,
                                            hand_number=number, samples=samples)
    finally:
        teacher.EVIDENCE = previous
    actual = args.get("action")
    verdict = {"rule": "allin_call", "preferred": None, "actual": actual, "agree": None} | result | {
        "shover": f"seat{spot['shover']}", "to_call": state["to_call"], "pot": state["pot"]}
    verdict["agree"] = None if verdict["preferred"] is None else verdict["preferred"] == actual
    return record | {"source": "synth", "match": None, "hand_number": number, "hand_code": hand_code(state["hole"]),
                     "verdict": verdict}


def field_codes(db_path: Path) -> list[str]:
    from poker import teacher
    from poker.arena import db

    previous, teacher.PER_OPPONENT = teacher.PER_OPPONENT, False
    try:
        return teacher.shove_range(db.connect(db_path), "")[0]
    finally:
        teacher.PER_OPPONENT = previous


def judge_spans(spans, codes: list[str], *, samples: int = 600) -> tuple[list[dict], dict]:
    out, stats = [], {"spans": 0, "actions": 0, "spots": 0}
    for record in spans:
        stats["spans"] += 1
        stats["actions"] += record.get("kind") == "action"
        judged = judge_record(record, codes, samples=samples)
        if judged:
            out.append(judged)
    stats["spots"] = len(out)
    return out, stats


def summarize(records: list[dict]) -> dict:
    counts: dict = {}
    for r in records:
        v = r["verdict"]
        key = f"{v['preferred'] or 'abstain'}:{v['actual']}"
        counts[key] = counts.get(key, 0) + 1
    fold = [r for r in records if r["verdict"]["preferred"] == "fold"]
    return {"preferred_vs_actual": counts,
            "preferred": {k: sum(1 for r in records if (r["verdict"]["preferred"] or "abstain") == k)
                          for k in ("fold", "call", "abstain")},
            "teacher_fold_n": len(fold),
            "call_rate_when_teacher_folds": (round(sum(r["verdict"]["actual"] == "call" for r in fold) / len(fold), 3)
                                             if fold else None)}


def main() -> None:
    from poker.arena.pull_gateway import parse_when, pull

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deployment-id", required=True, help="gateway job id, e.g. infer_d3d2b6b92171")
    parser.add_argument("--manifest", type=Path, help="run_games output; supplies --since/--until")
    parser.add_argument("--since")
    parser.add_argument("--until")
    parser.add_argument("--db", type=Path, default=Path("data/arena/arena.db"))
    parser.add_argument("--spans", type=Path, help="raw pulled spans (default: data/synth/spans_<id>.jsonl)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=600)
    args = parser.parse_args()
    since, until = args.since, args.until
    if args.manifest:
        window = json.loads(args.manifest.read_text())
        since, until = since or window["since"], until or window["until"]
    spans_path = args.spans or Path(f"data/synth/spans_{args.deployment_id}.jsonl")
    pulled = pull(args.deployment_id, parse_when(since), parse_when(until) if until else None, spans_path)
    spans = [json.loads(line) for line in spans_path.open(encoding="utf-8")]
    low, high = parse_when(since) if since else "", parse_when(until) if until else "9"
    spans = [s for s in spans if s.get("source") != "arena" and low <= s["ts"].rstrip("Z") < high]
    records, stats = judge_spans(spans, field_codes(args.db), samples=args.samples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for r in records:
            stream.write(json.dumps(r) + "\n")
    print(json.dumps({"pull": pulled, "judge": stats, "summary": summarize(records), "out": str(args.out)}))


if __name__ == "__main__":
    main()
