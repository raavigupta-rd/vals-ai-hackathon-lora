"""Turn teacher verdicts on our arena decisions into SFT records for poker/sft/build_sft_dataset.py.

Mix per cycle:
- teacher_correction: the teacher disagreed. Target = a short reasoning that states the numbers, then the
  teacher's action. Weighted up, since these are the behaviors we want to change.
- teacher_agree: the teacher judged the spot and agreed. Target = the model's own reply.
- self_clean: a sample of decisions the teacher abstained on, kept as the model wrote them. They hold the reply
  format and general play steady, so corrections do not spread to unrelated spots. Never selected by outcome.

Replies cut off at the token cap, or without exactly one submit_action call, are never targets.

    uv run python -m poker.sft.make_records --joined data/arena/joined.jsonl --cycle 1 --out data/sft/cycle1.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from poker.hints import parse_prompt

CORRECTION_WEIGHT = 2.0  # a correction in the model's own words (resampled)
BUST_RULES = ("allin_call", "own_shove")  # mistakes that bust us: placement drops ~2.4 places
BUST_WEIGHT = 3.0
TEMPLATE_WEIGHT = 1.0  # fixed-sentence fallback: cycle 1 showed these get memorized, so they count less
AGREE_WEIGHT = 1.0
ABSTAIN_WEIGHT = 1.0


def _call(record: dict) -> dict | None:
    calls = [c for c in record.get("tool_calls") or [] if c.get("name") == "submit_action"]
    if len(calls) != 1:
        return None
    args = calls[0].get("arguments")
    try:
        args = json.loads(args) if isinstance(args, str) else args
    except json.JSONDecodeError:
        return None
    return args if isinstance(args, dict) else None


def target_from_reply(record: dict) -> dict | None:
    args = _call(record)
    if args is None or not (record.get("reasoning") or "").strip():
        return None
    if args.get("action") not in ("bet", "raise"):
        args = {k: v for k, v in args.items() if k != "amount"}  # the harness ignores it; teach the clean form
    return {"reasoning": record["reasoning"], "content": record.get("content") or "",
            "tool_call": {"name": "submit_action", "arguments": args}}


def correction_reasoning(verdict: dict, hand_code: str) -> str:
    rule, preferred = verdict["rule"], verdict["preferred"]
    if rule == "allin_call":
        price = verdict["to_call"] / (verdict["pot"] + verdict["to_call"])
        need = f"{verdict['required']:.0%}"
        have = f"{verdict['equity']:.0%}"
        if preferred == "fold":
            return (f"I face an all-in: {verdict['to_call']:g} more into a pot of {verdict['pot']:g}, so the price is "
                    f"{price:.0%} of the final pot. Calling risks my stack, and busting ranks last, so I need about "
                    f"{need} equity. Against this player's shoving range, {hand_code} has about {have}. Fold.")
        return (f"I face an all-in: {verdict['to_call']:g} more into a pot of {verdict['pot']:g}. I need about {need} "
                f"equity and {hand_code} has about {have} against this player's shoving range. This is a clear call.")
    if rule == "postflop_call":
        need, have = f"{verdict['required']:.0%}", f"{verdict['equity']:.0%}"
        facing = (f"On the {verdict['street']} I face a bet: {verdict['to_call']:g} more into a pot of "
                  f"{verdict['pot']:g}, so I need about {need} equity.")
        if preferred == "fold":
            return (f"{facing} Against the hands this player bets this way, {hand_code} has about {have} on this "
                    f"board. That is not enough. Fold.")
        return (f"{facing} Against the hands this player bets this way, {hand_code} has about {have} on this "
                f"board. That is enough, so call.")
    if rule == "unopened":
        if preferred == "raise":
            return f"{hand_code} is a premium hand and the pot is unopened. Raise for value."
        if preferred == "check":
            return f"{hand_code} is weak and checking is free. Check and see the flop."
        return (f"{hand_code} is too weak to play from this seat, and limping or raising it loses chips over time. "
                f"Fold and wait for a better spot.")
    if rule == "facing_raise":
        return (f"Someone has already raised. {hand_code} is not strong enough to continue against a raise, and "
                f"calling or re-raising with it loses chips and risks my stack. Fold.")
    if rule == "own_shove":
        return (f"Going all-in with {hand_code} at this stack depth risks my whole stack with a hand that is behind "
                f"most calling ranges. Busting ranks last. Fold and wait for a better spot.")
    if rule == "free_fold":
        return "Checking costs nothing, so folding can only lose. Check."
    raise ValueError(f"no correction text for rule {rule!r}")


def _correction_args(verdict: dict, record: dict) -> dict | None:
    preferred = verdict["preferred"]
    if preferred in ("fold", "check", "call"):
        return {"action": preferred}
    state = parse_prompt(record["messages"])
    if not state or state["raise_type"] != "raise" or state["raise_min"] is None:
        return None
    amount = min(max(state["raise_min"], 6), state["raise_max"])
    return {"action": "raise", "amount": int(amount)}


def select(rows, *, cycle: int, abstain_ratio: float = 2.0, seed: int = 0, resampled: dict | None = None,
           template_weight: float = TEMPLATE_WEIGHT, weight_scale: float = 1.0, rules: tuple | None = None):
    """rows: (joined record, teacher verdict, hand code). Returns (records, stats)."""
    resampled = resampled or {}
    stats = {"corrections": 0, "resampled": 0, "agreements": 0, "abstain_pool": 0, "abstain_kept": 0, "skipped": 0, "unvalidated_dropped": 0, "out_of_scope": 0}
    records, pool = [], []

    def base(record, source, weight, target):
        return {"decision_id": record["span_id"], "messages": record["messages"], "target": target,
                "weight": weight * weight_scale, "source": source, "cycle": cycle}

    for record, verdict, code in rows:
        if record.get("cutoff") or _call(record) is None:
            stats["skipped"] += 1
            continue
        if rules is not None and verdict.get("agree") is not None and verdict.get("rule") not in rules:
            # Ablation: this rule is switched off, so the spot is an anchor that keeps base's own action.
            stats["out_of_scope"] += verdict["agree"] is False
            verdict = {"rule": verdict.get("rule"), "preferred": None, "agree": None}
        if verdict.get("agree") is False:
            own = resampled.get(record["span_id"])
            if own and own.get("preferred", verdict["preferred"]) != verdict["preferred"]:
                own = None  # sampled for an older verdict: the teacher now prefers something else
            if own:
                target = {"reasoning": own["reasoning"], "content": own.get("content") or "",
                          "tool_call": {"name": "submit_action", "arguments": own["arguments"]}}
                weight = BUST_WEIGHT if verdict.get("rule") in BUST_RULES else CORRECTION_WEIGHT
                records.append(base(record, "teacher_resampled", weight, target))
                stats["resampled"] += 1
                stats["corrections"] += 1
                continue
            if template_weight <= 0:  # keep only corrections validated by a graded re-rollout
                stats["unvalidated_dropped"] += 1
                continue
            args = _correction_args(verdict, record)
            if args is None:
                stats["skipped"] += 1
                continue
            target = {"reasoning": correction_reasoning(verdict, code), "content": "",
                      "tool_call": {"name": "submit_action", "arguments": args}}
            records.append(base(record, "teacher_correction", template_weight, target))
            stats["corrections"] += 1
        elif verdict.get("agree") is True and (target := target_from_reply(record)):
            records.append(base(record, "teacher_agree", AGREE_WEIGHT, target))
            stats["agreements"] += 1
        elif verdict.get("agree") is None and (target := target_from_reply(record)):
            pool.append(base(record, "self_clean", ABSTAIN_WEIGHT, target))
    stats["abstain_pool"] = len(pool)
    keep = min(len(pool), round(abstain_ratio * (stats["corrections"] + stats["agreements"])))
    random.Random(seed).shuffle(pool)
    records += pool[:keep]
    stats["abstain_kept"] = keep
    return records, stats


def main() -> None:
    from poker.arena import db
    from poker.teacher import judge

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--joined", type=Path, default=Path("data/arena/joined.jsonl"))
    parser.add_argument("--db", type=Path, default=Path("data/arena/arena.db"))
    parser.add_argument("--deployment", help="only decisions by this deployment id (the current entry)")
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--abstain-ratio", type=float, default=2.0)
    parser.add_argument("--template-weight", type=float, default=0.0,
                        help="weight of unvalidated fixed-sentence corrections; 0 drops them (default)")
    parser.add_argument("--rules", help="comma list of teacher rules whose corrections to keep (ablations); others become anchors")
    parser.add_argument("--weight-scale", type=float, default=1.0, help="multiply every weight (newer cycles weigh more)")
    parser.add_argument("--resampled", type=Path, help="poker/sft/resample.py output: own-words corrections")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    resampled = {}
    if args.resampled:
        for line in args.resampled.open(encoding="utf-8"):
            item = json.loads(line)
            if item.get("reply"):
                resampled[item["span_id"]] = item["reply"] | {"preferred": item.get("preferred")}
    con = db.connect(args.db)
    rows = []
    for line in args.joined.open(encoding="utf-8"):
        record = json.loads(line)
        match = record.get("match") or {}
        if record.get("source") != "arena" or record.get("kind") != "action" or not match.get("verified"):
            continue
        if args.deployment and record.get("deployment") != args.deployment:
            continue
        verdict = judge(con, match["match_id"], match["hand"], match["seat"], match["action_index"])
        row = con.execute("SELECT card1, card2 FROM hole_cards WHERE match_id=? AND hand_number=? AND seat=?",
                          (match["match_id"], match["hand"], match["seat"])).fetchone()
        from poker.teacher import hand_code
        rows.append((record, verdict, hand_code([row[0], row[1]]) if row else "this hand"))
    records, stats = select(rows, cycle=args.cycle, abstain_ratio=args.abstain_ratio, resampled=resampled,
                            template_weight=args.template_weight, weight_scale=args.weight_scale,
                            rules=tuple(args.rules.split(",")) if args.rules else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    print(json.dumps(stats | {"records": len(records), "out": str(args.out)}))


if __name__ == "__main__":
    main()
