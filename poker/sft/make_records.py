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
    if rule == "river_fold":
        return "On the river a bet this size is rarely a bluff; a pair is not enough. Fold."
    if rule == "monster_call":
        return "I have a very strong hand and the price is fine; folding here gives up a pot I am well ahead in. Call."
    if rule == "river_air_fold":
        return "On the river my hand has nothing; calling only beats a bluff, and bets here are rarely bluffs. Fold."
    if rule == "steal":
        return "Folded to me in late position: the blinds fold most of the time to a small raise. Raise."
    if rule == "stab":
        return "They checked, which usually means weakness; a half-pot bet often takes it down. Bet."
    if rule == "oversize":
        return ("A smaller raise does the same job here and keeps the pot manageable; there is no need to build a "
                f"huge pot before the flop. Raise to {verdict['target_amount']}.")
    if rule == "free_fold":
        return "Checking costs nothing, so folding can only lose. Check."
    raise ValueError(f"no correction text for rule {rule!r}")


def _correction_args(verdict: dict, record: dict) -> dict | None:
    preferred = verdict["preferred"]
    if preferred in ("fold", "check", "call"):
        return {"action": preferred}
    if verdict["rule"] == "oversize":
        state = parse_prompt(record["messages"])
        if not state or state["raise_min"] is None or state["raise_type"] != "raise":
            return None
        return {"action": "raise", "amount": int(min(max(state["raise_min"], verdict["target_amount"]), state["raise_max"]))}
    state = parse_prompt(record["messages"])
    if not state or state["raise_min"] is None or state["raise_type"] not in ("raise", "bet"):
        return None
    if state["raise_type"] == "bet" and verdict["rule"] != "stab":
        return None
    if state["raise_type"] == "raise" and verdict["rule"] == "stab":
        return None
    if verdict["rule"] == "steal":
        amount = min(max(state["raise_min"], 5), state["raise_max"])  # 2.5bb
    elif verdict["rule"] == "stab":
        if state["raise_type"] != "bet":
            return None
        amount = min(max(state["raise_min"], round(0.5 * state["pot"])), state["raise_max"])
        return {"action": "bet", "amount": int(amount)}
    else:
        amount = min(max(state["raise_min"], 6), state["raise_max"])
    return {"action": "raise", "amount": int(amount)}


def select(rows, *, cycle: int, abstain_ratio: float = 2.0, seed: int = 0, resampled: dict | None = None,
           template_weight: float = TEMPLATE_WEIGHT, weight_scale: float = 1.0, rules: tuple | None = None,
           strip_say: bool = True):
    """rows: (joined record, teacher verdict, hand code). Returns (records, stats).

    strip_say removes the table-talk "say" argument from every target's tool call: the best arena teams never talk."""
    resampled = resampled or {}
    stats = {"corrections": 0, "resampled": 0, "agreements": 0, "abstain_pool": 0, "abstain_kept": 0, "skipped": 0, "unvalidated_dropped": 0, "out_of_scope": 0,
             "say_stripped": 0}
    records, pool, stripped = [], [], set()

    def base(record, source, weight, target):
        args = target["tool_call"]["arguments"]
        if strip_say and "say" in args:
            target = target | {"tool_call": target["tool_call"] | {"arguments": {k: v for k, v in args.items() if k != "say"}}}
            stripped.add(record["span_id"])
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
    stats["say_stripped"] = sum(r["decision_id"] in stripped for r in records)
    return records, stats


def cleanup_anchors(records, *, n: int, cycle: int, seed: int = 0) -> list[dict]:
    """Up to n after-hand cleanup turns answered with exactly one valid update_scratchpad call, as SFT anchors.
    SFT on action turns alone made fine-tunes answer cleanup turns with submit_action (c2b-s12: 67% correct)."""
    pool = []
    for r in records:
        if r.get("kind") != "cleanup" or r.get("cutoff") or not (r.get("reasoning") or "").strip():
            continue
        calls = r.get("tool_calls") or []
        if len(calls) != 1 or calls[0].get("name") != "update_scratchpad":
            continue
        args = calls[0].get("arguments")
        try:
            args = json.loads(args) if isinstance(args, str) else args
        except json.JSONDecodeError:
            continue
        if not isinstance(args, dict) or set(args) != {"contents"} or len(args["contents"]) > 5000:
            continue
        pool.append({"decision_id": r["span_id"], "messages": r["messages"], "weight": 1.0, "source": "cleanup_anchor",
                     "cycle": cycle, "target": {"reasoning": r["reasoning"], "content": r.get("content") or "",
                                                "tool_call": {"name": "update_scratchpad", "arguments": args}}})
    random.Random(seed).shuffle(pool)
    return pool[:n]


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
    parser.add_argument("--exploits", help="comma list of teacher v5 exploit rules to enable: river_fold,steal,stab,oversize,monster_call,river_air_fold")
    parser.add_argument("--cleanup-ratio", type=float, default=0.3,
                        help="cleanup-turn anchors per action record (keeps update_scratchpad working)")
    parser.add_argument("--cleanup-deployment", default="infer_d3d2b6b92171",
                        help="whose cleanup replies to use as anchors (base answers 98.5% correctly)")
    parser.add_argument("--rules", help="comma list of teacher rules whose corrections to keep (ablations); others become anchors")
    parser.add_argument("--keep-say", action="store_true", help="keep the table-talk 'say' argument in targets (default: removed)")
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
    if args.exploits:
        import poker.teacher as teacher

        teacher.EXPLOITS = {x for x in args.exploits.split(",") if x}
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
                            rules=tuple(args.rules.split(",")) if args.rules else None,
                            strip_say=not args.keep_say)
    if args.cleanup_ratio > 0:
        cleanups = [json.loads(line) for line in args.joined.open(encoding="utf-8")]
        cleanups = [r for r in cleanups if r.get("source") == "arena" and r.get("deployment") == args.cleanup_deployment]
        extra = cleanup_anchors(cleanups, n=round(args.cleanup_ratio * len(records)), cycle=args.cycle)
        records += extra
        stats["cleanup_anchors"] = len(extra)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    print(json.dumps(stats | {"records": len(records), "out": str(args.out)}))


if __name__ == "__main__":
    main()
