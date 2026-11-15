"""Prepare blinded batches of decisions for Claude critic subagents (no model calls here).

Reads arena decisions from data/arena/joined.jsonl (and optionally synth spots), renders the decision-time state
without the scratchpad, adds deterministic FACTS (legal actions, pot odds, teacher equity numbers when known), and
writes N batch files plus an answer key. The key maps each blinded id back to the span id and the teacher verdict;
it must not be shown to the critics. `poker.critic.merge` reads their rated*.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

INSTRUCTIONS = """\
# Critic task

Each line of a batch file is one poker decision: {id, state, action_taken, facts}.
`state` is everything the player saw at decision time (no scratchpad). `facts` is deterministic and trusted;
equity numbers, when present, come from a Monte Carlo model against an estimated opponent range.

Rate the action taken. Write one JSON line per input to rated_<batch>.jsonl:
  {"id": <same id>, "score": 1-10, "better_action": "fold|check|call|raise|bet or null",
   "confidence": 0-1 or "low|medium|high", "reason": "one short sentence"}
score 8-10: clearly good. 4-7: defensible or unclear. 1-3: clearly wrong.
Set better_action only when you are sure a different action is better. Do not guess. Use confidence honestly.
The goal is to maximize final placement in a 30-hand, 6-player game (busting early is costly).
"""

FIELDS = ("equity", "required", "equity_lo", "equity_hi")


def action_taken(record: dict) -> str | None:
    for call in record.get("tool_calls") or []:
        if call.get("name") == "submit_action":
            try:
                return json.loads(call.get("arguments") or "{}").get("action")
            except (ValueError, AttributeError):
                return None
    return None


def render_state(messages: list[dict]) -> str:
    """All user messages of the hand, in order, minus the scratchpad block."""
    parts = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content") or ""
        if content.lstrip().startswith("## Your scratchpad"):
            continue
        parts.append(content.strip())
    return "\n\n".join(parts)


def facts_for(state: str, verdict: dict | None) -> dict:
    """Facts from the last decision prompt in `state`, plus the teacher's numbers (never its preferred action)."""
    facts: dict = {}
    legal = re.findall(r"Legal actions:\s*(.+)", state)
    if legal:
        facts["legal_actions"] = [a.strip() for a in legal[-1].split(",")]
        call = re.search(r"call (\d+)", legal[-1])
        facts["to_call"] = int(call.group(1)) if call else 0
    pot = re.findall(r"Pot \(including current bets\):\s*(\d+)", state)
    if pot:
        facts["pot"] = int(pot[-1])
    if facts.get("pot") is not None and facts.get("to_call"):
        facts["pot_odds"] = round(facts["to_call"] / (facts["pot"] + facts["to_call"]), 4)
    if verdict:
        if verdict.get("equity") is not None:
            facts["equity"] = verdict["equity"]
        if verdict.get("required") is not None:
            facts["required_equity"] = verdict["required"]
        if verdict.get("equity_lo") is not None and verdict.get("equity_hi") is not None:
            facts["equity_interval"] = [verdict["equity_lo"], verdict["equity_hi"]]
    return facts


def facts_block(facts: dict) -> str:
    lines = ["FACTS"]
    if "legal_actions" in facts:
        lines.append("Legal actions: " + ", ".join(facts["legal_actions"]))
    if "pot" in facts:
        lines.append(f"Pot: {facts['pot']}; to call: {facts.get('to_call', 0)}")
    if "pot_odds" in facts:
        lines.append(f"Pot odds (break-even equity before bust margin): {facts['pot_odds']:.1%}")
    if "equity" in facts:
        interval = facts.get("equity_interval")
        span = f" (interval {interval[0]:.1%} to {interval[1]:.1%})" if interval else ""
        lines.append(f"Estimated equity: {facts['equity']:.1%}{span}")
    if "required_equity" in facts:
        lines.append(f"Required equity incl. bust margin: {facts['required_equity']:.1%}")
    return "\n".join(lines)


def load_decisions(joined: Path, synth_paths=(), db_path: Path | None = None, verdicts: dict | None = None,
                   deployment: str | None = None) -> list[dict]:
    """Decision records with `action`, `messages` and `verdict` (None when unknown).

    Arena verdicts come from `verdicts` ({span_id: verdict}) or, when `db_path` is given, from the teacher.
    Synth spots carry their own verdict.
    """
    con = None
    if db_path is not None:
        from poker.arena import db

        con = db.connect(db_path)
    out = []

    def add(record, verdict):
        action = action_taken(record)
        if action is None or not record.get("messages"):
            return
        out.append({"span_id": record["span_id"], "source": record.get("source"),
                    "deployment": record.get("deployment"), "messages": record["messages"], "action": action,
                    "verdict": verdict})

    if joined and Path(joined).exists():
        for line in Path(joined).open(encoding="utf-8"):
            record = json.loads(line)
            if record.get("source") != "arena" or record.get("kind") != "action":
                continue
            if deployment and record.get("deployment") != deployment:
                continue
            verdict = (verdicts or {}).get(record["span_id"])
            match = record.get("match") or {}
            if verdict is None and con is not None and match.get("verified"):
                from poker.teacher import judge

                verdict = judge(con, match["match_id"], match["hand"], match["seat"], match["action_index"])
            add(record, verdict)
    for path in synth_paths:
        for line in Path(path).open(encoding="utf-8"):
            record = json.loads(line)
            if record.get("kind", "action") == "action":
                add(record, record.get("verdict"))
    return out


def _rule(item) -> str:
    return ((item.get("verdict") or {}).get("rule")) or "none"


def select(items: list[dict], n: int | None, seed: int = 0, per_rule: int | None = None) -> list[dict]:
    """Round-robin over teacher rules so small rule families are not drowned; deterministic for a seed."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for item in sorted(items, key=lambda i: i["span_id"]):
        groups[_rule(item)].append(item)
    for rule in groups:
        rng.shuffle(groups[rule])
        if per_rule:
            groups[rule] = groups[rule][:per_rule]
    picked, queues = [], [groups[r] for r in sorted(groups)]
    while queues and (n is None or len(picked) < n):
        for queue in list(queues):
            if n is not None and len(picked) >= n:
                break
            picked.append(queue.pop())
            if not queue:
                queues.remove(queue)
    return picked


def write_batches(items: list[dict], out_dir: Path, n_batches: int, seed: int = 0) -> dict:
    """Write batch_XX.jsonl files (blinded), answer_key.json and INSTRUCTIONS.md; return the key."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    order = list(items)
    rng.shuffle(order)  # mix rules and sources so a batch does not reveal them
    key, rows = {}, []
    for item in order:
        bid = hashlib.sha1(f"{seed}:{item['span_id']}".encode()).hexdigest()[:10]
        state = render_state(item["messages"])
        facts = facts_for(state, item.get("verdict"))
        rows.append({"id": bid, "state": state, "action_taken": item["action"], "facts": facts_block(facts),
                     "facts_json": facts})
        key[bid] = {"span_id": item["span_id"], "rule": _rule(item), "actual": item["action"],
                    "deployment": item.get("deployment"), "source": item.get("source"),
                    "verdict": item.get("verdict")}
    for b in range(max(1, n_batches)):
        chunk = rows[b::max(1, n_batches)]
        with (out_dir / f"batch_{b:02d}.jsonl").open("w", encoding="utf-8") as stream:
            for row in chunk:
                stream.write(json.dumps(row) + "\n")
    (out_dir / "answer_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    (out_dir / "INSTRUCTIONS.md").write_text(INSTRUCTIONS, encoding="utf-8")
    return key


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--joined", type=Path, default=Path("data/arena/joined.jsonl"))
    parser.add_argument("--synth", type=Path, action="append", default=[], help="synth spots jsonl; repeatable")
    parser.add_argument("--db", type=Path, default=Path("data/arena/arena.db"), help="arena.db for teacher verdicts")
    parser.add_argument("--no-db", action="store_true", help="skip teacher verdicts for arena spots")
    parser.add_argument("--verdicts", type=Path, help="JSON {span_id: verdict} instead of calling the teacher")
    parser.add_argument("--deployment")
    parser.add_argument("--n", type=int, default=400, help="decisions to select (0 = all)")
    parser.add_argument("--per-rule", type=int, default=None, help="cap per teacher rule")
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("data/critic/batches"))
    args = parser.parse_args(argv)
    verdicts = json.loads(args.verdicts.read_text()) if args.verdicts else None
    db_path = None if (args.no_db or verdicts is not None or not args.db.exists()) else args.db
    items = load_decisions(args.joined, args.synth, db_path, verdicts, args.deployment)
    picked = select(items, args.n or None, args.seed, args.per_rule)
    key = write_batches(picked, args.out_dir, args.batches, args.seed)
    print(json.dumps({"candidates": len(items), "selected": len(key), "batches": args.batches,
                      "out_dir": str(args.out_dir)}))


if __name__ == "__main__":
    main()
