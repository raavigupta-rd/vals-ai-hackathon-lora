"""Build SFT cleanup records (teacher scratchpads) from our arena cleanup prompts.

    uv run python -m poker.pad.build_records --joined data/arena/joined.jsonl --out data/pad_records.jsonl
    uv run python -m poker.pad.build_records --db data/arena/arena.db --multi-only --limit 500 --out ...

Each output line is a record `poker/sft/build_sft_dataset.py` accepts: the real cleanup prompt as `messages`
and a target with a short reasoning note, empty content and one update_scratchpad call. Multi-turn hands
(3 or more hero turns, or the last hero action was a check) come first and get a higher weight, because that is
where fine-tunes fail the cleanup turn. With `--chain` (default) the pad shown in each prompt is replaced by the
teacher pad written for the same seat's previous hand, so the counts the target accumulates are derivable from
the prompt; hands whose predecessor is missing keep the original pad. `--no-chain` keeps every original prompt.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from poker.pad.writer import parse_cleanup, teacher_pad, teacher_reasoning

ROOT = Path(__file__).resolve().parents[2]
MULTI_WEIGHT, SINGLE_WEIGHT = 2.0, 1.0
PAD_HEAD = "## Your scratchpad (private: only you can see it)\n<<<SCRATCHPAD\n"


def is_multi_turn(hero_actions: list[str]) -> bool:
    return len(hero_actions) >= 3 or (bool(hero_actions) and hero_actions[-1] == "check")


def with_pad(messages: list[dict], pad: str) -> list[dict]:
    """Copy of the prompt whose first user message shows `pad` as the current scratchpad."""
    out = copy.deepcopy(messages)
    for m in out:
        if m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].startswith("## Your scratchpad"):
            m["content"] = (f"{PAD_HEAD}{pad}\nSCRATCHPAD>>>\n{len(pad)}/5000 characters. "
                            "You can rewrite it after this hand.")
            break
    return out


def db_shown(db: sqlite3.Connection, match_id: str, hand: int) -> dict[int, frozenset] | None:
    rows = db.execute("select seat, card1, card2 from hole_cards where match_id=? and hand_number=? and shown=1",
                      (match_id, hand)).fetchall()
    if not rows and not db.execute("select 1 from hands where match_id=? and hand_number=?", (match_id, hand)).fetchone():
        return None
    return {seat: frozenset((a, b)) for seat, a, b in rows}


def db_agrees(db, match_id: str, summary) -> bool | None:
    """True/False when the arena DB confirms/contradicts the showdown cards in the prompt; None if the hand is absent."""
    shown = db_shown(db, match_id, summary.hand)
    if shown is None:
        return None
    parsed = {s: frozenset(c.split()) for s, c in summary.showdown.items()}
    return all(shown.get(s) == c for s, c in parsed.items())


def build(joined: Path, out: Path, *, chain: bool = True, sources=("arena",), multi_only: bool = False,
          limit: int | None = None, db_path: Path | None = None) -> dict:
    stats: Counter = Counter()
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) if db_path else None
    parsed = []
    seen_ids = set()
    with joined.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_json"] += 1
                continue
            if row.get("kind") != "cleanup" or row.get("source") not in sources:
                continue
            stats["cleanup_rows"] += 1
            try:
                summary = parse_cleanup(row["messages"])
            except (ValueError, KeyError, AttributeError):
                stats["unparseable"] += 1
                continue
            if summary.stacks.get(summary.hero, 0) <= 0:
                stats["hero_out"] += 1
                continue
            match = row.get("match") or {}
            match_id = match.get("match_id")
            decision_id = f"pad:{match_id}:h{summary.hand}:s{summary.hero}" if match_id else f"pad:{row.get('span_id')}"
            if decision_id in seen_ids:
                stats["duplicate"] += 1
                continue
            seen_ids.add(decision_id)
            if db is not None and match_id:
                verdict = db_agrees(db, match_id, summary)
                stats["db_verified" if verdict else "db_missing" if verdict is None else "db_mismatch"] += 1
                if verdict is False:
                    continue
            parsed.append((match_id, summary.hero, summary.hand, decision_id, row, summary))
    parsed.sort(key=lambda t: (str(t[0]), t[1], t[2]))
    records, last_pad = [], {}
    for match_id, hero, hand, decision_id, row, summary in parsed:
        prev = last_pad.get((match_id, hero)) if chain and match_id else None
        messages = with_pad(row["messages"], prev) if prev is not None else row["messages"]
        pad, state = teacher_pad(messages)
        if chain and match_id:
            last_pad[(match_id, hero)] = pad
        multi = is_multi_turn(summary.hero_actions)
        stats["multi_turn" if multi else "single_turn"] += 1
        stats["chained"] += prev is not None
        records.append({
            "messages": messages,
            "target": {"reasoning": teacher_reasoning(summary, state), "content": "",
                       "tool_call": {"name": "update_scratchpad", "arguments": {"contents": pad}}},
            "weight": MULTI_WEIGHT if multi else SINGLE_WEIGHT,
            "source": "pad_teacher", "decision_id": decision_id,
            "meta": {"match_id": match_id, "hand": hand, "seat": hero, "hero_turns": summary.hero_turns,
                     "priority": "multi_turn" if multi else "single_turn", "chained": prev is not None,
                     "pad_chars": len(pad)},
        })
    records.sort(key=lambda r: r["meta"]["priority"] != "multi_turn")  # stable: multi-turn first
    if multi_only:
        records = [r for r in records if r["meta"]["priority"] == "multi_turn"]
    if limit is not None:
        records = records[:limit]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as stream:
        for r in records:
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["written"] = len(records)
    return dict(stats)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joined", type=Path, default=ROOT / "data/arena/joined.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "data/pad_records.jsonl")
    ap.add_argument("--db", type=Path, default=None, help="arena.db: drop records whose showdown cards disagree with it")
    ap.add_argument("--sources", default="arena", help="comma list of joined 'source' values (arena, other)")
    ap.add_argument("--no-chain", action="store_true", help="keep each prompt's original pad")
    ap.add_argument("--multi-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    stats = build(a.joined, a.out, chain=not a.no_chain, sources=tuple(a.sources.split(",")), multi_only=a.multi_only,
                  limit=a.limit, db_path=a.db)
    print(json.dumps(stats, indent=1, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
