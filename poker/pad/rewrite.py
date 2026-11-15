"""Make SFT action records pad-consistent: show the teacher pad the deterministic writer would have left.

Action turns in joined.jsonl carry whatever pad the model wrote itself. The teacher judges with facts the model
only sees through its pad (seats are anonymous), so a target that depends on opponent stats needs the pad to hold
them. This chains `poker.pad.writer.teacher_pad` over the match's cleanup prompts (hands 1..h-1 for one seat) and
swaps the scratchpad block of the hand-h action prompt for the result, in the exact harness wrapper.

The chain needs every earlier hand of that (match, seat). Hand 1 starts from the empty pad. When any earlier
cleanup prompt is missing or unparseable, the record keeps its original block and is counted.
"""

from __future__ import annotations

import copy
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "harness/src"))  # the harness is not an installed package

from poker.pad.writer import parse_cleanup, teacher_pad  # noqa: E402
from poker_harness.prompts import scratchpad_block  # noqa: E402

LIMIT = 5000
HEAD = "## Your scratchpad"


def chain_index(rows) -> dict:
    """{(match_id, seat): {hand: parsed cleanup messages}} from arena cleanup rows. First row per hand wins."""
    index: dict = {}
    for row in rows:
        match = row.get("match") or {}
        if row.get("kind") != "cleanup" or row.get("source") != "arena" or not match.get("match_id"):
            continue
        try:
            summary = parse_cleanup(row["messages"])
        except (ValueError, KeyError, AttributeError):
            continue
        index.setdefault((match["match_id"], summary.hero), {}).setdefault(summary.hand, row["messages"])
    return index


def _pads(index: dict, match_id, seat) -> dict[int, str]:
    """{h: pad shown at the start of hand h} for the longest gap-free chain from hand 1 (cached on the index)."""
    cache = index.setdefault("_pads", {})
    key = (match_id, seat)
    if key not in cache:
        hands = index.get(key, {})
        pads, prev, h = {1: ""}, "", 1
        while h in hands:
            try:
                prev, _ = teacher_pad(hands[h], prev)
            except (ValueError, KeyError, AttributeError):
                break
            h += 1
            pads[h] = prev
        cache[key] = pads
    return cache[key]


def teacher_pad_at(index: dict, match_id, seat: int, hand: int) -> str | None:
    """The pad the writer leaves at the start of `hand` ('' for hand 1); None when the chain cannot be built."""
    return _pads(index, match_id, seat).get(hand)


def rewrite_messages(messages: list[dict], pad: str) -> list[dict]:
    out = copy.deepcopy(messages)
    for m in out:
        if m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].startswith(HEAD):
            m["content"] = scratchpad_block(pad, LIMIT)
            break
    return out


def rewrite_records(records, matches: dict, index: dict) -> tuple[list[dict], dict]:
    """matches: decision_id -> (match_id, seat, hand). Only submit_action records are touched."""
    stats: Counter = Counter()
    out = []
    for r in records:
        if (r.get("target") or {}).get("tool_call", {}).get("name") != "submit_action":
            stats["skipped_non_action"] += 1
            out.append(r)
            continue
        stats["action_records"] += 1
        key = matches.get(r.get("decision_id"))
        pad = teacher_pad_at(index, key[0], key[1], key[2]) if key else None
        if pad is None:
            stats["kept_no_chain"] += 1
            out.append(r)
            continue
        stats["rewritten"] += 1
        out.append(r | {"messages": rewrite_messages(r["messages"], pad)})
    return out, dict(stats)
