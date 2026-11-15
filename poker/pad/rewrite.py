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
    """{(match_id, seat): {hand: cleanup messages}} from arena cleanup rows. First row per hand wins.

    The "_meta" entry counts rows skipped for incomplete match metadata and cleanups that did not parse."""
    index: dict = {"_meta": {"incomplete": 0, "unparseable": 0}}
    meta = index["_meta"]
    for row in rows:
        if row.get("kind") != "cleanup" or row.get("source") != "arena":
            continue
        match = row.get("match") or {}
        if match.get("match_id") is None:
            meta["incomplete"] += 1
            continue
        try:
            summary = parse_cleanup(row["messages"])
        except (ValueError, KeyError, AttributeError):
            meta["unparseable"] += 1
            continue
        index.setdefault((match["match_id"], summary.hero), {}).setdefault(summary.hand, row["messages"])
    return index


def _pads(index: dict, match_id, seat, version: int = 1) -> dict[int, str]:
    """{h: pad shown at the start of hand h} for the longest gap-free chain from hand 1 (cached on the index)."""
    cache = index.setdefault("_pads", {})
    key = (match_id, seat, version)
    if key not in cache:
        hands = index.get((match_id, seat), {})
        pads, prev, h = {1: ""}, "", 1
        while h in hands:
            try:
                prev, _ = teacher_pad(hands[h], prev, version)
            except (ValueError, KeyError, AttributeError):
                break
            h += 1
            pads[h] = prev
        cache[key] = pads
    return cache[key]


def truncated_chains(index: dict, version: int = 1) -> dict:
    """{(match_id, seat): {"chain_end", "last_hand"}} for chains that stop before the last cleanup seen: a missing or
    unparseable cleanup in the middle. chain_end is the last hand whose start pad the chain can give."""
    found = {}
    for key, hands in list(index.items()):
        if not isinstance(key, tuple) or len(key) != 2:
            continue
        chain_end, last_hand = max(_pads(index, *key, version)), max(hands)
        if last_hand >= chain_end:
            found[key] = {"chain_end": chain_end, "last_hand": last_hand}
    return found


def teacher_pad_at(index: dict, match_id, seat: int, hand: int, version: int = 1) -> str | None:
    """The pad the writer leaves at the start of `hand` ('' for hand 1); None when the chain cannot be built."""
    return _pads(index, match_id, seat, version).get(hand)


def rewrite_messages(messages: list[dict], pad: str) -> list[dict]:
    assert len(pad) <= LIMIT, f"rendered pad is {len(pad)} chars, over the {LIMIT} limit"
    out = copy.deepcopy(messages)
    for m in out:
        if m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].startswith(HEAD):
            m["content"] = scratchpad_block(pad, LIMIT)
            break
    return out


def rewrite_records(records, matches: dict, index: dict, strict_ids=frozenset(), version: int = 1) -> tuple[list[dict], dict]:
    """matches: decision_id -> (match_id, seat, hand); a missing part (None) skips the record and counts it.
    Only submit_action records are touched. A record whose pad cannot be rewritten is kept as is, unless its
    decision_id is in `strict_ids` (its target depends on the pad): then it is dropped and counted.
    `version` 2 renders the v2 pad schema (default 1)."""
    stats: Counter = Counter(rewritten=0, kept_no_chain=0, dropped_no_chain=0, incomplete_match=0)
    out = []
    for r in records:
        if (r.get("target") or {}).get("tool_call", {}).get("name") != "submit_action":
            stats["skipped_non_action"] += 1
            out.append(r)
            continue
        stats["action_records"] += 1
        key = matches.get(r.get("decision_id"))
        if key is not None and None in key:
            stats["incomplete_match"] += 1
            key = None
        pad = teacher_pad_at(index, key[0], key[1], key[2], version) if key else None
        if pad is None:
            if r.get("decision_id") in strict_ids:
                stats["dropped_no_chain"] += 1
                continue
            stats["kept_no_chain"] += 1
            out.append(r)
            continue
        stats["rewritten"] += 1
        out.append(r | {"messages": rewrite_messages(r["messages"], pad)})
    truncated = truncated_chains(index, version)
    stats["truncated_chains"] = len(truncated)
    stats["truncated_matches"] = sorted({m for m, _ in truncated})
    stats["unparseable_cleanups"] = index.get("_meta", {}).get("unparseable", 0)
    stats["incomplete_cleanup_rows"] = index.get("_meta", {}).get("incomplete", 0)
    return out, dict(stats)
