"""Join normalized gateway decisions to public arena match JSON (match, hand, seat, outcome).

The student-visible prompt stays in `messages`; hindsight fields (hand delta, final place, actual
action) live only under `outcome` so they cannot leak into training prompts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import re

from poker.hints import parse_prompt

SLACK = timedelta(minutes=2)


def _time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def _state(record):
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None, None
    state = parse_prompt(messages)
    text = "\n".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
    hand = re.search(r"^## Hand (\d+) of", text, re.M)
    return state, int(hand[1]) if hand else None


def _find_hand(match, hand_number, seat, hole):
    for hand in match.get("hands", []):
        if hand_number is not None and hand["hand_number"] != hand_number:
            continue
        for player in hand.get("players", []):
            if player["seat"] == seat and (hole is None or sorted(player["hole_cards"]) == sorted(hole)):
                return hand
    return None


def _candidates(record, matches):
    ts, dep = _time(record.get("ts")), record.get("deployment")
    for match in matches:
        start, end = _time(match.get("started_at")), _time(match.get("finished_at"))
        if not (start and end and ts and start - SLACK <= ts <= end + SLACK):
            continue
        for seat in match.get("seats", []):
            if (seat.get("entry") or {}).get("deployment_id") == dep:
                yield match, seat["seat"]


def _locate(record, by_id, matches):
    state, hand_number = _state(record)
    hole = state["hole"] if state else None
    arena = record.get("arena")
    if arena and arena["match_id"] in by_id:
        match = by_id[arena["match_id"]]
        hand = _find_hand(match, arena["hand"], arena["seat"], None)
        if hand:
            return match, hand, arena["seat"], "conversation_id"
    for match, seat in _candidates(record, matches):
        seat = state["seat"] if state else seat
        hand = _find_hand(match, hand_number, seat, hole)
        if hand and hole:
            return match, hand, seat, "window+cards"
    return None


def _outcome(match, hand, seat, action):
    result = next((r for r in match.get("results", []) if r["seat"] == seat), {})
    out = {"hand_delta": (hand.get("deltas") or {}).get(str(seat)),
           "stack_after": (hand.get("stacks_after") or {}).get(str(seat)),
           "showdown": hand.get("showdown"), "final_place": result.get("place"),
           "match_net_chips": result.get("net_chips"), "match_final_chips": result.get("final_chips")}
    if action:
        out["actual_action"] = {"action": action.get("action"), "amount": action.get("amount")}
        for key in ("fallback", "coerced", "format_error", "afk", "token_cap", "timed_out"):
            out[key] = action.get(key)
    return out


def join(records, matches):
    """Return one joined record per input; `match` is None when nothing matched."""
    by_id = {m["id"]: m for m in matches}
    located = {}
    groups = {}
    for record in records:
        found = _locate(record, by_id, matches)
        located[record["span_id"]] = found
        if found and record.get("kind") == "action":
            match, hand, seat, _ = found
            groups.setdefault((match["id"], hand["hand_number"], seat), []).append(record)
    index = {}
    for key, group in groups.items():
        for i, record in enumerate(sorted(group, key=lambda r: r["ts"])):
            index[record["span_id"]] = i
    result = []
    for record in records:
        joined = dict(record)
        found = located[record["span_id"]]
        if not found:
            joined["match"], joined["outcome"] = None, None
            result.append(joined)
            continue
        match, hand, seat, method = found
        i = index.get(record["span_id"])
        actions = [a for a in hand.get("actions", []) if a["seat"] == seat and not a.get("scripted")]
        action = actions[i] if i is not None and i < len(actions) else None
        state, _ = _state(record)
        verified = bool(action and state and action.get("street") == state["street"]) if i is not None else None
        if method != "conversation_id" and not verified:
            joined["match"], joined["outcome"] = None, None  # a time+cards guess needs the street to agree
            result.append(joined)
            continue
        joined["match"] = {"match_id": match["id"], "hand": hand["hand_number"], "seat": seat,
                           "action_index": i, "method": method, "verified": verified}
        joined["outcome"] = _outcome(match, hand, seat, action)
        result.append(joined)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", type=Path, required=True, help="gateway JSONL from pull_gateway.py")
    parser.add_argument("--matches", type=Path, default=Path("data/arena/raw"), help="dir of match JSON files")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    matches = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.matches.glob("*.json"))]
    records = [json.loads(line) for line in args.gateway.open(encoding="utf-8")]
    joined = join(records, matches)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for record in joined:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    hit = sum(1 for r in joined if r["match"])
    print(json.dumps({"records": len(joined), "joined": hit, "verified": sum(1 for r in joined if r["match"] and r["match"]["verified"]),
                      "matches_loaded": len(matches)}))
