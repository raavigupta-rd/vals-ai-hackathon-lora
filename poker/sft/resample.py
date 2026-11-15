"""Corrections in the model's own words: resample a deployment on spots the teacher corrects.

Cycle 1 showed that fixed correction sentences get memorized verbatim and shorten or break replies. At arena
temperature the model often picks the teacher's action on its own, so for each corrected spot we sample it a few
times and keep the first valid, uncut reply that does. make_records then uses that reply as the target and falls
back to the fixed sentences only where no sample agreed.

    uv run python -m poker.sft.resample --deployment vals-hackathon/sft-c1-s10 --samples 6 \
        --out data/sft/resampled_c2.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MAX_REASONING_CHARS = 2500 * 3.5  # the same budget build_sft_dataset enforces, in characters
HINT_HEADER = ("\n\nPrivate coaching for this decision only. Use it in your own reasoning, but never mention "
               "coaching, notes or hints: ")
HINT_WORDS = re.compile(r"\b(coach\w*|hint\w*|note[sd]?|told|instruction\w*)\b", re.I)


def hinted_messages(messages: list[dict], hint: str) -> list[dict]:
    """Teacher-only context: the hint is appended to the system message (as in OPSD). The student never sees it."""
    out = [dict(m) for m in messages]
    out[0]["content"] = (out[0].get("content") or "") + HINT_HEADER + hint
    return out


def mentions_hint(reasoning: str) -> bool:
    """A target that refers to a hint would teach the student to cite context it never has."""
    return bool(HINT_WORDS.search(reasoning or ""))


def _valid_by_gate(prompt, response) -> bool:
    from poker.gate import parse_reply

    return parse_reply(prompt, response)["valid"]


def _legal_min(prompt) -> int | None:
    from poker.hints import parse_prompt

    try:
        state = parse_prompt(prompt)
    except Exception:  # an odd prompt only means we cannot read the minimum
        state = None
    if state and state["raise_min"] is not None:
        return state["raise_min"]
    found = re.findall(r"(?:raise|bet) to (\d+)-\d+", json.dumps(prompt[-1].get("content") if prompt else ""))
    return int(found[-1]) if found else None


def pick_reply(prompt, samples, *, preferred: str, validate=_valid_by_gate, max_amount=None,
               min_target=None) -> dict | None:
    """First sample whose single submit_action call chooses `preferred`, is valid, not cut off, and short enough.
    With `max_amount` (a verdict's target_amount) the call must also be a raise or bet of at most that amount and
    at least the legal minimum. `min_target` (a verdict's floor, e.g. no min-raise opens) sets a lower bound on that amount."""
    for response in samples:
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        calls = message.get("tool_calls") or []
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        if choice.get("finish_reason") == "length" or len(calls) != 1 or not reasoning.strip():
            continue
        if len(reasoning) > MAX_REASONING_CHARS or mentions_hint(reasoning) or not validate(prompt, response):
            continue
        try:
            args = json.loads(calls[0]["function"]["arguments"])
        except (KeyError, TypeError, ValueError):
            continue
        action = args.get("action") if isinstance(args, dict) else None
        if action == preferred or (preferred == "raise" and action == "bet"):
            if max_amount is not None:
                amount = args.get("amount")
                low = _legal_min(prompt)
                if (action not in ("bet", "raise") or not isinstance(amount, (int, float)) or amount > max_amount
                        or (low is not None and amount < low)):
                    continue
            if min_target is not None:
                amount = args.get("amount")
                if action not in ("bet", "raise") or not isinstance(amount, (int, float)) or amount < min_target:
                    continue
            if action not in ("bet", "raise"):
                args = {k: v for k, v in args.items() if k != "amount"}
            return {"reasoning": reasoning, "content": message.get("content") or "", "arguments": args}
    return None


def pick_shortest(prompt, samples, *, preferred: str, validate=_valid_by_gate, max_amount=None,
                  min_target=None) -> dict | None:
    """Among all samples that pass pick_reply's checks, the one with the shortest reasoning. Teams that improved
    on the arena ladder moved to short, decisive replies; picking the shortest valid target teaches brevity."""
    picks = [p for s in samples if (p := pick_reply(prompt, [s], preferred=preferred, validate=validate, max_amount=max_amount,
                                                  min_target=min_target))]
    return min(picks, key=lambda p: len(p["reasoning"])) if picks else None


def main() -> None:
    from poker.gate import HttpClient, chat_messages, chat_tools, load_entries, load_api_key

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--joined", type=Path, default=Path("data/arena/joined.jsonl"))
    parser.add_argument("--db", type=Path, default=Path("data/arena/arena.db"))
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--hinted-samples", type=int, default=4, help="re-rolls with a coaching hint after plain ones fail")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--shortest", action="store_true", help="draw all plain samples and keep the shortest valid one")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    client, tools = HttpClient(load_api_key()), chat_tools()
    spots = [e for e in load_entries(args.joined, args.db) if e["verdict"].get("agree") is False]
    done = set()
    if args.out.exists():
        done = {(item["span_id"], item.get("preferred")) for item in map(json.loads, args.out.open(encoding="utf-8"))}
    # A spot is redone when the teacher's preferred action changed since it was sampled.
    spots = [e for e in spots if (e["record"]["span_id"], e["verdict"]["preferred"]) not in done]

    from poker.sft.make_records import correction_reasoning
    from poker.teacher import hand_code

    def code_for(record):
        m = record["match"]
        c = sqlite3.connect(args.db)
        row = c.execute("SELECT card1, card2 FROM hole_cards WHERE match_id=? AND hand_number=? AND seat=?",
                        (m["match_id"], m["hand"], m["seat"])).fetchone()
        c.close()
        return hand_code([row[0], row[1]]) if row and row[0] else "this hand"

    def one(entry):
        record, verdict = entry["record"], entry["verdict"]
        preferred = verdict["preferred"]
        cap = verdict.get("target_amount")
        floor = verdict.get("min_target")
        messages = chat_messages(record["messages"])
        tries = 0
        if args.shortest:
            drawn = [client.complete(args.deployment, messages, tools) for _ in range(args.samples)]
            tries = len(drawn)
            picked = pick_shortest(record["messages"], drawn, preferred=preferred, max_amount=cap, min_target=floor)
            if picked:
                return record["span_id"], preferred, picked | {"hinted": False}, tries
        for _ in range(0 if args.shortest else args.samples):
            tries += 1
            picked = pick_reply(record["messages"], [client.complete(args.deployment, messages, tools)], preferred=preferred,
                               max_amount=cap, min_target=floor)
            if picked:
                return record["span_id"], preferred, picked | {"hinted": False}, tries
        try:
            hint = correction_reasoning(verdict, code_for(record))
        except ValueError:
            return record["span_id"], preferred, None, tries
        teacher = hinted_messages(messages, hint)
        for _ in range(args.hinted_samples):
            tries += 1
            picked = pick_reply(record["messages"], [client.complete(args.deployment, teacher, tools)], preferred=preferred,
                               max_amount=cap, min_target=floor)
            if picked:
                return record["span_id"], preferred, picked | {"hinted": True}, tries
        return record["span_id"], preferred, None, tries

    args.out.parent.mkdir(parents=True, exist_ok=True)
    found = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool, args.out.open("a", encoding="utf-8") as out:
        for span_id, preferred, picked, tries in pool.map(one, spots):
            found += picked is not None
            out.write(json.dumps({"span_id": span_id, "preferred": preferred, "tries": tries, "reply": picked}) + "\n")
    print(json.dumps({"spots": len(spots), "found": found, "skipped_done": len(done), "out": str(args.out)}))


if __name__ == "__main__":
    main()
