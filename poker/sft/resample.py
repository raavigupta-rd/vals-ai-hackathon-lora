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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MAX_REASONING_CHARS = 2500 * 3.5  # the same budget build_sft_dataset enforces, in characters


def _valid_by_gate(prompt, response) -> bool:
    from poker.gate import parse_reply

    return parse_reply(prompt, response)["valid"]


def pick_reply(prompt, samples, *, preferred: str, validate=_valid_by_gate) -> dict | None:
    """First sample whose single submit_action call chooses `preferred`, is valid, not cut off, and short enough."""
    for response in samples:
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        calls = message.get("tool_calls") or []
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        if choice.get("finish_reason") == "length" or len(calls) != 1 or not reasoning.strip():
            continue
        if len(reasoning) > MAX_REASONING_CHARS or not validate(prompt, response):
            continue
        try:
            args = json.loads(calls[0]["function"]["arguments"])
        except (KeyError, TypeError, ValueError):
            continue
        action = args.get("action") if isinstance(args, dict) else None
        if action == preferred or (preferred == "raise" and action == "bet"):
            if action not in ("bet", "raise"):
                args = {k: v for k, v in args.items() if k != "amount"}
            return {"reasoning": reasoning, "content": message.get("content") or "", "arguments": args}
    return None


def main() -> None:
    from poker.gate import HttpClient, chat_messages, chat_tools, load_entries, load_api_key

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--joined", type=Path, default=Path("data/arena/joined.jsonl"))
    parser.add_argument("--db", type=Path, default=Path("data/arena/arena.db"))
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    client, tools = HttpClient(load_api_key()), chat_tools()
    spots = [e for e in load_entries(args.joined, args.db) if e["verdict"].get("agree") is False]
    done = set()
    if args.out.exists():
        done = {json.loads(line)["span_id"] for line in args.out.open(encoding="utf-8")}
    spots = [e for e in spots if e["record"]["span_id"] not in done]

    def one(entry):
        record, preferred = entry["record"], entry["verdict"]["preferred"]
        messages = chat_messages(record["messages"])
        samples = []
        for _ in range(args.samples):
            samples.append(client.complete(args.deployment, messages, tools))
            picked = pick_reply(record["messages"], samples[-1:], preferred=preferred)
            if picked:
                return record["span_id"], preferred, picked, len(samples)
        return record["span_id"], preferred, None, len(samples)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    found = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool, args.out.open("a", encoding="utf-8") as out:
        for span_id, preferred, picked, tries in pool.map(one, spots):
            found += picked is not None
            out.write(json.dumps({"span_id": span_id, "preferred": preferred, "tries": tries, "reply": picked}) + "\n")
    print(json.dumps({"spots": len(spots), "found": found, "skipped_done": len(done), "out": str(args.out)}))


if __name__ == "__main__":
    main()
