"""Check converted test records against shuffled original JSON prompts, using content rather than row order."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re

from poker.pokerbench_to_records import action, chips
from poker.records import POSITIONS

RANKS = dict(zip(("two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "jack", "queen", "king", "ace"), "23456789TJQKA"))
SUITS = {"club": "c", "diamond": "d", "heart": "h", "spade": "s"}


def natural_cards(text):
    return [RANKS[r] + SUITS[s] for r, s in re.findall(r"(two|three|four|five|six|seven|eight|nine|ten|jack|queen|king|ace) of (club|diamond|heart|spade)", text.lower())]


def prompt_key(row):
    text = row["instruction"]
    position = re.search(r"your position is (\w+),", text)[1]
    hole = natural_cards(re.search(r"your holding is \[([^]]+)\]", text)[1])
    board = []
    for street in ("flop", "turn", "river"):
        match = re.search(r"The " + street + r" comes (.*?)(?:, then|\.\n)", text)
        if match:
            board += natural_cards(match[1])
    pot = chips(re.search(r"current pot size is ([\d.]+) chips", text)[1])
    move = action(row["output"])
    return tuple(sorted(hole)), tuple(board), POSITIONS.index(position), pot, move["type"], move["to"]


def record_key(row):
    state, move = row["state"], row["action"]
    return tuple(sorted(state["hole"])), tuple(state["board"]), state["seat"], state["pot"], move["type"], move["to"]


def audit(source: Path, records: Path, out: Path, count=20):
    keys = set()
    for path in source.glob("*test_set_prompt_and_label.json"):
        for row in json.loads(path.read_text(encoding="utf-8")):
            keys.add(prompt_key(row))
    if not keys:
        raise ValueError("no reference JSON prompts")
    pools = {"preflop": [], "postflop": []}
    with records.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            pools["preflop" if record["state"]["street"] == "preflop" else "postflop"].append(record)
    rng = random.Random(0)
    selected = []
    for pool in pools.values():
        selected += rng.sample(pool, min(count // 2, len(pool)))
    checks = [{"id": r["id"], "matched_original_prompt": record_key(r) in keys,
               "street": r["state"]["street"], "hole": r["state"]["hole"], "board": r["state"]["board"],
               "pot": r["state"]["pot"], "position": r["state"]["position"], "action": r["action"]} for r in selected]
    report = {"checked": len(checks), "matched": sum(c["matched_original_prompt"] for c in checks),
              "fields": ["hole", "board", "position", "pot", "action"],
              "note": "JSON and CSV rows are shuffled independently; match visible state and label by content.", "checks": checks}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/pokerbench"))
    parser.add_argument("--records", type=Path, default=Path("out/eval/pokerbench.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("out/pokerbench_audit.json"))
    args = parser.parse_args()
    result = audit(args.source, args.records, args.out)
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))
    if result["checked"] < 20 or result["matched"] != result["checked"]:
        raise SystemExit(1)
