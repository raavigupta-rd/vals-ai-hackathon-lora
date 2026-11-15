"""Feed synthetic, pre-judged spots (poker/synth/judge_spots.py output) to the make_records / resample logic.

make_records.select and resample.pick_reply need no DB row once the verdict and hand code are on the record, so
this wraps them without changing either module.

    uv run python -m poker.synth.adapter resample --deployment vals-hackathon/base \
        --spots data/synth/spots.jsonl --out data/synth/resampled.jsonl
    uv run python -m poker.synth.adapter select --spots data/synth/spots.jsonl --resampled data/synth/resampled.jsonl \
        --cycle 3 --out data/sft/synth_c3.jsonl
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from poker.sft import make_records, resample


def load_spots(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def to_rows(spots: list[dict]) -> list[tuple[dict, dict, str]]:
    """(record, verdict, hand code) rows, the shape make_records.select takes."""
    return [(s, s["verdict"], s.get("hand_code") or "this hand") for s in spots]


def load_resampled(path: Path | None) -> dict:
    out = {}
    if path and Path(path).exists():
        for item in map(json.loads, Path(path).open(encoding="utf-8")):
            if item.get("reply"):
                out[item["span_id"]] = item["reply"] | {"preferred": item.get("preferred")}
    return out


def select(spots: list[dict], *, cycle: int, resampled: dict | None = None, **kwargs):
    """make_records.select on synthetic spots."""
    return make_records.select(to_rows(spots), cycle=cycle, resampled=resampled, **kwargs)


def resample_spots(spots: list[dict], client, tools, deployment: str, *, samples: int = 6, hinted_samples: int = 4,
                   concurrency: int = 8, done: set | None = None) -> list[dict]:
    """The same plain-then-hinted loop as resample.main, on spots the teacher corrects (agree is False)."""
    from poker.gate import chat_messages

    done = done or set()
    todo = [s for s in spots if s["verdict"].get("agree") is False and (s["span_id"], s["verdict"]["preferred"]) not in done]

    def one(record):
        preferred = record["verdict"]["preferred"]
        messages = chat_messages(record["messages"])
        tries = 0
        for _ in range(samples):
            tries += 1
            picked = resample.pick_reply(record["messages"], [client.complete(deployment, messages, tools)], preferred=preferred)
            if picked:
                return {"span_id": record["span_id"], "preferred": preferred, "tries": tries, "reply": picked | {"hinted": False}}
        try:
            hint = make_records.correction_reasoning(record["verdict"], record.get("hand_code") or "this hand")
        except ValueError:
            return {"span_id": record["span_id"], "preferred": preferred, "tries": tries, "reply": None}
        teacher = resample.hinted_messages(messages, hint)
        for _ in range(hinted_samples):
            tries += 1
            picked = resample.pick_reply(record["messages"], [client.complete(deployment, teacher, tools)], preferred=preferred)
            if picked:
                return {"span_id": record["span_id"], "preferred": preferred, "tries": tries, "reply": picked | {"hinted": True}}
        return {"span_id": record["span_id"], "preferred": preferred, "tries": tries, "reply": None}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(one, todo))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    rs = sub.add_parser("resample")
    rs.add_argument("--deployment", required=True)
    rs.add_argument("--spots", type=Path, required=True)
    rs.add_argument("--samples", type=int, default=6)
    rs.add_argument("--hinted-samples", type=int, default=4)
    rs.add_argument("--concurrency", type=int, default=8)
    rs.add_argument("--out", type=Path, required=True)
    sel = sub.add_parser("select")
    sel.add_argument("--spots", type=Path, required=True)
    sel.add_argument("--resampled", type=Path)
    sel.add_argument("--cycle", type=int, required=True)
    sel.add_argument("--abstain-ratio", type=float, default=2.0)
    sel.add_argument("--template-weight", type=float, default=0.0)
    sel.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spots = load_spots(args.spots)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.cmd == "resample":
        from poker.gate import HttpClient, chat_tools, load_api_key

        done = set()
        if args.out.exists():
            done = {(i["span_id"], i.get("preferred")) for i in map(json.loads, args.out.open(encoding="utf-8"))}
        results = resample_spots(spots, HttpClient(load_api_key()), chat_tools(), args.deployment, samples=args.samples,
                                 hinted_samples=args.hinted_samples, concurrency=args.concurrency, done=done)
        with args.out.open("a", encoding="utf-8") as stream:
            for item in results:
                stream.write(json.dumps(item) + "\n")
        found = [r for r in results if r["reply"]]
        print(json.dumps({"spots": len(results), "validated": len(found),
                          "plain": sum(not r["reply"]["hinted"] for r in found),
                          "hinted": sum(r["reply"]["hinted"] for r in found), "out": str(args.out)}))
    else:
        records, stats = select(spots, cycle=args.cycle, resampled=load_resampled(args.resampled),
                                abstain_ratio=args.abstain_ratio, template_weight=args.template_weight)
        with args.out.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")
        print(json.dumps(stats | {"records": len(records), "out": str(args.out)}))


if __name__ == "__main__":
    main()
