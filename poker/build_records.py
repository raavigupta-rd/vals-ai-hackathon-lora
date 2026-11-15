"""Validate, deduplicate and split engine-verified labels without test/hand leakage."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import tempfile

from jsonschema import ValidationError
from poker.records import digest, fingerprint, hand_key, legal_action, validate


def rows(paths):
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    yield json.loads(line), None
                except json.JSONDecodeError:
                    yield None, f"{path}:{number}: malformed JSON"


def build(inputs: list[Path], held_out: list[Path], output: Path, *, val_fraction=.1, fold_cap=.25) -> dict:
    if not 0 < val_fraction < 1 or not 0 <= fold_cap < 1:
        raise ValueError("val fraction must be between 0 and 1; fold cap must be in [0,1)")
    output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    distributions = {s: {k: Counter() for k in ("source", "street", "action", "stack_bb")} for s in ("train", "val")}
    # Disk-backed staging keeps a half-million-record conversion from exhausting a laptop's RAM.
    with tempfile.TemporaryDirectory(prefix="poker-clean-") as temp:
        db = sqlite3.connect(str(Path(temp) / "records.sqlite"))
        db.execute("create table held (kind text, key text, primary key(kind,key))")
        db.execute("create table records (fp text primary key, split text, fold integer, data text)")
        for record, error in rows(held_out + inputs):
            if error or not isinstance(record, dict):
                continue
            if record.get("meta", {}).get("split") != "test":
                continue
            validate(record)
            for kind, key in (("hand", hand_key(record)), ("state", fingerprint(record))):
                db.execute("insert or ignore into held values (?,?)", (kind, key))
        db.commit()
        for record, error in rows(inputs):
            counts["input"] += 1
            try:
                if error:
                    raise ValueError(error)
                validate(record)
            except (ValueError, ValidationError, TypeError):
                counts["drop_schema"] += 1
                continue
            if record["meta"]["split"] == "test":
                counts["drop_test"] += 1
                continue
            fp, hand = fingerprint(record), hand_key(record)
            if db.execute("select 1 from held where (kind='hand' and key=?) or (kind='state' and key=?)", (hand, fp)).fetchone():
                counts["drop_test_overlap"] += 1
                continue
            if not record["meta"]["engine_verified"]:
                counts["drop_unverified"] += 1
                continue
            if not legal_action(record):
                counts["drop_illegal"] += 1
                continue
            amounts = [record["state"]["pot"], record["state"]["to_call"], record["action"]["to"],
                       *record["state"]["stacks"], *[h["to"] for h in record["state"]["history"]]]
            if any(v is not None and not float(v).is_integer() for v in amounts):
                # Solver sizes such as 2.3 BB become 4.6 event chips. The event tool takes integers; do not
                # silently round the history, alter pot odds, or call those states reachable event examples.
                counts["drop_fractional_event_chips"] += 1
                continue
            split = "val" if int(digest(hand)[:8], 16) / 2**32 < val_fraction else "train"
            fold = record["state"]["street"] == "preflop" and record["action"]["type"] == "fold"
            inserted = db.execute("insert or ignore into records values (?,?,?,?)", (fp, split, fold, json.dumps(record))).rowcount
            if not inserted:
                counts["drop_duplicate"] += 1
        db.commit()
        for split in ("train", "val"):
            other = db.execute("select count(*) from records where split=? and fold=0", (split,)).fetchone()[0]
            allowed = int(other * fold_cap / (1 - fold_cap))
            fold_kept = 0
            with (output / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
                for is_fold, text in db.execute("select fold,data from records where split=? order by fp", (split,)):
                    if is_fold and fold_kept >= allowed:
                        counts["drop_fold_cap"] += 1
                        continue
                    fold_kept += is_fold
                    record = json.loads(text)
                    record["meta"]["partition"] = split
                    state = record["state"]
                    stats = distributions[split]
                    for key, value in (("source", record["source"]), ("street", state["street"]), ("action", record["action"]["type"])):
                        stats[key][value] += 1
                    depth = state["effective_stack"]
                    bucket = "unknown" if depth is None else "<=20" if depth <= 40 else "<=50" if depth <= 100 else "<=100" if depth <= 200 else ">100"
                    stats["stack_bb"][bucket] += 1
                    counts[split] += 1
                    stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        db.close()
    stats = {"counts": dict(counts), "distributions": distributions, "val_fraction": val_fraction,
             "preflop_fold_cap": fold_cap, "inputs": [str(p) for p in inputs], "held_out": [str(p) for p in held_out],
             "note": "PokerBench hand IDs are conservative scenario families; all test families are excluded."}
    (output / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--held-out", nargs="*", type=Path, default=[])
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--val-fraction", type=float, default=.1)
    parser.add_argument("--fold-cap", type=float, default=.25)
    args = parser.parse_args()
    protected = {p.resolve() for p in args.inputs + args.held_out}
    if any((args.out / name).resolve() in protected for name in ("train.jsonl", "val.jsonl")):
        parser.error("output would overwrite an input")
    print(json.dumps(build(args.inputs, args.held_out, args.out, val_fraction=args.val_fraction, fold_cap=args.fold_cap), indent=2))


if __name__ == "__main__":
    main()
