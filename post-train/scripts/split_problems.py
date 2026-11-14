"""Stable train / held-out split of a problems.jsonl.

A problem is held out iff md5(id) % k == 0, so adding new problems never moves old ones
between splits (safe while the synthetic set keeps growing).

    uv run python scripts/split_problems.py data/synth_ioi/problems.jsonl --k 4
    -> data/synth_ioi/train.jsonl, data/synth_ioi/heldout.jsonl
"""

import argparse
import hashlib
import json
from pathlib import Path


def is_heldout(pid: str, k: int) -> bool:
    return int(hashlib.md5(pid.encode()).hexdigest(), 16) % k == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("problems")
    ap.add_argument("--k", type=int, default=4, help="hold out ~1/k of problems")
    a = ap.parse_args()
    src = Path(a.problems)
    rows = [json.loads(line) for line in src.open()]
    split = {"train": [], "heldout": []}
    for p in rows:
        split["heldout" if is_heldout(p["id"], a.k) else "train"].append(p)
    for name, ps in split.items():
        with (src.parent / f"{name}.jsonl").open("w") as f:
            for p in ps:
                f.write(json.dumps(p) + "\n")
        print(f"{name}: {len(ps)}  ratings={sorted(p['rating'] for p in ps)}")
