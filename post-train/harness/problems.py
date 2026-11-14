"""Build a local eval set shaped like the Vals Track-2 tasks from open-r1/codeforces (test split).

Each problem: a plain-text statement with samples, limits, and hidden tests split into
"small" (40 pts) and "large" (60 pts) halves by input size. Tests are written to the sandbox's
read-only tests dir; Python checkers (for problems with several valid answers) are kept.

The test split is 2024-2025 Codeforces contests. Keep these problems OUT of any training data.

    python -m harness.problems --out data/eval/cf_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd

from harness import sandbox

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "open_r1_cf"


def _clean(s: str | None) -> str:
    if not isinstance(s, str) or not s:
        return ""
    s = s.replace("$$$", "$")
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def render_statement(row, mem_mib: int) -> str:
    parts = [row.title.upper(), "", _clean(row.description), "", "Input", _clean(row.input_format), "",
             "Output", _clean(row.output_format)]
    for i, ex in enumerate(row.examples, 1):
        parts += ["", f"Sample input {i} (file sample_{i}.in)", ex["input"].rstrip(), "",
                  f"Sample output {i} (file sample_{i}.out)", ex["output"].rstrip()]
    if isinstance(row.note, str) and row.note.strip():
        parts += ["", "Note", _clean(row.note)]
    parts += ["", f"Limits: {row.time_limit:g} seconds and {mem_mib} MiB of memory per test."]
    return "\n".join(parts)


def _write_pairs(d: Path, prefix: str, pairs: list[tuple[str, str]]) -> list[list[str]]:
    out = []
    for i, (inp, exp) in enumerate(pairs):
        fi, fo = d / f"{prefix}{i}.in", d / f"{prefix}{i}.out"
        fi.write_text(inp)
        fo.write_text(exp)
        out.append([str(fi.relative_to(sandbox.tests_dir())), str(fo.relative_to(sandbox.tests_dir()))])
    return out


def build(out: Path, min_rating: int, max_rating: int, max_tests: int, seed: int) -> None:
    df = pd.read_parquet(DATA / "test.parquet")
    df = df[df.executable & df.interaction_format.isna() & df.rating.between(min_rating, max_rating)]
    contests = [c.strip() for c in (DATA / "selected_contests.txt").read_text().split()]
    df = df[df.contest_id.isin(contests)]
    rng = random.Random(seed)
    probs = []
    for cid, group in df.groupby("contest_id"):
        f = DATA / "generated_tests" / f"test_cases_{int(cid):04d}.parquet"
        if not f.exists():
            print("missing tests for contest", cid)
            continue
        gen = pd.read_parquet(f)
        for _, row in group.iterrows():
            g = gen[gen.problem_id == row.id]
            hidden = [(t["input"], t["output"]) for t in row.official_tests
                      if not t["input"].rstrip().endswith("...") and not t["output"].rstrip().endswith("...")]
            hidden += list(zip(g.input, g.output))
            seen, uniq = set(), []
            for t in hidden:
                if t[0] not in seen and t[1].strip():
                    seen.add(t[0])
                    uniq.append(t)
            if len(uniq) < 6:
                print("skip (few tests)", row.id, len(uniq))
                continue
            uniq.sort(key=lambda t: len(t[0]))
            half = len(uniq) // 2
            small, large = uniq[:half], uniq[half:]
            k = max_tests // 2
            small = sorted(rng.sample(small, min(k, len(small))), key=lambda t: len(t[0]))
            large = large[-3:] + rng.sample(large[:-3], min(max(0, k - 3), len(large) - 3))
            large.sort(key=lambda t: len(t[0]))
            pid = row.id.replace("/", "_")
            d = sandbox.tests_dir() / pid
            d.mkdir(parents=True, exist_ok=True)
            checker = None
            if isinstance(row.generated_checker, str) and row.generated_checker.strip():
                (d / "checker.py").write_text(row.generated_checker)
                checker = str((d / "checker.py").relative_to(sandbox.tests_dir()))
            mem_mib = int(row.memory_limit)
            probs.append({
                "id": pid,
                "title": row.title,
                "rating": int(row.rating),
                "tags": list(row.tags),
                "time_limit": float(row.time_limit),
                "memory_limit": mem_mib * 1024 * 1024,
                "statement": render_statement(row, mem_mib),
                "samples": [[ex["input"], ex["output"]] for ex in row.examples],
                "checker": checker,
                "hidden_small": _write_pairs(d, "s", small),
                "hidden_large": _write_pairs(d, "l", large),
                "max_input_bytes": len(large[-1][0]),
            })
    probs.sort(key=lambda p: (p["rating"], p["id"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for p in probs:
            fh.write(json.dumps(p) + "\n")
    print(f"wrote {len(probs)} problems -> {out}")


def load(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def test_paths(rel_pairs: list[list[str]]) -> list[tuple[Path, Path]]:
    base = sandbox.tests_dir()
    return [(base / i, base / o) for i, o in rel_pairs]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "eval" / "cf_test.jsonl"))
    ap.add_argument("--min-rating", type=int, default=1400)
    ap.add_argument("--max-rating", type=int, default=2600)
    ap.add_argument("--max-tests", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(Path(a.out), a.min_rating, a.max_rating, a.max_tests, a.seed)
