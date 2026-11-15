"""Compare judged results: Qwen vs R1, by rating bucket.

Usage: python report.py <qwen_results_name> <r1_results_name>
"""

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"
BUCKETS = [(1600, 1999), (2000, 2399), (2400, 2800)]
LIMIT = 16384


def load(name: str) -> list[dict]:
    return [json.loads(l) for l in open(RES / f"{name}.jsonl")]


def per_problem(rows: list[dict]) -> dict[str, list[bool]]:
    out: dict[str, list[bool]] = {}
    for r in rows:
        out.setdefault(r["id"], []).append(r["verdict"] == "AC")
    return out


def summarize(label: str, rows: list[dict]) -> None:
    print(f"\n== {label}: {len(rows)} samples ==")
    print("verdicts:", dict(Counter(r["verdict"] for r in rows).most_common()))
    toks = [r["n_tokens"] for r in rows if r.get("n_tokens")]
    print(f"tokens: median {statistics.median(toks):.0f}, "
          f"share <= {LIMIT}: {sum(t <= LIMIT for t in toks) / len(toks):.0%}")
    for lo, hi in BUCKETS + [(0, 9999)]:
        b = [r for r in rows if lo <= r["rating"] <= hi]
        pp = per_problem(b)
        name = "all" if lo == 0 else f"{lo}-{hi}"
        print(f"  {name:>9}: pass@1 {sum(r['verdict'] == 'AC' for r in b) / len(b):5.0%}  "
              f"solved>=1 {sum(any(v) for v in pp.values()):3d}/{len(pp)}")


def main() -> None:
    qwen, r1 = load(sys.argv[1]), load(sys.argv[2])
    summarize("Qwen3.5-9B", qwen)
    summarize("DeepSeek-R1", r1)

    q, r = per_problem(qwen), per_problem(r1)
    r_short = per_problem([x for x in r1 if (x.get("n_tokens") or 10**9) <= LIMIT])
    both = sum(any(q[i]) and any(r[i]) for i in q)
    only_r1 = [i for i in q if not any(q[i]) and any(r.get(i, []))]
    only_r1_short = [i for i in only_r1 if any(r_short.get(i, []))]
    only_q = sum(any(q[i]) and not any(r.get(i, [])) for i in q)
    print("\n== overlap (problem solved at least once) ==")
    print(f"both: {both}, only Qwen: {only_q}, only R1: {len(only_r1)} "
          f"(of those, R1 has a passing trace <= {LIMIT} tokens: {len(only_r1_short)})")


if __name__ == "__main__":
    main()
