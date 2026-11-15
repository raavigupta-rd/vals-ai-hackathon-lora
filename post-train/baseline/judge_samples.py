"""Judge a file of model samples against data/problems.jsonl.

Input jsonl rows: {"id", "sample", "text", "n_tokens", "finish"}.
  --r1 builds these rows from data/r1_traces.jsonl instead.
Output: results/<name>.jsonl with the judge verdict added to each row.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from judge import extract_code, judge

ROOT = Path(__file__).resolve().parent.parent


def r1_rows() -> list[dict]:
    rows, seen = [], {}
    for line in open(ROOT / "data" / "r1_traces.jsonl"):
        t = json.loads(line)
        k = seen[t["id"]] = seen.get(t["id"], -1) + 1
        meta = t.get("api_metadata") or {}
        rows.append({"id": t["id"], "sample": k, "text": t["generation"],
                     "n_tokens": meta.get("completion_tokens"), "finish": t["finish_reason"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", help="jsonl of model samples")
    ap.add_argument("--r1", action="store_true", help="judge the R1 traces")
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    problems = {p["id"]: p for p in map(json.loads, open(ROOT / "data" / "problems.jsonl"))}
    rows = r1_rows() if args.r1 else [json.loads(l) for l in open(args.samples)]
    rows = [r for r in rows if r["id"] in problems]

    def run(row: dict) -> dict:
        res = judge(problems[row["id"]], extract_code(row["text"] or ""))
        out = {k: v for k, v in row.items() if k != "text"}
        out.update(res, rating=problems[row["id"]]["rating"])
        return out

    (ROOT / "results").mkdir(exist_ok=True)
    out_path = ROOT / "results" / f"{args.name}.jsonl"
    with ThreadPoolExecutor(args.workers) as pool, open(out_path, "w") as f:
        for i, res in enumerate(pool.map(run, rows), 1):
            f.write(json.dumps(res) + "\n")
            if i % 50 == 0:
                print(f"{i}/{len(rows)} judged", flush=True)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
