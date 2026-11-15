"""Pick the baseline problem sample (spec agreed 2026-11-14).

Spec:
- open-r1/codeforces `verifiable`, stdin/stdout, not interactive, executable.
- Ratings: 1600-1999 x30, 2000-2399 x40, 2400-2800 x30.
- Contests from 2023 on; fill a short bucket from 2022.
- Multi-answer problems are included (judged with `generated_checker`).
- The problem must have >= 1 DeepSeek-R1 trace in open-r1/codeforces-cots
  (solutions_decontaminated), so Qwen and R1 are judged on the same problems.
- Tests: official tests + generated tests (downloaded per contest).

Outputs in data/:
  problems.jsonl   one problem per line, with `tests` = official + generated
  r1_traces.jsonl  all R1 generations for the picked problems
"""

import argparse
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, snapshot_download

DATA = Path(__file__).resolve().parent.parent / "data"
CF = "open-r1/codeforces"
COTS = "open-r1/codeforces-cots"
BUCKETS = [((1600, 1999), 30), ((2000, 2399), 40), ((2400, 2800), 30)]
FILTER_COLS = ["id", "contest_id", "contest_start_year", "rating", "executable", "input_mode",
               "interaction_format", "generated_tests"]
FULL_COLS = ["id", "contest_id", "contest_start_year", "title", "time_limit", "memory_limit",
             "description", "input_format", "output_format", "note", "examples", "rating",
             "tags", "official_tests", "official_tests_complete", "generated_checker",
             "generated_tests"]

def read_cols(repo: str, pattern: str, columns: list[str], ids: set | None = None) -> list[dict]:
    """Download matching parquet files in parallel (cached), then read only `columns`."""
    root = Path(snapshot_download(repo, repo_type="dataset", allow_patterns=[pattern], max_workers=16))
    rows = []
    for path in sorted(root.glob(pattern)):
        table = pq.read_table(path, columns=columns)
        if ids is not None:
            table = table.filter(pc.is_in(table["id"], value_set=pa.array(sorted(ids))))
        rows += table.to_pylist()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cached-only", action="store_true",
                    help="use only generated-test files already in the local HF cache")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)

    meta = read_cols(CF, "verifiable/train-*.parquet", FILTER_COLS)
    eligible = {r["id"]: r for r in meta
                if r["executable"] and r["input_mode"] == "stdio" and not r["interaction_format"]
                and r["rating"] and (r["contest_start_year"] or 0) >= 2022}
    print(f"eligible (stdio, 2022+): {len(eligible)}")

    cots_ids = {r["id"] for r in read_cols(COTS, "solutions_decontaminated/train-*.parquet", ["id"])}
    eligible = {i: r for i, r in eligible.items() if i in cots_ids}
    print(f"with R1 traces: {len(eligible)}")

    rng = random.Random(args.seed)
    picked: list[str] = []
    for (lo, hi), want in BUCKETS:
        in_bucket = sorted(i for i, r in eligible.items() if lo <= r["rating"] <= hi)
        new = [i for i in in_bucket if eligible[i]["contest_start_year"] >= 2023]
        old = [i for i in in_bucket if eligible[i]["contest_start_year"] == 2022]
        rng.shuffle(new)
        rng.shuffle(old)
        chosen = (new + old)[:want]
        n_old = sum(eligible[i]["contest_start_year"] == 2022 for i in chosen)
        print(f"bucket {lo}-{hi}: 2023+ {len(new)}, 2022 {len(old)} -> picked {len(chosen)} "
              f"({n_old} from 2022)")
        picked += chosen

    full = {r["id"]: r for r in read_cols(CF, "verifiable/train-*.parquet", FULL_COLS, set(picked))}

    # Generated tests are stored per contest: generated_tests/test_cases_<contest_id>.parquet
    gen: dict[str, list] = {}
    missing_contests: list = []
    for contest in sorted({full[i]["contest_id"] for i in picked if full[i]["generated_tests"]}):
        try:
            path = hf_hub_download("open-r1/codeforces", f"generated_tests/test_cases_{contest}.parquet",
                                   repo_type="dataset", local_files_only=args.cached_only)
        except Exception as e:  # noqa: BLE001 - report and continue with official tests only
            missing_contests.append(contest)
            if not args.cached_only:
                print(f"no generated tests for contest {contest}: {e}")
            continue
        for t in pq.read_table(path, columns=["problem_id", "input", "output", "test_i"]).to_pylist():
            if t["problem_id"] in full:
                gen.setdefault(t["problem_id"], []).append(t)

    with open(DATA / "problems.jsonl", "w") as f:
        for pid in picked:
            p = full[pid]
            extra = sorted(gen.get(pid, []), key=lambda t: t.get("test_i", t.get("test_case_i", 0)))
            p["tests"] = ([{"input": t["input"], "output": t["output"], "kind": "official"}
                           for t in p["official_tests"]]
                          + [{"input": t["input"], "output": t["output"], "kind": "generated"}
                             for t in extra])
            del p["official_tests"]
            p["tests_complete"] = bool(p["official_tests_complete"]) or bool(extra)
            f.write(json.dumps(p) + "\n")

    traces = read_cols(COTS, "solutions_decontaminated/train-*.parquet",
                       ["id", "generation", "finish_reason", "api_metadata"], set(picked))
    with open(DATA / "r1_traces.jsonl", "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")

    if missing_contests:
        print(f"generated tests missing for {len(missing_contests)} contests "
              f"(those problems use official tests only for now)")
    n_tests = [len(json.loads(l)["tests"]) for l in open(DATA / "problems.jsonl")]
    print(f"wrote {len(picked)} problems (tests per problem: min {min(n_tests)}, "
          f"median {sorted(n_tests)[len(n_tests) // 2]}, max {max(n_tests)})")
    print(f"wrote {len(traces)} R1 traces")


if __name__ == "__main__":
    main()
