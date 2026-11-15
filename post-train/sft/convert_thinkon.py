"""Option A: convert option-B rows (reasoning as visible text) to thinking-on rows.

For each row of an option-B train.jsonl, find the original R1 trace (same problem id, whose
<think> text starts the assistant content), then rebuild the assistant turns as:
  turn 1: thinking = R1 reasoning, text = R1 explanation, tool call = same submission
  turn 2: thinking = short fixed note, text = "EXIT"
Order is kept, so a curriculum file stays a curriculum file. No re-judging: the code is unchanged.

Usage: python sft/convert_thinkon.py runs/sft_b_v1/train.jsonl data/sft_r1_thinkon_3000.jsonl
"""

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sft"))
from build_sft_data import split_trace  # noqa: E402

EXIT_THOUGHT = "The submission compiled and passed all sample tests, so I am done."


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    rows = [json.loads(l) for l in open(src)]
    want = {r["id"] for r in rows}
    cots = Path(snapshot_download("open-r1/codeforces-cots", repo_type="dataset",
                                  allow_patterns=["solutions_decontaminated/*.parquet"]))
    traces: dict[str, list] = {}
    for f in sorted(cots.glob("solutions_decontaminated/*.parquet")):
        for t in pq.read_table(f, columns=["id", "generation"]).to_pylist():
            if t["id"] in want:
                s = split_trace(t["generation"])
                if s:
                    traces.setdefault(t["id"], []).append(s)

    out, missing = [], 0
    for r in rows:
        msgs = r["messages"]
        content = msgs[2]["content"]
        match = next(((th, ex) for th, ex, _ in traces.get(r["id"], []) if th and content.startswith(th)), None)
        if match is None:
            missing += 1
            continue
        think, explanation = match
        msgs[2]["content"] = [{"type": "thinking", "thinking": think},
                              {"type": "text", "text": explanation or "Submitting the solution."}]
        msgs[4]["content"] = [{"type": "thinking", "thinking": EXIT_THOUGHT}, {"type": "text", "text": "EXIT"}]
        out.append(r)
    with open(dst, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"converted {len(out)} of {len(rows)} rows (unmatched: {missing}) -> {dst}")


if __name__ == "__main__":
    main()
