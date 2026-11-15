"""Build SFT data (option B): R1 traces in thinking-off, tool-loop format.

Steps:
1. Take R1 traces from open-r1/codeforces-cots (solutions_decontaminated) that finished, closed
   </think>, contain a code block, are <= --max-trace-tokens, and are NOT in the 100 eval problems.
2. Keep at most --per-problem traces per problem (shortest first).
3. Judge each trace's final code on the problem's official tests (with the dataset checker when
   there is one). Keep only accepted traces. Generated tests are not used here (110 GB).
4. Convert to a conversation:
     system (tool declarations) -> user (instructions + problem)
     -> assistant: R1 reasoning + explanation, then a `submission` tool call with the code
     -> tool: sample-test result
     -> assistant: "EXIT"
   The renderer `qwen3_5_disable_thinking` adds an empty <think></think> to each assistant turn.

Output: data/sft_r1_thinkoff.jsonl, one {"id", "n_tokens", "messages"} per line.
"""

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "baseline"))
sys.path.insert(0, str(ROOT / "sft"))
from agent_format import prompt_messages, sample_result_json, submission_call  # noqa: E402
from judge import judge  # noqa: E402

FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
PROBLEM_COLS = ["id", "title", "time_limit", "memory_limit", "description", "input_format",
                "output_format", "note", "examples", "rating", "official_tests",
                "generated_checker", "input_mode", "interaction_format", "executable"]


def split_trace(gen: str) -> tuple[str, str, str] | None:
    """Return (reasoning, explanation_without_code, code) or None."""
    if "</think>" not in gen:
        return None
    think, answer = gen.split("</think>", 1)
    think = think.replace("<think>", "", 1).strip()
    blocks = list(FENCE.finditer(answer))
    if not blocks:
        return None
    last = blocks[-1]
    code = last.group(1).strip()
    explanation = (answer[: last.start()] + answer[last.end():]).strip()
    return think, explanation, code


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-trace-tokens", type=int, default=12000)
    ap.add_argument("--per-problem", type=int, default=2)
    ap.add_argument("--max-problems", type=int, default=None, help="limit for a quick pilot")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--min-rating", type=int, default=None, help="only problems rated >= this")
    ap.add_argument("--skip-file", default=None, help="skip problem ids already in this jsonl")
    ap.add_argument("--out", default=str(ROOT / "data" / "sft_r1_thinkoff.jsonl"))
    args = ap.parse_args()

    eval_ids = {json.loads(l)["id"] for l in open(ROOT / "data" / "problems.jsonl")}

    cf = Path(snapshot_download("open-r1/codeforces", repo_type="dataset",
                                allow_patterns=["verifiable/train-*.parquet"]))
    problems = {}
    for f in sorted(cf.glob("verifiable/train-*.parquet")):
        for p in pq.read_table(f, columns=PROBLEM_COLS).to_pylist():
            if (p["executable"] and p["input_mode"] == "stdio" and not p["interaction_format"]
                    and p["official_tests"] and p["id"] not in eval_ids):
                problems[p["id"]] = p
    print(f"candidate problems with official tests: {len(problems)}")

    cots = Path(snapshot_download("open-r1/codeforces-cots", repo_type="dataset",
                                  allow_patterns=["solutions_decontaminated/*.parquet"]))
    by_problem: dict[str, list] = {}
    for f in sorted(cots.glob("solutions_decontaminated/*.parquet")):
        for t in pq.read_table(f, columns=["id", "generation", "finish_reason",
                                           "api_metadata"]).to_pylist():
            n = (t["api_metadata"] or {}).get("completion_tokens") or 10**9
            if (t["id"] in problems and t["finish_reason"] == "stop"
                    and n <= args.max_trace_tokens and split_trace(t["generation"])):
                t["n"] = n
                by_problem.setdefault(t["id"], []).append(t)

    ids = sorted(by_problem)
    if args.min_rating is not None:
        ids = [i for i in ids if (problems[i]["rating"] or 0) >= args.min_rating]
    random.Random(0).shuffle(ids)
    ids = ids[: args.max_problems]
    if args.skip_file and Path(args.skip_file).exists():
        done_ids = {json.loads(l)["id"] for l in open(args.skip_file)}
        # The main builder shuffles with the same seed; skip ids it has judged or will judge soon.
        ids = [i for i in ids if i not in done_ids]
        ids = ids[::-1]  # judge from the other end of the main builder's order
    cands = [t for i in ids for t in sorted(by_problem[i], key=lambda t: t["n"])[: args.per_problem]]
    print(f"traces to judge: {len(cands)} from {len(ids)} problems")

    def run(t: dict):
        p = problems[t["id"]]
        think, explanation, code = split_trace(t["generation"])
        prob = {"time_limit": p["time_limit"], "generated_checker": p["generated_checker"],
                "tests": [{**x, "kind": "official"} for x in p["official_tests"]]}
        res = judge(prob, code)
        return t, p, think, explanation, code, res["verdict"]

    from tinker_cookbook import renderers
    from tinker_cookbook.image_processing_utils import get_image_processor
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    model = "Qwen/Qwen3.5-9B"
    tok = get_tokenizer(model)
    renderer = renderers.get_renderer("qwen3_5_disable_thinking", tok, get_image_processor(model))

    kept, verdicts = 0, {}
    with ThreadPoolExecutor(args.workers) as pool, open(args.out, "w") as out:
        for i, (t, p, think, explanation, code, verdict) in enumerate(pool.map(run, cands), 1):
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
            if verdict == "AC":
                n_samples = len(p["examples"] or []) or 1
                content = think + ("\n\n" + explanation if explanation else "")
                msgs = prompt_messages(renderer, p) + [
                    {"role": "assistant", "content": content, "tool_calls": [submission_call(code)]},
                    {"role": "tool", "content": sample_result_json(n_samples, n_samples),
                     "tool_call_id": "call_0", "name": "submission"},
                    {"role": "assistant", "content": "EXIT"},
                ]
                n_tok = renderer.build_supervised_example(msgs)[0].length
                for m in msgs:  # JSON-serializable tool calls
                    if "tool_calls" in m:
                        m["tool_calls"] = [tc.model_dump() for tc in m["tool_calls"]]
                out.write(json.dumps({"id": t["id"], "rating": p["rating"], "n_tokens": n_tok,
                                      "messages": msgs}) + "\n")
                kept += 1
            if i % 200 == 0:
                print(f"{i}/{len(cands)} judged, kept {kept}, verdicts {verdicts}", flush=True)
    print(f"done: kept {kept} of {len(cands)}; verdicts {verdicts}; wrote {args.out}")


if __name__ == "__main__":
    main()
