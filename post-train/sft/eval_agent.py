"""Evaluate a model on the 100 held-out problems with the agent (tool) prompt.

First-turn evaluation: the model gets the system tool declarations + task, and we parse its first
reply. If it calls `submission`, we judge that code on the full tests. This measures solving
skill plus tool-call format. (A full multi-turn loop comes later.)

Usage:
  python sft/eval_agent.py --name base_agent                       # base model
  python sft/eval_agent.py --name sft_v1 --model-path tinker://... # fine-tuned sampler weights
Output: results/<name>.jsonl (judge rows) and a summary.
"""

import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.tokenizer_utils import get_tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "baseline"))
sys.path.insert(0, str(ROOT / "sft"))
from agent_format import prompt_messages  # noqa: E402
from judge import judge  # noqa: E402
from train_sft import MODEL, RENDERER, load_env  # noqa: E402

FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def code_from_args(arguments: str) -> str | None:
    try:
        raw = json.loads(arguments).get("cpp_code", "")
    except (json.JSONDecodeError, AttributeError):
        return None
    blocks = FENCE.findall(raw)
    return (blocks[-1] if blocks else raw).strip() or None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    load_env()
    problems = [json.loads(l) for l in open(ROOT / "data" / "problems.jsonl")][: args.limit]
    tok = get_tokenizer(MODEL)
    renderer = renderers.get_renderer(RENDERER, tok, get_image_processor(MODEL))
    client = tinker.ServiceClient().create_sampling_client(
        base_model=MODEL if not args.model_path else None, model_path=args.model_path)
    params = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature,
                                   top_p=args.top_p, stop=renderer.get_stop_sequences())

    async def run(p: dict) -> list[dict]:
        res = await client.sample_async(renderer.build_generation_prompt(prompt_messages(renderer, p)),
                                        num_samples=args.samples, sampling_params=params)
        out = []
        for i, seq in enumerate(res.sequences):
            msg, _ = renderer.parse_response(seq.tokens)
            calls = msg.get("tool_calls") or []
            sub = next((c for c in calls if c.function.name == "submission"), None)
            text = msg.get("content") if isinstance(msg.get("content"), str) else ""
            out.append({"id": p["id"], "sample": i, "rating": p["rating"], "n_tokens": len(seq.tokens),
                        "finish": str(seq.stop_reason),
                        "first_action": (sub and "submission") or (calls and calls[0].function.name)
                        or ("exit" if re.search(r"\bEXIT\b", text or "", re.I) else "text"),
                        "malformed": bool(msg.get("unparsed_tool_calls")),
                        "code": code_from_args(sub.function.arguments) if sub else None})
        print(f"sampled {p['id']}", flush=True)
        return out

    rows = [r for rs in await asyncio.gather(*(run(p) for p in problems)) for r in rs]
    by_id = {p["id"]: p for p in problems}

    def judge_row(r: dict) -> dict:
        r.update(judge(by_id[r["id"]], r.pop("code")))
        return r

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(judge_row, rows))
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / f"{args.name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\n== {args.name} ({len(rows)} samples) ==")
    print("first action:", dict(Counter(r["first_action"] for r in rows)))
    print("malformed tool calls:", sum(r["malformed"] for r in rows),
          "| cut off at max tokens:", sum(r["finish"] == "length" for r in rows))
    print("verdicts:", dict(Counter(r["verdict"] for r in rows).most_common()))
    for lo, hi in [(1600, 1999), (2000, 2399), (2400, 2800), (0, 9999)]:
        b = [r for r in rows if lo <= r["rating"] <= hi]
        solved = {r["id"] for r in b if r["verdict"] == "AC"}
        print(f"  {'all' if lo == 0 else f'{lo}-{hi}':>9}: pass@1 "
              f"{sum(r['verdict'] == 'AC' for r in b) / len(b):5.0%}  "
              f"solved>=1 {len(solved)}/{len({r['id'] for r in b})}")


if __name__ == "__main__":
    asyncio.run(main())
