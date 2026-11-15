"""Sample Qwen3.5-9B on the baseline problems through Tinker (single-turn).

Spec: thinking On (`qwen3_5` renderer), max_tokens 16384, temperature 0.6, top_p 0.95,
4 samples per problem. Needs TINKER_API_KEY in the environment or in post-train/.env.

Output: data/qwen_samples.jsonl rows {"id", "sample", "text", "n_tokens", "finish"}.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.tokenizer_utils import get_tokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen3.5-9B"

INSTRUCTIONS = (
    "Solve the following competitive programming problem. Reason about the solution, "
    "then give one complete C++20 program in a single ```cpp code block. The program must "
    "read from standard input and write to standard output, and it must run within the "
    "time and memory limits."
)


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def format_problem(p: dict) -> str:
    parts = [f"# {p['title']}",
             f"Time limit per test: {p['time_limit']} seconds. "
             f"Memory limit per test: {int(p['memory_limit'])} megabytes.",
             p["description"] or ""]
    if p.get("input_format"):
        parts += ["## Input", p["input_format"]]
    if p.get("output_format"):
        parts += ["## Output", p["output_format"]]
    for i, ex in enumerate(p.get("examples") or [], 1):
        parts += [f"## Example {i}", "Input:", f"```\n{ex['input']}\n```",
                  "Output:", f"```\n{ex['output']}\n```"]
    if p.get("note"):
        parts += ["## Note", p["note"]]
    return INSTRUCTIONS + "\n\n" + "\n\n".join(parts)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--limit", type=int, default=None, help="only the first N problems")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--out", default=str(ROOT / "data" / "qwen_samples.jsonl"))
    ap.add_argument("--renderer", default="qwen3_5",
                    choices=["qwen3_5", "qwen3_5_disable_thinking"])
    ap.add_argument("--dry-run", action="store_true", help="render one prompt and exit")
    args = ap.parse_args()

    problems = [json.loads(l) for l in open(ROOT / "data" / "problems.jsonl")][: args.limit]
    tokenizer = get_tokenizer(MODEL)
    renderer = renderers.get_renderer(args.renderer, tokenizer, get_image_processor(MODEL))

    def prompt_for(p: dict):
        return renderer.build_generation_prompt([{"role": "user", "content": format_problem(p)}])

    if args.dry_run:
        mi = prompt_for(problems[0])
        print(tokenizer.decode(mi.to_ints())[-1500:])
        print(f"\nprompt tokens: {mi.length}")
        return

    load_env()
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not set. Put it in post-train/.env")

    client = tinker.ServiceClient().create_sampling_client(base_model=MODEL)
    params = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature,
                                   top_p=args.top_p, stop=renderer.get_stop_sequences())
    # Thinking mode: the prompt ends with an open <think>, so add it back to the text.
    prefix = "<think>\n" if args.renderer == "qwen3_5" else ""
    sem = asyncio.Semaphore(args.concurrency)
    done = 0

    async def run(p: dict) -> list[dict]:
        nonlocal done
        async with sem:
            res = await client.sample_async(prompt_for(p), num_samples=args.samples,
                                            sampling_params=params)
        done += 1
        print(f"{done}/{len(problems)} problems sampled", flush=True)
        return [{"id": p["id"], "sample": i,
                 # The prompt ends with an open <think>; add it back so the judge
                 # can tell a closed reasoning block from a cut-off one.
                 "text": (prefix + tokenizer.decode(seq.tokens)),
                 "n_tokens": len(seq.tokens), "finish": str(seq.stop_reason)}
                for i, seq in enumerate(res.sequences)]

    results = await asyncio.gather(*(run(p) for p in problems))
    with open(args.out, "w") as f:
        for rows in results:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print(f"wrote {sum(map(len, results))} samples to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
