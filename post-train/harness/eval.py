"""Run a Tinker model (base or tinker:// checkpoint) through the local Vals-like harness.

    uv run python -m harness.eval --problems data/eval/cf_test.jsonl --n 60 --tag base
    uv run python -m harness.eval --model-path tinker://.../sampler_weights/xxx --tag sft1

Writes runs/<tag>/results.jsonl (one row per problem, with full transcript) and summary.json.
The last submission that compiled is graded post-hoc on hidden tests, as in the real grader,
regardless of how the episode ended (EXIT, turn limit, context overflow, parse errors).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.renderers.base import format_content_as_string
from tinker_cookbook.rl.rollout_runner import run_rollout
from tinker_cookbook.tokenizer_utils import get_tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from config import BASE_MODEL, PROJECT_ID  # noqa: E402

from harness import sandbox  # noqa: E402
from harness.env import build_env, grade  # noqa: E402
from harness.problems import load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _msg_to_json(m) -> dict:
    out = {"role": m.get("role")}
    content = m.get("content")
    out["content"] = content if isinstance(content, str) else format_content_as_string(content)
    for k in ("thinking", "reasoning_content"):
        if m.get(k):
            out[k] = m[k]
    if m.get("tool_calls"):
        out["tool_calls"] = [{"name": tc.function.name, "arguments": tc.function.arguments} for tc in m["tool_calls"]]
    if m.get("unparsed_tool_calls"):
        out["unparsed_tool_calls"] = [tc.error for tc in m["unparsed_tool_calls"]]
    return out


async def run_one(problem, policy, renderer, args, sem):
    async with sem:
        env, msg_env = build_env(problem, renderer, max_turns=args.max_turns,
                                 max_trajectory_tokens=args.max_context, max_generation_tokens=args.max_tokens)
        t0 = time.time()
        err = None
        traj = None
        try:
            traj = await run_rollout(policy, env)
        except Exception as e:  # keep the batch going; still grade whatever was submitted
            err = f"{type(e).__name__}: {e}"
        tools = msg_env.cp_tools
        g = await grade(problem, tools.last_compiled_code, "subtask")
        gf = await grade(problem, tools.last_compiled_code, "fraction")
        sampled = sum(len(t.ac.tokens) for t in traj.transitions) if traj else 0
        row = {
            "id": problem["id"], "title": problem["title"], "rating": problem["rating"],
            "score": g["score"], "score_fraction": gf["score"],
            "small_pass": g.get("small_pass", 0.0), "large_pass": g.get("large_pass", 0.0),
            "verdicts_small": g.get("small_verdicts", ""), "verdicts_large": g.get("large_verdicts", ""),
            "submitted": g["submitted"], "n_submissions": tools.n_submissions, "n_runs": tools.n_runs,
            "turns": msg_env._turn_count, "sampled_tokens": sampled,
            "stop_reason": traj.stop_reason if traj else "error", "error": err,
            "seconds": round(time.time() - t0, 1),
            "final_code": tools.last_compiled_code,
            "transcript": [_msg_to_json(m) for m in msg_env.history[1:]],
        }
        print(f"[{problem['rating']}] {problem['id']:>10} score={row['score']:5.1f} frac={row['score_fraction']:5.1f} "
              f"turns={row['turns']:3d} subs={row['n_submissions']:2d} runs={row['n_runs']:2d} "
              f"tok={sampled:6d} stop={row['stop_reason']} {err or ''}", flush=True)
        return row


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    by_band: dict[str, list[float]] = {}
    for r in rows:
        band = f"{(r['rating'] // 400) * 400}-{(r['rating'] // 400) * 400 + 399}"
        by_band.setdefault(band, []).append(r["score"])
    return {
        "n": n,
        "mean_score": statistics.mean(r["score"] for r in rows),
        "mean_score_fraction": statistics.mean(r["score_fraction"] for r in rows),
        "solved_full": sum(r["score"] == 100 for r in rows) / n,
        "small_only": sum(r["score"] == 40 for r in rows) / n,
        "submitted": sum(r["submitted"] for r in rows) / n,
        "mean_turns": statistics.mean(r["turns"] for r in rows),
        "mean_sampled_tokens": statistics.mean(r["sampled_tokens"] for r in rows),
        "total_sampled_tokens": sum(r["sampled_tokens"] for r in rows),
        "stop_reasons": {s: sum(r["stop_reason"] == s for r in rows) for s in {r["stop_reason"] for r in rows}},
        "score_by_rating": {b: round(statistics.mean(v), 1) for b, v in sorted(by_band.items())},
        "errors": sum(r["error"] is not None for r in rows),
    }


async def main(args):
    sandbox.ensure_container()
    problems = load(args.problems)
    if args.ids:
        keep = set(args.ids.split(","))
        problems = [p for p in problems if p["id"] in keep]
    if args.n:
        step = max(1, len(problems) // args.n)
        problems = problems[::step][: args.n]  # spread across the rating range
    out_dir = ROOT / "runs" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    service = tinker.ServiceClient(project_id=PROJECT_ID, display_name=f"eval-{args.tag}")
    if args.model_path:
        sampling_client = await service.create_sampling_client(model_path=args.model_path)
    else:
        sampling_client = await service.create_sampling_client(base_model=args.base_model)
    tokenizer = get_tokenizer(args.base_model)
    renderer_name = args.renderer or model_info.get_recommended_renderer_name(args.base_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    policy = TinkerTokenCompleter(sampling_client, max_tokens=args.max_tokens, temperature=args.temperature,
                                  context_window=args.max_context)
    print(f"model={args.model_path or args.base_model} renderer={renderer_name} problems={len(problems)}")
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    sem = asyncio.Semaphore(args.concurrency)
    rows = []
    with (out_dir / "results.jsonl").open("w") as f:
        for coro in asyncio.as_completed([run_one(p, policy, renderer, args, sem) for p in problems]):
            row = await coro
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
    summary = summarize(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default=str(ROOT / "data" / "eval" / "cf_test.jsonl"))
    ap.add_argument("--n", type=int, default=None, help="subsample n problems spread over ratings")
    ap.add_argument("--ids", default=None)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--model-path", default=None, help="tinker:// sampler checkpoint")
    ap.add_argument("--renderer", default=None, help="e.g. qwen3_5 (thinking) or qwen3_5_disable_thinking")
    ap.add_argument("--max-tokens", type=int, default=16384, help="per-turn generation cap")
    ap.add_argument("--max-context", type=int, default=65536)
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--concurrency", type=int, default=16)
    asyncio.run(main(ap.parse_args()))
