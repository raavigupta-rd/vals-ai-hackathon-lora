"""Generate teacher traces with MiMo-V2.6-Distill-Qwen-9B (served by harness/mimo_server.py) in our harness.

MiMo plays the same agentic task as the student: run_cpp / submit tools (executed by harness.env.CPTools,
judged in the Modal sandbox), at most --max-turns turns and --max-output total output tokens per task.
Each episode is graded on hidden tests exactly like the real grader (last compiling submission). Every
episode is appended to --out as structured turns (reasoning, text, tool calls, tool results), so it can be
re-rendered in the Qwen3.5 harness format for SFT (harness/sft_traces.py). Held-out problems are excluded.

    MIMO_URL=https://lenz-pracher--mimo-teacher-serve.us-east.modal.run HARNESS_BACKEND=modal \
      uv run python -m harness.mimo_traces --problems data/synth_ioi/problems.jsonl data/cc_train/problems.jsonl \
        --samples 4 --out data/mimo_traces/traces.jsonl --budget 60
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time
from pathlib import Path

import httpx
from tinker_cookbook.renderers.base import ToolCall
from tinker_cookbook.tool_use.tools import handle_tool_call
from transformers import AutoTokenizer

from harness import sandbox
from harness.env import EXIT_RE, SYSTEM_PROMPT, CPTools, grade_with_retry
from harness.problems import load

ROOT = Path(__file__).resolve().parent.parent
MODEL = "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"
STOP_IDS = [248046, 248044]  # MiMo generation_config eos ids
TOOL_RE = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)
GPU_USD_PER_HOUR = {"H100": 3.95, "A100-80GB": 2.50, "L40S": 1.95}

# Teacher-only steering (NOT saved in traces; the student is trained on the plain harness context).
TEACHER_NOTE = """

IMPORTANT - hard limit: you have exactly {max_turns} turns (assistant replies) for this task, including the final EXIT.
Only your LAST submission that compiles is graded; if you never call submit you get 0 points.
Plan: think through the algorithm first, write a complete solution, check it on the samples with run_cpp at most once
or twice, and call submit by turn {submit_by} at the latest. After a successful submit, reply EXIT."""


def turn_reminder(used: int, max_turns: int) -> str:
    left = max_turns - used
    if left <= 1:
        return (f"\n\n[Turn {used} of {max_turns} used. This is your LAST turn: call submit with your best complete "
                f"solution now (or reply EXIT if you already submitted it).]")
    return (f"\n\n[Turn {used} of {max_turns} used, {left} left. If you have a working solution, submit it now; "
            f"do not spend your remaining turns on more test runs.]")


def parse_completion(text: str) -> tuple[str, str, list[dict], bool]:
    """MiMo output -> (reasoning, visible text, tool calls, malformed?)."""
    t = text[len("<think>"):] if text.startswith("<think>") else text
    if "</think>" in t:
        reasoning, rest = t.split("</think>", 1)
    else:  # cut off while still thinking
        reasoning, rest = t, ""
    calls = [{"name": m.group(1), "arguments": dict(PARAM_RE.findall(m.group(2)))} for m in TOOL_RE.finditer(rest)]
    content = TOOL_RE.sub("", rest).strip()
    malformed = "<tool_call>" in content or "<function=" in content
    return reasoning.strip(), content, calls, malformed


async def stream_completion(client: httpx.AsyncClient, url: str, body: dict) -> tuple[dict, int]:
    """Streamed /v1/completions (Modal web endpoints drop requests that stay silent for ~150 s)."""
    parts, finish, n_out = [], None, None
    async with client.stream("POST", url, json=body, timeout=httpx.Timeout(1800, read=600)) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            for ch in ev.get("choices") or []:
                parts.append(ch.get("text") or "")
                finish = ch.get("finish_reason") or finish
            if ev.get("usage"):
                n_out = ev["usage"]["completion_tokens"]
    if finish is None:
        raise RuntimeError("stream ended without finish_reason")
    return {"text": "".join(parts), "finish_reason": finish}, n_out or 0


class Budget:
    def __init__(self, usd: float, containers: int, gpu: str):
        self.t0, self.usd, self.rate = time.time(), usd, containers * GPU_USD_PER_HOUR.get(gpu, 4.0)

    def spent(self) -> float:
        return (time.time() - self.t0) / 3600 * self.rate

    def exceeded(self) -> bool:
        return self.spent() >= self.usd


async def episode(problem: dict, sample: int, client: httpx.AsyncClient, tok, a, sem: asyncio.Semaphore,
                  budget: Budget) -> dict | None:
    async with sem:
        if budget.exceeded():
            return None
        tools = CPTools(problem)
        tool_fns = {"run_cpp": tools.run_cpp, "submit": tools.submit}
        mimo_tools = [{"type": "function", "function": t.to_spec()} for t in tool_fns.values()]
        system = SYSTEM_PROMPT.format(max_turns=a.max_turns)
        if a.teacher_steering:
            system += TEACHER_NOTE.format(max_turns=a.max_turns, submit_by=max(1, a.max_turns - 2))
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": problem["statement"]}]
        turns, out_tok, stop, bad_streak = [], 0, "max_turns", 0
        t0 = time.time()
        for t in range(a.max_turns):
            left = a.max_output - out_tok
            if left < 256:
                stop = "output_budget"
                break
            ids = tok.apply_chat_template(msgs, tools=mimo_tools, add_generation_prompt=True, tokenize=True,
                                          return_dict=False)
            if len(ids) + min(a.max_turn_tokens, left) > a.max_context:
                stop = "context"
                break
            body = {"model": "mimo", "prompt": ids, "max_tokens": min(a.max_turn_tokens, left), "temperature": 0.6,
                    "top_p": 0.95, "top_k": 20, "stop_token_ids": STOP_IDS, "skip_special_tokens": False,
                    "stream": True, "stream_options": {"include_usage": True}}
            for attempt in range(4):
                try:
                    choice, n_out = await stream_completion(client, f"{a.url}/v1/completions", body)
                    break
                except Exception as e:  # server cold start / transient
                    if attempt == 3:
                        print(f"[{problem['id']}#{sample}] teacher call failed: {type(e).__name__}: {e}", flush=True)
                        return None
                    await asyncio.sleep(10 * (attempt + 1))
            out_tok += n_out
            text = choice["text"]
            reasoning, content, calls, malformed = parse_completion(text)
            turn = {"reasoning": reasoning, "content": content, "tool_calls": calls, "tool_results": [],
                    "finish": choice.get("finish_reason"), "out_tokens": n_out}
            msgs.append({"role": "assistant", "content": content, "reasoning_content": reasoning,
                         "tool_calls": [{"type": "function", "function": c} for c in calls]})
            for i, c in enumerate(calls):
                if c["name"] in tool_fns:
                    tc = ToolCall(function=ToolCall.FunctionBody(name=c["name"], arguments=json.dumps(c["arguments"])),
                                  id=f"call_{t}_{i}")
                    res = await handle_tool_call(tool_fns, tc)
                    result = res.messages[0]["content"]
                else:
                    result = json.dumps({"error": f"Unknown tool {c['name']}"})
                turn["tool_results"].append(result)  # saved clean (student sees the plain harness output)
                shown = result + (turn_reminder(t + 1, a.max_turns) if a.teacher_steering and i == len(calls) - 1 else "")
                msgs.append({"role": "tool", "content": shown})
            turns.append(turn)
            if malformed and not calls:
                bad_streak += 1
                if bad_streak >= 2:
                    stop = "malformed"
                    break
            else:
                bad_streak = 0
            if not calls and EXIT_RE.search(content or ""):
                stop = "exit"
                break
        try:
            g = await grade_with_retry(problem, tools.last_compiled_code)
        except sandbox.SandboxError as e:
            print(f"[{problem['id']}#{sample}] grading infra error, dropped: {e}", flush=True)
            return None
        rec = {"id": problem["id"], "sample": sample, "score": g["score"], "small_pass": g.get("small_pass"),
               "large_pass": g.get("large_pass"), "stop": stop, "turns": len(turns), "out_tokens": out_tok,
               "n_submissions": tools.n_submissions, "seconds": round(time.time() - t0, 1), "turns_data": turns}
        print(f"[{problem['id']}#{sample}] score={g['score']:5.1f} turns={len(turns)} out={out_tok:6d} stop={stop} "
              f"spent≈${budget.spent():.2f}", flush=True)
        return rec


async def main(a):
    sandbox.ensure_container()
    held = {p["id"] for p in load(a.heldout)} if a.heldout else set()  # --heldout "" = evaluate on everything
    problems, seen = [], set()
    for f in a.problems:
        for p in load(f):
            if p["id"] not in held and p["id"] not in seen:
                seen.add(p["id"])
                problems.append(p)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["id"], r["sample"]))
    jobs = [(p, k) for k in range(a.samples) for p in problems if (p["id"], k) not in done]
    random.Random(0).shuffle(jobs)
    print(f"{len(problems)} problems (held-out excluded), {len(jobs)} episodes to run, budget ${a.budget}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    budget = Budget(a.budget, a.containers, a.gpu)
    sem = asyncio.Semaphore(a.concurrency)
    n_ok = n_full = 0
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=a.concurrency + 8)) as client:
        with out.open("a") as fh:
            for coro in asyncio.as_completed([episode(p, k, client, tok, a, sem, budget) for p, k in jobs]):
                rec = await coro
                if rec is None:
                    continue
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                n_ok += 1
                n_full += rec["score"] == 100
    print(f"done: {n_ok} episodes written, {n_full} full solves; est. GPU spend ${budget.spent():.2f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="+", default=[str(ROOT / "data" / "synth_ioi" / "problems.jsonl")])
    ap.add_argument("--heldout", default=str(ROOT / "data" / "synth_ioi" / "heldout.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data" / "mimo_traces" / "traces.jsonl"))
    ap.add_argument("--url", default=os.environ.get("MIMO_URL", "https://lenz-pracher--mimo-teacher-serve.us-east.modal.run"))
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--max-output", type=int, default=32_000, help="total output tokens per task")
    ap.add_argument("--max-turn-tokens", type=int, default=32_000)
    ap.add_argument("--max-context", type=int, default=65_536)
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--budget", type=float, default=60.0, help="stop starting episodes past this est. GPU USD")
    ap.add_argument("--containers", type=int, default=4, help="GPU containers assumed for the cost estimate")
    ap.add_argument("--gpu", default="H100")
    ap.add_argument("--no-teacher-steering", dest="teacher_steering", action="store_false",
                    help="disable the teacher-only turn-limit note and per-turn reminders")
    asyncio.run(main(ap.parse_args()))
