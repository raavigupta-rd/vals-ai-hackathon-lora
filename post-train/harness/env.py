"""Agentic C++ competitive-programming environment that mimics the Vals Track-2 grader.

Per task the model sees the statement (with samples) and has two tools:
  * run_cpp(code, input): compile C++20 and run it on an arbitrary stdin.
  * submit(code): compile and judge on the SAMPLE tests only; up to 50 submissions.
The task ends when the model replies with EXIT (and no tool call), runs out of turns, or makes
repeated malformed tool calls. The LAST submission that compiled is graded on hidden tests:
40 points for the small half, 60 for the large half (all-or-nothing per half by default,
see `score_mode`).

Built on tinker_cookbook's AgentToolMessageEnv, so the same env serves eval and RL.
Exact tool names/schemas of the real grader are unconfirmed (ask organizers); override
TOOL_RUN / TOOL_SUBMIT if they differ.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from tinker_cookbook.renderers import Renderer, get_text_content
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.rl import types
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageStepResult
from tinker_cookbook.rl.rollout_limits import ParseErrorPolicy, RolloutLimits
from tinker_cookbook.tool_use import ToolResult, simple_tool_result, tool
from tinker_cookbook.tool_use.agent_tool_message_env import AgentToolMessageEnv

from harness import sandbox
from harness.problems import test_paths

MAX_SUBMISSIONS = 50
MAX_TURNS = int(os.environ.get("HARNESS_MAX_TURNS", 100))
SHOW = 2000
EXIT_RE = re.compile(r"\bEXIT\b")

SYSTEM_PROMPT = """You are an expert competitive programmer. You will solve one hard competitive programming problem in C++20.

The solution is compiled with: g++ -std=c++20 -O2 -include bits/stdc++.h
It must read from standard input and write to standard output.

You have two tools:
- run_cpp: compile a C++ program and run it on any input you choose. Use it to test ideas, check the samples, compare against a brute force, and measure speed on large inputs.
- submit: submit a full solution. It is judged on the sample tests only and you see the result. You may submit up to 50 times.

Your LAST submission that compiles is graded on hidden tests (small tests are worth 40 points, large tests 60 points), under the time and memory limits in the statement. Make sure your final submission is your best, fastest, correct solution.

When you are finished, reply with EXIT (and no tool call). Fewer turns is better."""


def _trunc(s: str, n: int = SHOW) -> str:
    return s if len(s) <= n else s[:n] + f"\n...[truncated {len(s) - n} chars]"


class CPTools:
    """Stateful tools for one episode."""

    def __init__(self, problem: dict):
        self.p = problem
        self.n_submissions = 0
        self.last_compiled_code: str | None = None
        self.n_runs = 0

    @tool
    async def run_cpp(
        self,
        code: Annotated[str, "Complete C++20 program (with main) to compile and run."],
        input: Annotated[str, "Text passed to the program on standard input."] = "",
    ) -> ToolResult:
        """Compile a C++20 program and run it on the given standard input. Returns stdout, stderr, exit code and running time."""
        self.n_runs += 1
        comp = await sandbox.compile_cpp(code)
        if not comp.ok:
            return simple_tool_result(json.dumps({"compiled": False, "compile_error": _trunc(comp.error)}))
        # a generous limit for experiments; the statement's limit is reported for comparison
        r = await sandbox.run_stdin(comp.binary, input, max(10.0, self.p["time_limit"] * 3), self.p["memory_limit"])
        return simple_tool_result(json.dumps({
            "compiled": True, "status": r.status, "exit_code": r.exit_code, "time_seconds": r.time_s,
            "time_limit_seconds": self.p["time_limit"], "stdout": _trunc(r.stdout), "stderr": _trunc(r.stderr, 1000),
        }))

    @tool
    async def submit(
        self,
        code: Annotated[str, "Complete C++20 solution to submit."],
    ) -> ToolResult:
        """Submit a solution. It is judged on the sample tests only; your last submission that compiles is graded on the hidden tests."""
        if self.n_submissions >= MAX_SUBMISSIONS:
            return simple_tool_result(json.dumps({"error": f"Maximum submissions ({MAX_SUBMISSIONS}) reached."}))
        self.n_submissions += 1
        comp = await sandbox.compile_cpp(code)
        remaining = MAX_SUBMISSIONS - self.n_submissions
        if not comp.ok:
            return simple_tool_result(json.dumps({
                "compiled": False, "compile_error": _trunc(comp.error), "submissions_remaining": remaining,
                "note": "This submission did not compile, so it will not be graded.",
            }))
        self.last_compiled_code = code
        # Judge samples exactly like hidden tests: inside the sandbox, with the problem's checker if any.
        checker = sandbox.tests_dir() / self.p["checker"] if self.p.get("checker") else None
        runs = await sandbox.run_tests_inline(comp.binary, [tuple(x) for x in self.p["samples"]], self.p["time_limit"],
                                              self.p["memory_limit"], checker=checker, keep_stdout=True)
        results = []
        for i, ((_, exp), r) in enumerate(zip(self.p["samples"], runs), 1):
            ok = bool(r.passed)
            entry = {"sample": i, "verdict": "AC" if ok else r.status, "time_seconds": r.time_s}
            if not ok:
                entry.update({"your_output": _trunc(r.stdout, 800), "stderr": _trunc(r.stderr, 400)})
                if checker is not None:
                    entry["note"] = "Several answers may be valid; a checker rejected your output."
                    entry["example_correct_output"] = _trunc(exp, 800)
                else:
                    entry["expected_output"] = _trunc(exp, 800)
            results.append(entry)
        passed = sum(e["verdict"] == "AC" for e in results)
        return simple_tool_result(json.dumps({
            "compiled": True, "samples_passed": f"{passed}/{len(results)}", "results": results,
            "submissions_remaining": remaining,
        }))


def sample_files(problem: dict) -> list[tuple[Path, Path]]:
    """Materialize the problem's samples as files in the sandbox tests dir (idempotent, race-safe)."""
    d = sandbox.tests_dir() / problem["id"]
    d.mkdir(parents=True, exist_ok=True)
    pairs = []
    for i, (inp, exp) in enumerate(problem["samples"], 1):
        fi, fo = d / f"sample_{i}.in", d / f"sample_{i}.out"
        for f, text in ((fi, inp), (fo, exp)):
            if not f.exists() or f.read_text() != text:
                tmp = f.with_suffix(f.suffix + f".{os.getpid()}.{id(problem)}.tmp")
                tmp.write_text(text)
                os.replace(tmp, f)
        pairs.append((fi, fo))
    return pairs


async def grade(problem: dict, code: str | None, score_mode: Literal["subtask", "fraction"] = "subtask") -> dict:
    """Grade code on hidden tests. Returns dict with score (0-100) and details."""
    if code is None:
        return {"score": 0.0, "small_pass": 0.0, "large_pass": 0.0, "compiled": False, "submitted": False}
    comp = await sandbox.compile_cpp(code)
    if not comp.ok:
        return {"score": 0.0, "small_pass": 0.0, "large_pass": 0.0, "compiled": False, "submitted": True}
    checker = sandbox.tests_dir() / problem["checker"] if problem.get("checker") else None
    out = {"compiled": True, "submitted": True}
    score = 0.0
    for part, pts in (("small", 40.0), ("large", 60.0)):
        tests = test_paths(problem[f"hidden_{part}"])
        # all-or-nothing halves can stop at the first failing test (big CPU saving on brute-force TLEs)
        res = await sandbox.run_tests(comp.binary, tests, problem["time_limit"], problem["memory_limit"], checker=checker,
                                      stop_on_fail=(score_mode == "subtask"))
        frac = sum(bool(r.passed) for r in res) / max(1, len(tests))
        out[f"{part}_pass"] = frac
        out[f"{part}_verdicts"] = "".join({"OK": "A"}.get(r.status, r.status[0]) for r in res)
        score += pts * (frac if score_mode == "fraction" else float(frac == 1.0))
    out["score"] = score
    return out


@dataclass
class ValsMessageEnv(AgentToolMessageEnv):
    """AgentToolMessageEnv with Vals' termination semantics:
    a text-only reply continues the episode unless it contains EXIT."""

    cp_tools: CPTools | None = None
    result: dict = field(default_factory=dict)

    async def step(self, message: Message) -> MessageStepResult:
        has_calls = bool(message.get("tool_calls") or message.get("unparsed_tool_calls"))
        if has_calls:
            return await super().step(message)
        # text-only turn
        self._turn_count += 1
        self.history.append(message)
        text = get_text_content(message) or ""
        done = bool(EXIT_RE.search(text)) or self._turn_count >= self.max_turns
        metrics: dict[str, float] = {}
        if not done:
            return MessageStepResult(reward=0.0, episode_done=False, next_messages=self.history, metrics=metrics, logs={})
        reason = types.StopReason.COMPLETED if EXIT_RE.search(text) else types.StopReason.MAX_TURNS
        metrics[f"{types.STOP_METRIC_PREFIX}{reason}"] = 1.0
        reward, m = await self._grade()
        metrics.update(m)
        return MessageStepResult(reward=reward, episode_done=True, next_messages=self.history, metrics=metrics, logs={})


def hint_text(reference_code: str, level: float) -> str:
    """Truncation hint (Hinted Scaling Laws, arXiv 2611.02270): the first `level` fraction of a
    reference solution's lines. Eval-only: used to compare checkpoints with less noise."""
    lines = reference_code.rstrip().splitlines()
    k = max(1, int(round(len(lines) * level)))
    return ("\n\nHint: here are the first " + f"{k} of {len(lines)}" + " lines of a correct reference solution.\n"
            "```cpp\n" + "\n".join(lines[:k]) + "\n```")


THINK_SHORT = ("\n\nThink briefly and concretely. Once you have a candidate approach, stop reasoning and call "
               "run_cpp to test it instead of deliberating at length. Never restate the same idea twice; "
               "if you are going in circles, write the code and test it.")


def initial_messages(problem: dict, renderer: Renderer, tools: CPTools, hint: str = "",
                     system_extra: str = "") -> list[Message]:
    specs = [tools.run_cpp.to_spec(), tools.submit.to_spec()]
    prefix = renderer.create_conversation_prefix_with_tools(tools=specs, system_prompt=SYSTEM_PROMPT + system_extra)
    return prefix + [{"role": "user", "content": problem["statement"] + hint}]


def build_env(problem: dict, renderer: Renderer, *, max_turns: int = MAX_TURNS,
              max_trajectory_tokens: int = 60_000, max_generation_tokens: int = 16_384,
              score_mode: str = "subtask", hint: str = "",
              system_extra: str = "") -> tuple[EnvFromMessageEnv, ValsMessageEnv]:
    tools = CPTools(problem)
    holder: dict = {}

    async def reward_fn(history: list[Message]) -> tuple[float, dict[str, float]]:
        g = await grade(problem, tools.last_compiled_code, score_mode)
        g.update({"n_submissions": tools.n_submissions, "n_runs": tools.n_runs})
        holder.update(g)
        return g["score"] / 100.0, {k: float(v) for k, v in g.items() if isinstance(v, (int, float, bool))}

    msg_env = ValsMessageEnv(
        tools=[tools.run_cpp, tools.submit],
        initial_messages=initial_messages(problem, renderer, tools, hint, system_extra),
        max_turns=max_turns,
        reward_fn=reward_fn,
        tool_execution="sequential",
        parse_error_policy=ParseErrorPolicy(max_consecutive=2),
        cp_tools=tools,
        result=holder,
    )
    # Runner-owned budgets: each turn is capped at min(max_generation_tokens, max_trajectory_tokens -
    # current context), so episodes can use the whole context instead of stopping once
    # context + a full per-turn budget no longer fits. Ending by budget is fine: grading is done from
    # the last compiling submission afterwards (eval.py / rl_env.compute_group_rewards).
    env = EnvFromMessageEnv(
        renderer=renderer, message_env=msg_env, failed_parse_reward=0.0,
        context_overflow_reward=0.0, terminate_on_length=False,
        rollout_limits=RolloutLimits(max_turn_tokens=max_generation_tokens,
                                     max_trajectory_tokens=max_trajectory_tokens),
    )
    return env, msg_env
