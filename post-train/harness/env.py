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

import asyncio
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
# Mirrors the real grader: github.com/vals-ai/hackathon-post-training-agent (src/post_training_agent/*).
# 100 turns; per-turn output cap is the submission's max_output_tokens (default 16384). Every sampling call
# counts as a turn. (The 5-turn loop is only Vals' endpoint smoke check.)
MAX_TURNS = int(os.environ.get("HARNESS_MAX_TURNS", 100))
MAX_OUTPUT_TOKENS = int(os.environ.get("HARNESS_MAX_OUTPUT_TOKENS", 16384))
MAX_SAMPLED_TOKENS = int(os.environ.get("HARNESS_MAX_SAMPLED_TOKENS", 400_000))  # our per-episode spend cap
RUN_TIMEOUT = 10  # cpp_executor / sample runs are killed after 10 s
OUTPUT_LIMIT = 4000
EXIT_RE = re.compile(r"\bEXIT\b", re.IGNORECASE)
MALFORMED_TOOL_CALL_PROMPT = "Your last tool call was malformed and could not be parsed. Retry it with valid arguments."

# Verbatim INSTRUCTIONS_PROMPT of the real harness (sent as the user message; there is no system prompt).
INSTRUCTIONS_PROMPT = """
You will solve a competitive programming problem.

Write your solution as a complete C++20 program that reads standard input and writes standard output. Enclose code in three backticks, for example:

```
[your code here]
```

You have two tools. cpp_executor compiles a program and runs it on stdin you choose, for experiments, brute forces and stress tests. submission saves a solution as your answer and runs it on the sample tests from the statement. Your last submission that compiles is graded on hidden tests, under the time limit, with partial credit for each test it passes.

You may make at most 50 submissions in at most 100 turns. When you are done and do not want to submit again, respond with 'EXIT'.

The problem statement follows:
{question}
"""
SYSTEM_PROMPT = ""  # kept for backwards compatibility (the real harness has no system prompt)


def simple_extract_code(model_output: str) -> str:
    """Real harness: code = the last block between triple backticks (else the raw text)."""
    lines = model_output.split("\n")
    idx = [i for i, line in enumerate(lines) if "```" in line]
    if len(idx) < 2:
        return ""
    return "\n".join(lines[idx[-2] + 1: idx[-1]])


def _extract(code: str) -> str:
    return simple_extract_code(code) or code


def _clip(text: str) -> str:
    return text if len(text) <= OUTPUT_LIMIT else text[:OUTPUT_LIMIT] + f"\n... [{len(text) - OUTPUT_LIMIT} more chars]"


_trunc = _clip  # old name


class CPTools:
    """Stateful tools for one episode; names, arguments, descriptions and outputs mirror the real harness."""

    def __init__(self, problem: dict):
        self.p = problem
        self.n_submissions = 0
        self.last_compiled_code: str | None = None
        self.n_runs = 0
        self.infra_errors = 0

    @tool
    async def cpp_executor(
        self,
        cpp_code: Annotated[str, "The C++20 program, optionally inside a code block."],
        stdin: Annotated[str, "Text passed to the program on standard input."] = "",
    ) -> ToolResult:
        """Compile a C++20 program (g++ -O2) and run it once with the given stdin. Returns stdout, stderr and the exit code. Runs are killed after 10 seconds. Use it for experiments, brute-force checkers and generators."""
        self.n_runs += 1
        code = _extract(cpp_code)
        if not code.strip():
            return simple_tool_result("No C++ code found in the input")
        try:
            comp = await sandbox.compile_cpp(code)
            if not comp.ok:
                return simple_tool_result(f"Compilation failed:\n{_clip(comp.error)}")
            r = await sandbox.run_stdin(comp.binary, stdin, RUN_TIMEOUT, self.p["memory_limit"])
        except sandbox.SandboxError:
            self.infra_errors += 1
            return simple_tool_result("Temporary infrastructure error (not your code). Please retry.")
        if r.status == "TLE":
            return simple_tool_result(f"Execution timed out after {RUN_TIMEOUT} seconds")
        return simple_tool_result(json.dumps({"exit_code": r.exit_code, "stdout": _clip(r.stdout),
                                              "stderr": _clip(r.stderr)}))

    @tool
    async def submission(
        self,
        cpp_code: Annotated[str, "The full C++20 solution, optionally inside a code block."],
    ) -> ToolResult:
        """Submit a complete C++20 solution. It is compiled, run on the sample tests from the statement, and saved as your final answer; your last submission that compiles is the one graded on the hidden tests. Only sample results are reported, never hidden test results."""
        if self.n_submissions >= MAX_SUBMISSIONS:
            return simple_tool_result(f"Submission limit of {MAX_SUBMISSIONS} reached; your last accepted "
                                      "submission stands.", should_stop=True)
        code = _extract(cpp_code)
        if not code.strip():
            return simple_tool_result("No C++ code found in the input")
        try:
            comp = await sandbox.compile_cpp(code)
        except sandbox.SandboxError:  # not counted
            self.infra_errors += 1
            return simple_tool_result("Temporary infrastructure error; this submission was not counted. "
                                      "Please submit again.")
        self.n_submissions += 1
        remaining = MAX_SUBMISSIONS - self.n_submissions
        if not comp.ok:
            return simple_tool_result(f"Compilation failed:\n{_clip(comp.error)}\nNot saved; your previous submission "
                                      f"stands. Submissions remaining: {remaining}", should_stop=remaining == 0)
        self.last_compiled_code = code
        try:
            runs = await sandbox.run_tests_inline(comp.binary, [tuple(x) for x in self.p["samples"]], RUN_TIMEOUT,
                                                  self.p["memory_limit"], keep_stdout=True)
        except sandbox.SandboxError:
            self.infra_errors += 1
            return simple_tool_result(json.dumps({"saved": True, "samples": {}, "submissions_remaining": remaining,
                                                  "note": "sample check hit a temporary infrastructure error"}))
        samples = {}
        for i, r in enumerate(runs, 1):
            if r.passed:
                v = "passed"
            elif r.status == "TLE":
                v = "time limit exceeded"
            elif r.status in ("RE", "MLE", "OLE"):
                v = f"runtime error (exit code {r.exit_code}): {_clip(r.stderr)}"
            else:
                v = f"wrong answer, got:\n{_clip(r.stdout)}"
            samples[f"sample_{i}"] = v
        return simple_tool_result(json.dumps({"saved": True, "samples": samples, "submissions_remaining": remaining}),
                                  should_stop=remaining == 0)

    # old names, so older scripts keep working
    @property
    def run_cpp(self):
        return self.cpp_executor

    @property
    def submit(self):
        return self.submission


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


async def grade_with_retry(problem: dict, code: str | None, score_mode: str = "subtask", attempts: int = 3) -> dict:
    """grade() that retries infrastructure failures (they must never become a 0 score). Raises SandboxError if
    the sandbox stays down, so the caller can drop the sample instead of training/evaluating on a fake score."""
    for k in range(attempts):
        try:
            return await grade(problem, code, score_mode)
        except sandbox.SandboxError as e:
            if k == attempts - 1:
                raise
            print(f"[grade] infra failure, retrying in 30s: {e}", flush=True)
            await asyncio.sleep(30)


@dataclass
class ValsMessageEnv(AgentToolMessageEnv):
    """AgentToolMessageEnv with Vals' termination semantics:
    a text-only reply continues the episode unless it contains EXIT."""

    cp_tools: CPTools | None = None
    result: dict = field(default_factory=dict)
    instructed: bool = False  # whether the think-short instruction was in this episode's prompt

    async def step(self, message: Message) -> MessageStepResult:
        has_calls = bool(message.get("tool_calls") or message.get("unparsed_tool_calls"))
        if has_calls:
            return await super().step(message)
        # text-only turn (a clean turn: resets the malformed-call streak, like the parent does)
        self._consecutive_parse_errors = 0
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
               "cpp_executor to test it instead of deliberating at length. Never restate the same idea twice; "
               "if you are going in circles, write the code and test it.")


def initial_messages(problem: dict, renderer: Renderer, tools: CPTools, hint: str = "",
                     system_extra: str = "", max_turns: int = MAX_TURNS) -> list[Message]:
    """Like the real harness: tool declarations (no system text) + INSTRUCTIONS_PROMPT with the statement.
    system_extra (e.g. the RL think-short cue) is the only thing we ever add as system text."""
    specs = [tools.cpp_executor.to_spec(), tools.submission.to_spec()]
    prefix = renderer.create_conversation_prefix_with_tools(tools=specs, system_prompt=system_extra.strip())
    return prefix + [{"role": "user", "content": INSTRUCTIONS_PROMPT.format(question=problem["statement"] + hint)}]


def build_env(problem: dict, renderer: Renderer, *, max_turns: int = MAX_TURNS,
              max_trajectory_tokens: int = 64_000, max_generation_tokens: int = MAX_OUTPUT_TOKENS,
              max_sampled_tokens: int = MAX_SAMPLED_TOKENS, score_mode: str = "subtask", hint: str = "",
              system_extra: str = "") -> tuple[EnvFromMessageEnv, ValsMessageEnv]:
    tools = CPTools(problem)
    holder: dict = {}

    async def reward_fn(history: list[Message]) -> tuple[float, dict[str, float]]:
        g = await grade(problem, tools.last_compiled_code, score_mode)
        g.update({"n_submissions": tools.n_submissions, "n_runs": tools.n_runs})
        holder.update(g)
        return g["score"] / 100.0, {k: float(v) for k, v in g.items() if isinstance(v, (int, float, bool))}

    msg_env = ValsMessageEnv(
        tools=[tools.cpp_executor, tools.submission],
        initial_messages=initial_messages(problem, renderer, tools, hint, system_extra, max_turns),
        max_turns=max_turns,
        reward_fn=reward_fn,
        tool_execution="sequential",
        # real harness: retry prompt after a malformed call; the 3rd consecutive one ends the task
        parse_error_policy=ParseErrorPolicy(max_consecutive=2, retry_message_template=MALFORMED_TOOL_CALL_PROMPT),
        cp_tools=tools,
        result=holder,
        instructed=bool(system_extra),
    )
    # Runner-owned budgets: each turn is capped at min(max_generation_tokens, max_trajectory_tokens -
    # current context), so episodes can use the whole context instead of stopping once
    # context + a full per-turn budget no longer fits. Ending by budget is fine: grading is done from
    # the last compiling submission afterwards (eval.py / rl_env.compute_group_rewards).
    env = EnvFromMessageEnv(
        renderer=renderer, message_env=msg_env, failed_parse_reward=0.0,
        context_overflow_reward=0.0, terminate_on_length=False,
        # max_turns here counts every sampling call (incl. length-truncated turns, which the env doesn't see);
        # max_sampled_tokens bounds spend: earlier turns' thinking is dropped from the context, so the context
        # cap alone doesn't bound how much a looping episode samples.
        rollout_limits=RolloutLimits(max_turns=max_turns, max_turn_tokens=max_generation_tokens,
                                     max_trajectory_tokens=max_trajectory_tokens,
                                     max_sampled_tokens=max_sampled_tokens),
    )
    return env, msg_env
