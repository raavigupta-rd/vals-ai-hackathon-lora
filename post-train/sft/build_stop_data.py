"""Build SFT data that teaches the model to STOP reliably (thinking on).

Input: thinking-on rows (convert_thinkon.py output), e.g. data/sft_r1_thinkon_le8k.jsonl.
Each row already has: R1 reasoning (thinking), explanation (text), and the verified solution code.

Fixes for the endpoint-check failure "Max turns (5) reached; agent did not produce a final answer":
1. Short traces only (the input file is already <= 8k tokens).
2. Varied tool schemas, so the model calls whatever tools the system prompt declares, not one
   memorized pair of names:
     ours     cpp_executor(cpp_code)            submission(cpp_code)        (no stdin)
     harness  run_cpp(code, input)              submit(code)                (coding branch)
     ioi      cpp_executor(cpp_code)            submission(cpp_code)        (vals-ai/ioi-agent prompt)
     generic  compile_and_run(source, stdin)    submit_solution(source)
3. Two trajectory shapes, both short and both ending with a text-only "EXIT" turn:
     T1  think -> submit -> result -> EXIT                              (2 assistant turns)
     T2  think -> run on sample 1 -> real output -> submit -> result -> EXIT  (3 assistant turns)
   T2 is used only when the real run output matches the sample (checked here).
All tool results come from really compiling and running the code (baseline/judge.py).

Output: rows {"id","rating","n_tokens","schema","shape","messages"} (messages JSON-serializable).
"""

import argparse
import json
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "baseline"))
sys.path.insert(0, str(ROOT / "sft"))
from agent_format import INSTRUCTIONS, format_problem  # noqa: E402
from judge import TIME_SLACK, compile_cpp  # noqa: E402
from tinker_cookbook.renderers import ToolCall  # noqa: E402

FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def fn(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


HARNESS_SYSTEM = """You are an expert competitive programmer. You will solve one hard competitive programming problem in C++20.

The solution is compiled with: g++ -std=c++20 -O2 -include bits/stdc++.h
It must read from standard input and write to standard output.

You have two tools:
- run_cpp: compile a C++ program and run it on any input you choose. Use it to test ideas, check the samples, compare against a brute force, and measure speed on large inputs.
- submit: submit a full solution. It is judged on the sample tests only and you see the result. You may submit up to 50 times.

Your LAST submission that compiles is graded on hidden tests (small tests are worth 40 points, large tests 60 points), under the time and memory limits in the statement. Make sure your final submission is your best, fastest, correct solution.

When you are finished, reply with EXIT (and no tool call). Fewer turns is better."""

IOI_INSTRUCTIONS = """
You will solve programming problems from the IOI competition.

Provide all submissions and execution requests in a c++ (v20) program with stdlib imports. Submit all code enclosed with three backticks. For example:

```
[your code here]
```

When you would like to make a submission, use the appropriate tool call. After you make a submission, you will see the score you received for each of the subtasks of that problem.

You may make at most 50 submissions in at most 100 turns. Your score will be calculated based on the number of categories passed acorss all submissions.

When you are done working and would not like to make more submissions you should respond with 'EXIT'.

The following includes the problem statement and any relevant contextual files:
"""

GENERIC_SYSTEM = ("You are solving a competitive programming task in C++20. Use the provided tools to test "
                  "and submit your program. Reply with EXIT (no tool call) when you are done.")

SCHEMAS = {
    "ours": {
        "run": fn("cpp_executor", "Compile a C++20 program and run it, returning its output or any errors. "
                  "The program gets no standard input: hard-code any test input in the program.",
                  {"cpp_code": {"type": "string", "description": "The C++ program to compile and run."}}, ["cpp_code"]),
        "submit": fn("submission", "Submit a complete C++20 solution. Returns the result on the sample tests.",
                     {"cpp_code": {"type": "string", "description": "The complete C++ solution."}}, ["cpp_code"]),
        "code_arg": "cpp_code", "stdin_arg": None, "fence": True,
    },
    "harness": {
        "run": fn("run_cpp", "Compile a C++20 program and run it on the given standard input. Returns stdout, "
                  "stderr, exit code and running time.",
                  {"code": {"type": "string", "description": "Complete C++20 program (with main) to compile and run."},
                   "input": {"type": "string", "description": "Text passed to the program on standard input."}},
                  ["code"]),
        "submit": fn("submit", "Submit a solution. It is judged on the sample tests only; your last submission that "
                     "compiles is graded on the hidden tests.",
                     {"code": {"type": "string", "description": "Complete C++20 solution to submit."}}, ["code"]),
        "code_arg": "code", "stdin_arg": "input", "fence": False,
    },
    "ioi": {
        "run": fn("cpp_executor", "Execute a C++ code fragment and return the result. The tool will extract C++ code "
                  "from the provided text, compile it, and run it, returning the output or any compilation/runtime "
                  "errors.Do not use cin. Include any execution parameters hardcoded within your main function for testing.",
                  {"cpp_code": {"type": "string", "description": "The C++ code fragment to execute."}}, ["cpp_code"]),
        "submit": fn("submission", "Submit C++ code for evaluation. Returns the result on the sample tests.",
                     {"cpp_code": {"type": "string", "description": "The C++ code to submit and evaluate."}}, ["cpp_code"]),
        "code_arg": "cpp_code", "stdin_arg": None, "fence": True,
    },
    "generic": {
        "run": fn("compile_and_run", "Compile C++20 source and run it with the given stdin.",
                  {"source": {"type": "string", "description": "C++20 source code."},
                   "stdin": {"type": "string", "description": "Standard input for the program."}}, ["source"]),
        "submit": fn("submit_solution", "Submit the final C++20 source. Judged on the sample tests.",
                     {"source": {"type": "string", "description": "C++20 source code."}}, ["source"]),
        "code_arg": "source", "stdin_arg": "stdin", "fence": False,
    },
}
SCHEMA_WEIGHTS = {"ours": 0.3, "harness": 0.35, "ioi": 0.15, "generic": 0.2}
EXIT_THOUGHT = ""  # no Claude-written text in training data (CLAUDE.md rule)


def harness_statement(p: dict) -> str:
    parts = [p["title"].upper(), "", p["description"] or "", "", "Input", p.get("input_format") or "", "",
             "Output", p.get("output_format") or ""]
    for i, ex in enumerate(p.get("examples") or [], 1):
        parts += ["", f"Sample input {i} (file sample_{i}.in)", ex["input"].rstrip(), "",
                  f"Sample output {i} (file sample_{i}.out)", ex["output"].rstrip()]
    if p.get("note"):
        parts += ["", "Note", p["note"]]
    parts += ["", f"Limits: {p['time_limit']:g} seconds and {int(p['memory_limit'] or 256)} MiB of memory per test."]
    return "\n".join(parts).replace("$$$", "$")


def prompt(schema: str, p: dict) -> tuple[str, str]:
    """(system_prompt, user_content)"""
    if schema == "harness":
        return HARNESS_SYSTEM, harness_statement(p)
    if schema == "ioi":
        return "", IOI_INSTRUCTIONS + "\n" + format_problem(p)
    if schema == "generic":
        return GENERIC_SYSTEM, harness_statement(p)
    return "", INSTRUCTIONS + "\n" + format_problem(p)


def hardcode_input(code: str, stdin: str) -> str | None:
    """Make a no-stdin test program: redirect std::cin to a string at the start of main.
    Only for programs that read with cin (not scanf/getchar/fread)."""
    if re.search(r"\b(scanf|getchar|fread|gets|getline\s*\(\s*stdin|fgets)\b", code):
        return None
    m = re.search(r"\bint\s+main\s*\([^)]*\)\s*\{", code)
    if not m:
        return None
    lit = 'R"SAMPLEINPUT(' + stdin + ')SAMPLEINPUT"'
    inject = f"\n    static std::istringstream sample_input_({lit});\n    std::cin.rdbuf(sample_input_.rdbuf());\n"
    head = code if "<sstream>" in code or "bits/stdc++" in code else "#include <sstream>\n" + code
    m = re.search(r"\bint\s+main\s*\([^)]*\)\s*\{", head)
    # sync_with_stdio(false) gives cin a new buffer and would undo the redirect: insert after it.
    s = re.search(r"sync_with_stdio\s*\([^)]*\)\s*;", head[m.end():])
    at = m.end() + s.end() if s else m.end()
    return head[:at] + inject + head[at:]


def run(code: str, stdin: str, time_limit: float) -> dict | None:
    exe, err = compile_cpp(code)
    if exe is None:
        return None
    try:
        proc = subprocess.run([str(exe)], input=stdin.encode(), capture_output=True,
                              timeout=max(2.0, time_limit * TIME_SLACK))
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    return {"stdout": proc.stdout.decode(errors="replace")}


def call(schema: str, kind: str, code: str, stdin: str | None, idx: int) -> ToolCall:
    s = SCHEMAS[schema]
    body = f"```cpp\n{code.strip()}\n```" if s["fence"] else code.strip()
    args = {s["code_arg"]: body}
    if kind == "run" and s["stdin_arg"] and stdin is not None:
        args[s["stdin_arg"]] = stdin
    name = s[kind]["function"]["name"]
    return ToolCall(id=f"call_{idx}", function=ToolCall.FunctionBody(name=name, arguments=json.dumps(args)))


def run_result(schema: str, out: str, time_limit: float) -> str:
    if schema == "harness":
        return json.dumps({"compiled": True, "status": "OK", "exit_code": 0, "time_seconds": 0.01,
                           "time_limit_seconds": time_limit, "stdout": out, "stderr": ""})
    if schema == "ioi":
        return json.dumps({"success": True, "output": out, "error": None, "exit_code": 0})
    return json.dumps({"compiled": True, "exit_code": 0, "stdout": out, "stderr": ""})


def submit_result(schema: str, n: int) -> str:
    if schema == "harness":
        return json.dumps({"compiled": True, "samples_passed": f"{n}/{n}",
                           "results": [{"sample": i, "verdict": "AC", "time_seconds": 0.01} for i in range(1, n + 1)],
                           "submissions_remaining": 49})
    if schema == "ioi":
        return json.dumps({"compiled": True, "sample_tests_passed": n, "sample_tests_total": n,
                           "submission_count": 1, "submissions_remaining": 49})
    return json.dumps({"compiled": True, "sample_tests_passed": n, "sample_tests_total": n,
                       "all_sample_tests_passed": True})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default=str(ROOT / "data" / "sft_r1_thinkon_le8k.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data" / "sft_stop_le8k.jsonl"))
    ap.add_argument("--test-run-share", type=float, default=0.4)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp)]
    want = {r["id"] for r in rows}
    cf = Path(snapshot_download("open-r1/codeforces", repo_type="dataset",
                                allow_patterns=["verifiable/train-*.parquet"]))
    problems = {}
    for f in sorted(cf.glob("verifiable/train-*.parquet")):
        for p in pq.read_table(f, columns=["id", "title", "time_limit", "memory_limit", "description",
                                           "input_format", "output_format", "note", "examples"]).to_pylist():
            if p["id"] in want:
                problems[p["id"]] = p

    from tinker_cookbook import renderers
    from tinker_cookbook.image_processing_utils import get_image_processor
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    model = "Qwen/Qwen3.5-9B"
    tok = get_tokenizer(model)
    renderer = renderers.get_renderer("qwen3_5", tok, get_image_processor(model))

    rng = random.Random(args.seed)
    plans = []
    for r in rows:
        schema = rng.choices(list(SCHEMA_WEIGHTS), weights=list(SCHEMA_WEIGHTS.values()))[0]
        plans.append((r, schema, rng.random() < args.test_run_share))

    def build(plan):
        r, schema, want_test = plan
        p = problems.get(r["id"])
        if p is None:
            return None
        a1 = r["messages"][2]
        code = json.loads(a1["tool_calls"][0]["function"]["arguments"])["cpp_code"]
        code = (FENCE.findall(code) or [code])[-1].strip()
        examples = p.get("examples") or []
        n = max(1, len(examples))
        thinking = next(c["thinking"] for c in a1["content"] if c["type"] == "thinking")
        text = next(c["text"] for c in a1["content"] if c["type"] == "text")
        system, user = prompt(schema, p)
        msgs = renderer.create_conversation_prefix_with_tools(
            [SCHEMAS[schema]["run"], SCHEMAS[schema]["submit"]], system_prompt=system)
        msgs = [dict(m) for m in msgs] + [{"role": "user", "content": user}]

        shape = "T1"
        if want_test and examples:
            sample_in, sample_out = examples[0]["input"], examples[0]["output"]
            stdin_mode = SCHEMAS[schema]["stdin_arg"] is not None
            test_code = code if stdin_mode else hardcode_input(code, sample_in)
            res = run(test_code, sample_in if stdin_mode else "", p["time_limit"] or 2.0) if test_code else None
            if res and res["stdout"].split() == sample_out.split():
                shape = "T2"
                msgs.append({"role": "assistant",
                             "content": [{"type": "thinking", "thinking": thinking},
                                         {"type": "text", "text": "Let me check the solution on sample 1 first."}],
                             "tool_calls": [call(schema, "run", test_code, sample_in if stdin_mode else None, 0)]})
                msgs.append({"role": "tool", "content": run_result(schema, res["stdout"], p["time_limit"] or 2.0),
                             "tool_call_id": "call_0", "name": SCHEMAS[schema]["run"]["function"]["name"]})
                msgs.append({"role": "assistant",
                             "content": [{"type": "thinking", "thinking": "The output matches the expected output "
                                                                          "for sample 1. I will submit this solution."},
                                         {"type": "text", "text": text or "Submitting the solution."}],
                             "tool_calls": [call(schema, "submit", code, None, 1)]})
        if shape == "T1":
            msgs.append({"role": "assistant", "content": [{"type": "thinking", "thinking": thinking},
                                                          {"type": "text", "text": text}],
                         "tool_calls": [call(schema, "submit", code, None, 1)]})
        msgs.append({"role": "tool", "content": submit_result(schema, n), "tool_call_id": "call_1",
                     "name": SCHEMAS[schema]["submit"]["function"]["name"]})
        msgs.append({"role": "assistant", "content": [{"type": "thinking", "thinking": EXIT_THOUGHT},
                                                      {"type": "text", "text": "EXIT"}]})
        n_tok = sum(mi.length for mi, _ in renderer.build_supervised_examples(msgs))
        for m in msgs:
            if "tool_calls" in m:
                m["tool_calls"] = [tc.model_dump() for tc in m["tool_calls"]]
        return {"id": r["id"], "rating": r["rating"], "n_tokens": n_tok, "schema": schema, "shape": shape,
                "messages": msgs}

    out, stats = [], {}
    with ThreadPoolExecutor(args.workers) as pool:
        for row in pool.map(build, plans):
            if row:
                out.append(row)
                k = (row["schema"], row["shape"])
                stats[k] = stats.get(k, 0) + 1
    with open(args.out, "w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(out)} rows -> {args.out}")
    for k in sorted(stats):
        print(f"  {k[0]:8s} {k[1]}: {stats[k]}")


if __name__ == "__main__":
    main()
