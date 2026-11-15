"""Shared agent format: tools, prompt, and conversation builder.

Follows post-train/CLAUDE.md: track rules win; vals-ai/ioi-agent supplies the format.
- Tools: `cpp_executor(cpp_code)` and `submission(cpp_code)`. No stdin argument.
- The model ends the task with a text-only reply "EXIT".
Training (build_sft_data.py) and evaluation (eval_agent.py) both import this file, so the
prompt the model is trained on is the prompt it is evaluated on.
"""

import json

from tinker_cookbook.renderers import Message, ToolCall


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "cpp_executor",
            "description": (
                "Compile a C++20 program and run it, returning its output or any compilation or "
                "runtime errors. The tool takes the code from the last code block in cpp_code. "
                "The program gets no standard input: hard-code any test input in the program "
                "(for example, read from a std::istringstream instead of std::cin)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"cpp_code": {"type": "string",
                                            "description": "The C++ program to compile and run."}},
                "required": ["cpp_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submission",
            "description": (
                "Submit a complete C++20 solution that reads standard input and writes standard "
                "output. Returns the result on the sample tests. The last submission that compiles "
                "is graded on the hidden tests. At most 50 submissions."
            ),
            "parameters": {
                "type": "object",
                "properties": {"cpp_code": {"type": "string",
                                            "description": "The complete C++ solution."}},
                "required": ["cpp_code"],
            },
        },
    },
]

INSTRUCTIONS = """You will solve a hard competitive programming problem.

Write your solution as a C++20 program that reads standard input and writes standard output. Put all code in a block enclosed with three backticks.

Use the submission tool to submit a solution. After a submission, you will see the result on the sample tests. The last submission that compiles is graded on hidden tests, which include large inputs, so the solution must be efficient. You may make at most 50 submissions in at most 100 turns.

When you are done and do not want to make more submissions, respond with 'EXIT'.

The problem statement follows:
"""


def format_problem(p: dict) -> str:
    parts = [p["title"].upper() if p.get("title") else "",
             p.get("description") or ""]
    if p.get("input_format"):
        parts += ["Input", p["input_format"]]
    if p.get("output_format"):
        parts += ["Output", p["output_format"]]
    for i, ex in enumerate(p.get("examples") or [], 1):
        parts += [f"Sample input {i}", ex["input"].strip(), f"Sample output {i}", ex["output"].strip()]
    if p.get("note"):
        parts += ["Note", p["note"]]
    parts.append(f"Time limit: {p.get('time_limit')} seconds per test. "
                 f"Memory limit: {int(p.get('memory_limit') or 256)} MB per test.")
    return "\n\n".join(x for x in parts if x)


def prompt_messages(renderer, problem: dict) -> list[Message]:
    """System message with tool declarations, then the user task."""
    prefix = renderer.create_conversation_prefix_with_tools(TOOLS, system_prompt="")
    return prefix + [{"role": "user", "content": INSTRUCTIONS + "\n" + format_problem(problem)}]


def submission_call(code: str, call_id: str = "call_0") -> ToolCall:
    args = json.dumps({"cpp_code": f"```cpp\n{code.strip()}\n```"})
    return ToolCall(id=call_id, function=ToolCall.FunctionBody(name="submission", arguments=args))


def sample_result_json(passed: int, total: int) -> str:
    return json.dumps({"compiled": True, "sample_tests_passed": passed, "sample_tests_total": total,
                       "all_sample_tests_passed": passed == total})
