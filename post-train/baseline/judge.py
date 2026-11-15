"""Local C++ judge for Codeforces-style stdin/stdout problems.

Compile flags match vals-ai/ioi-agent: -std=c++20 -O2 -include bits/stdc++.h.
Set CXX to a real GCC (Apple clang has no bits/stdc++.h). Default: g++-16.

Verdicts: AC, WA, TLE, RE, CE, NO_CODE, CHECKER_ERROR.
Limits: time limit x TIME_SLACK (no sandbox, so this machine is a bit noisy).
Memory limits are not enforced (macOS has no reliable RLIMIT_AS).
"""

import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

CXX = os.environ.get("CXX", "g++-16")
CXX_FLAGS = ["-std=c++20", "-O2", "-include", "bits/stdc++.h"]
TIME_SLACK = float(os.environ.get("TIME_SLACK", "2.0"))
BUILD_DIR = Path(tempfile.gettempdir()) / "cf_judge_build"
BUILD_DIR.mkdir(exist_ok=True)

_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str | None:
    """Return the last fenced code block after the reasoning, or None.

    If the reasoning never closed (output cut off), there is no answer: return None.
    """
    if "<think>" in text and "</think>" not in text:
        return None
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    blocks = _FENCE.findall(text)
    return blocks[-1] if blocks else None


def compile_cpp(code: str) -> tuple[Path | None, str]:
    digest = hashlib.sha256(code.encode()).hexdigest()[:20]
    exe = BUILD_DIR / f"{digest}.out"
    if exe.exists():
        return exe, ""
    src = BUILD_DIR / f"{digest}.cpp"
    src.write_text(code)
    try:
        proc = subprocess.run([CXX, *CXX_FLAGS, str(src), "-o", str(exe)],
                              capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None, "compile timeout"
    if proc.returncode != 0:
        return None, proc.stderr[-2000:]
    return exe, ""


def _tokens(s: str) -> list[str]:
    return s.split()


def _check(test: dict, output: str, checker: str | None) -> tuple[bool, str]:
    if not checker:
        return _tokens(output) == _tokens(test["output"]), ""
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        (dp / "checker.py").write_text(checker)
        (dp / "input.txt").write_text(test["input"])
        (dp / "correct_output.txt").write_text(test["output"])
        (dp / "solution_output.txt").write_text(output)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "checker.py", "input.txt", "correct_output.txt",
                 "solution_output.txt"],
                cwd=d, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return False, "checker timeout"
    try:
        score = float(proc.stdout.split()[0])
    except (IndexError, ValueError):
        return False, f"checker error: {proc.stderr[-300:]}"
    # Checkers print 0-1 or 0-100; full marks only.
    return (score >= 0.999 if score <= 1 else score >= 99.9), ""


def judge(problem: dict, code: str | None, max_tests: int | None = None) -> dict:
    if not code:
        return {"verdict": "NO_CODE", "passed": 0, "total": len(problem["tests"])}
    exe, err = compile_cpp(code)
    tests = problem["tests"][:max_tests] if max_tests else problem["tests"]
    if exe is None:
        return {"verdict": "CE", "passed": 0, "total": len(tests), "detail": err[-500:]}
    limit = float(problem["time_limit"] or 2.0) * TIME_SLACK
    checker = problem.get("generated_checker")
    for i, test in enumerate(tests):
        try:
            proc = subprocess.run([str(exe)], input=test["input"].encode(),
                                  capture_output=True, timeout=limit)
        except subprocess.TimeoutExpired:
            return {"verdict": "TLE", "passed": i, "total": len(tests), "fail_kind": test["kind"]}
        if proc.returncode != 0:
            return {"verdict": "RE", "passed": i, "total": len(tests), "fail_kind": test["kind"],
                    "detail": proc.stderr.decode(errors="replace")[-300:]}
        ok, note = _check(test, proc.stdout.decode(errors="replace"), checker)
        if not ok:
            verdict = "CHECKER_ERROR" if note.startswith("checker") else "WA"
            return {"verdict": verdict, "passed": i, "total": len(tests), "fail_kind": test["kind"],
                    "detail": note}
    return {"verdict": "AC", "passed": len(tests), "total": len(tests)}
