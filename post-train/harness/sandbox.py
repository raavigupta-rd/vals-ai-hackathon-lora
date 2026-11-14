"""C++ compile/run sandbox mirroring the Vals Track-2 grader.

Everything runs in one long-lived `gcc:14` Docker container (no network, non-root execution,
RLIMIT_AS memory cap, CPU-time cap). Compiled with the same flags as vals-ai/ioi-agent:
    g++ -std=c++20 -O2 -include bits/stdc++.h
Host WORK_DIR/ro is mounted read-only at /work (runner, binaries, tests); WORK_DIR/rw at /rw
(compiler output, program stdout). Binaries are moved from rw to ro by the host after compiling.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

IMAGE = os.environ.get("HARNESS_IMAGE", "gcc:14")
CONTAINER = os.environ.get("HARNESS_CONTAINER", "cp-harness")
WORK_DIR = Path(os.environ.get("HARNESS_WORK", Path(__file__).resolve().parent.parent / ".cache" / "work"))
CXX_FLAGS = ["-std=c++20", "-O2", "-include", "bits/stdc++.h"]
COMPILE_TIMEOUT = 60.0
DEFAULT_MEM = 1024 * 1024 * 1024  # 1 GiB, as in the Vals example problem

_EXEC_SEM: asyncio.Semaphore | None = None


def _sem() -> asyncio.Semaphore:
    global _EXEC_SEM
    if _EXEC_SEM is None:
        _EXEC_SEM = asyncio.Semaphore(int(os.environ.get("HARNESS_PARALLEL", max(1, (os.cpu_count() or 4) - 1))))
    return _EXEC_SEM


def ensure_container() -> None:
    """Start the sandbox container if it isn't running (idempotent)."""
    for sub in ("ro/bin", "ro/tests", "rw/src", "rw/bin", "rw/out"):
        (WORK_DIR / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__).with_name("runner_in_container.py"), WORK_DIR / "ro" / "runner.py")
    r = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER], capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip() == "true":
        return
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
    subprocess.run(
        ["docker", "run", "-d", "--name", CONTAINER, "--network", "none",
         "--memory", os.environ.get("HARNESS_MEM", "6g"), "--pids-limit", "2048",
         "-v", f"{WORK_DIR / 'ro'}:/work:ro", "-v", f"{WORK_DIR / 'rw'}:/rw", IMAGE, "sleep", "infinity"],
        check=True, capture_output=True,
    )


@dataclass
class CompileResult:
    ok: bool
    binary: str | None  # container path
    error: str


@dataclass
class RunResult:
    status: str  # OK | RE | TLE | MLE | OLE
    stdout: str
    stderr: str
    exit_code: int
    time_s: float


async def _exec(args: list[str], timeout: float) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        "docker", "exec", CONTAINER, *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -9, "", "timeout"
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


async def compile_cpp(code: str) -> CompileResult:
    h = hashlib.sha256((" ".join(CXX_FLAGS) + "\n" + code).encode()).hexdigest()[:24]
    bin_host = WORK_DIR / "ro" / "bin" / h
    err_host = WORK_DIR / "ro" / "bin" / f"{h}.err"
    if bin_host.exists():
        return CompileResult(True, f"/work/bin/{h}", "")
    if err_host.exists():
        return CompileResult(False, None, err_host.read_text())
    (WORK_DIR / "rw" / "src" / f"{h}.cpp").write_text(code)
    async with _sem():
        rc, _, err = await _exec(
            ["timeout", str(int(COMPILE_TIMEOUT)), "g++", *CXX_FLAGS, f"/rw/src/{h}.cpp", "-o", f"/rw/bin/{h}"],
            COMPILE_TIMEOUT + 10,
        )
    if rc != 0:
        msg = err.replace(f"/rw/src/{h}.cpp", "solution.cpp")[:8000] or "Compilation failed"
        err_host.write_text(msg)
        return CompileResult(False, None, msg)
    os.replace(WORK_DIR / "rw" / "bin" / h, bin_host)
    return CompileResult(True, f"/work/bin/{h}", "")


def _to_container(p: Path) -> str:
    return "/work/" + str(Path(p).resolve().relative_to((WORK_DIR / "ro").resolve()))


async def _run_spec(spec: dict, n: int) -> list[RunResult]:
    run_id = uuid.uuid4().hex[:12]
    spec["out_dir"] = f"/rw/out/{run_id}"
    timeout = n * (spec["time_limit"] * 2 + 2) + 30
    async with _sem():
        rc, out, err = await _exec(["python3", "/work/runner.py", json.dumps(spec)], timeout)
    out_dir = WORK_DIR / "rw" / "out" / run_id
    try:
        if rc != 0:
            return [RunResult("RE", "", f"sandbox error: {err[-500:]}", rc, 0.0)]
        results = []
        for r in json.loads(out):
            p = out_dir / Path(r["stdout_path"]).name
            stdout = p.read_text(errors="replace") if p.exists() else ""
            results.append(RunResult(r["status"], stdout, r["stderr"], r["exit_code"], r["time"]))
        return results
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


async def run_stdin(binary: str, stdin: str, time_limit: float, mem_bytes: int = DEFAULT_MEM) -> RunResult:
    spec = {"binary": binary, "inputs": None, "stdin": stdin, "time_limit": time_limit, "mem_bytes": mem_bytes}
    return (await _run_spec(spec, 1))[0]


async def run_files(binary: str, input_files: list[Path], time_limit: float, mem_bytes: int = DEFAULT_MEM,
                    stop_on_fail: bool = False) -> list[RunResult]:
    spec = {"binary": binary, "inputs": [_to_container(p) for p in input_files], "stdin": None,
            "time_limit": time_limit, "mem_bytes": mem_bytes, "stop_on_fail": stop_on_fail}
    return await _run_spec(spec, len(input_files))


def _tok_eq(a: str, b: str) -> bool:
    if a == b:
        return True
    if not any(c in a + b for c in ".eE"):
        return False  # integers / words must match exactly
    try:
        fa, fb = float(a), float(b)
    except ValueError:
        return False
    return abs(fa - fb) <= 1e-6 * max(1.0, abs(fb))


def outputs_match(got: str, expected: str) -> bool:
    g, e = got.split(), expected.split()
    return len(g) == len(e) and all(_tok_eq(x, y) for x, y in zip(g, e))


def tests_dir() -> Path:
    """Host dir for test files; visible read-only to the sandbox."""
    d = WORK_DIR / "ro" / "tests"
    d.mkdir(parents=True, exist_ok=True)
    return d
