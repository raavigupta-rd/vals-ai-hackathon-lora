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
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

IMAGE = os.environ.get("HARNESS_IMAGE", "gcc:14")
CONTAINER = os.environ.get("HARNESS_CONTAINER", "cp-harness")
WORK_DIR = Path(os.environ.get("HARNESS_WORK", Path(__file__).resolve().parent.parent / ".cache" / "work"))
CXX_FLAGS = ["-std=c++20", "-O2", "-include", "bits/stdc++.h", "-static"]  # static: program runs chrooted in an empty dir
COMPILE_TIMEOUT = 60.0
DEFAULT_MEM = 1024 * 1024 * 1024  # 1 GiB, as in the Vals example problem

_EXEC_SEM: asyncio.Semaphore | None = None
BACKEND = os.environ.get("HARNESS_BACKEND", "docker")  # "docker" (local) or "modal"
_MODAL_SEM: asyncio.Semaphore | None = None
_CODE: dict[str, str] = {}  # hash -> source, for the modal backend (binaries are compiled remotely)
_MODAL_COMPILED: dict[str, tuple[bool, str]] = {}


def _sem() -> asyncio.Semaphore:
    global _EXEC_SEM
    if _EXEC_SEM is None:
        _EXEC_SEM = asyncio.Semaphore(int(os.environ.get("HARNESS_PARALLEL", max(1, (os.cpu_count() or 4) - 1))))
    return _EXEC_SEM


def ensure_container() -> None:
    """Start the sandbox container if it isn't running (idempotent). No-op for the modal backend."""
    for sub in ("ro/bin", "ro/tests", "rw/src", "rw/bin", "rw/out"):
        (WORK_DIR / sub).mkdir(parents=True, exist_ok=True)
    # Hidden tests (incl. expected outputs) must not be readable by sandboxed programs (run as nobody).
    os.chmod(WORK_DIR / "ro" / "tests", 0o700)
    if BACKEND == "modal":
        return
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
    status: str  # OK | WA | RE | TLE | MLE | OLE
    stdout: str  # truncated to ~4 KB (only when requested)
    stderr: str
    exit_code: int
    time_s: float
    passed: bool | None = None  # None for plain stdin runs


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


async def _modal_call(code: str, spec: dict | None) -> dict:
    global _MODAL_SEM
    from harness.modal_backend import remote_execute
    if _MODAL_SEM is None:
        _MODAL_SEM = asyncio.Semaphore(int(os.environ.get("MODAL_PARALLEL", 96)))
    err = None
    for attempt in range(3):
        try:
            async with _MODAL_SEM:
                return await remote_execute(code, spec)
        except Exception as e:  # transient infra errors must not count as wrong answers
            err = e
            print(f"[sandbox/modal] call failed (attempt {attempt + 1}): {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            await asyncio.sleep(2 + 2 * attempt)
    raise RuntimeError(f"modal sandbox failed: {err}")


async def compile_cpp(code: str) -> CompileResult:
    h = hashlib.sha256((" ".join(CXX_FLAGS) + "\n" + code).encode()).hexdigest()[:24]
    if BACKEND == "modal":
        _CODE[h] = code
        if h not in _MODAL_COMPILED:
            r = await _modal_call(code, None)
            _MODAL_COMPILED[h] = (r["compiled"], r.get("error", ""))
        ok, msg = _MODAL_COMPILED[h]
        return CompileResult(ok, f"modal:{h}" if ok else None, msg)
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
    if spec["binary"].startswith("modal:"):
        try:
            r = await _modal_call(_CODE[spec["binary"][6:]], spec)
        except RuntimeError as e:
            return [RunResult("RE", "", f"sandbox error: {e}", -1, 0.0, False)]
        return [RunResult(x["status"], x.get("stdout", ""), x["stderr"], x["exit_code"], x["time"], x["passed"])
                for x in r["results"]]
    run_id = uuid.uuid4().hex[:12]
    spec["out_dir"] = f"/rw/out/{run_id}"
    timeout = n * (spec["time_limit"] * 4 + 4) + 30  # must exceed the runner's per-test wall stop (4*tl+2)
    for attempt in range(3):  # transient docker-exec failures must not count as wrong answers
        async with _sem():
            rc, out, err = await _exec(["python3", "/work/runner.py", json.dumps(spec)], timeout)
        shutil.rmtree(WORK_DIR / "rw" / "out" / run_id, ignore_errors=True)
        if rc == 0:
            break
        print(f"[sandbox] runner failed (attempt {attempt + 1}, rc={rc}): {err[-300:]}", file=sys.stderr, flush=True)
        await asyncio.sleep(1 + attempt)
    if rc != 0:
        return [RunResult("RE", "", f"sandbox error: {err[-500:]}", rc, 0.0, False)]
    return [RunResult(r["status"], r.get("stdout", ""), r["stderr"], r["exit_code"], r["time"], r["passed"])
            for r in json.loads(out)]


async def run_stdin(binary: str, stdin: str, time_limit: float, mem_bytes: int = DEFAULT_MEM) -> RunResult:
    if binary.startswith("modal:"):
        return (await _run_spec({"binary": binary, "stdin": stdin, "time_limit": time_limit, "mem_bytes": mem_bytes}, 1))[0]
    in_dir = WORK_DIR / "rw" / "in"
    in_dir.mkdir(parents=True, exist_ok=True)
    name = uuid.uuid4().hex[:12] + ".in"
    (in_dir / name).write_text(stdin)
    spec = {"binary": binary, "stdin_file": f"/rw/in/{name}", "time_limit": time_limit, "mem_bytes": mem_bytes}
    try:
        return (await _run_spec(spec, 1))[0]
    finally:
        (in_dir / name).unlink(missing_ok=True)


async def run_tests(binary: str, tests: list[tuple[Path, Path]], time_limit: float, mem_bytes: int = DEFAULT_MEM,
                    checker: Path | None = None, stop_on_fail: bool = False, keep_stdout: bool = False) -> list[RunResult]:
    """Grade a compiled binary on (input_file, expected_file) pairs that live under tests_dir()."""
    spec = {"binary": binary, "tests": [{"in": _to_container(i), "out": _to_container(o)} for i, o in tests],
            "checker": _to_container(checker) if checker else None, "time_limit": time_limit,
            "mem_bytes": mem_bytes, "stop_on_fail": stop_on_fail, "keep_stdout": keep_stdout}
    return await _run_spec(spec, len(tests))


def tests_dir() -> Path:
    """Host dir for test files; visible read-only to the sandbox."""
    d = WORK_DIR / "ro" / "tests"
    d.mkdir(parents=True, exist_ok=True)
    return d


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
    """Whitespace-insensitive token compare with 1e-6 tolerance for floats (same as the runner)."""
    g, e = got.split(), expected.split()
    return len(g) == len(e) and all(_tok_eq(x, y) for x, y in zip(g, e))
