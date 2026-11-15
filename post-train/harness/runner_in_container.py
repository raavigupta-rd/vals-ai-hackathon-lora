"""Runs inside the gcc:14 container. Usage: python3 runner.py '<json spec>'

spec = {
  "binary": "/work/bin/x", "time_limit": 2.0, "mem_bytes": 1073741824, "out_dir": "/rw/out/<id>",
  # either a single stdin run ...
  "stdin_file": "/rw/in/x.in",
  # ... or graded tests
  "tests": [{"in": "/work/tests/p/1.in", "out": "/work/tests/p/1.out"}, ...],
  "checker": "/work/tests/p/checker.py" | null, "stop_on_fail": bool, "keep_stdout": bool,
}
Prints JSON list of {status, time, exit_code, passed, stdout (truncated, if keep_stdout or stdin mode), stderr}.
status: OK / WA / RE / TLE / MLE / OLE.
"""

import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time

MAX_OUT = 64 * 1024 * 1024
SHOW = 4000


def _tok_eq(a, b):
    if a == b:
        return True
    if not any(c in a + b for c in ".eE"):
        return False
    try:
        fa, fb = float(a), float(b)
    except ValueError:
        return False
    return abs(fa - fb) <= 1e-6 * max(1.0, abs(fb))


def tokens_match(got, exp):
    g, e = got.split(), exp.split()
    return len(g) == len(e) and all(_tok_eq(x, y) for x, y in zip(g, e))


def run_one(binary, stdin_path_or_bytes, tl, mem, out_path, timing="cpu"):
    """timing="cpu": judge user+sys CPU time (accurate under host load; local Docker).
    timing="wall": judge wall time (for gVisor/Modal, whose rusage CPU accounting is unreliable;
    run one program per container so wall time isn't inflated by contention)."""
    def pre():
        # with wall timing the (possibly bogus) CPU counter must never kill a valid run early
        cpu = int(tl + 0.999) + 1 if timing == "cpu" else int(tl * 4) + 5
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_AS, (mem + 64 * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY,) * 2)
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUT,) * 2)
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        # Jail the (statically linked) program in its empty temp dir: it must not be able to read
        # hidden tests / expected outputs or other binaries. Then drop root.
        os.chroot(td)
        os.chdir("/")
        os.setgid(65534)
        os.setuid(65534)

    with tempfile.TemporaryDirectory() as td, open(out_path, "wb") as fout, tempfile.TemporaryFile() as ferr:
        os.chmod(td, 0o777)
        if isinstance(stdin_path_or_bytes, bytes):
            fin = tempfile.TemporaryFile()
            fin.write(stdin_path_or_bytes)
            fin.seek(0)
        else:
            fin = open(stdin_path_or_bytes, "rb")
        # Time limit is judged on this child's own CPU time (user+sys, via wait4) so host load and
        # concurrent runs don't cause false TLEs; wall clock is only a generous hard stop.
        jailed = os.path.join(td, "prog")
        shutil.copy(binary, jailed)
        os.chmod(jailed, 0o755)
        p = subprocess.Popen(["/prog"], executable="/prog", stdin=fin, stdout=fout, stderr=ferr, cwd=td, preexec_fn=pre)
        killed = threading.Event()

        def _kill():
            killed.set()
            p.kill()

        timer = threading.Timer(tl * 4 + 2 if timing == "cpu" else tl * 1.5 + 1, _kill)
        t0 = time.perf_counter()
        timer.start()
        _, status_code, ru = os.wait4(p.pid, 0)
        timer.cancel()
        wall = time.perf_counter() - t0
        rc = -os.WTERMSIG(status_code) if os.WIFSIGNALED(status_code) else os.WEXITSTATUS(status_code)
        p.returncode = rc  # already reaped
        timed_out = killed.is_set()
        cpu = ru.ru_utime + ru.ru_stime
        if timing == "wall":
            dt = wall
        else:
            dt = cpu if not timed_out else max(cpu, wall)
        fin.close()
        ferr.seek(0)
        err = ferr.read(SHOW).decode(errors="replace")
    if timed_out or dt > tl or rc == -24:  # -24 = SIGXCPU from RLIMIT_CPU
        status = "TLE"
    elif rc == -9:  # SIGKILL that wasn't our timer: cgroup OOM killer
        status = "MLE"
    elif rc == -25:
        status = "OLE"
    elif rc != 0:
        status = "MLE" if "bad_alloc" in err else "RE"
    else:
        status = "OK"
    return {"status": status, "time": round(dt, 4), "exit_code": rc, "stderr": err,
            "utime": round(ru.ru_utime, 4), "stime": round(ru.ru_stime, 4), "wall": round(wall, 4)}


def run_one_judged(binary, stdin, tl, mem, out_path, timing="cpu"):
    """run_one, but under wall timing a TLE that finished (not killed) is re-run once and the
    faster run kept: guards against brief contention from a co-located input."""
    r = run_one(binary, stdin, tl, mem, out_path, timing)
    if timing == "wall" and r["status"] == "TLE" and r["exit_code"] == 0:
        r2 = run_one(binary, stdin, tl, mem, out_path, timing)
        if r2["time"] < r["time"]:
            r = r2
    return r


def check(checker, in_path, exp_path, got_path):
    if checker:
        try:
            p = subprocess.run(["python3", checker, in_path, exp_path, got_path], capture_output=True, text=True,
                               timeout=60)
            toks = p.stdout.split()
            return bool(toks) and toks[-1] in ("1", "1.0", "True", "AC", "OK")
        except subprocess.TimeoutExpired:
            return False
    with open(got_path, errors="replace") as g, open(exp_path, errors="replace") as e:
        return tokens_match(g.read(), e.read())


def head(path):
    with open(path, "rb") as f:
        data = f.read(SHOW + 1)
    s = data[:SHOW].decode(errors="replace")
    return s + ("\n...[truncated]" if len(data) > SHOW else "")


def run_spec(spec):
    out_dir = spec["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir, 0o777)
    tl, mem, binary = spec["time_limit"], spec["mem_bytes"], spec["binary"]
    res = []
    if spec.get("tests") is None:
        op = os.path.join(out_dir, "0.out")
        r = run_one_judged(binary, spec["stdin_file"], tl, mem, op, spec.get("timing", "cpu"))
        r["stdout"] = head(op)
        r["passed"] = None
        res.append(r)
    else:
        for i, t in enumerate(spec["tests"]):
            op = os.path.join(out_dir, f"{i}.out")
            r = run_one_judged(binary, t["in"], tl, mem, op, spec.get("timing", "cpu"))
            if r["status"] == "OK":
                r["passed"] = check(spec.get("checker"), t["in"], t["out"], op)
                if not r["passed"]:
                    r["status"] = "WA"
            else:
                r["passed"] = False
            if spec.get("keep_stdout"):
                r["stdout"] = head(op)
            os.unlink(op)
            res.append(r)
            if spec.get("stop_on_fail") and not r["passed"]:
                break
    return res


def main():
    print(json.dumps(run_spec(json.loads(sys.argv[1]))))


if __name__ == "__main__":
    main()
