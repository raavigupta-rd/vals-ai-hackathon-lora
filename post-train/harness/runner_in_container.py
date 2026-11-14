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
import subprocess
import sys
import tempfile
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


def run_one(binary, stdin_path_or_bytes, tl, mem, out_path):
    def pre():
        cpu = int(tl * 2) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_AS, (mem + 64 * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY,) * 2)
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUT,) * 2)
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
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
        # Time limit is judged on CPU time (user+sys) so host load doesn't cause false TLEs;
        # wall clock only acts as a generous hard stop (e.g. for programs blocked on input).
        ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        t0 = time.perf_counter()
        try:
            p = subprocess.run([binary], stdin=fin, stdout=fout, stderr=ferr, cwd=td, preexec_fn=pre,
                               timeout=tl * 4 + 2)
            rc, timed_out = p.returncode, False
        except subprocess.TimeoutExpired:
            rc, timed_out = -9, True
        wall = time.perf_counter() - t0
        ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)
        dt = cpu if not timed_out else max(cpu, wall)
        fin.close()
        ferr.seek(0)
        err = ferr.read(SHOW).decode(errors="replace")
    if timed_out or dt > tl or rc in (-24, -9):  # -24 = SIGXCPU from RLIMIT_CPU
        status = "TLE"
    elif rc == -25:
        status = "OLE"
    elif rc != 0:
        status = "MLE" if "bad_alloc" in err else "RE"
    else:
        status = "OK"
    return {"status": status, "time": round(dt, 4), "exit_code": rc, "stderr": err}


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


def main():
    spec = json.loads(sys.argv[1])
    out_dir = spec["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir, 0o777)
    tl, mem, binary = spec["time_limit"], spec["mem_bytes"], spec["binary"]
    res = []
    if spec.get("tests") is None:
        op = os.path.join(out_dir, "0.out")
        r = run_one(binary, spec["stdin_file"], tl, mem, op)
        r["stdout"] = head(op)
        r["passed"] = None
        res.append(r)
    else:
        for i, t in enumerate(spec["tests"]):
            op = os.path.join(out_dir, f"{i}.out")
            r = run_one(binary, t["in"], tl, mem, op)
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
    print(json.dumps(res))


if __name__ == "__main__":
    main()
