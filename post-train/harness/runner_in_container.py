"""Runs inside the gcc:14 container. Usage: python3 runner.py '<json spec>'

spec = {"binary": "/work/bin/x", "inputs": ["/work/tests/a.in", ...] | null, "stdin": str | null,
        "time_limit": 2.0, "mem_bytes": 1073741824, "out_dir": "/work/out/<id>"}
Prints a JSON list of {status, time, exit_code, stdout_path, stderr} (status: OK/RE/TLE/MLE/OLE).
"""

import json
import os
import resource
import subprocess
import sys
import tempfile
import time

MAX_OUT = 64 * 1024 * 1024


def run_one(binary, stdin_bytes, tl, mem, out_path):
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
        t0 = time.perf_counter()
        try:
            p = subprocess.run([binary], input=stdin_bytes, stdout=fout, stderr=ferr, cwd=td,
                               preexec_fn=pre, timeout=tl * 2 + 1)
            rc = p.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            rc, timed_out = -9, True
        dt = time.perf_counter() - t0
        ferr.seek(0)
        err = ferr.read(4000).decode(errors="replace")
    if timed_out or dt > tl or rc in (-24, -9) and dt > tl * 0.9:
        status = "TLE"
    elif rc == -25:
        status = "OLE"
    elif rc != 0:
        status = "MLE" if ("bad_alloc" in err or rc == -6 and "alloc" in err) else "RE"
    else:
        status = "OK"
    return {"status": status, "time": round(dt, 4), "exit_code": rc, "stdout_path": out_path, "stderr": err}


def main():
    spec = json.loads(sys.argv[1])
    os.makedirs(spec["out_dir"], exist_ok=True)
    os.chmod(spec["out_dir"], 0o777)
    res = []
    if spec.get("inputs") is None:
        res.append(run_one(spec["binary"], spec["stdin"].encode(), spec["time_limit"], spec["mem_bytes"],
                           os.path.join(spec["out_dir"], "0.out")))
    else:
        for i, path in enumerate(spec["inputs"]):
            with open(path, "rb") as f:
                data = f.read()
            r = run_one(spec["binary"], data, spec["time_limit"], spec["mem_bytes"],
                        os.path.join(spec["out_dir"], f"{i}.out"))
            res.append(r)
            if spec.get("stop_on_fail") and r["status"] != "OK":
                break
    print(json.dumps(res))


if __name__ == "__main__":
    main()
