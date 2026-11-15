"""Modal backend for the C++ sandbox: compile + run/judge in Modal containers instead of local Docker.

Enable with HARNESS_BACKEND=modal. Hidden test files live in the Modal Volume `cp-tests`
(mirror of .cache/work/ro/tests), mounted read-only at /work/tests; upload with
    uv run python -m harness.modal_backend upload [problem_dir_prefix ...]

Each call ships the source code; containers cache compiled binaries by hash, so repeated
calls on a warm container skip compilation. Programs run as `nobody` with the same limits and
judging logic as the local runner (runner_in_container.py), and CPU-time-based TLE.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import modal

APP_NAME = "cp-harness"
VOLUME_NAME = "cp-tests"
HERE = Path(__file__).resolve().parent
CXX_FLAGS = ["-std=c++20", "-O2", "-pipe", "-static"]  # = real grader flags (no auto bits/stdc++.h) + -static for the chroot jail  # static: program runs chrooted in an empty dir

image = (
    modal.Image.from_registry("gcc:14", add_python="3.12")
    .add_local_file(HERE / "runner_in_container.py", "/opt/runner_in_container.py")
)
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# Stay under the workspace container cap: 50 containers x 2 inputs, 2 CPUs each (1 core per program).
@app.function(image=image, volumes={"/work/tests": volume}, cpu=2.0, memory=6144, timeout=900,
              max_containers=int(os.environ.get("MODAL_MAX_CONTAINERS", 50)), block_network=True)
@modal.concurrent(max_inputs=2)
def execute(code: str, spec: dict | None = None) -> dict:
    """Compile `code`; if `spec` is given, run it (stdin_file / tests) like runner_in_container."""
    import subprocess
    import tempfile
    import threading

    sys.path.insert(0, "/opt")
    import runner_in_container as R  # noqa: E402

    # Hidden tests (incl. expected outputs) must not be readable by the program (it runs as nobody).
    try:
        os.chmod("/work/tests", 0o700)
    except OSError:
        pass
    h = hashlib.sha256((" ".join(CXX_FLAGS) + "\n" + code).encode()).hexdigest()[:24]
    bdir = Path("/tmp/bin")
    bdir.mkdir(exist_ok=True)
    os.chmod(bdir, 0o755)
    binary = bdir / h
    if not binary.exists():
        # unique temp names: concurrent inputs in one container may compile the same code at once
        tag = f"{h}.{os.getpid()}.{threading.get_ident()}"
        src = bdir / f"{tag}.cpp"
        src.write_text(code)
        tmp = bdir / f"{tag}.tmp"
        p = subprocess.run(["g++", *CXX_FLAGS, str(src), "-o", str(tmp)], capture_output=True, text=True, timeout=60)
        src.unlink(missing_ok=True)
        if p.returncode != 0:
            return {"compiled": False, "error": p.stderr.replace(str(src), "solution.cpp")[:8000] or "Compilation failed"}
        os.chmod(tmp, 0o755)
        os.replace(tmp, binary)  # atomic; last writer wins with an identical binary
    if spec is None:
        return {"compiled": True}

    work = Path(tempfile.mkdtemp(prefix="run-"))
    os.chmod(work, 0o777)
    # gVisor's rusage CPU accounting is unreliable (trivial programs report ~2.5 s user time),
    # so judge wall time; 2 inputs on 2 CPUs keeps each program on its own core.
    spec = dict(spec, binary=str(binary), out_dir=str(work / "out"), timing="wall")
    if spec.get("stdin") is not None:
        f = work / "in.txt"
        f.write_text(spec.pop("stdin"))
        os.chmod(f, 0o644)
        spec["stdin_file"] = str(f)
    if spec.get("inline_tests") is not None:
        # tests shipped with the call (e.g. samples, which aren't on the volume); root-only perms
        tests = []
        for i, (inp, exp) in enumerate(spec.pop("inline_tests")):
            fi, fo = work / f"t{i}.in", work / f"t{i}.out"
            fi.write_text(inp)
            fo.write_text(exp)
            os.chmod(fi, 0o600)
            os.chmod(fo, 0o600)
            tests.append({"in": str(fi), "out": str(fo)})
        spec["tests"] = tests
    try:
        results = R.run_spec(spec)  # same judging code as the local runner
    finally:
        subprocess.run(["rm", "-rf", str(work)])
    return {"compiled": True, "results": results}


_fn = None


def _remote():
    global _fn
    if _fn is None:
        _fn = modal.Function.from_name(APP_NAME, "execute")
    return _fn


async def remote_execute(code: str, spec: dict | None = None) -> dict:
    return await _remote().remote.aio(code, spec)


def upload(prefixes: list[str]) -> None:
    from harness import sandbox
    base = sandbox.tests_dir()
    dirs = [d for d in sorted(base.iterdir()) if d.is_dir() and (not prefixes or any(d.name.startswith(p) for p in prefixes))]
    with volume.batch_upload(force=True) as batch:
        for d in dirs:
            batch.put_directory(str(d), f"/{d.name}")
    print(f"uploaded {len(dirs)} problem dirs to volume {VOLUME_NAME}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "upload":
        upload(sys.argv[2:])
    else:
        print("usage: python -m harness.modal_backend upload [prefix ...]   |   modal deploy harness/modal_backend.py")
