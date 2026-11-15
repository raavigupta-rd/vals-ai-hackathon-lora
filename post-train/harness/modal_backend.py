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
CXX_FLAGS = ["-std=c++20", "-O2", "-include", "bits/stdc++.h"]

image = (
    modal.Image.from_registry("gcc:14", add_python="3.12")
    .add_local_file(HERE / "runner_in_container.py", "/opt/runner_in_container.py")
)
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(image=image, volumes={"/work/tests": volume}, cpu=2.0, memory=4096, timeout=900,
              max_containers=int(os.environ.get("MODAL_MAX_CONTAINERS", 64)), block_network=True)
@modal.concurrent(max_inputs=2)
def execute(code: str, spec: dict | None = None) -> dict:
    """Compile `code`; if `spec` is given, run it (stdin_file / tests) like runner_in_container."""
    import subprocess
    import tempfile

    sys.path.insert(0, "/opt")
    import runner_in_container as R  # noqa: E402

    h = hashlib.sha256((" ".join(CXX_FLAGS) + "\n" + code).encode()).hexdigest()[:24]
    bdir = Path("/tmp/bin")
    bdir.mkdir(exist_ok=True)
    os.chmod(bdir, 0o755)
    binary = bdir / h
    if not binary.exists():
        src = bdir / f"{h}.cpp"
        src.write_text(code)
        p = subprocess.run(["g++", *CXX_FLAGS, str(src), "-o", str(binary) + ".tmp"],
                           capture_output=True, text=True, timeout=60)
        if p.returncode != 0:
            return {"compiled": False, "error": p.stderr.replace(str(src), "solution.cpp")[:8000] or "Compilation failed"}
        os.replace(str(binary) + ".tmp", binary)
        os.chmod(binary, 0o755)
    if spec is None:
        return {"compiled": True}

    work = Path(tempfile.mkdtemp(prefix="run-"))
    os.chmod(work, 0o777)
    spec = dict(spec, binary=str(binary), out_dir=str(work / "out"))
    if spec.get("stdin") is not None:
        f = work / "in.txt"
        f.write_text(spec.pop("stdin"))
        os.chmod(f, 0o644)
        spec["stdin_file"] = str(f)
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
