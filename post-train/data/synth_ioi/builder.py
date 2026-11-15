"""Builder and verifier for synthetic Vals Track-2 / IOI-style problems.

Uses an isolated gcc:14 Docker container (`cp-synth-builder`) so it never
contends with an active `cp-harness` evaluation run, while writing verified
hidden test files directly into `harness.sandbox.tests_dir()` so that
`python -m harness.eval --problems data/synth_ioi/problems.jsonl` works
immediately out of the box.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TESTS_RO = ROOT / ".cache" / "work" / "ro" / "tests"
SYNTH_DIR = Path(__file__).resolve().parent
BUILD_RW = SYNTH_DIR / ".build_rw"
CONTAINER = "cp-synth-builder"
CXX_FLAGS = ["-std=c++20", "-O2", "-include", "bits/stdc++.h"]
DEFAULT_TL = 2.0
DEFAULT_MEM_MIB = 1024
DEFAULT_MEM_BYTES = DEFAULT_MEM_MIB * 1024 * 1024


@dataclass
class ProblemSpec:
    pid: str               # e.g. "synth_01"
    title: str             # e.g. "Orchard MEX"
    rating: int            # e.g. 1800
    tags: list[str]
    story_and_spec: str    # Body between TITLE and Sample input 1 (includes Input, Output, Constraints)
    samples: list[str]     # Sample input strings (expected outputs generated & cross-checked by fast+brute)
    small_inputs: list[str]
    large_inputs: list[str]
    fast_cpp: str
    brute_cpp: str
    notes: str = ""
    time_limit: float = DEFAULT_TL
    mem_mib: int = DEFAULT_MEM_MIB
    expected_samples: list[str] | None = None  # optional explicit check on sample outputs


def ensure_builder_container() -> None:
    TESTS_RO.mkdir(parents=True, exist_ok=True)
    for sub in ("src", "bin", "in", "out"):
        (BUILD_RW / sub).mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER],
                       capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip() == "true":
        return
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
    subprocess.run([
        "docker", "run", "-d", "--name", CONTAINER, "--network", "none",
        "--memory", "4g", "--pids-limit", "512",
        "-v", f"{BUILD_RW}:/rw",
        "gcc:14", "sleep", "infinity",
    ], check=True, capture_output=True)
    # Precompile bits/stdc++.h inside cp-synth-builder for 10x faster compiles
    subprocess.run([
        "docker", "exec", CONTAINER, "bash", "-lc",
        "HDR=$(g++ -std=c++20 -O2 -xc++ -E -include bits/stdc++.h /dev/null -M | tr ' ' '\\n' | grep 'bits/stdc++.h$' | head -n1) "
        "&& [ -n \"$HDR\" ] && [ ! -f \"$HDR.gch\" ] && g++ -std=c++20 -O2 \"$HDR\" || true"
    ], check=True, capture_output=True)


def compile_in_container(name: str, code: str) -> str:
    src = BUILD_RW / "src" / f"{name}.cpp"
    src.write_text(code)
    bin_c = f"/rw/bin/{name}"
    r = subprocess.run(
        ["docker", "exec", CONTAINER, "g++", *CXX_FLAGS, f"/rw/src/{name}.cpp", "-o", bin_c],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Compile failed for {name}:\n{r.stderr}")
    return bin_c


def run_in_container(bin_c: str, inp: str, tl: float) -> tuple[str, str, float]:
    in_host = BUILD_RW / "in" / "cur.in"
    out_host = BUILD_RW / "out" / "cur.out"
    in_host.write_text(inp)
    if out_host.exists():
        out_host.unlink()
    t0 = time.perf_counter()
    r = subprocess.run(
        ["docker", "exec", CONTAINER, "bash", "-lc",
         f"ulimit -s unlimited && timeout {tl + 0.5:.2f}s {bin_c} < /rw/in/cur.in > /rw/out/cur.out"],
        capture_output=True, text=True,
    )
    dt = time.perf_counter() - t0
    if r.returncode == 124 or dt > tl:
        return "TLE", "", dt
    if r.returncode != 0:
        return f"RE({r.returncode})", r.stderr, dt
    return "OK", out_host.read_text(), dt


def render_statement(spec: ProblemSpec, sample_pairs: list[tuple[str, str]]) -> str:
    parts = [spec.title.upper(), "", spec.story_and_spec.strip()]
    for i, (sin, sout) in enumerate(sample_pairs, 1):
        parts += [
            "",
            f"Sample input {i} (file sample_{i}.in)",
            sin.rstrip(),
            "",
            f"Sample output {i} (file sample_{i}.out)",
            sout.rstrip(),
        ]
    parts += ["", f"Limits: {spec.time_limit:g} seconds and {spec.mem_mib} MiB of memory per test."]
    return "\n".join(parts)


def build_and_verify_batch(specs: list[ProblemSpec]) -> list[dict]:
    ensure_builder_container()
    problems_jsonl = SYNTH_DIR / "problems.jsonl"
    solutions_jsonl = SYNTH_DIR / "solutions.jsonl"

    existing_p = {}
    if problems_jsonl.exists():
        for line in problems_jsonl.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                existing_p[row["id"]] = row

    existing_s = {}
    if solutions_jsonl.exists():
        for line in solutions_jsonl.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                existing_s[row["id"]] = row

    summaries = []
    for spec in specs:
        fast_bin = compile_in_container(f"{spec.pid}_fast", spec.fast_cpp)
        brute_bin = compile_in_container(f"{spec.pid}_brute", spec.brute_cpp)

        # 1. Verify samples: fast == brute (and == expected_samples if given)
        sample_pairs: list[tuple[str, str]] = []
        for idx, sin in enumerate(spec.samples):
            st_f, out_f, _ = run_in_container(fast_bin, sin, spec.time_limit)
            st_b, out_b, _ = run_in_container(brute_bin, sin, spec.time_limit)
            assert st_f == "OK", f"{spec.pid} sample {idx} fast failed: {st_f}"
            assert st_b == "OK", f"{spec.pid} sample {idx} brute failed: {st_b}"
            assert out_f.split() == out_b.split(), (
                f"{spec.pid} sample {idx} mismatch!\nfast: {out_f}\nbrute: {out_b}"
            )
            if spec.expected_samples is not None:
                assert out_f.split() == spec.expected_samples[idx].split(), (
                    f"{spec.pid} sample {idx} != hand-expected!\ngot: {out_f}\nexp: {spec.expected_samples[idx]}"
                )
            sample_pairs.append((sin if sin.endswith("\n") else sin + "\n",
                                 out_f if out_f.endswith("\n") else out_f + "\n"))

        # 2. Verify small hidden tests: both fast and brute must be OK and identical
        small_pairs: list[tuple[str, str]] = []
        max_brute_small_t = 0.0
        for idx, sin in enumerate(spec.small_inputs):
            st_f, out_f, _ = run_in_container(fast_bin, sin, spec.time_limit)
            st_b, out_b, dt_b = run_in_container(brute_bin, sin, spec.time_limit)
            assert st_f == "OK", f"{spec.pid} small {idx} fast failed: {st_f}"
            assert st_b == "OK", f"{spec.pid} small {idx} brute failed ({st_b}, {dt_b:.2f}s)"
            assert out_f.split() == out_b.split(), (
                f"{spec.pid} small {idx} mismatch!\nfast: {out_f[:300]}\nbrute: {out_b[:300]}"
            )
            max_brute_small_t = max(max_brute_small_t, dt_b)
            small_pairs.append((sin if sin.endswith("\n") else sin + "\n",
                                out_f if out_f.endswith("\n") else out_f + "\n"))

        # 3. Verify large hidden tests: fast must finish well within TL; brute must TLE on large[0]
        large_pairs: list[tuple[str, str]] = []
        max_fast_large_t = 0.0
        for idx, lin in enumerate(spec.large_inputs):
            st_f, out_f, dt_f = run_in_container(fast_bin, lin, spec.time_limit)
            assert st_f == "OK" and dt_f <= spec.time_limit * 0.75, (
                f"{spec.pid} large {idx} fast status={st_f} time={dt_f:.3f}s (too slow or failed)"
            )
            max_fast_large_t = max(max_fast_large_t, dt_f)
            large_pairs.append((lin if lin.endswith("\n") else lin + "\n",
                                out_f if out_f.endswith("\n") else out_f + "\n"))

        st_b_large, _, dt_b_large = run_in_container(brute_bin, spec.large_inputs[0], spec.time_limit)
        assert st_b_large == "TLE", (
            f"{spec.pid} brute did NOT TLE on large[0]! status={st_b_large} dt={dt_b_large:.3f}s"
        )

        # 4. Write test files to harness's read-only tests_dir
        tdir = TESTS_RO / spec.pid
        if tdir.exists():
            shutil.rmtree(tdir)
        tdir.mkdir(parents=True, exist_ok=True)

        def write_split(prefix: str, pairs: list[tuple[str, str]]) -> list[list[str]]:
            rels = []
            for i, (inp_s, out_s) in enumerate(pairs):
                fi, fo = tdir / f"{prefix}{i}.in", tdir / f"{prefix}{i}.out"
                fi.write_text(inp_s)
                fo.write_text(out_s)
                rels.append([f"{spec.pid}/{prefix}{i}.in", f"{spec.pid}/{prefix}{i}.out"])
            return rels

        hidden_small_rel = write_split("s", small_pairs)
        hidden_large_rel = write_split("l", large_pairs)
        statement = render_statement(spec, sample_pairs)

        # Save human-readable bundle under solutions/<pid>/
        sdir = SYNTH_DIR / "problems" / spec.pid
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "statement.txt").write_text(statement + "\n")
        (sdir / "fast.cpp").write_text(spec.fast_cpp.strip() + "\n")
        (sdir / "brute.cpp").write_text(spec.brute_cpp.strip() + "\n")

        prob_row = {
            "id": spec.pid,
            "title": spec.title,
            "rating": spec.rating,
            "tags": spec.tags,
            "time_limit": spec.time_limit,
            "memory_limit": spec.mem_mib * 1024 * 1024,
            "statement": statement,
            "samples": [[a, b] for a, b in sample_pairs],
            "checker": None,
            "hidden_small": hidden_small_rel,
            "hidden_large": hidden_large_rel,
            "max_input_bytes": max(len(x[0]) for x in large_pairs),
        }
        sol_row = {
            "id": spec.pid,
            "title": spec.title,
            "rating": spec.rating,
            "tags": spec.tags,
            "notes": spec.notes,
            "fast_cpp": spec.fast_cpp.strip() + "\n",
            "brute_cpp": spec.brute_cpp.strip() + "\n",
        }
        existing_p[spec.pid] = prob_row
        existing_s[spec.pid] = sol_row

        summary = {
            "id": spec.pid,
            "title": spec.title,
            "rating": spec.rating,
            "brute_small_max_s": round(max_brute_small_t, 3),
            "brute_large0": st_b_large,
            "fast_large_max_s": round(max_fast_large_t, 3),
        }
        print(f"OK {spec.pid} ({spec.title}, {spec.rating}): "
              f"fast_large={max_fast_large_t:.3f}s brute_small={max_brute_small_t:.3f}s brute_large0={st_b_large}",
              flush=True)
        summaries.append(summary)

    with problems_jsonl.open("w") as fp:
        for k in sorted(existing_p):
            fp.write(json.dumps(existing_p[k]) + "\n")
    with solutions_jsonl.open("w") as fs:
        for k in sorted(existing_s):
            fs.write(json.dumps(existing_s[k]) + "\n")

    return summaries
