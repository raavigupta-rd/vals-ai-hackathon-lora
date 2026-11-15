"""Check every problem's reference solutions score as calibrated: fast = 100, brute = 40.

    HARNESS_BACKEND=modal uv run python scripts/calibrate_problems.py data/synth_ioi/problems.jsonl \
        data/synth_ioi/solutions.jsonl [--parallel 32]
Prints one line per problem and a summary; exits non-zero if any problem is off.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import sandbox  # noqa: E402
from harness.env import grade  # noqa: E402
from harness.problems import load  # noqa: E402


async def main(a):
    sandbox.ensure_container()
    ps = {p["id"]: p for p in load(a.problems)}
    sols = {s["id"]: s for s in map(json.loads, open(a.solutions))}
    sem = asyncio.Semaphore(a.parallel)

    async def one(pid):
        async with sem:
            p, s = ps[pid], sols.get(pid)
            if s is None:
                return pid, None, None, "no solution"
            f, b = await asyncio.gather(grade(p, s["fast_cpp"]), grade(p, s["brute_cpp"]))
            return pid, f["score"], b["score"], f"{f.get('small_verdicts')}/{f.get('large_verdicts')} {b.get('small_verdicts')}/{b.get('large_verdicts')}"

    rows = sorted(await asyncio.gather(*(one(i) for i in ps)), key=lambda r: int(r[0].split("_")[1]))
    bad = [r for r in rows if not (r[1] == 100 and r[2] == 40)]
    for r in rows:
        print(("OK " if r not in bad else "BAD"), *r, flush=True)
    print(f"{len(rows) - len(bad)}/{len(rows)} calibrated; bad: {[r[0] for r in bad]}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("problems")
    ap.add_argument("solutions")
    ap.add_argument("--parallel", type=int, default=32)
    asyncio.run(main(ap.parse_args()))
