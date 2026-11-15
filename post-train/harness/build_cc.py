"""Build a training problem set from DeepMind CodeContests (train split) in the harness format.

    uv run python -m harness.build_cc [--target 650] [--shards 0 1 2 3 4 5]

Downloads parquet shards to data/code_contests/, filters Codeforces problems (1300-2400, stdin/stdout,
>=2 C++ refs), builds hidden small/large tests, validates every test with 3 C++ references in the sandbox
(majority must match), writes tests to .cache/work/ro/tests/cc_<id>/ and data/cc_train/{problems,solutions}.jsonl.
Shards are processed incrementally; existing results are resumed (problems.jsonl is appended per problem).
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

from harness import sandbox
from harness.problems import _clean

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "code_contests"
OUT = ROOT / "data" / "cc_train"
MIN_R, MAX_R = 1300, 2400
MAX_FILE = 8_000_000  # skip individual tests bigger than this (bytes)
NREF = 3
STATS: collections.Counter = collections.Counter()


def shard_files() -> list[str]:
    from huggingface_hub import HfApi
    return sorted(f.path for f in HfApi().list_repo_tree("deepmind/code_contests", repo_type="dataset", path_in_repo="data")
                  if "train-" in f.path)


def get_shard(path: str) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download("deepmind/code_contests", path, repo_type="dataset", local_dir=str(DATA)))


def split_description(desc: str):
    """-> (body_before_examples, note) or None."""
    desc = desc.replace("\r\n", "\n")
    m = re.search(r"\n\s*Examples?\s*\n\s*\nInput\n", desc)
    if not m:
        return None
    body, rest = desc[: m.start()], desc[m.start():]
    nm = list(re.finditer(r"\n\s*Notes?\s*\n", rest))
    note = rest[nm[-1].end():] if nm else ""
    if "\nInput\n" not in "\n" + body or "\nOutput\n" not in "\n" + body:
        return None
    body = re.sub(r"\n(Input|Output)\n\n+", r"\n\1\n", "\n" + body).strip()
    return body, note.strip()


def render(title: str, body: str, note: str, samples, tl: float, mem_mib: int) -> str:
    parts = [title.upper(), "", _clean(body)]
    for i, (a, b) in enumerate(samples, 1):
        parts += ["", f"Sample input {i} (file sample_{i}.in)", a.rstrip(), "", f"Sample output {i} (file sample_{i}.out)", b.rstrip()]
    if note:
        parts += ["", "Note", _clean(note)]
    parts += ["", f"Limits: {tl:g} seconds and {mem_mib} MiB of memory per test."]
    return "\n".join(parts)


def pick_tests(row):
    seen, uniq = set(), []
    for col in ("private_tests", "generated_tests"):
        t = row[col]
        for i, o in zip(t["input"], t["output"]):
            if i in seen or not o.strip() or len(i) > MAX_FILE or len(o) > MAX_FILE:
                continue
            seen.add(i)
            uniq.append((i, o))
    uniq.sort(key=lambda t: len(t[0]))
    return uniq


def select(uniq, k=8):
    half = len(uniq) // 2
    small, large = uniq[:half], uniq[half:]
    def spread(lst, n, keep_last=0):
        keep = lst[len(lst) - keep_last:] if keep_last else []
        rest = lst[: len(lst) - keep_last]
        m = n - len(keep)
        if len(rest) > m:
            idx = sorted({round(j * (len(rest) - 1) / max(1, m - 1)) for j in range(m)}) if m > 1 else [0]
            rest = [rest[j] for j in idx]
        return rest + keep
    return spread(small, k), spread(large, k, keep_last=3)


async def validate(refs, tests, tl, mem, tmp: Path):
    """tests: list of (in,out). returns per-test pass-count across refs, n compiled."""
    tmp.mkdir(parents=True, exist_ok=True)
    pairs = []
    for i, (a, b) in enumerate(tests):
        (tmp / f"{i}.in").write_text(a)
        (tmp / f"{i}.out").write_text(b)
        pairs.append((tmp / f"{i}.in", tmp / f"{i}.out"))
    counts = [0] * len(tests)
    for bin_ in refs:
        res = await sandbox.run_tests(bin_, pairs, tl, mem)
        for i, r in enumerate(res):
            counts[i] += bool(r.passed)
    return counts


async def process(row, sem):
    name = row["name"]
    async with sem:
        try:
            return await _process(row, name)
        except sandbox.SandboxError as e:
            STATS["drop: sandbox error"] += 1
            print("sandbox error", name, e, file=sys.stderr)
            return None


async def _process(row, name):
    pid = "cc_" + hashlib.md5(name.encode()).hexdigest()[:10]
    sp = split_description(row["description"])
    if sp is None:
        STATS["drop: statement unparsable"] += 1
        return None
    body, note = sp
    samples = [[i, o] for i, o in zip(row["public_tests"]["input"], row["public_tests"]["output"])]
    cpp = [s for l, s in zip(row["solutions"]["language"], row["solutions"]["solution"]) if l == 2]
    tl = float(row["time_limit"]["seconds"]) + float(row["time_limit"]["nanos"]) / 1e9
    if tl <= 0:
        tl = 2.0
    mem = max(int(row["memory_limit_bytes"]), 256 * 1024 * 1024)
    uniq = pick_tests(row)
    if len(uniq) < 6:
        STATS["drop: too few hidden tests"] += 1
        return None
    small, large = select(uniq)
    # compile references (shortest first: fewer templates/macros that break), keep first NREF compiled
    refs, ref_src = [], []
    for code in sorted(cpp, key=len)[:8]:
        c = await sandbox.compile_cpp(code)
        if c.ok:
            refs.append(c.binary)
            ref_src.append(code)
        if len(refs) == NREF:
            break
    if len(refs) < 2:
        STATS["drop: <2 compiled refs"] += 1
        return None
    tmp = sandbox.tests_dir() / "_cctmp" / pid
    try:
        need = len(refs) // 2 + 1
        sc = await validate(refs, [(a, b) for a, b in samples], tl * 2, mem, tmp)
        if any(c < len(refs) for c in sc):  # references must agree on every sample
            STATS["drop: refs disagree on sample"] += 1
            return None
        allt = small + large
        cs = await validate(refs, allt, tl * 2, mem, tmp)
        ok = [c >= need for c in cs]
        vs = [t for t, k in zip(small, ok[: len(small)]) if k]
        vl = [t for t, k in zip(large, ok[len(small):]) if k]
        if len(vs) < 3 or len(vl) < 3:
            STATS["drop: <3 valid small/large tests"] += 1
            return None
        if sum(ok) / len(ok) < 0.7:
            STATS["drop: <70% tests validate"] += 1
            return None
        STATS["tests dropped by validation"] += len(ok) - sum(ok)
    finally:
        shutil.rmtree(sandbox.tests_dir() / "_cctmp" / pid, ignore_errors=True)
    d = sandbox.tests_dir() / pid
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    rel = lambda p: str(p.relative_to(sandbox.tests_dir()))
    hs, hl = [], []
    for pre, lst, dst in (("s", vs, hs), ("l", vl, hl)):
        for i, (a, b) in enumerate(lst):
            (d / f"{pre}{i}.in").write_text(a)
            (d / f"{pre}{i}.out").write_text(b)
            dst.append([rel(d / f"{pre}{i}.in"), rel(d / f"{pre}{i}.out")])
    title = re.sub(r"^\d+_[A-Za-z0-9]+\.\s*", "", name)
    mem_mib = mem // (1024 * 1024)
    prob = {"id": pid, "title": title, "rating": int(row["cf_rating"]), "tags": [str(t) for t in row["cf_tags"]],
            "time_limit": tl, "memory_limit": mem_mib * 1024 * 1024,
            "statement": render(title, body, note, samples, tl, mem_mib), "samples": samples, "checker": None,
            "hidden_small": hs, "hidden_large": hl, "max_input_bytes": max(len(t[0]) for t in vl)}
    return prob, {"id": pid, "ref_cpp": ref_src[0]}


async def run_shard(df, done: set, sem, pf, sf, kept: list):
    rows = [r for _, r in df.iterrows()]
    results = await asyncio.gather(*[process(r, sem) for r in rows])
    for res in results:
        if res:
            pf.write(json.dumps(res[0]) + "\n")
            sf.write(json.dumps(res[1]) + "\n")
            kept.append(res[0])
    pf.flush(); sf.flush()


def candidates(path: Path, done: set) -> pd.DataFrame:
    df = pd.read_parquet(path)
    n0 = len(df)
    df = df[df.source == 2]
    df = df[df.cf_rating.between(MIN_R, MAX_R)]
    df = df[(df.input_file.fillna("") == "") & (df.output_file.fillna("") == "")]
    df = df[df.public_tests.map(lambda t: len(t["input"]) >= 1)]
    df = df[df.solutions.map(lambda s: sum(1 for l in s["language"] if l == 2) >= 2)]
    df = df[~df["name"].map(lambda n: "cc_" + hashlib.md5(n.encode()).hexdigest()[:10] in done)]
    STATS["filtered out by metadata"] += n0 - len(df)
    return df


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=650)
    ap.add_argument("--first-shard", type=int, default=0)
    ap.add_argument("--max-shards", type=int, default=6)
    ap.add_argument("--conc", type=int, default=7)
    ap.add_argument("--verify", action="store_true", help="post-pass: check/replace stored references")
    a = ap.parse_args()
    if a.verify:
        await verify_refs()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    sandbox.ensure_container()
    pfp, sfp = OUT / "problems.jsonl", OUT / "solutions.jsonl"
    kept = [json.loads(l) for l in pfp.open()] if pfp.exists() else []
    done = {p["id"] for p in kept}
    files = shard_files()
    sem = asyncio.Semaphore(a.conc)
    with pfp.open("a") as pf, sfp.open("a") as sf:
        for si in range(a.first_shard, min(len(files), a.first_shard + a.max_shards)):
            if len(kept) >= a.target:
                break
            t0 = time.time()
            df = candidates(get_shard(files[si]), done)
            print(f"[shard {si}] {len(df)} candidates; kept so far {len(kept)}", flush=True)
            await run_shard(df, done, sem, pf, sf, kept)
            print(f"[shard {si}] done in {time.time() - t0:.0f}s; kept {len(kept)}; stats {dict(STATS)}", flush=True)
    print("rating hist:", sorted(collections.Counter(p["rating"] for p in kept).items()))
    print("final stats:", dict(STATS))




async def verify_refs():
    """Post-pass (`python -m harness.build_cc --verify`): make sure each stored ref_cpp scores 100 on the kept tests;
    otherwise try the problem's other C++ solutions; drop problems where none does (and delete their tests)."""
    from harness import env
    sandbox.ensure_container()
    P = [json.loads(l) for l in (OUT / "problems.jsonl").open()]
    S = {j["id"]: j for j in map(json.loads, (OUT / "solutions.jsonl").open())}
    allcpp = {}
    for f in sorted(DATA.glob("data/train-*.parquet")):
        df = pd.read_parquet(f, columns=["name", "solutions"])
        for n, s in zip(df["name"], df["solutions"]):
            pid = "cc_" + hashlib.md5(n.encode()).hexdigest()[:10]
            if pid in S:
                allcpp[pid] = sorted((c for l, c in zip(s["language"], s["solution"]) if l == 2), key=len)
    sem = asyncio.Semaphore(7)

    async def one(p):
        async with sem:
            tried = [S[p["id"]]["ref_cpp"]] + [c for c in allcpp[p["id"]][:12] if c != S[p["id"]]["ref_cpp"]]
            for code in tried:
                try:
                    r = await env.grade(p, code)
                except sandbox.SandboxError:
                    continue
                if r["score"] == 100.0:
                    return p, code
            return p, None

    res = await asyncio.gather(*[one(p) for p in P])
    keep = [(p, c) for p, c in res if c]
    fixed = sum(1 for p, c in keep if c != S[p["id"]]["ref_cpp"])
    for p, c in res:
        if not c:
            shutil.rmtree(sandbox.tests_dir() / p["id"], ignore_errors=True)
    with (OUT / "problems.jsonl").open("w") as pf, (OUT / "solutions.jsonl").open("w") as sf:
        for p, c in keep:
            pf.write(json.dumps(p) + "\n")
            sf.write(json.dumps({"id": p["id"], "ref_cpp": c}) + "\n")
    print(f"verify: kept {len(keep)}, ref replaced for {fixed}, dropped {len(res) - len(keep)} (no ref scores 100)")


if __name__ == "__main__":
    asyncio.run(main())
