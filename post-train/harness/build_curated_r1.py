"""Cloud-native dataset curation & C++20 Modal calibration pipeline for `lenz-sft-2`.

Pulls public human-authored competitive programming tasks directly inside Modal containers from:
  - `open-r1/codeforces` (`train` + `test` splits + `generated_tests/test_cases_XXXX.parquet`)
  - `open-r1/codeforces-submissions` (`selected_accepted` + `selected_incorrect`)

Applies strict Vals Track-2 alignment & anti-leakage filters:
  1. 100% deterministic stdio exact/token match (`generated_checker` is null, non-interactive, no float-tolerance
     or multi-answer `"print any"` tasks).
  2. Strict zero-leakage separation between `train.jsonl` (pre-cutoff `train` split) and `eval.jsonl` (2024-2025
     `test` split), deduplicated across `id`, `contest_id`, Div1/Div2 `aliases`, normalized `title`, and prior
     repo splits (`cf_test.jsonl` and `sft_b_v1.problem_ids.txt`).
  3. IOI / `DISTINCT STALLS` structural ranking: prioritizes single-instance-per-file tasks, core algorithmic tags
     (DP, trees, graphs, data structures, DSU, binary search, divide & conquer, strings), and concise C++20 solutions.
  4. True asymptotic efficiency tests: requires megabyte/tens-of-KB `generated_tests` (`>= 25 KB` train / `>= 15 KB`
     eval, or empirically verified TLE against a real human TLE C++ submission on compact-input tasks).
  5. End-to-end Modal `gcc:14` verification: every kept problem is verified inside the exact chroot runner
     (`g++ -std=c++20 -O2 -pipe -static`, NO auto-included `bits/stdc++.h`, wall-clock timing) to ensure a real
     human C++ solution scores 100/100 within `0.80 * time_limit`, staging hidden tests directly onto the `cp-tests`
     Modal Volume (`/work/tests/r1_<cid>_<idx>/`).

Usage (from `post-train/`):
    .venv/bin/modal run harness/build_curated_r1.py
"""

from __future__ import annotations

import collections
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_DIR = ROOT / "data" / "curated_r1"
VOLUME_NAME = "cp-tests"
CXX_FLAGS = ["-std=c++20", "-O2", "-pipe", "-static"]
MAX_SINGLE_TEST_BYTES = 6_000_000  # 6 MB cap per test file for fast volume I/O

IOI_GOOD_TAGS = {
    "data structures",
    "trees",
    "dp",
    "graphs",
    "shortest paths",
    "dsu",
    "dfs and similar",
    "divide and conquer",
    "binary search",
    "two pointers",
    "strings",
    "bitmasks",
    "combinatorics",
}
NOISY_TAGS = {
    "games",
    "probabilities",
    "expression parsing",
    "*special",
    "interactive",
}
REJECT_PHRASES = (
    "interactive",
    "flush the output",
    "fflush",
    "cout.flush",
    "absolute or relative error",
    "10^{-6}",
    "10^-6",
    "10^{-9}",
    "10^-9",
    "10^{-4}",
    "10^-4",
    "print any",
    "output any",
    "any valid",
    "if there are multiple",
    "if there are several",
    "in any order",
)
MULTI_TEST_PHRASES = (
    "multiple test cases",
    "number of test cases",
    "each test contains multiple",
    "t test cases",
)

app = modal.App("curate-r1-dataset")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

curate_image = (
    modal.Image.from_registry("gcc:14", add_python="3.12")
    .pip_install("huggingface_hub>=0.26", "pandas", "pyarrow", "numpy")
    .add_local_file(HERE / "runner_in_container.py", "/opt/runner_in_container.py")
)


def _norm_title(t: str) -> str:
    return re.sub(r"\W+", "", (t or "").lower())


def _clean_text(s: Any) -> str:
    if not isinstance(s, str) or not s:
        return ""
    s = s.replace("\r\n", "\n").replace("$$$", "$")
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def _to_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, float) and math.isnan(x):
        return []
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return []


def _render_statement(title: str, desc: str, in_fmt: str, out_fmt: str, examples: list[list[str]],
                      note: str, tl: float, mem_mib: int) -> str:
    parts = [title.upper(), "", _clean_text(desc), "", "Input", _clean_text(in_fmt), "", "Output", _clean_text(out_fmt)]
    for i, (sin, sout) in enumerate(examples, 1):
        parts += [
            "",
            f"Sample input {i} (file sample_{i}.in)",
            sin.rstrip(),
            "",
            f"Sample output {i} (file sample_{i}.out)",
            sout.rstrip(),
        ]
    if note and _clean_text(note):
        parts += ["", "Note", _clean_text(note)]
    parts += ["", f"Limits: {tl:g} seconds and {mem_mib} MiB of memory per test."]
    return "\n".join(parts)


def _lang_priority(lang: str) -> int:
    s = lang or ""
    if "C++20" in s or "C++23" in s:
        return 0
    if "C++17" in s:
        return 1
    if "C++14" in s:
        return 2
    return 3


@app.function(image=curate_image, cpu=4.0, memory=16384, timeout=900)
def plan_curation(
    sft_b_v1_ids: list[str],
    legacy_eval_ids: list[str],
    train_quotas: dict[str, int],
    batch_contests: int = 8,
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]], dict[str, Any]]:
    """Download metadata + C++ submissions inside Modal and build contest-grouped verification batches."""
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import HfApi, hf_hub_download
    import pandas as pd

    t0 = time.time()
    api = HfApi()
    work = Path(tempfile.mkdtemp(prefix="plan-"))

    # 1. Discover available generated_tests parquet files (skipping freak >150 MB blobs)
    gt_tree = list(api.list_repo_tree("open-r1/codeforces", path_in_repo="generated_tests", repo_type="dataset"))
    gt_files: dict[int, tuple[str, int]] = {}
    for item in gt_tree:
        m = re.search(r"test_cases_(\d+)\.parquet$", item.path)
        sz = getattr(item, "size", 0) or 0
        if m and 0 < sz <= 150_000_000:
            gt_files[int(m.group(1))] = (item.path, sz)
    print(f"[plan] {len(gt_files)} generated_tests contest files available (<=150MB)", flush=True)

    # 2. Download C++ accepted & incorrect submissions + metadata parquets in parallel
    train_shards = [f"data/train-{i:05d}-of-00011.parquet" for i in range(11)]
    dl_jobs = [
        ("open-r1/codeforces-submissions", "selected_accepted/train-00000-of-00001.parquet"),
        ("open-r1/codeforces-submissions", "selected_incorrect/train-00000-of-00001.parquet"),
        ("open-r1/codeforces", "data/test-00000-of-00001.parquet"),
        *[("open-r1/codeforces", s) for s in train_shards],
    ]

    def _dl(job: tuple[str, str]) -> tuple[str, Path]:
        repo, rel = job
        sub = work / repo.replace("/", "__")
        for att in range(5):
            try:
                p = hf_hub_download(repo, rel, repo_type="dataset", local_dir=str(sub))
                return f"{repo}:{rel}", Path(p)
            except Exception:
                if att == 4:
                    raise
                time.sleep(2.0 * (att + 1) + random.random())
        raise RuntimeError("unreachable")

    with ThreadPoolExecutor(max_workers=4) as ex:
        paths = dict(ex.map(_dl, dl_jobs))
    print(f"[plan] downloaded metadata & submissions in {time.time() - t0:.1f}s", flush=True)

    # 3. Index C++ accepted and TLE/incorrect submissions by problem_id
    sa_df = pd.read_parquet(
        paths["open-r1/codeforces-submissions:selected_accepted/train-00000-of-00001.parquet"],
        columns=["problem_id", "programmingLanguage", "source"],
    )
    sa_df = sa_df[
        sa_df["programmingLanguage"].fillna("").str.contains(r"C\+\+", regex=True)
        & ~sa_df["programmingLanguage"].fillna("").str.contains("MS C", regex=False)
    ]
    ac_by_pid: dict[str, list[tuple[int, int, str]]] = collections.defaultdict(list)
    for pid, lang, src in zip(sa_df["problem_id"], sa_df["programmingLanguage"], sa_df["source"]):
        if isinstance(src, str) and 100 <= len(src) <= 16_000:
            ac_by_pid[str(pid)].append((_lang_priority(str(lang)), len(src), src))
    ac_cpp: dict[str, list[str]] = {}
    for pid, items in ac_by_pid.items():
        items.sort(key=lambda x: (x[0], x[1]))
        seen_s: set[str] = set()
        kept_s: list[str] = []
        for _, _, src in items:
            if src not in seen_s:
                seen_s.add(src)
                kept_s.append(src)
            if len(kept_s) >= 5:
                break
        ac_cpp[pid] = kept_s
    del sa_df, ac_by_pid

    si_df = pd.read_parquet(
        paths["open-r1/codeforces-submissions:selected_incorrect/train-00000-of-00001.parquet"],
        columns=["problem_id", "programmingLanguage", "verdict", "passedTestCount", "source"],
    )
    si_df = si_df[
        si_df["programmingLanguage"].fillna("").str.contains(r"C\+\+", regex=True)
        & ~si_df["programmingLanguage"].fillna("").str.contains("MS C", regex=False)
    ]
    tle_by_pid: dict[str, list[tuple[int, int, str]]] = collections.defaultdict(list)
    for pid, verd, ptc, src in zip(si_df["problem_id"], si_df["verdict"], si_df["passedTestCount"], si_df["source"]):
        if isinstance(src, str) and 80 <= len(src) <= 14_000:
            is_tle = 0 if str(verd) == "TIME_LIMIT_EXCEEDED" else 1
            passed_cnt = int(ptc) if pd.notna(ptc) else 0
            if is_tle == 0 or passed_cnt >= 2:
                tle_by_pid[str(pid)].append((is_tle, -passed_cnt, src))
    tle_cpp: dict[str, list[str]] = {
        pid: [x[2] for x in sorted(items)[:2]] for pid, items in tle_by_pid.items()
    }
    del si_df, tle_by_pid

    # 4. Parse test split first and build the strict anti-leakage shield
    test_df = pd.read_parquet(paths["open-r1/codeforces:data/test-00000-of-00001.parquet"])
    blocked_ids: set[str] = set()
    blocked_cids: set[int] = set()
    blocked_titles: set[str] = set()
    sft_b_set = set(sft_b_v1_ids)

    for leg in legacy_eval_ids:
        # e.g. "2008_F" -> "2008/F" and cid 2008
        parts = leg.split("_", 1)
        if len(parts) == 2 and parts[0].isdigit():
            blocked_ids.add(f"{parts[0]}/{parts[1]}")
            blocked_cids.add(int(parts[0]))

    for _, row in test_df.iterrows():
        rid = str(row["id"])
        blocked_ids.add(rid)
        for al in _to_list(row.get("aliases")):
            blocked_ids.add(str(al))
        try:
            blocked_cids.add(int(row["contest_id"]))
        except Exception:
            pass
        nt = _norm_title(str(row.get("title") or ""))
        if nt:
            blocked_titles.add(nt)

    stats: collections.Counter[str] = collections.Counter()

    def extract_candidate(row: Any, is_eval: bool) -> dict[str, Any] | None:
        rid = str(row["id"])
        if not bool(row.get("executable")):
            stats["drop_not_exec"] += 1
            return None
        if str(row.get("input_mode") or "") != "stdio":
            stats["drop_not_stdio"] += 1
            return None
        if pd.notna(row.get("interaction_format")) and str(row.get("interaction_format")).strip():
            stats["drop_interactive"] += 1
            return None
        if pd.notna(row.get("generated_checker")) and str(row.get("generated_checker")).strip():
            stats["drop_custom_checker"] += 1
            return None
        if pd.isna(row.get("rating")):
            stats["drop_unrated"] += 1
            return None
        rating = int(row["rating"])
        if is_eval and not (1400 <= rating <= 2600):
            return None
        if (not is_eval) and not (1500 <= rating <= 2600):
            return None
        if is_eval and rid in sft_b_set:
            stats["drop_eval_in_sft_b"] += 1
            return None

        try:
            cid = int(row["contest_id"])
        except Exception:
            return None
        if cid not in gt_files:
            stats["drop_no_gt_parquet"] += 1
            return None
        if int(row.get("generated_tests") or 0) < 4:
            stats["drop_few_gt"] += 1
            return None
        if rid not in ac_cpp:
            stats["drop_no_ac_cpp"] += 1
            return None

        desc = str(row.get("description") or "")
        in_fmt = str(row.get("input_format") or "")
        out_fmt = str(row.get("output_format") or "")
        if len(desc.strip()) < 80:
            stats["drop_short_desc"] += 1
            return None
        combined_low = f"{desc}\n{in_fmt}\n{out_fmt}".lower()
        if any(ph in combined_low for ph in REJECT_PHRASES):
            stats["drop_reject_phrase"] += 1
            return None

        raw_ex = _to_list(row.get("examples"))
        samples = [
            [str(e["input"]), str(e["output"])]
            for e in raw_ex
            if isinstance(e, dict) and str(e.get("input", "")).strip() and str(e.get("output", "")).strip()
        ]
        if not samples:
            stats["drop_no_samples"] += 1
            return None

        raw_off = _to_list(row.get("official_tests"))
        off_tests = []
        for t in raw_off:
            if not isinstance(t, dict):
                continue
            ti, to = str(t.get("input", "")), str(t.get("output", ""))
            if not ti.strip() or not to.strip():
                continue
            if ti.rstrip().endswith("...") or to.rstrip().endswith("..."):
                continue
            if len(ti) <= MAX_SINGLE_TEST_BYTES and len(to) <= MAX_SINGLE_TEST_BYTES:
                off_tests.append([ti, to])

        tl = float(row.get("time_limit") or 2.0)
        if tl <= 0 or tl > 8.0:
            stats["drop_bad_tl"] += 1
            return None
        mem_mib = max(256, int(float(row.get("memory_limit") or 256.0)))
        title = str(row.get("title") or rid).strip()
        note = str(row.get("note") or "") if pd.notna(row.get("note")) else ""
        tags = [str(t) for t in _to_list(row.get("tags"))]
        aliases = [str(a) for a in _to_list(row.get("aliases"))]

        single_inst = not any(ph in f"{in_fmt}\n{desc[:400]}".lower() for ph in MULTI_TEST_PHRASES)
        min_lines = min(sum(1 for ln in c.splitlines() if ln.strip()) for c in ac_cpp[rid])

        # IOI / Vals Track-2 structural alignment score
        ioi_score = 0.0
        if single_inst:
            ioi_score += 25.0
        good_hits = sum(1 for t in tags if t in IOI_GOOD_TAGS)
        bad_hits = sum(1 for t in tags if t in NOISY_TAGS)
        ioi_score += min(27.0, good_hits * 9.0) - bad_hits * 12.0
        ioi_score += max(0.0, 20.0 - 0.30 * abs(min_lines - 50.0))
        if 1.0 <= tl <= 4.0:
            ioi_score += 5.0

        idx_clean = re.sub(r"[^A-Za-z0-9]", "", str(row.get("index") or rid.split("/")[-1]))
        pid = f"r1_{cid}_{idx_clean}"

        return {
            "pid": pid,
            "or1_id": rid,
            "cid": cid,
            "gt_path": gt_files[cid][0],
            "aliases": aliases,
            "norm_title": _norm_title(title),
            "title": title,
            "rating": rating,
            "tags": tags,
            "time_limit": tl,
            "memory_limit": mem_mib * 1024 * 1024,
            "statement": _render_statement(title, desc, in_fmt, out_fmt, samples, note, tl, mem_mib),
            "samples": samples,
            "official_tests": off_tests[:20],
            "single_instance": single_inst,
            "ioi_score": round(ioi_score, 2),
            "ac_candidates": ac_cpp[rid],
            "tle_candidates": tle_cpp.get(rid, []),
            "is_eval": is_eval,
        }

    eval_cands = []
    for _, row in test_df.iterrows():
        c = extract_candidate(row, is_eval=True)
        if c is not None:
            eval_cands.append(c)
    del test_df

    # 5. Extract & deduplicate train split candidates
    raw_train_cands = []
    for s in train_shards:
        df = pd.read_parquet(paths[f"open-r1/codeforces:{s}"])
        for _, row in df.iterrows():
            rid = str(row["id"])
            try:
                cid = int(row["contest_id"])
            except Exception:
                continue
            nt = _norm_title(str(row.get("title") or ""))
            als = [str(a) for a in _to_list(row.get("aliases"))]
            if rid in blocked_ids or cid in blocked_cids or (nt and nt in blocked_titles) or any(a in blocked_ids for a in als):
                stats["drop_eval_shield"] += 1
                continue
            c = extract_candidate(row, is_eval=False)
            if c is not None:
                raw_train_cands.append(c)
        del df

    # Deduplicate Div1/Div2 aliases within train (prefer higher rating, then higher ioi_score)
    raw_train_cands.sort(key=lambda c: (-c["rating"], -c["ioi_score"], c["pid"]))
    seen_train_keys: set[str] = set()
    seen_train_titles: set[str] = set()
    seen_train_pids: set[str] = set()
    train_dedup = []
    for c in raw_train_cands:
        keys = {c["or1_id"], *c["aliases"]}
        if c["pid"] in seen_train_pids or (keys & seen_train_keys) or (c["norm_title"] and c["norm_title"] in seen_train_titles):
            stats["drop_alias_dup"] += 1
            continue
        seen_train_pids.add(c["pid"])
        seen_train_keys.update(keys)
        if c["norm_title"]:
            seen_train_titles.add(c["norm_title"])
        train_dedup.append(c)

    # Mild contest-density bonus on train so selected problems cluster into fewer parquet downloads
    cid_counts = collections.Counter(c["cid"] for c in train_dedup)
    for c in train_dedup:
        c["rank_score"] = round(c["ioi_score"] + 3.5 * min(5, cid_counts[c["cid"]]), 2)

    selected_train = []
    for band_str, quota in train_quotas.items():
        lo, hi = map(int, band_str.split("-"))
        pool = [c for c in train_dedup if lo <= c["rating"] <= hi]
        pool.sort(key=lambda c: (-c["rank_score"], -c["ioi_score"], c["pid"]))
        selected_train.extend(pool[:quota])

    def pack_batches(cands: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        by_cid: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
        for c in cands:
            by_cid[c["cid"]].append(c)
        cids_sorted = sorted(by_cid.keys())
        batches = []
        for i in range(0, len(cids_sorted), batch_contests):
            chunk_cids = cids_sorted[i : i + batch_contests]
            batches.append([ {"cid": cid, "gt_path": by_cid[cid][0]["gt_path"], "problems": by_cid[cid]} for cid in chunk_cids ])
        return batches

    eval_batches = pack_batches(eval_cands)
    train_batches = pack_batches(selected_train)

    shutil.rmtree(work, ignore_errors=True)
    plan_summary = {
        "eval_candidates": len(eval_cands),
        "eval_contests": len({c["cid"] for c in eval_cands}),
        "eval_batches": len(eval_batches),
        "train_dedup_pool": len(train_dedup),
        "train_selected_candidates": len(selected_train),
        "train_contests": len({c["cid"] for c in selected_train}),
        "train_batches": len(train_batches),
        "filter_stats": dict(stats),
        "plan_seconds": round(time.time() - t0, 1),
    }
    print(f"[plan] summary: {json.dumps(plan_summary)}", flush=True)
    return eval_batches, train_batches, plan_summary


@app.function(
    image=curate_image,
    volumes={"/work/tests": volume},
    cpu=2.0,
    memory=8192,
    timeout=1500,
    max_containers=16,
)
def verify_and_stage_batch(contest_groups: list[dict[str, Any]]) -> dict[str, Any]:
    """Download contest generated_tests, verify C++20 reference solutions in chroot jail, and stage tests to cp-tests."""
    from huggingface_hub import hf_hub_download
    import pandas as pd

    sys.path.insert(0, "/opt")
    import runner_in_container as R  # noqa: E402

    work = Path(tempfile.mkdtemp(prefix="verify-"))
    os.chmod(work, 0o755)
    bdir = work / "bin"
    bdir.mkdir(exist_ok=True)
    tdir = work / "tmp_tests"
    tdir.mkdir(exist_ok=True)
    outdir = work / "out"
    outdir.mkdir(exist_ok=True)
    os.chmod(outdir, 0o777)

    kept_problems: list[dict[str, Any]] = []
    kept_solutions: list[dict[str, Any]] = []
    counts: collections.Counter[str] = collections.Counter()

    def compile_static(code: str, tag: str) -> Path | None:
        src = bdir / f"{tag}.cpp"
        exe = bdir / tag
        src.write_text(code)
        try:
            p = subprocess.run(
                ["g++", *CXX_FLAGS, str(src), "-o", str(exe)],
                capture_output=True,
                text=True,
                timeout=45,
            )
        except subprocess.TimeoutExpired:
            src.unlink(missing_ok=True)
            return None
        src.unlink(missing_ok=True)
        if p.returncode != 0 or not exe.exists():
            return None
        os.chmod(exe, 0o755)
        return exe

    def judge_pairs(exe: Path, pairs: list[list[str]], tl: float, mem_bytes: int, prefix: str) -> tuple[bool, float, list[str]]:
        test_specs = []
        created: list[Path] = []
        for i, (inp, exp) in enumerate(pairs):
            fi = tdir / f"{prefix}_{i}.in"
            fo = tdir / f"{prefix}_{i}.out"
            fi.write_text(inp)
            fo.write_text(exp)
            os.chmod(fi, 0o600)
            os.chmod(fo, 0o600)
            created.extend([fi, fo])
            test_specs.append({"in": str(fi), "out": str(fo)})
        try:
            res = R.run_spec(
                {
                    "binary": str(exe),
                    "tests": test_specs,
                    "checker": None,
                    "time_limit": tl,
                    "mem_bytes": mem_bytes,
                    "out_dir": str(outdir),
                    "stop_on_fail": True,
                    "keep_stdout": False,
                    "timing": "wall",
                }
            )
        finally:
            for p in created:
                p.unlink(missing_ok=True)
        all_ok = len(res) == len(pairs) and all(bool(r.get("passed")) for r in res)
        max_t = max((float(r.get("time", 0.0)) for r in res), default=0.0)
        verdicts = [str(r.get("status", "?")) for r in res]
        return all_ok, max_t, verdicts

    def spread_pick(lst: list[list[str]], k: int, keep_last: int = 0) -> list[list[str]]:
        if len(lst) <= k:
            return lst
        tail = lst[len(lst) - keep_last :] if keep_last > 0 else []
        head = lst[: len(lst) - keep_last]
        m = k - len(tail)
        if m <= 0:
            return tail[-k:]
        if m == 1:
            return [head[0]] + tail
        idxs = sorted({round(j * (len(head) - 1) / (m - 1)) for j in range(m)})
        return [head[j] for j in idxs] + tail

    try:
        os.makedirs("/work/tests", exist_ok=True)
        os.chmod("/work/tests", 0o700)
        rng = random.Random(42)

        for grp in contest_groups:
            gt_rel = grp["gt_path"]
            dl_dir = work / "hf"
            gt_df = None
            for att in range(4):
                try:
                    gt_local = Path(
                        hf_hub_download("open-r1/codeforces", gt_rel, repo_type="dataset", local_dir=str(dl_dir))
                    )
                    gt_df = pd.read_parquet(gt_local, columns=["problem_id", "input", "output"])
                    break
                except Exception as e:
                    if att == 3:
                        print(f"[verify] failed to load {gt_rel}: {e}", flush=True)
                    else:
                        time.sleep(2.0 * (att + 1) + random.random())
            if gt_df is None:
                counts["drop_gt_download_err"] += len(grp["problems"])
                shutil.rmtree(dl_dir, ignore_errors=True)
                continue

            for cand in grp["problems"]:
                pid = cand["pid"]
                or1_id = cand["or1_id"]
                sub = gt_df[gt_df["problem_id"] == or1_id]

                seen_in = {s[0] for s in cand["samples"]}
                uniq: list[list[str]] = []
                for ti, to in cand["official_tests"]:
                    if ti not in seen_in and to.strip():
                        seen_in.add(ti)
                        uniq.append([ti, to])
                for ti, to in zip(sub["input"], sub["output"]):
                    tis, tos = str(ti), str(to)
                    if (
                        tis not in seen_in
                        and tis.strip()
                        and tos.strip()
                        and len(tis) <= MAX_SINGLE_TEST_BYTES
                        and len(tos) <= MAX_SINGLE_TEST_BYTES
                        and not tis.rstrip().endswith("...")
                        and not tos.rstrip().endswith("...")
                    ):
                        seen_in.add(tis)
                        uniq.append([tis, tos])

                if len(uniq) < 8:
                    counts["drop_too_few_tests"] += 1
                    continue

                uniq.sort(key=lambda pair: len(pair[0]))
                half = len(uniq) // 2
                small_pairs = spread_pick(uniq[:half], k=6, keep_last=0)
                large_pairs = spread_pick(uniq[half:], k=6, keep_last=3)
                max_bytes = max(len(x[0]) for x in large_pairs)
                min_req_bytes = 15_000 if cand["is_eval"] else 25_000

                # Test C++20 reference candidates
                tl = float(cand["time_limit"])
                mem = int(cand["memory_limit"])
                winning_cpp: str | None = None
                winning_time = 0.0
                compiled_any = False

                for ci, cpp_src in enumerate(cand["ac_candidates"]):
                    exe = compile_static(cpp_src, f"{pid}_ac{ci}")
                    if exe is None:
                        continue
                    compiled_any = True
                    try:
                        s_ok, _, _ = judge_pairs(exe, cand["samples"], max(tl, 2.0), mem, f"{pid}_smp")
                        if not s_ok:
                            continue
                        sm_ok, sm_t, _ = judge_pairs(exe, small_pairs, tl, mem, f"{pid}_sm")
                        if not sm_ok or sm_t > tl * 0.80:
                            continue
                        lg_ok, lg_t, _ = judge_pairs(exe, large_pairs, tl, mem, f"{pid}_lg")
                        if not lg_ok or lg_t > tl * 0.80:
                            continue
                        winning_cpp = cpp_src
                        winning_time = round(max(sm_t, lg_t), 4)
                        break
                    finally:
                        exe.unlink(missing_ok=True)

                if winning_cpp is None:
                    counts["drop_no_cpp20_compile" if not compiled_any else "drop_ref_failed_tests"] += 1
                    continue

                # Optional empirical TLE check (REQUIRED if max_bytes < min_req_bytes!)
                tle_verified = False
                tle_small_pass = False
                for ti_idx, tle_src in enumerate(cand["tle_candidates"]):
                    texe = compile_static(tle_src, f"{pid}_tle{ti_idx}")
                    if texe is None:
                        continue
                    try:
                        smp_ok, _, _ = judge_pairs(texe, cand["samples"], max(tl, 2.0), mem, f"{pid}_tsmp")
                        if not smp_ok:
                            continue
                        sm_ok, _, _ = judge_pairs(texe, small_pairs, tl, mem, f"{pid}_tsm")
                        lg_ok, _, lg_verdicts = judge_pairs(texe, large_pairs, tl, mem, f"{pid}_tlg")
                        if (not lg_ok) and ("TLE" in lg_verdicts):
                            tle_verified = True
                            tle_small_pass = sm_ok
                            break
                    finally:
                        texe.unlink(missing_ok=True)

                if max_bytes < min_req_bytes and not tle_verified:
                    counts["drop_small_max_bytes_and_no_tle"] += 1
                    continue

                # Stage verified hidden tests directly onto /work/tests/<pid>
                dest = Path("/work/tests") / pid
                shutil.rmtree(dest, ignore_errors=True)
                dest.mkdir(parents=True, exist_ok=True)
                os.chmod(dest, 0o700)

                hs_rel: list[list[str]] = []
                hl_rel: list[list[str]] = []
                for pre, pairs, out_list in (("s", small_pairs, hs_rel), ("l", large_pairs, hl_rel)):
                    for i, (inp, exp) in enumerate(pairs):
                        fi, fo = dest / f"{pre}{i}.in", dest / f"{pre}{i}.out"
                        fi.write_text(inp)
                        fo.write_text(exp)
                        os.chmod(fi, 0o600)
                        os.chmod(fo, 0o600)
                        out_list.append([f"{pid}/{pre}{i}.in", f"{pid}/{pre}{i}.out"])

                kept_problems.append(
                    {
                        "id": pid,
                        "or1_id": or1_id,
                        "title": cand["title"],
                        "rating": cand["rating"],
                        "tags": cand["tags"],
                        "time_limit": tl,
                        "memory_limit": mem,
                        "statement": cand["statement"],
                        "samples": cand["samples"],
                        "checker": None,
                        "hidden_small": hs_rel,
                        "hidden_large": hl_rel,
                        "max_input_bytes": max_bytes,
                        "single_instance": cand["single_instance"],
                        "ioi_score": cand["ioi_score"],
                        "tle_verified": tle_verified,
                        "tle_40pt_subtask": tle_small_pass,
                        "ref_max_time_s": winning_time,
                    }
                )
                kept_solutions.append(
                    {
                        "id": pid,
                        "or1_id": or1_id,
                        "rating": cand["rating"],
                        "fast_cpp": winning_cpp,
                        "ref_max_time_s": winning_time,
                    }
                )
                counts["kept"] += 1

            del gt_df
            shutil.rmtree(dl_dir, ignore_errors=True)

        if kept_problems:
            for attempt in range(6):
                try:
                    volume.commit()
                    break
                except Exception as e:
                    if attempt == 5:
                        raise
                    time.sleep(2.0 * (attempt + 1) + random.random())

        print(f"[verify] batch done: {dict(counts)}", flush=True)
        return {"problems": kept_problems, "solutions": kept_solutions, "counts": dict(counts)}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _stratify_eval(probs: list[dict[str, Any]], target_per_band: int = 10) -> list[dict[str, Any]]:
    bands = [(1400, 1599), (1600, 1799), (1800, 1999), (2000, 2199), (2200, 2399), (2400, 2600)]
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    for lo, hi in bands:
        pool = [p for p in probs if lo <= p["rating"] <= hi]
        pool.sort(key=lambda p: (-p["ioi_score"], -p["max_input_bytes"], p["id"]))
        for p in pool[:target_per_band]:
            chosen.append(p)
            used.add(p["id"])
    target_total = target_per_band * len(bands)
    if len(chosen) < target_total:
        rem = [p for p in probs if p["id"] not in used]
        rem.sort(key=lambda p: (-p["ioi_score"], -p["max_input_bytes"], p["id"]))
        chosen.extend(rem[: target_total - len(chosen)])
    chosen.sort(key=lambda p: (p["rating"], -p["ioi_score"], p["id"]))
    return chosen


@app.local_entrypoint()
def main(eval_per_band: int = 10, max_train: int = 600):
    t0 = time.time()
    # Collect any existing local eval / sft IDs for cross-repo non-contamination
    main_repo = Path("/Users/pracher/vals-ai/vals-ai-hackathon-lora")
    legacy_eval_ids: list[str] = []
    cf_test_file = main_repo / "post-train" / "data" / "eval" / "cf_test.jsonl"
    if cf_test_file.exists():
        legacy_eval_ids = [json.loads(line)["id"] for line in cf_test_file.open()]

    sft_b_ids: list[str] = []
    try:
        out = subprocess.check_output(
            ["git", "-C", str(main_repo), "show", "origin/sft-data:post-train/sft-data/sft_b_v1.problem_ids.txt"],
            text=True,
        )
        sft_b_ids = [ln.strip() for ln in out.splitlines() if ln.strip()]
    except Exception:
        pass

    train_quotas = {
        "1500-1699": 135,
        "1700-1899": 210,
        "1900-2099": 210,
        "2100-2299": 170,
        "2300-2600": 115,
    }
    print(f"Starting cloud curation on Modal (sft_b_ids={len(sft_b_ids)}, legacy_eval_ids={len(legacy_eval_ids)})...")
    eval_batches, train_batches, plan_summary = plan_curation.remote(
        sft_b_ids, legacy_eval_ids, train_quotas, batch_contests=8
    )

    print(f"Verifying & staging {len(eval_batches)} eval batches + {len(train_batches)} train batches across Modal...")
    all_batches = eval_batches + train_batches
    n_eval_b = len(eval_batches)

    eval_probs, eval_sols = [], {}
    train_probs, train_sols = [], {}
    agg_eval_counts: collections.Counter[str] = collections.Counter()
    agg_train_counts: collections.Counter[str] = collections.Counter()

    for idx, res in enumerate(verify_and_stage_batch.map(all_batches)):
        is_ev = idx < n_eval_b
        target_p = eval_probs if is_ev else train_probs
        target_s = eval_sols if is_ev else train_sols
        target_c = agg_eval_counts if is_ev else agg_train_counts
        target_p.extend(res["problems"])
        for s in res["solutions"]:
            target_s[s["id"]] = s
        target_c.update(res["counts"])

    final_eval = _stratify_eval(eval_probs, target_per_band=eval_per_band)
    train_probs.sort(key=lambda p: (p["rating"], -p["ioi_score"], p["id"]))
    if max_train and len(train_probs) > max_train:
        # Proportional stratification across ratings
        step = len(train_probs) / max_train
        final_train = [train_probs[min(len(train_probs) - 1, int(i * step))] for i in range(max_train)]
        # Deduplicate if rounding collided
        seen_ids = {p["id"] for p in final_train}
        if len(seen_ids) < max_train:
            dedup = []
            s2 = set()
            for p in final_train + train_probs:
                if p["id"] not in s2:
                    s2.add(p["id"])
                    dedup.append(p)
                if len(dedup) == max_train:
                    break
            final_train = sorted(dedup, key=lambda p: (p["rating"], -p["ioi_score"], p["id"]))
    else:
        final_train = train_probs

    # Final hard assertion: zero ID or title overlap between train and eval
    eval_id_set = {p["id"] for p in final_eval}
    eval_or1_set = {p["or1_id"] for p in final_eval}
    eval_titles = {_norm_title(p["title"]) for p in final_eval}
    final_train = [
        p
        for p in final_train
        if p["id"] not in eval_id_set
        and p["or1_id"] not in eval_or1_set
        and _norm_title(p["title"]) not in eval_titles
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "eval.jsonl").write_text("".join(json.dumps(p) + "\n" for p in final_eval))
    (OUT_DIR / "train.jsonl").write_text("".join(json.dumps(p) + "\n" for p in final_train))
    (OUT_DIR / "eval_solutions.jsonl").write_text(
        "".join(json.dumps(eval_sols[p["id"]]) + "\n" for p in final_eval)
    )
    (OUT_DIR / "train_solutions.jsonl").write_text(
        "".join(json.dumps(train_sols[p["id"]]) + "\n" for p in final_train)
    )

    def _dist(ps: list[dict[str, Any]]) -> dict[str, Any]:
        if not ps:
            return {}
        mb = sorted(p["max_input_bytes"] for p in ps)
        rc = dict(sorted(collections.Counter(p["rating"] for p in ps).items()))
        return {
            "count": len(ps),
            "single_instance_count": sum(bool(p["single_instance"]) for p in ps),
            "single_instance_frac": round(sum(bool(p["single_instance"]) for p in ps) / len(ps), 3),
            "tle_empirically_verified_count": sum(bool(p["tle_verified"]) for p in ps),
            "tle_40pt_subtask_count": sum(bool(p["tle_40pt_subtask"]) for p in ps),
            "max_input_bytes_p10": mb[len(mb) // 10],
            "max_input_bytes_median": mb[len(mb) // 2],
            "max_input_bytes_p90": mb[int(len(mb) * 0.9)],
            "ratings": rc,
        }

    summary = {
        "elapsed_seconds": round(time.time() - t0, 1),
        "plan": plan_summary,
        "eval_verify_counts": dict(agg_eval_counts),
        "train_verify_counts": dict(agg_train_counts),
        "eval": _dist(final_eval),
        "train": _dist(final_train),
    }
    (OUT_DIR / "stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
