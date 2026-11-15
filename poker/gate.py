"""Fast offline gate: replay real arena prompts against one or more deployments and compare them.

    uv run python -m poker.gate --deployment vals-hackathon/base --deployment vals-hackathon/cand --n-other 150

Requests match the arena: exact recorded `messages`, both tools, max_completion_tokens 5000, thinking on.
Judged spots (teacher has a preferred action) give an agreement rate; spots where the teacher abstains give an
action distribution and drift from the action base originally played. Needs AC2_API_KEY (or ~/.ac2_api_key).
"""

from __future__ import annotations

import argparse
import json
import os
import math
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from poker.trace_quality import legal_reply

INFERENCE_URL = "https://inference.appliedcompute.com/v1"
TOOLS_PATH = Path(__file__).resolve().parent.parent / "opsd" / "tools.json"
ACTIONS = ("fold", "check", "call", "bet", "raise")
RULES = ("allin_call", "own_shove", "unopened", "facing_raise", "postflop_call", "free_fold")
RETRY_STATUS = {429, 500, 502, 503, 504}


def load_api_key() -> str:
    key = os.environ.get("AC2_API_KEY")
    if not key:
        path = Path.home() / ".ac2_api_key"
        if path.exists():
            key = path.read_text().strip()
            os.environ["AC2_API_KEY"] = key
    if not key:
        raise SystemExit("set AC2_API_KEY or put the key in ~/.ac2_api_key")
    return key


def chat_tools(path: Path = TOOLS_PATH) -> list[dict]:
    """opsd/tools.json is in flat (Responses) form; chat completions wants {"type", "function": {...}}."""
    out = []
    for tool in json.loads(path.read_text()):
        if "function" in tool:
            out.append(tool)
        else:
            out.append({"type": "function", "function": {k: v for k, v in tool.items() if k != "type"}})
    return out


def _text(content) -> str:
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def chat_messages(items: list[dict]) -> list[dict]:
    """Recorded Responses-style items -> chat messages. function_call items become tool_calls on the preceding
    assistant message (or a new empty one); function_call_output items become tool messages."""
    out = []
    for m in items:
        kind = m.get("type")
        if kind == "function_call":
            call = {"id": m["call_id"], "type": "function",
                    "function": {"name": m["name"], "arguments": m["arguments"] if isinstance(m["arguments"], str) else json.dumps(m["arguments"])}}
            if out and out[-1]["role"] == "assistant":
                out[-1].setdefault("tool_calls", []).append(call)
            else:
                out.append({"role": "assistant", "content": "", "tool_calls": [call]})
        elif kind == "function_call_output":
            out.append({"role": "tool", "tool_call_id": m["call_id"], "content": _text(m.get("output"))})
        else:
            out.append({"role": m["role"], "content": _text(m.get("content"))})
    return out


class HttpClient:
    """POSTs one chat completion, retrying 429/5xx with exponential backoff. Never logs the key."""

    def __init__(self, api_key: str, *, url: str = INFERENCE_URL, retries: int = 6, timeout: float = 600,
                 sleep=time.sleep):
        import httpx

        self.httpx, self.url, self.retries, self.sleep = httpx, url, retries, sleep
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.timeout = timeout

    def complete(self, model: str, messages: list[dict], tools: list[dict]) -> dict:
        body = {"model": model, "messages": messages, "tools": tools, "max_completion_tokens": 5000,
                "chat_template_kwargs": {"enable_thinking": True}}
        wait = 2.0
        for attempt in range(self.retries + 1):
            try:
                resp = self.httpx.post(f"{self.url}/chat/completions", json=body, headers=self.headers,
                                       timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()
                retry = resp.status_code in RETRY_STATUS
                err = f"HTTP {resp.status_code}"
            except self.httpx.HTTPError as exc:
                retry, err = True, type(exc).__name__
            if not retry or attempt == self.retries:
                return {"error": err}
            self.sleep(wait)
            wait = min(wait * 2, 60)
        return {"error": "unreachable"}


def parse_reply(prompt: list[dict], response: dict) -> dict:
    """Reduce a chat-completions response to {valid, cutoff, tokens, action, error}."""
    if "choices" not in response:
        return {"valid": False, "cutoff": False, "tokens": None, "action": None, "error": response.get("error", "no choices")}
    choice = response["choices"][0]
    message = choice.get("message") or {}
    calls = [{"type": "function_call", "name": c["function"]["name"], "arguments": c["function"]["arguments"]}
             for c in message.get("tool_calls") or []]
    problem = legal_reply(prompt, calls)
    action = None
    if len(calls) == 1:
        try:
            args = json.loads(calls[0]["arguments"]) if isinstance(calls[0]["arguments"], str) else calls[0]["arguments"]
            action = args.get("action") if isinstance(args, dict) else None
        except ValueError:
            action = None
    usage = response.get("usage") or {}
    return {"valid": problem is None, "cutoff": choice.get("finish_reason") == "length",
            "tokens": usage.get("completion_tokens"), "action": action if action in ACTIONS else None, "error": problem}


def cleanup_ok(response: dict) -> bool:
    """A valid after-hand reply: exactly one update_scratchpad call with string contents of at most 5,000 chars."""
    if "choices" not in response:
        return False
    choice = response["choices"][0]
    calls = (choice.get("message") or {}).get("tool_calls") or []
    if choice.get("finish_reason") == "length" or len(calls) != 1 or calls[0]["function"]["name"] != "update_scratchpad":
        return False
    try:
        args = json.loads(calls[0]["function"]["arguments"])
    except (TypeError, ValueError):
        return False
    return isinstance(args, dict) and isinstance(args.get("contents"), str) and len(args["contents"]) <= 5000


def run_cleanup_check(joined: Path, deployments: list[str], client, tools: list[dict], *, n: int, seed: int,
                      source_deployment: str = "infer_d3d2b6b92171", concurrency: int = 8) -> dict:
    """Replay n real cleanup prompts (from base's arena games) on each deployment; share answered correctly."""
    pool = []
    for line in joined.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("source") == "arena" and r.get("kind") == "cleanup" and r.get("deployment") == source_deployment:
            pool.append(r)
    pool.sort(key=lambda r: r["span_id"])
    picks = random.Random(seed).sample(pool, min(n, len(pool)))
    jobs = [(d, r) for d in deployments for r in picks]

    def one(job):
        d, r = job
        return d, cleanup_ok(client.complete(d, chat_messages(r["messages"]), tools))

    out = {d: [0, 0] for d in deployments}
    with ThreadPoolExecutor(max_workers=concurrency) as pool_:
        for d, ok in pool_.map(one, jobs):
            out[d][0] += ok
            out[d][1] += 1
    return {d: {"ok": k, "n": t, "rate": k / t if t else None} for d, (k, t) in out.items()}


def original_action(record: dict) -> str | None:
    calls = [c for c in record.get("tool_calls") or [] if c.get("name") == "submit_action"]
    if len(calls) != 1:
        return None
    args = calls[0].get("arguments")
    try:
        args = json.loads(args) if isinstance(args, str) else args
    except ValueError:
        return None
    action = args.get("action") if isinstance(args, dict) else None
    return action if action in ACTIONS else None


def select_spots(entries: list[dict], *, n_judged: int | None, n_other: int, seed: int, repeats: int = 1,
                 other_repeats: int = 1, bust_repeats: int | None = None) -> list[dict]:
    """entries: {record, verdict}. Judged = verdict.preferred is not None. Deterministic for a fixed seed,
    whatever the input order."""
    key = lambda e: e["record"]["span_id"]  # noqa: E731
    judged = sorted((e for e in entries if e["verdict"].get("preferred") is not None), key=key)
    other = sorted((e for e in entries if e["verdict"].get("preferred") is None), key=key)
    rng = random.Random(seed)
    if n_judged is not None and n_judged < len(judged):
        judged = rng.sample(judged, n_judged)
    other = rng.sample(other, min(n_other, len(other)))
    # Sampling at temperature is noisy; replaying each judged spot `repeats` times averages that noise out.
    # Bust spots (calling or making preflop all-ins) decide placement, so they get extra replays.
    reps = lambda e: bust_repeats if bust_repeats and e["verdict"].get("rule") in BUST_RULES else repeats  # noqa: E731
    return ([dict(e, group="judged") for e in judged for _ in range(reps(e))]
            + [dict(e, group="other") for e in other for _ in range(other_repeats)])


def quantile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p * (len(ordered) - 1)))]


def aggregate(results: list[dict]) -> dict:
    """results: one deployment's per-spot dicts with group, rule, preferred, orig, valid, cutoff, tokens, action."""
    n = len(results)
    tokens = [r["tokens"] for r in results if r["tokens"] is not None]
    out = {"n": n, "errors": sum(1 for r in results if r.get("error") in ("http",)),
           "valid_rate": sum(r["valid"] for r in results) / n if n else None,
           "cutoff_rate": sum(r["cutoff"] for r in results) / n if n else None,
           "tokens_median": statistics.median(tokens) if tokens else None, "tokens_p90": quantile(tokens, 0.9)}
    judged = [r for r in results if r["group"] == "judged"]
    agree = lambda rs: sum(1 for r in rs if r["action"] == r["preferred"])  # noqa: E731
    out["judged"] = {"n": len(judged), "agree": agree(judged), "agree_rate": agree(judged) / len(judged) if judged else None,
                     "by_rule": {}}
    for rule in sorted({r["rule"] for r in judged}):
        rs = [r for r in judged if r["rule"] == rule]
        out["judged"]["by_rule"][rule] = {"n": len(rs), "agree": agree(rs), "agree_rate": agree(rs) / len(rs)}
    # Call rate when facing an all-in: improving arena teams call ~30-35%; our base calls ~58% and busts more.
    allin = [r for r in judged if r["rule"] == "allin_call" and r["action"]]
    out["allin_call_rate"] = sum(1 for r in allin if r["action"] in ("call", "raise")) / len(allin) if allin else None
    other = [r for r in results if r["group"] == "other"]
    counts = {a: sum(1 for r in other if r["action"] == a) for a in ACTIONS}
    out["other"] = {"n": len(other), "dist": {a: c / len(other) if other else None for a, c in counts.items()},
                    "drift": sum(1 for r in other if r["action"] != r["orig"]) / len(other) if other else None}
    return out


def bootstrap_ci(values: list[float], *, seed: int = 0, n: int = 2000, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of `values`."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(k)) / k for _ in range(n))
    tail = (1 - level) / 2
    return means[int(tail * (n - 1))], means[int((1 - tail) * (n - 1))]


def _spot_agreement(rows: list[dict]) -> dict[str, float]:
    by: dict[str, list[int]] = {}
    for r in rows:
        if r["group"] == "judged":
            by.setdefault(r["span_id"], []).append(int(r["action"] == r["preferred"]))
    return {k: sum(v) / len(v) for k, v in by.items()}


def paired_difference(reference: list[dict], candidate: list[dict], *, seed: int = 0) -> dict:
    """Candidate minus reference teacher agreement, averaged per judged spot, with a bootstrap 95% interval
    over spots. Spots are the unit, so the repeats of one spot do not count as independent evidence."""
    ref, cand = _spot_agreement(reference), _spot_agreement(candidate)
    common = sorted(set(ref) & set(cand))
    diffs = [cand[k] - ref[k] for k in common]
    if not diffs:
        return {"n_spots": 0, "diff": None, "lo": None, "hi": None}
    lo, hi = bootstrap_ci(diffs, seed=seed)
    return {"n_spots": len(common), "diff": sum(diffs) / len(diffs), "lo": lo, "hi": hi}


def mixing(rows: list[dict]) -> dict:
    """How much the policy still mixes on spots the teacher does not judge: the share of spots whose replays
    chose more than one action, and the entropy of the pooled action distribution."""
    by: dict[str, list[str]] = {}
    for r in rows:
        if r["group"] == "other" and r["action"]:
            by.setdefault(r["span_id"], []).append(r["action"])
    multi = [v for v in by.values() if len(v) > 1]
    pooled = [a for v in by.values() for a in v]
    entropy = 0.0
    for a in set(pooled):
        q = pooled.count(a) / len(pooled)
        entropy -= q * math.log2(q)
    return {"spots": len(multi), "mixed_rate": sum(len(set(v)) > 1 for v in multi) / len(multi) if multi else None,
            "entropy_bits": round(entropy, 3)}


VALID_SLACK, CUTOFF_SLACK, MIXING_FLOOR, WORSE_MARGIN = 0.01, 0.01, 0.6, -0.05
BUST_RULES = ("allin_call", "own_shove")
BUST_SLACK = 0.05  # bust-spot agreement may not fall more than 5 points below the reference


def verdict(reference: dict, candidate: dict, diff: dict) -> dict:
    """Pass when the candidate is about as valid as the reference (within 1 point), rarely cut off, not clearly worse than the reference on judged spots,
    and still mixes on other spots (at least 60% of the reference's mixed rate). Determinism is not a goal."""
    reasons = []
    min_valid = (reference.get("valid_rate") or 0) - VALID_SLACK
    max_cutoff = max(0.02, (reference.get("cutoff_rate") or 0) + CUTOFF_SLACK)
    if (candidate["valid_rate"] or 0) < min_valid:
        reasons.append(f"valid rate {candidate['valid_rate']:.1%} below reference minus 1 point ({min_valid:.1%})")
    if (candidate["cutoff_rate"] or 0) > max_cutoff:
        reasons.append(f"cut-off rate {candidate['cutoff_rate']:.1%} above {max_cutoff:.1%}")
    if diff.get("hi") is not None and diff["hi"] < 0:
        reasons.append(f"teacher agreement clearly worse ({diff['diff']:+.0%}, CI {diff['lo']:+.0%}..{diff['hi']:+.0%})")
    elif diff.get("diff") is not None and diff["diff"] < WORSE_MARGIN:
        reasons.append(f"teacher agreement lower ({diff['diff']:+.0%})")
    for rule in BUST_RULES:
        r = ((reference.get("judged") or {}).get("by_rule") or {}).get(rule)
        c = ((candidate.get("judged") or {}).get("by_rule") or {}).get(rule)
        if r and c and r["n"] >= 10 and c["agree_rate"] < r["agree_rate"] - BUST_SLACK:
            reasons.append(f"bust spots ({rule}) worse: {c['agree_rate']:.0%} vs {r['agree_rate']:.0%}")
    ref_mix = (reference.get("mixing") or {}).get("mixed_rate")
    cand_mix = (candidate.get("mixing") or {}).get("mixed_rate")
    if ref_mix and cand_mix is not None and cand_mix < MIXING_FLOOR * ref_mix:
        reasons.append(f"mixing collapsed: {cand_mix:.0%} of spots mixed vs {ref_mix:.0%} for the reference")
    return {"pass": not reasons, "reasons": reasons}


def run_gate(spots: list[dict], deployments: list[str], client, tools: list[dict], concurrency: int = 8) -> dict:
    jobs = [(d, s) for d in deployments for s in spots]

    def one(job):
        dep, spot = job
        record, verdict = spot["record"], spot["verdict"]
        resp = client.complete(dep, chat_messages(record["messages"]), tools)
        parsed = parse_reply(record["messages"], resp)
        if "choices" not in resp:
            parsed["error"] = "http"
        return dep, {"span_id": record["span_id"], "group": spot["group"], "rule": verdict.get("rule"),
                     "preferred": verdict.get("preferred"), "orig": original_action(record)} | parsed

    per = {d: [] for d in deployments}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for dep, row in pool.map(one, jobs):
            per[dep].append(row)
    return {"spots": per, "summary": {d: aggregate(rows) for d, rows in per.items()}}


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def format_table(summary: dict) -> str:
    lines = []
    deps = list(summary)
    w = max([len(d) for d in deps] + [10])

    def row(label, f):
        lines.append(f"{label:<28}" + "".join(f"{f(summary[d]):>{w + 2}}" for d in deps))

    lines.append(f"{'':<28}" + "".join(f"{d:>{w + 2}}" for d in deps))
    row("spots", lambda s: str(s["n"]))
    row("valid", lambda s: _pct(s["valid_rate"]))
    row("cutoff", lambda s: _pct(s["cutoff_rate"]))
    row("tokens median / p90", lambda s: f"{s['tokens_median']:.0f} / {s['tokens_p90']:.0f}" if s["tokens_median"] is not None else "-")
    row("judged agree (all)", lambda s: f"{_pct(s['judged']['agree_rate'])} ({s['judged']['agree']}/{s['judged']['n']})")
    for rule in RULES:
        row(f"  {rule}", lambda s, rule=rule: (lambda b: f"{_pct(b['agree_rate'])} ({b['agree']}/{b['n']})" if b else "-")(s["judged"]["by_rule"].get(rule)))
    for a in ACTIONS:
        row(f"other: {a}", lambda s, a=a: _pct(s["other"]["dist"][a]))
    row("other: drift vs original", lambda s: _pct(s["other"]["drift"]))
    row("all-in spots: call rate", lambda s: _pct(s.get("allin_call_rate")))
    return "\n".join(lines)


def load_entries(joined: Path, db_path: Path, source_deployment: str | None = None) -> list[dict]:
    from poker.arena import db
    from poker.teacher import judge

    con = db.connect(db_path)
    entries = []
    for line in joined.open(encoding="utf-8"):
        record = json.loads(line)
        match = record.get("match") or {}
        if record.get("source") != "arena" or record.get("kind") != "action" or not match.get("verified"):
            continue
        if source_deployment and record.get("deployment") != source_deployment:
            continue
        verdict = judge(con, match["match_id"], match["hand"], match["seat"], match["action_index"])
        entries.append({"record": record, "verdict": verdict})
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deployment", action="append", required=True, help="vals-hackathon/<name>; repeatable")
    parser.add_argument("--joined", type=Path, default=Path("data/arena/joined.jsonl"))
    parser.add_argument("--db", type=Path, default=Path("data/arena/arena.db"))
    parser.add_argument("--source-deployment", help="only arena decisions made by this deployment id")
    parser.add_argument("--n-judged", type=int, default=None, help="default: all teacher-judged spots")
    parser.add_argument("--n-other", type=int, default=150)
    parser.add_argument("--repeats", type=int, default=3, help="replays per judged spot")
    parser.add_argument("--other-repeats", type=int, default=2, help="replays per other spot (measures mixing)")
    parser.add_argument("--bust-repeats", type=int, default=5, help="replays per all-in spot (decides placement)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--n-cleanup", type=int, default=30, help="cleanup prompts replayed per deployment (0 skips)")
    parser.add_argument("--exploits", default="", help="teacher rules to enable, comma list; 'simple' = the v6 set")
    args = parser.parse_args()
    client = HttpClient(load_api_key())
    started = time.monotonic()
    if args.exploits:
        from poker import teacher

        if args.exploits == "simple":
            from poker.sft.make_records import SIMPLE_EXPLOITS

            teacher.EXPLOITS = set(SIMPLE_EXPLOITS)
        else:
            teacher.EXPLOITS = set(args.exploits.split(","))
    entries = load_entries(args.joined, args.db, args.source_deployment)
    spots = select_spots(entries, n_judged=args.n_judged, n_other=args.n_other, seed=args.seed, repeats=args.repeats,
                         other_repeats=args.other_repeats, bust_repeats=args.bust_repeats)
    print(f"{len(entries)} arena decisions; replaying {len(spots)} x {len(args.deployment)} deployments", flush=True)
    result = run_gate(spots, args.deployment, client, chat_tools(), args.concurrency)
    out = args.out or Path("data/gate") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"deployments": args.deployment, "seed": args.seed, "n_judged": args.n_judged, "n_other": args.n_other,
            "runtime_s": round(time.monotonic() - started, 1)}
    for dep, rows in result["spots"].items():
        result["summary"][dep]["mixing"] = mixing(rows)
    reference = args.deployment[0]
    result["comparisons"] = {}
    for dep in args.deployment[1:]:
        diff = paired_difference(result["spots"][reference], result["spots"][dep], seed=args.seed)
        result["comparisons"][dep] = {"vs": reference, "diff": diff,
                                      "verdict": verdict(result["summary"][reference], result["summary"][dep], diff)}
    if args.n_cleanup:
        result["cleanup"] = run_cleanup_check(args.joined, args.deployment, client, chat_tools(), n=args.n_cleanup,
                                              seed=args.seed, concurrency=args.concurrency)
        ref_rate = result["cleanup"][args.deployment[0]]["rate"] or 0
        for dep, c in result["comparisons"].items():
            rate = result["cleanup"][dep]["rate"] or 0
            if rate < max(0.9, ref_rate - 0.05):
                c["verdict"]["pass"] = False
                c["verdict"]["reasons"].append(f"cleanup turns broken: {rate:.0%} valid update_scratchpad")
    out.write_text(json.dumps(meta | result, indent=1))
    print(format_table(result["summary"]))
    for dep, c in (result.get("cleanup") or {}).items():
        print(f"{dep}: cleanup turns valid {c['ok']}/{c['n']}")
    for dep, s in result["summary"].items():
        lo, hi = bootstrap_ci(list(_spot_agreement(result["spots"][dep]).values()), seed=args.seed)
        m = s["mixing"]
        mixed = _pct(m["mixed_rate"])
        print(f"{dep}: judged agree 95% CI {lo:.0%}..{hi:.0%}; other spots mixed {mixed} of {m['spots']}, "
              f"entropy {m['entropy_bits']} bits")
    for dep, c in result["comparisons"].items():
        d, v = c["diff"], c["verdict"]
        if d["diff"] is not None:
            print(f"{dep} vs {reference}: agreement {d['diff']:+.0%} (95% CI {d['lo']:+.0%}..{d['hi']:+.0%}, {d['n_spots']} spots)")
        print(f"{dep}: {'PASS' if v['pass'] else 'FAIL'}" + ("" if v["pass"] else " - " + "; ".join(v["reasons"])))
    print(f"runtime {meta['runtime_s']}s; wrote {out}")


if __name__ == "__main__":
    main()
