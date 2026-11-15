"""Pre-deploy gate for scratchpad writing: replay real cleanup turns and compare the written pad with the teacher pad.

    export AC2_API_KEY=$(cat ~/.ac2_api_key)
    uv run python -m poker.pad.check --deployment vals-hackathon/c7x-s18 --n 60

Samples `--n` arena cleanup prompts (hand 2 or later) whose previous pad is replaced by the teacher v2 pad chained
from hand 1 of the same (match, seat), plus `--n-first` hand-1 cleanups with an empty pad. Each prompt is sent to the
deployment. The script prints the valid tool-call rate, the strict parse rate, the mean count errors and 3 raw
failures. `pad_check` is the per-pad scorer; a count error is one per-seat count field (or blinds field) that differs
from the teacher pad for the same history. The prompts are our own arena games, so the check does not depend on
whether the opponents were team or house bots.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from poker.gate import HttpClient, chat_messages, chat_tools, cleanup_ok, load_api_key
from poker.pad.rewrite import chain_index, rewrite_messages, teacher_pad_at
from poker.pad.writer import V2_HEAD, V2_OPP, parse_cleanup, parse_pad, teacher_pad

ROOT = Path(__file__).resolve().parents[2]
COUNT_FIELDS = ("va", "vb", "opens", "open_opp", "three_bet", "three_bet_opp", "jams", "ftr_f", "ftr_g")
KNOWN_LINE = re.compile(r"blinds: fold-to-steal \d+/\d+|out: .*|> .*")


def pad_check(prev_pad: str, cleanup_messages: list[dict], written_pad: str) -> dict:
    """Score a written pad against the teacher pad for the same history.

    parses: `parse_pad` reads it. header_ok: the first line is the exact v2 header shape with the teacher's hand,
    total, seat and stack. seats_ok: same opponent seat set. count_errors: wrong count fields over the teacher's seats
    (a missing seat costs all its fields; the blinds line counts too); None when the pad does not parse.
    exploit_lines_ok: the "> " lines equal the teacher's. extra_lines: lines that are not header, seat, blinds, out or
    exploit lines, plus repeated seat lines."""
    teacher_text, teacher = teacher_pad(cleanup_messages, prev_pad or "", version=2)
    written_pad = written_pad or ""
    lines = written_pad.split("\n")
    got = parse_pad(written_pad)
    head = V2_HEAD.fullmatch(lines[0]) if lines else None
    header_ok = bool(head and got and (got.hand, got.total, got.hero, got.stack)
                     == (teacher.hand, teacher.total, teacher.hero, teacher.stack))
    if got is None or got.version != 2:
        return {"parses": False, "header_ok": False, "seats_ok": False, "count_errors": None,
                "exploit_lines_ok": False, "extra_lines": None}
    errors = 0
    for seat, want in teacher.opps.items():
        have = got.opps.get(seat)
        errors += len(COUNT_FIELDS) if have is None else sum(getattr(have, f) != getattr(want, f) for f in COUNT_FIELDS)
    errors += (got.fts_f != teacher.fts_f) + (got.fts_g != teacher.fts_g)
    seen_seats: set[str] = set()
    extra = 0
    for ln in lines[1:]:
        m = V2_OPP.fullmatch(ln)
        if m:
            extra += m.group(1) in seen_seats
            seen_seats.add(m.group(1))
        elif not KNOWN_LINE.fullmatch(ln):
            extra += 1
    exploits = [ln for ln in lines if ln.startswith("> ")]
    return {"parses": True, "header_ok": header_ok, "seats_ok": set(got.opps) == set(teacher.opps),
            "count_errors": errors, "exploit_lines_ok": exploits == [ln for ln in teacher_text.split("\n") if ln.startswith("> ")],
            "extra_lines": extra}


def build_cases(rows, *, n: int, n_first: int = 20, seed: int = 0) -> list[dict]:
    """Replay cases from joined arena cleanup rows: n for hand 2+ (teacher v2 previous pad), n_first for hand 1 (empty pad)."""
    rows = [r for r in rows if r.get("kind") == "cleanup" and r.get("source") == "arena"]
    index = chain_index(rows)
    later, first, seen = [], [], set()
    for r in sorted(rows, key=lambda r: str(r.get("span_id"))):
        match = r.get("match") or {}
        try:
            s = parse_cleanup(r["messages"])
        except (ValueError, KeyError, AttributeError):
            continue
        key = (match.get("match_id"), s.hero, s.hand)
        if key[0] is None or key in seen:
            continue
        seen.add(key)
        prev = teacher_pad_at(index, *key, 2)
        if prev is None:
            continue
        case = {"span_id": r.get("span_id"), "hand": s.hand, "prev_pad": prev, "messages": rewrite_messages(r["messages"], prev)}
        (first if s.hand == 1 else later).append(case)
    rng = random.Random(seed)
    return rng.sample(later, min(n, len(later))) + rng.sample(first, min(n_first, len(first)))


def _contents(response: dict) -> str | None:
    try:
        return json.loads(response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])["contents"]
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def run(cases: list[dict], client, deployment: str, *, tools: list[dict], concurrency: int = 8) -> dict:
    def one(case):
        response = client.complete(deployment, chat_messages(case["messages"]), tools)
        valid = cleanup_ok(response)
        text = _contents(response) if valid else None
        score = pad_check(case["prev_pad"], case["messages"], text) if text is not None else None
        raw = text if text is not None else json.dumps(response)[:800]
        return {"hand": case["hand"], "valid": valid, "score": score, "raw": raw}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one, cases))
    n = len(results)
    scored = [r["score"] for r in results if r["score"]]
    parsed = [s for s in scored if s["parses"]]
    strict = [s for s in parsed if s["extra_lines"] == 0]
    errors = [s["count_errors"] for s in parsed]

    def rate(k):
        return k / n if n else None

    by_hand = {}
    for label, pick in (("hand1", lambda r: r["hand"] == 1), ("later", lambda r: r["hand"] != 1)):
        part = [r for r in results if pick(r)]
        ok = [r["score"] for r in part if r["score"] and r["score"]["parses"]]
        by_hand[label] = {"n": len(part), "parse_rate": len(ok) / len(part) if part else None,
                          "mean_count_errors": sum(s["count_errors"] for s in ok) / len(ok) if ok else None}
    bad = [r for r in results if not r["valid"] or not r["score"]["parses"]]
    bad += sorted((r for r in results if r["score"] and r["score"]["parses"] and r not in bad),
                  key=lambda r: -(r["score"]["count_errors"] or 0))
    return {"deployment": deployment, "n": n, "valid_rate": rate(sum(r["valid"] for r in results)),
            "parse_rate": rate(len(parsed)), "strict_parse_rate": rate(len(strict)),
            "header_ok_rate": rate(sum(s["header_ok"] for s in scored)), "seats_ok_rate": rate(sum(s["seats_ok"] for s in scored)),
            "exploit_ok_rate": rate(sum(s["exploit_lines_ok"] for s in scored)),
            "mean_count_errors": sum(errors) / len(errors) if errors else None, "n_parsed": len(parsed),
            "n_hand1": sum(r["hand"] == 1 for r in results), "by_hand": by_hand,
            "failures": [{"hand": r["hand"], "valid": r["valid"], "score": r["score"], "raw": r["raw"][:800]} for r in bad[:3]]}


def pick_chains(rows, *, games: int, hands: int, seed: int = 0) -> list[dict]:
    """Real (match, seat) arena chains with a cleanup prompt for every hand 1..hands: {"key", "msgs": {hand: messages},
    "teacher_prev": {hand: teacher v2 pad shown at the start of that hand}}."""
    index = chain_index([r for r in rows if r.get("kind") == "cleanup" and r.get("source") == "arena"])
    found = []
    for key in sorted((k for k in index if isinstance(k, tuple)), key=str):
        if all(h in index[key] for h in range(1, hands + 1)) and teacher_pad_at(index, *key, hands, 2) is not None:
            found.append({"key": key, "msgs": {h: index[key][h] for h in range(1, hands + 1)},
                          "teacher_prev": {h: teacher_pad_at(index, *key, h, 2) for h in range(1, hands + 1)}})
    return random.Random(seed).sample(found, min(games, len(found)))


def run_chain(chain: dict, complete) -> list[dict]:
    """Self-chained replay of one game. `complete(messages) -> pad text or None` (None: no valid update_scratchpad call).
    Hand 1 starts with an empty pad; each next hand shows the pad the model wrote (the previous pad if the call failed).
    Each written pad is scored against the teacher pad chained from hand 1 for the same history."""
    pad, out = "", []
    for hand in sorted(chain["msgs"]):
        orig = chain["msgs"][hand]
        written = complete(rewrite_messages(orig, pad))
        teacher_prev = chain["teacher_prev"][hand]
        score = pad_check(teacher_prev, rewrite_messages(orig, teacher_prev), written) if written is not None else None
        out.append({"hand": hand, "valid": written is not None, "pad": written, "score": score})
        if written is not None:
            pad = written
    return out


def summarize_chains(chains: list[list[dict]]) -> dict:
    steps = [s for c in chains for s in c]

    def parsed(s):
        return bool(s["score"] and s["score"]["parses"])

    def rate(pick):
        part = [s for s in steps if pick(s["hand"])]
        return sum(parsed(s) for s in part) / len(part) if part else None

    late = [s["score"]["count_errors"] for s in steps if s["hand"] >= 5 and parsed(s)]
    firsts = []
    for c in chains:
        bad = [s["hand"] for s in c if not parsed(s)]
        firsts.append(bad[0] if bad else None)
    return {"games": len(chains), "steps": len(steps),
            "valid_rate": sum(s["valid"] for s in steps) / len(steps) if steps else None,
            "parse_rate_hand1": rate(lambda h: h == 1), "parse_rate_hands2_4": rate(lambda h: 2 <= h <= 4),
            "parse_rate_hands5plus": rate(lambda h: h >= 5),
            "mean_count_errors_hands5plus": sum(late) / len(late) if late else None, "n_parsed_hands5plus": len(late),
            "games_all_parse": sum(f is None for f in firsts) / len(firsts) if firsts else None,
            "first_break_hand": firsts}


def self_chain(chains: list[dict], complete, *, concurrency: int = 8) -> dict:
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda c: run_chain(c, complete), chains))
    summary = summarize_chains(results)
    summary["transcripts"] = [[{"hand": s["hand"], "pad": s["pad"], "parses": bool(s["score"] and s["score"]["parses"]),
                                "count_errors": s["score"]["count_errors"] if s["score"] else None} for s in c] for c in results]
    return summary


def client_complete(client, deployment: str, tools: list[dict]):
    def complete(messages):
        response = client.complete(deployment, chat_messages(messages), tools)
        return _contents(response) if cleanup_ok(response) else None
    return complete


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deployment", action="append", required=True, help="deployment name; repeat to compare")
    ap.add_argument("--n", type=int, default=60, help="cleanups from hand 2 or later (teacher v2 previous pad)")
    ap.add_argument("--n-first", type=int, default=20, help="hand-1 cleanups with an empty pad")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--joined", type=Path, default=ROOT / "data/arena/joined.jsonl")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--self-chain", action="store_true", help="replay whole games with the model's own previous pad")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--hands", type=int, default=12)
    ap.add_argument("--json", action="store_true", help="print the raw result dicts as JSON")
    a = ap.parse_args(argv)
    with a.joined.open(encoding="utf-8") as stream:
        rows = [r for r in map(json.loads, stream) if r.get("kind") == "cleanup" and r.get("source") == "arena"]
    if a.self_chain:
        chains = pick_chains(rows, games=a.games, hands=a.hands, seed=a.seed)
        print(f"{len(chains)} chains of {a.hands} hands", file=sys.stderr)
        client, tools = HttpClient(load_api_key()), chat_tools()
        for dep in a.deployment:
            res = self_chain(chains, client_complete(client, dep, tools), concurrency=a.concurrency)
            if a.json:
                print(json.dumps({"deployment": dep, **res}, indent=1))
                continue
            print(f"== {dep} self-chain ({res['games']} games, {res['steps']} pads)")
            for key in ("valid_rate", "parse_rate_hand1", "parse_rate_hands2_4", "parse_rate_hands5plus",
                        "mean_count_errors_hands5plus", "games_all_parse"):
                print(f"{key}: {_fmt(res[key])}")
            print(f"first_break_hand per game: {res['first_break_hand']}")
            if res["transcripts"]:
                print("--- game 1 transcript")
                for s in res["transcripts"][0]:
                    print(f"[hand {s['hand']}] parses={s['parses']} count_errors={s['count_errors']}\n{s['pad']}")
        return 0
    cases = build_cases(rows, n=a.n, n_first=a.n_first, seed=a.seed)
    print(f"{len(cases)} cases ({sum(c['hand'] == 1 for c in cases)} hand-1)", file=sys.stderr)
    client, tools = HttpClient(load_api_key()), chat_tools()
    for dep in a.deployment:
        res = run(cases, client, dep, tools=tools, concurrency=a.concurrency)
        if a.json:
            print(json.dumps(res, indent=1))
            continue
        print(f"== {dep} (n={res['n']}, {res['n_hand1']} hand-1)")
        for key in ("valid_rate", "parse_rate", "strict_parse_rate", "header_ok_rate", "seats_ok_rate", "exploit_ok_rate"):
            print(f"{key}: {_fmt(res[key])}")
        print(f"mean_count_errors (parsed pads, n={res['n_parsed']}): {_fmt(res['mean_count_errors'])}")
        for label, part in res["by_hand"].items():
            print(f"  {label}: n={part['n']} parse_rate={_fmt(part['parse_rate'])} mean_count_errors={_fmt(part['mean_count_errors'])}")
        for i, f in enumerate(res["failures"], 1):
            print(f"--- failure {i} (hand {f['hand']}, valid_call={f['valid']}, score={f['score']})\n{f['raw']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
