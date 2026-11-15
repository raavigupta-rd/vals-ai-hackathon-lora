"""Per-submission scoreboard with match-level uncertainty (games are the unit, never single actions).

Run: uv run python -m poker.arena.scoreboard [--ours] [--since ISO] [--min-matches N]
Writes data/arena/scoreboard.json. House bots (entries.kind != 'team') are excluded from the field.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from poker.arena import db
from poker.profile_logs import rate as _wilson

OUT_PATH = db.DATA_DIR / "scoreboard.json"
OUR_TEAM = "LoRA"
N_BOOT = 2000


def bootstrap_dist(xs, n_boot: int = N_BOOT, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    n = len(xs)
    return sorted(sum(rng.choices(xs, k=n)) / n for _ in range(n_boot))


def _pct(dist, lo=0.025, hi=0.975):
    return dist[int(lo * (len(dist) - 1))], dist[int(round(hi * (len(dist) - 1)))]


def bootstrap_mean_ci(xs, n_boot: int = N_BOOT, seed: int = 0):
    """95% percentile bootstrap CI of the mean, resampling matches. None if empty."""
    if not xs:
        return None
    lo, hi = _pct(bootstrap_dist(list(xs), n_boot, seed))
    return (float(lo), float(hi))


def wilson(k: int, n: int) -> dict:
    r = _wilson(k, n)
    return dict(k=k, n=n, rate=r["rate"], ci95=r["ci95"])


def _seat_rows(con, since):
    q = ("SELECT s.*, e.name, e.submission_name, e.deployment_id, e.kind AS ekind FROM seats s "
         "JOIN matches m ON m.match_id=s.match_id JOIN entries e ON e.sub_key=s.sub_key "
         "WHERE m.status='complete' AND s.place IS NOT NULL")
    args = []
    if since:
        q += " AND m.started_at >= ?"
        args.append(since)
    return [dict(r) for r in con.execute(q + " ORDER BY s.match_id, s.seat", args)]


def _action_diag(con, keys_by_match_seat):
    """Per sub_key: all-in-call counts, shoves, fallback counts, out tokens (our-spot diagnostics)."""
    d = defaultdict(lambda: dict(faced=0, called=0, hands=Counter(), shoves=0, fb=0, dec=0, tok=[]))
    for a in con.execute("SELECT * FROM actions WHERE scripted=0 AND sat_out=0"):
        if (a["match_id"], a["seat"]) not in keys_by_match_seat:
            continue
        x = d[a["sub_key"]]
        x["dec"] += 1
        # An AFK action is a failed reply the arena played as check or fold: it is a fallback, and it is not a
        # real choice, so it stays out of the action diagnostics below.
        x["fb"] += int(bool(a["afk"] or a["fallback"] or a["coerced"] or a["format_error"] or a["token_cap"]))
        if a["afk"]:
            continue
        if a["tok_out"] is not None:
            x["tok"].append(a["tok_out"])
        if a["street"] == "preflop":
            if a["action"] in ("bet", "raise") and a["stack_after"] == 0:
                x["shoves"] += 1
            if a["to_call"] and a["facing_allin"]:
                x["faced"] += 1
                if a["action"] == "call":
                    x["called"] += 1
                    if a["hand_code"]:
                        x["hands"][a["hand_code"]] += 1
    return d


def _summ(rows) -> dict:
    places = [r["place"] for r in rows]
    n = len(rows)
    return dict(
        matches=n, mean_place=statistics.fmean(places) if n else None,
        place_ci95=bootstrap_mean_ci(places),
        bust_rate=wilson(sum(r["eliminated_on_hand"] is not None for r in rows), n),
        top3_rate=wilson(sum(p <= 3 for p in places), n),
        mean_net_bb=statistics.fmean([r["net_bb"] or 0 for r in rows]) if n else None,
        mean_ev_bb=statistics.fmean([r["ev_net_bb"] or 0 for r in rows]) if n else None)


def build(con, min_matches: int = 3, since: str | None = None, ours: bool = False) -> dict:
    seats = _seat_rows(con, since)
    team_seats = [r for r in seats if r["ekind"] == "team"]
    by_key = defaultdict(list)
    for r in team_seats:
        by_key[r["sub_key"]].append(r)
    keep = {k: v for k, v in by_key.items() if len(v) >= min_matches}
    if ours:
        keep = {k: v for k, v in keep.items() if v[0]["name"] == OUR_TEAM}
    ms = {(r["match_id"], r["seat"]) for v in keep.values() for r in v}
    diag = _action_diag(con, ms)
    subs = []
    for k, rows in keep.items():
        s = _summ(rows)
        x = diag.get(k) or dict(faced=0, called=0, hands=Counter(), shoves=0, fb=0, dec=0, tok=[])
        s.update(sub_key=k, name=rows[0]["name"], submission_name=rows[0]["submission_name"],
                 deployment_id=rows[0]["deployment_id"], kind="team",
                 allin_call=wilson(x["called"], x["faced"]), allin_call_hands=dict(x["hands"]),
                 preflop_shoves=x["shoves"], fallback_rate=wilson(x["fb"], x["dec"]),
                 median_out_tokens=statistics.median(x["tok"]) if x["tok"] else None)
        subs.append(s)
    subs.sort(key=lambda s: s["mean_place"])
    fs = _summ(team_seats)
    fs["seat_matches"] = len(team_seats)
    fs["n_submissions"] = len(by_key)
    fs["note"] = "all kind=team seats; unit is seat-in-match, matches share tables so CI is approximate"
    paired = []
    mine = [s for s in subs if s["name"] == OUR_TEAM]
    for a, b in itertools.combinations(mine, 2):
        pa = [r["place"] for r in keep[a["sub_key"]]]
        pb = [r["place"] for r in keep[b["sub_key"]]]
        rng = random.Random(3)
        diffs = sorted(sum(rng.choices(pa, k=len(pa))) / len(pa) - sum(rng.choices(pb, k=len(pb))) / len(pb)
                       for _ in range(N_BOOT))
        lo, hi = _pct(diffs)
        paired.append(dict(a=_label(a), b=_label(b), diff_mean_place=statistics.fmean(pb) - statistics.fmean(pa),
                           ci95=None, method="independent-samples bootstrap over matches (matches not paired)",
                           n_a=len(pa), n_b=len(pb)))
        # diff defined as mean(b) - mean(a); the bootstrap above is a - b, so negate
        paired[-1]["ci95"] = (-hi, -lo)
    return dict(field=fs, submissions=subs, paired=paired,
                params=dict(min_matches=min_matches, since=since, ours=ours, n_boot=N_BOOT))


def _label(s) -> str:
    return f"{s['name']}/{s['submission_name']}/{(s['deployment_id'] or '')[:8]}"


def _ci(c) -> str:
    return "-" if not c else f"[{c[0]:.2f},{c[1]:.2f}]"


def _rate(r) -> str:
    return "-" if r["rate"] is None else f"{100 * r['rate']:.0f}%({r['k']}/{r['n']})"


def render(out: dict) -> str:
    hdr = (f"{'team':14}{'submission':16}{'dep':9}{'n':>4}{'place':>6} {'95% CI':13}{'bust':>11}{'top3':>11}"
           f"{'netbb':>7}{'evbb':>7}{'AIcall':>12}{'shoves':>7}{'fb':>14}{'tok':>5}")
    lines = [hdr]
    for s in out["submissions"]:
        lines.append(f"{(s['name'] or '')[:13]:14}{(s['submission_name'] or '')[:15]:16}{(s['deployment_id'] or '')[:8]:9}"
                     f"{s['matches']:>4}{s['mean_place']:>6.2f} {_ci(s['place_ci95']):13}{_rate(s['bust_rate']):>11}"
                     f"{_rate(s['top3_rate']):>11}{s['mean_net_bb']:>7.1f}{s['mean_ev_bb']:>7.1f}"
                     f"{_rate(s['allin_call']):>12}{s['preflop_shoves']:>7}{_rate(s['fallback_rate']):>14}"
                     f"{'-' if s['median_out_tokens'] is None else int(s['median_out_tokens']):>5}")
        if s["allin_call_hands"]:
            lines.append("    AI-called: " + " ".join(f"{h}x{c}" for h, c in sorted(s["allin_call_hands"].items())))
    f = out["field"]
    lines.append(f"FIELD (LLM teams, {f['n_submissions']} subs, {f['seat_matches']} seat-matches): place {f['mean_place']:.2f} "
                 f"{_ci(f['place_ci95'])} bust {_rate(f['bust_rate'])} top3 {_rate(f['top3_rate'])} "
                 f"netbb {f['mean_net_bb']:.1f} evbb {f['mean_ev_bb']:.1f}")
    for p in out["paired"]:
        lines.append(f"PAIRED {p['b']} minus {p['a']}: diff mean place {p['diff_mean_place']:+.2f} {_ci(p['ci95'])} "
                     f"({p['method']}; n={p['n_b']} vs {p['n_a']})")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(db.DB_PATH))
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--ours", action="store_true", help=f"only team {OUR_TEAM!r}, plus paired comparisons")
    ap.add_argument("--since", help="ISO timestamp; keep matches with started_at >= this")
    ap.add_argument("--min-matches", type=int, default=3)
    a = ap.parse_args(argv)
    out = build(db.connect(a.db), a.min_matches, a.since, a.ours)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(render(out))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
