"""Opponent statistics from the arena database, per submission (sub_key) or per team (entry_id).

Every rate carries n (opportunities), k (successes) and a 95% Wilson interval (via profile_logs.rate).
Run: uv run python -m poker.arena.stats  -> data/arena/report.json plus a table on stdout.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

from poker.arena import db
from poker.profile_logs import rate as _wilson

REPORT_PATH = db.DATA_DIR / "report.json"
RATE_STATS = ["vpip", "pfr", "3bet", "fold_to_3bet", "open_shove", "cbet", "fold_to_cbet",
              "fold_to_bet_all", "fold_to_bet_lt50", "fold_to_bet_50_100", "fold_to_bet_gt100",
              "fold_to_bet_allin", "call_vs_allin", "postflop_agg_share", "wtsd",
              "fallback_rate", "token_cap_rate", "bust_rate"]


def rate(k: int, n: int) -> dict:
    r = _wilson(k, n)
    return dict(n=n, k=k, rate=r["rate"], ci95=r["ci95"])


def _bucket(a: dict) -> str | None:
    if not a["to_call"] or a["street"] == "preflop":
        return None
    if a["facing_allin"]:
        return "allin"
    f = a["facing_frac"]
    if f is None:
        return None
    return "lt50" if f < 0.5 else "50_100" if f <= 1 else "gt100"


def tally_hand(acts: list[dict], board: str, showdown: bool) -> list[tuple[str, int, bool]]:
    """Pure: (stat, seat, success) opportunity events for one hand. acts are action rows in order."""
    ev: list[tuple[str, int, bool]] = []
    acts = [a for a in acts if not (a["sat_out"] or a["afk"])]
    seen, vpip, pfr = set(), {}, {}
    first_3bet_seen, openers, f3_seen = set(), set(), set()
    cbet_seat = None
    cbet_opp_seen = set()
    folded = set()
    saw_flop_candidates = set()
    for a in acts:
        s, st, act = a["seat"], a["street"], a["action"]
        aggressive = act in ("bet", "raise")
        if st == "preflop":
            seen.add(s)
            vpip[s] = vpip.get(s, False) or act in ("call", "bet", "raise")
            pfr[s] = pfr.get(s, False) or aggressive
            rb = a["raises_before"]
            if rb == 0:
                ev.append(("open_shove", s, aggressive and a["stack_after"] == 0))
                if aggressive:
                    openers.add(s)
            if rb == 1 and a["to_call"] and s not in first_3bet_seen:
                first_3bet_seen.add(s)
                ev.append(("3bet", s, aggressive))
            if rb >= 2 and s in openers and s not in f3_seen:
                f3_seen.add(s)
                ev.append(("fold_to_3bet", s, act == "fold"))
        else:
            if s in folded:
                continue
            if st == "flop":
                if a["preflop_aggressor"] and a["raises_before"] == 0 and not a["to_call"] and s not in cbet_opp_seen:
                    cbet_opp_seen.add(s)
                    ev.append(("cbet", s, aggressive))
                    if aggressive:
                        cbet_seat = s
                elif a["to_call"] and a["raises_before"] == 1 and cbet_seat is not None and s != cbet_seat:
                    ev.append(("fold_to_cbet", s, act == "fold"))
            if a["to_call"]:
                b = _bucket(a)
                if b:
                    ev.append(("fold_to_bet_all", s, act == "fold"))
                    ev.append((f"fold_to_bet_{b}", s, act == "fold"))
            if aggressive or act == "call":
                ev.append(("postflop_agg_share", s, aggressive))
        if a["to_call"] and a["facing_allin"]:
            ev.append(("call_vs_allin", s, act == "call"))
        if act == "fold":
            folded.add(s)
    for s in seen:
        ev.append(("vpip", s, vpip[s]))
        ev.append(("pfr", s, pfr[s]))
    if board:
        pre_folded = {a["seat"] for a in acts if a["street"] == "preflop" and a["action"] == "fold"}
        for s in seen - pre_folded:
            ev.append(("wtsd", s, bool(showdown) and s not in folded))
    return ev


def _blank():
    return dict(counts=defaultdict(lambda: [0, 0]), hands=set(), matches=set(), out_tokens=[],
                n_dec=0, calls=0, aggs=0, places=[], busts=[0, 0], net_bb=0.0, ev_bb=0.0, hand_n=0,
                names=set(), models=set(), kinds=set(), submissions=set())


def compute(con: sqlite3.Connection, by: str = "sub_key") -> dict:
    assert by in ("sub_key", "entry_id")
    acc: dict[str, dict] = defaultdict(_blank)
    hands = {(r["match_id"], r["hand_number"]): (r["board"], bool(r["showdown"]))
             for r in con.execute("SELECT match_id, hand_number, board, showdown FROM hands")}
    seat_key = {}
    for r in con.execute("SELECT match_id, seat, sub_key, entry_id FROM seats"):
        seat_key[(r["match_id"], r["seat"])] = r[by]
    cur = con.execute("SELECT * FROM actions ORDER BY match_id, hand_number, idx")
    group, gid = [], None

    def flush():
        if not group:
            return
        board, sd = hands[gid]
        for stat, seat, ok in tally_hand(group, board, sd):
            key = group_key(gid[0], seat)
            if key is not None:
                c = acc[key]["counts"][stat]
                c[0] += int(ok)
                c[1] += 1
        for a in group:
            key = group_key(gid[0], a["seat"])
            if key is None:
                continue
            d = acc[key]
            d["hands"].add((gid[0], gid[1]))
            d["matches"].add(gid[0])
            if not a["scripted"]:
                d["n_dec"] += 1
                d["counts"]["fallback_rate"][0] += a["fallback"]
                d["counts"]["fallback_rate"][1] += 1
                d["counts"]["token_cap_rate"][0] += a["token_cap"]
                d["counts"]["token_cap_rate"][1] += 1
                if a["tok_out"] is not None:
                    d["out_tokens"].append(a["tok_out"])
            if a["street"] != "preflop" and not (a["sat_out"] or a["afk"]):
                d["calls"] += a["action"] == "call"
                d["aggs"] += a["action"] in ("bet", "raise")

    def group_key(match_id, seat):
        k = seat_key.get((match_id, seat))
        return k

    for a in cur:
        a = dict(a)
        g = (a["match_id"], a["hand_number"])
        if g != gid:
            flush()
            group, gid = [], g
        group.append(a)
    flush()

    meta = {}
    for r in con.execute("SELECT * FROM entries"):
        meta[r["sub_key"]] = dict(r)
    for r in con.execute("SELECT * FROM seats"):
        key = r[by]
        if key is None:
            continue
        d = acc[key]
        d["matches"].add(r["match_id"])
        if r["place"] is not None:
            d["places"].append(r["place"])
        d["busts"][1] += 1
        d["busts"][0] += r["eliminated_on_hand"] is not None
        d["net_bb"] += r["net_bb"] or 0
        d["ev_bb"] += r["ev_net_bb"] or 0
        d["hand_n"] += r["hands"] or 0
        m = meta.get(r["sub_key"], {})
        d["names"].add(m.get("name")); d["models"].add(m.get("model")); d["kinds"].add(m.get("kind"))
        d["submissions"].add(r["sub_key"])

    out = {}
    for key, d in acc.items():
        c = d["counts"]
        s = dict(key=key, name=", ".join(sorted(x for x in d["names"] if x)),
                 model=", ".join(sorted(x for x in d["models"] if x)),
                 kind=", ".join(sorted(x for x in d["kinds"] if x)),
                 submissions=sorted(d["submissions"]) if by == "entry_id" else None,
                 n_matches=len(d["matches"]), n_hands=len(d["hands"]), n_actions=d["n_dec"] + 0)
        for name in RATE_STATS:
            if name == "bust_rate":
                s[name] = rate(*d["busts"])
            else:
                k, n = c[name] if name in c else (0, 0)
                s[name] = rate(k, n)
        s["aggression_factor"] = dict(value=(d["aggs"] / d["calls"]) if d["calls"] else None,
                                      bets_raises=d["aggs"], calls=d["calls"])
        s["median_out_tokens"] = statistics.median(d["out_tokens"]) if d["out_tokens"] else None
        s["mean_place"] = statistics.fmean(d["places"]) if d["places"] else None
        s["net_bb_per_100"] = dict(value=(d["net_bb"] / d["hand_n"] * 100) if d["hand_n"] else None,
                                   ev_value=(d["ev_bb"] / d["hand_n"] * 100) if d["hand_n"] else None,
                                   n_hands=d["hand_n"])
        s["probe_flags"] = probe_flags(s)
        out[key] = s
    return out


def probe_flags(s: dict, min_n: int = 50) -> list[str]:
    """Flag extreme one-note styles by Wilson interval. Never used to drop a submission."""
    flags = []

    def lo(name):
        r = s.get(name)
        return r["ci95"][0] if r and r.get("ci95") and r["n"] >= min_n else None

    def hi(name):
        r = s.get(name)
        return r["ci95"][1] if r and r.get("ci95") and r["n"] >= min_n else None

    if (lo("vpip") or 0) > 0.85:
        flags.append("maniac_vpip")
    if (lo("pfr") or 0) > 0.6:
        flags.append("raise_every_hand")
    if hi("vpip") is not None and hi("vpip") < 0.12:
        flags.append("nit_vpip")
    if hi("vpip") is not None and hi("pfr") is not None and lo("vpip") and lo("vpip") > 0.6 and hi("pfr") < 0.08:
        flags.append("calling_station")
    if (lo("open_shove") or 0) > 0.5:
        flags.append("open_shover")
    if (lo("fold_to_bet_all") or 0) > 0.9:
        flags.append("always_folds_to_bets")
    if hi("fold_to_bet_all") is not None and hi("fold_to_bet_all") < 0.1:
        flags.append("never_folds_to_bets")
    if (lo("call_vs_allin") or 0) > 0.9:
        flags.append("always_calls_allin")
    if hi("call_vs_allin") is not None and hi("call_vs_allin") < 0.1:
        flags.append("never_calls_allin")
    return flags


# ---- hand classes -----------------------------------------------------------------------

def hand_class(row) -> dict | None:
    """Hand class of the actor of an action row (needs card1/card2). kind: pair/suited/offsuit."""
    c1, c2 = row["card1"], row["card2"]
    code = db.hand_code(c1, c2)
    if code is None:
        return None
    r1, r2 = db.RANKS.index(code[0]), db.RANKS.index(code[1])
    hi_r, lo_r = max(r1, r2), min(r1, r2)
    kind = "pair" if code[0] == code[1] else "suited" if code.endswith("s") else "offsuit"
    tags = {kind}
    if lo_r >= db.RANKS.index("T"):
        tags.add("broadway")
    if kind == "pair":
        tags.add("big_pair" if hi_r >= db.RANKS.index("T") else "small_pair" if hi_r <= db.RANKS.index("7") else "mid_pair")
    else:
        if hi_r - lo_r == 1:
            tags.add("connector")
            if kind == "suited":
                tags.add("suited_connector")
        elif hi_r - lo_r == 2:
            tags.add("one_gapper")
        if hi_r == 12:
            tags.add("ace_x")
            if kind == "suited":
                tags.add("suited_ace")
        if hi_r >= db.RANKS.index("T") and lo_r < db.RANKS.index("T"):
            tags.add("one_broadway")
        if hi_r < db.RANKS.index("T") and not (hi_r - lo_r <= 2):
            tags.add("trash")
    return dict(code=code, kind=kind, tags=sorted(tags))


def range_for(con, key: str, action: str | None = None, street: str | None = None,
              position: str | None = None, tag: str | None = None, by: str = "sub_key") -> dict:
    """Counts of hand codes shown by a submission (or team) for the given action/street/position/tag."""
    q, args = f"SELECT card1, card2, hand_code FROM actions WHERE {by}=? AND card1 IS NOT NULL", [key]
    for col, v in (("action", action), ("street", street), ("position", position)):
        if v:
            q += f" AND {col}=?"
            args.append(v)
    out: dict[str, int] = defaultdict(int)
    for r in con.execute(q, args):
        if tag and tag not in hand_class(r)["tags"]:
            continue
        out[r["hand_code"]] += 1
    return dict(out)


# ---- report ------------------------------------------------------------------------------

def build_report(con) -> dict:
    return dict(by_submission=compute(con, "sub_key"), by_team=compute(con, "entry_id"),
                n_matches=con.execute("SELECT count(*) FROM matches").fetchone()[0])


def _pct(r) -> str:
    return "  -  " if r["rate"] is None else f"{100 * r['rate']:3.0f}%"


def table(stats_by_key: dict) -> str:
    cols = [("vpip", "VPIP"), ("pfr", "PFR"), ("3bet", "3B"), ("fold_to_3bet", "F3B"), ("open_shove", "SHOV"),
            ("cbet", "CB"), ("fold_to_cbet", "FCB"), ("fold_to_bet_all", "FtB"), ("call_vs_allin", "CvAI"),
            ("wtsd", "WTSD"), ("fallback_rate", "FB")]
    head = f"{'name':24}{'mtch':>5}{'hnds':>6}" + "".join(f"{h:>6}" for _, h in cols) + "  AF  tok  place  bb/100  flags"
    lines = [head]
    for s in sorted(stats_by_key.values(), key=lambda x: -x["n_hands"]):
        af = s["aggression_factor"]["value"]
        bb = s["net_bb_per_100"]["value"]
        lines.append(
            f"{(s['name'] or s['key'])[:23]:24}{s['n_matches']:>5}{s['n_hands']:>6}"
            + "".join(f"{_pct(s[c]):>6}" for c, _ in cols)
            + f" {'-' if af is None else f'{af:.1f}':>4}{'-' if s['median_out_tokens'] is None else int(s['median_out_tokens']):>5}"
            + f"{'-' if s['mean_place'] is None else f'{s['mean_place']:.1f}':>6}{'-' if bb is None else f'{bb:.0f}':>8}  "
            + ",".join(s["probe_flags"]))
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(db.DB_PATH))
    ap.add_argument("--out", default=str(REPORT_PATH))
    ap.add_argument("--by", choices=["submission", "team", "both"], default="both")
    args = ap.parse_args(argv)
    con = db.connect(args.db)
    report = build_report(con)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    if args.by in ("team", "both"):
        print("== per team (n small? see report.json for Wilson intervals)")
        print(table(report["by_team"]))
    if args.by in ("submission", "both"):
        print("\n== per submission")
        print(table(report["by_submission"]))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
