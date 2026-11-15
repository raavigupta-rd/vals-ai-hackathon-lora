"""Non-circular preflop score from real hole cards, independent of the teacher.

For preflop decisions facing a bet (to_call > 0) the arena DB holds every seat's real hole cards. We price a
replayed action in chips relative to folding:

    EV(fold) = 0
    EV(call) = equity * (pot + to_call) - to_call
    EV(raise) = EV(call)            (approximation: a raise is priced as a call; fold equity and the larger
                                     pot are ignored)

`equity` is our all-in equity against the ACTUAL hole cards of the opponents who continue, found from the real
later actions: every opponent dealt in (has hole cards that hand) is kept unless they folded preflop at any time
(before or after our decision). A blind who checks the option is therefore still in. Dead blind money is part of
`pot` (pot_before). If that leaves nobody (for example every opponent folded to our real raise), we fall back to
the opponents with money in before our decision that had not folded yet (`fallback: True`).

Caveats: this is an all-in-equity proxy that ignores postflop realization (position, implied odds, folding
equity), so it is fine for big preflop prices and rough for small ones. Equity is seeded Monte Carlo (n about 500)
over the 5-card board, so a spot carries about +-2 points of equity noise, shared by every deployment. A call is
capped at min(our starting stack, smallest villain starting stack) when the stacks are known (`call_amt`).
Only fold/call/raise are valid at a facing-bet spot; anything else (check, bet, ...) is invalid and scored as a
fold (EV 0), as are replies the gate marks invalid. Spots are weighted by to_call in the
`wmean_ev` columns so big decisions dominate.

    uv run python -m poker.realhands --gate data/gate/c5.json --reference vals-hackathon/c4-s6
"""

from __future__ import annotations

import argparse
import json
import random
import zlib
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

N_DEFAULT = 500
ALLOWED = ("fold", "call", "raise")


@lru_cache(maxsize=None)
def _equity(hero: tuple, villains: tuple, n: int, seed: int) -> float:
    from pokerkit import Card, Deck, StandardHighHand

    rng = random.Random(seed)
    known = set(hero) | {c for v in villains for c in v}
    deck = [c for c in (repr(x) for x in Deck.STANDARD) if c not in known]
    hero_cards = list(Card.parse("".join(hero)))
    vill_cards = [list(Card.parse("".join(v))) for v in villains]
    total = 0.0
    for _ in range(n):
        board = list(Card.parse("".join(rng.sample(deck, 5))))
        mine = StandardHighHand.from_game(hero_cards, board)
        theirs = [StandardHighHand.from_game(v, board) for v in vill_cards]
        best = max(theirs)
        if mine > best:
            total += 1.0
        elif mine == best:
            total += 1.0 / (1 + sum(t == best for t in theirs))
    return total / n


def equity_vs_hands(hero, villains, n: int = N_DEFAULT, seed: int = 0) -> float:
    """All-in equity from the preflop (random 5-card board) of `hero` against known villain hands; ties split."""
    return _equity(tuple(hero), tuple(tuple(v) for v in villains), n, seed)


def _hole(con, match_id, hand, seat):
    row = con.execute("SELECT card1, card2 FROM hole_cards WHERE match_id=? AND hand_number=? AND seat=?",
                      (match_id, hand, seat)).fetchone()
    return (row[0], row[1]) if row and row[0] and row[1] else None


def _stack(con, match_id, hand, seat):
    row = con.execute("SELECT starting_stack FROM hole_cards WHERE match_id=? AND hand_number=? AND seat=?",
                      (match_id, hand, seat)).fetchone()
    return row[0] if row and row[0] else None


def _skip(reason: str) -> dict:
    return {"ev": None, "reason": reason}


def folded_before(rows, idx) -> set:
    return {r["seat"] for r in rows if r["street"] == "preflop" and r["action"] == "fold" and r["idx"] < idx}


def spot_ev(con, match_id: str, hand: int, seat: int, action_index: int, action: str,
            n: int = N_DEFAULT, seed: int = 0) -> dict:
    """Chip EV of `action` (relative to folding) at one of our preflop decisions facing a bet.

    action_index counts our non-scripted actions in the hand, as in poker/teacher.judge. Returns a dict with
    `ev` (None when the spot is not scorable, with `reason`), plus equity, pot, to_call, n_villains, fallback.
    """
    rows = list(con.execute("SELECT * FROM actions WHERE match_id=? AND hand_number=? ORDER BY idx", (match_id, hand)))
    mine = [r for r in rows if r["seat"] == seat and not r["scripted"]]
    if action_index >= len(mine):
        return _skip("unmatched")
    row = mine[action_index]
    if row["street"] != "preflop":
        return _skip("not_preflop")
    to_call = row["to_call"] or 0
    if to_call <= 0:
        return _skip("no_bet_faced")
    hero = _hole(con, match_id, hand, seat)
    if hero is None:
        return _skip("no_hole_cards")
    base = {"pot": row["pot_before"], "to_call": to_call}
    folded, money_before = set(), set()
    dealt = {r["seat"] for r in con.execute("SELECT seat FROM hole_cards WHERE match_id=? AND hand_number=?",
                                            (match_id, hand))}
    for r in rows:
        if r["street"] != "preflop" or r["seat"] == seat:
            continue
        if r["action"] == "fold":
            folded.add(r["seat"])
        elif r["action"] in ("call", "raise", "bet") and r["idx"] < row["idx"]:
            money_before.add(r["seat"])
    fallback = False
    villain_seats = {s for s in dealt if s != seat} - folded
    if not villain_seats:
        villain_seats = money_before - folded_before(rows, row["idx"])
        fallback = True
    cards = [c for s in sorted(villain_seats) if (c := _hole(con, match_id, hand, s))]
    if not cards:
        return _skip("no_villain")
    key = zlib.crc32(f"{match_id}/{hand}/{seat}/{action_index}".encode()) ^ seed
    eq = equity_vs_hands(hero, cards, n=n, seed=key)
    stacks = [x for x in ([_stack(con, match_id, hand, seat)] + [_stack(con, match_id, hand, s) for s in sorted(villain_seats)])
              if x is not None]
    call_amt = min([to_call] + stacks)
    invalid = action not in ALLOWED
    ev = 0.0 if invalid or action == "fold" else eq * (row["pot_before"] + call_amt) - call_amt
    return base | {"ev": ev, "reason": None, "equity": eq, "n_villains": len(cards), "fallback": fallback,
                   "call_amt": call_amt, "invalid": invalid}


def _wmean(values: dict, weights: dict, keys) -> float:
    tot = sum(weights[k] for k in keys)
    return sum(values[k] * weights[k] for k in keys) / tot if tot else float("nan")


def paired_bootstrap(ref: dict, dep: dict, weights: dict, *, seed: int = 0, n: int = 2000, level: float = 0.95) -> dict:
    """Paired percentile bootstrap over spots of the weighted-mean EV difference dep - ref."""
    keys = sorted(set(ref) & set(dep))
    if not keys:
        return {"diff": None, "lo": None, "hi": None, "n": 0}
    rng = random.Random(seed)
    diffs = {k: dep[k] - ref[k] for k in keys}
    point = _wmean(diffs, weights, keys)
    means = []
    for _ in range(n):
        pick = [rng.choice(keys) for _ in keys]
        tot = sum(weights[k] for k in pick)
        means.append(sum(diffs[k] * weights[k] for k in pick) / tot if tot else 0.0)
    means.sort()
    tail = (1 - level) / 2
    return {"diff": point, "lo": means[int(tail * (n - 1))], "hi": means[int((1 - tail) * (n - 1))], "n": len(keys)}


def load_spot_map(joined: Path, span_ids: set) -> dict:
    out = {}
    for line in Path(joined).open(encoding="utf-8"):
        if '"span_id"' not in line:
            continue
        r = json.loads(line)
        m = r.get("match") or {}
        if r.get("span_id") in span_ids and r.get("source") == "arena" and r.get("kind") == "action" and m.get("verified"):
            out[r["span_id"]] = (m["match_id"], m["hand"], m["seat"], m["action_index"])
    return out


def score(gate_path, reference: str | None = None, *, joined="data/arena/joined.jsonl", con=None,
          db_path="data/arena/arena.db", n: int = N_DEFAULT, seed: int = 0) -> dict:
    """Per deployment: scored preflop spots, mean chip EV of the replayed action (vs fold = 0), mean EV of the
    original arena action on the same spots, and a paired bootstrap vs `reference`.

    Per span the replays (repeats) are averaged first, so each spot counts once. `mean_ev` is unweighted;
    `wmean_ev` weights spots by to_call."""
    gate = json.loads(Path(gate_path).read_text())
    if con is None:
        from poker.arena import db

        con = db.connect(db_path)
    spans = {r["span_id"] for rows in gate["spots"].values() for r in rows}
    spot_map = load_spot_map(joined, spans)
    orig_cache = {}
    by_bad = defaultdict(int)

    def ev_of(span, action):
        key = (span, action)
        if key not in orig_cache:
            orig_cache[key] = spot_ev(con, *spot_map[span], action, n=n, seed=seed)
        return orig_cache[key]

    def original(span):
        m = spot_map[span]
        rows = [r for r in con.execute("SELECT action, scripted FROM actions WHERE match_id=? AND hand_number=? "
                                       "AND seat=? ORDER BY idx", (m[0], m[1], m[2])) if not r["scripted"]]
        return rows[m[3]]["action"] if m[3] < len(rows) else None

    per_dep, weights, orig_ev, why = {}, {}, {}, {}
    for dep, rows in gate["spots"].items():
        by = defaultdict(list)
        n_invalid_all = 0
        for r in rows:
            if r["span_id"] not in spot_map:
                continue
            ok = bool(r.get("valid") and r.get("action") in ALLOWED)
            n_invalid_all += not ok
            by[r["span_id"]].append(r["action"] if ok else "fold")
            by_bad[(dep, r["span_id"])] += not ok
        evs = {}
        for span, acts in by.items():
            res = [ev_of(span, a) for a in acts]
            bad = next((x for x in res if x["ev"] is None), None)
            if bad:
                why.setdefault(span, bad["reason"])
                continue
            evs[span] = sum(x["ev"] for x in res) / len(res)
            weights[span] = res[0]["to_call"]
            if span not in orig_ev:
                o = original(span)
                orig_ev[span] = ev_of(span, o)["ev"] if o else None
        per_dep[dep] = {"evs": evs, "n_invalid_all": n_invalid_all, "n_total_spans": len(by)}
    reasons = defaultdict(int)
    for s_ in spot_map:
        if s_ not in weights:
            reasons[why[s_] if s_ in why else spot_ev(con, *spot_map[s_], "call", n=n, seed=seed)["reason"]] += 1
    out = {"reference": reference, "deployments": {}, "unscored_reasons": dict(reasons),
           "spots_in_gate": len(spans), "spots_matched": len(spot_map)}
    for dep, d in per_dep.items():
        evs = d["evs"]
        keys = sorted(k for k in evs if orig_ev.get(k) is not None)
        o = {k: orig_ev[k] for k in keys}
        row = {"n_spots": len(keys), "n_invalid_scored": sum(by_bad[(dep, k)] for k in keys),
               "n_invalid_all": d["n_invalid_all"],
               "mean_ev": sum(evs[k] for k in keys) / len(keys) if keys else None,
               "wmean_ev": _wmean(evs, weights, keys) if keys else None,
               "orig_mean_ev": sum(o.values()) / len(keys) if keys else None,
               "orig_wmean_ev": _wmean(o, weights, keys) if keys else None}
        out["deployments"][dep] = row
        d["scored"] = {k: evs[k] for k in keys}
    if reference in per_dep:
        ref = per_dep[reference]["scored"]
        for dep, d in per_dep.items():
            if dep != reference:
                out["deployments"][dep]["vs_ref"] = paired_bootstrap(
                    ref, d["scored"], weights, seed=seed) | {"unweighted": paired_bootstrap(
                        ref, d["scored"], {k: 1.0 for k in weights}, seed=seed)}
    return out


def format_table(result: dict) -> str:
    ref = result.get("reference")
    head = (f"{'deployment':<26}{'spots':>6}{'invalid s/a':>12}{'mean EV':>9}{'wmean EV':>10}{'orig EV':>9}{'orig wEV':>10}"
            f"  w diff vs {ref} (95% CI)")
    lines = [head]
    for dep, r in result["deployments"].items():
        v = r.get("vs_ref")
        tail = f"{v['diff']:+.2f} ({v['lo']:+.2f}..{v['hi']:+.2f}) n={v['n']}" if v and v["diff"] is not None else "-"
        inv = f"{r['n_invalid_scored']}/{r['n_invalid_all']}"
        lines.append(f"{dep:<26}{r['n_spots']:>6}{inv:>12}{r['mean_ev']:>9.2f}{r['wmean_ev']:>10.2f}"
                     f"{r['orig_mean_ev']:>9.2f}{r['orig_wmean_ev']:>10.2f}  {tail}")
    lines.append(f"unscored spots (gate spans {result['spots_in_gate']}, matched {result['spots_matched']}): "
                 f"{result['unscored_reasons']}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gate", required=True)
    p.add_argument("--reference", default=None)
    p.add_argument("--joined", default="data/arena/joined.jsonl")
    p.add_argument("--db", default="data/arena/arena.db")
    p.add_argument("--n", type=int, default=N_DEFAULT)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    print(format_table(score(a.gate, a.reference, joined=a.joined, db_path=a.db, n=a.n, seed=a.seed)))


if __name__ == "__main__":
    main()
