"""Per-team preflop/postflop profiles from the arena DB, and seat-adjusted ranges.

build_profiles() aggregates every team (not LoRA, not house bots) over all its submissions ("all") and over its
most recently played submission ("latest", by max match start time). adjusted_range() rescales a field range
to a team's observed frequency: hands are ordered by strength and the top posterior-width slice is kept.
"""
import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from poker import ranges as R
from poker.ranges import ALL_CLASSES, combos_of

POSITIONS = ("UTG", "HJ", "CO", "BTN", "SB", "BB", "HU_SB", "HU_BB")
SITS = ("open", "limp", "3bet", "call_vs_open", "jam")
ORDER_PATH = "data/ranges/hand_order.json"


# ---------------------------------------------------------------- accumulation
def _accumulate(rows, kinds):
    """rows: preflop+postflop action rows ordered by match, hand, idx. Returns {sub_key: Counter}."""
    acc = defaultdict(Counter)
    seen_vol = set()
    flags = set()
    for (mid, hn, sk, street, action, amount, to_call, stack_after, c1, c2, code, pos, rb) in rows:
        hand = (mid, hn)
        kind, name = kinds.get(sk, (None, None))
        is_team = kind == "team" and name != "LoRA"
        if street == "preflop":
            vol_before = hand in seen_vol
            if action in ("call", "raise"):
                seen_vol.add(hand)
        if not is_team:
            continue
        c = acc[sk]
        rb = rb or 0
        if street != "preflop":
            if action in ("bet", "raise", "call", "check"):
                c[("post", action)] += 1
            if (to_call or 0) > 0 and action in ("fold", "call", "raise"):
                c[("ftb", action)] += 1
            continue
        code = code or (R.hand_class(c1, c2) if c1 and c2 else None)
        key = (mid, hn, sk)
        if ("h", key) not in flags:
            flags.add(("h", key))
            c[("hands",)] += 1
        if action in ("call", "raise") and ("v", key) not in flags and not (action == "call" and False):
            flags.add(("v", key))
            c[("vpip",)] += 1
        if action == "raise" and ("p", key) not in flags:
            flags.add(("p", key))
            c[("pfr",)] += 1
        allin = action == "raise" and stack_after is not None and stack_after <= 0
        if rb == 0 and not vol_before and pos != "BB":
            out = "jam" if allin else "open" if action == "raise" else "limp" if action == "call" else "fold"
            c[("fi", pos, out)] += 1
            if code in ALL_CLASSES:
                c[("fi_cls", pos, code)] += 1
                if out == "open":
                    c[("fiopen", pos, code)] += 1
            if out == "open":
                c[("size", pos)] += (amount or 0) / R.BB
                c[("size_n", pos)] += 1
        if rb == 1:
            c[("fo", pos, "jam" if allin else "3bet" if action == "raise" else action)] += 1
        sit = R._situation(action, rb, allin)
        if sit and code in ALL_CLASSES:
            for p in (pos, "any"):
                c[("sit", sit, p, code)] += 1
    return acc


def _pct(n, d):
    return round(100 * n / d, 2) if d else None


def finalize(c: Counter) -> dict:
    """Turn an accumulated Counter into a profile dict."""
    positions = {}
    for pos in POSITIONS:
        fi = {o: c[("fi", pos, o)] for o in ("open", "limp", "jam", "fold")}
        fo = {o: c[("fo", pos, o)] for o in ("fold", "call", "3bet", "jam")}
        n, nf = sum(fi.values()), sum(fo.values())
        if not n and not nf:
            continue
        d = {"first_in_n": n, "facing_n": nf, "open_size_bb": None}
        d.update({k: _pct(v, n) for k, v in fi.items()})
        d.update({f"facing_{k}": _pct(v, nf) for k, v in fo.items()})
        if c[("size_n", pos)]:
            d["open_size_bb"] = round(c[("size", pos)] / c[("size_n", pos)], 2)
        positions[pos] = d
    ranges = {s: defaultdict(dict) for s in SITS}
    fib = defaultdict(dict)
    for k, v in c.items():
        if k[0] == "sit":
            ranges[k[1]][k[2]][k[3]] = v
        elif k[0] == "fi_cls":
            fib[k[1]][k[2]] = v
    fio = defaultdict(dict)
    for k, v in c.items():
        if k[0] == "fiopen":
            fio[k[1]][k[2]] = v
    sizes_n = sum(v for k, v in c.items() if k[0] == "size_n")
    sizes = sum(v for k, v in c.items() if k[0] == "size")
    pf = {a: c[("post", a)] for a in ("bet", "raise", "call", "check")}
    pn = sum(pf.values())
    fb = {a: c[("ftb", a)] for a in ("fold", "call", "raise")}
    fbn = sum(fb.values())
    h = c[("hands",)]
    return {
        "hands": h, "vpip": _pct(c[("vpip",)], h), "pfr": _pct(c[("pfr",)], h),
        "open_size_bb": round(sizes / sizes_n, 2) if sizes_n else None, "open_size_n": sizes_n,
        "postflop_aggr": _pct(pf["bet"] + pf["raise"], pn), "postflop_n": pn,
        "fold_to_bet": _pct(fb["fold"], fbn), "fold_to_bet_n": fbn,
        "positions": positions,
        "ranges": {s: dict(v) for s, v in ranges.items()},
        "first_in_by_class": dict(fib), "first_in_open_by_class": dict(fio),
    }


def build_profiles(db_path: str) -> dict:
    con = sqlite3.connect(db_path)
    kinds = {sk: (k, n) for sk, n, k in con.execute("select sub_key, name, kind from entries")}
    rows = con.execute(
        "select match_id, hand_number, sub_key, street, action, amount, to_call, stack_after, card1, card2, "
        "hand_code, position, raises_before from actions order by match_id, hand_number, idx").fetchall()
    last = dict(con.execute("select s.sub_key, max(m.started_at) from seats s join matches m using (match_id) "
                            "group by s.sub_key"))
    places = defaultdict(list)
    for sk, pl in con.execute("select sub_key, place from seats where place is not null"):
        places[kinds.get(sk, (None, None))[1]].append(pl)
    con.close()
    acc = _accumulate(rows, kinds)
    by_team = defaultdict(list)
    for sk in acc:
        by_team[kinds[sk][1]].append(sk)
    teams = {}
    for name, sks in sorted(by_team.items()):
        total = sum((acc[s] for s in sks), Counter())
        latest = max(sks, key=lambda s: (last.get(s) or "", s))
        pl = places.get(name, [])
        teams[name] = {"sub_keys": sorted(sks), "latest_sub_key": latest,
                       "mean_place": round(sum(pl) / len(pl), 3) if pl else None, "matches": len(pl),
                       "all": finalize(total), "latest": finalize(acc[latest])}
    return {"teams": teams, "_acc": acc, "_by_team": dict(by_team)}


# ---------------------------------------------------------------- seat-adjusted ranges
def hand_order(path: str = ORDER_PATH, n: int = 1500, seed: int = 7) -> list[str]:
    """169 classes, strongest first, by equity vs a random hand (computed once and cached)."""
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())["order"]
    uniform = {c: combos_of(c) / 1326 for c in ALL_CLASSES}
    eq = {c: R.equity_vs_range(R.expand(c)[0], uniform, n=n, seed=seed) for c in ALL_CLASSES}
    order = sorted(ALL_CLASSES, key=lambda c: -eq[c])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"n": n, "seed": seed, "order": order, "equity": eq}, indent=1))
    return order


def posterior_width(observed, opps, base_freq, prior_strength=10):
    """Posterior mean of a Beta prior centred on base_freq (prior_strength pseudo-hands) updated with the data."""
    return (base_freq * prior_strength + observed) / (prior_strength + opps)


def effective_width(probs) -> float:
    """Share of the 1326 combos a distribution spreads over: 1 / sum(p^2 / combo share)."""
    s = sum(p * p / (combos_of(c) / 1326) for c, p in probs.items() if p > 0)
    return 1 / s if s else 0.0


def _tilted(base_probs, w, base_freq, order, lam):
    """Keep the field's own shape: per-class action rate r_c = P(c | action) * freq / share, then shift every
    logit by a common offset (solved by bisection) so the mean rate over all combos equals w. lam > 0 also
    tilts toward strong hands (z-scored position in the strength order), which smooths classes the field rarely saw."""
    n = len(order)
    z = {c: (n / 2 - i) / n * 2 for i, c in enumerate(order)}  # +1 strongest ... -1 weakest
    share = {c: combos_of(c) / 1326 for c in order}
    lg = {}
    for c in order:
        r = min(max(base_probs.get(c, 0.0) * base_freq / share[c], 1e-4), 1 - 1e-4)
        lg[c] = math.log(r / (1 - r)) + lam * z[c]
    lo, hi = -30.0, 30.0
    for _ in range(60):
        a = (lo + hi) / 2
        m = sum(share[c] / (1 + math.exp(-(a + lg[c]))) for c in order)
        lo, hi = (a, hi) if m < w else (lo, a)
    out = {c: share[c] / (1 + math.exp(-(a + lg[c]))) for c in order}
    tot = sum(out.values())
    return {c: v / tot for c, v in out.items()}


def adjusted_range(base_probs, observed_open, observed_opps, prior_strength=10, order=None, base_freq=None,
                   taper=0.06, mode="tilt", lam=1.0):
    """mode="tilt" (default, fits real teams best): keep the field's shape and shift it to the posterior width,
    tilted toward strong hands by lam. mode="strength": the top `posterior width` of combos in strength order, combo weighted,
    with a logistic taper at the edge. base_freq is the field's frequency for this seat and action (defaults to the
    effective width of base_probs). Works for open, 3bet and jam alike."""
    order = order or hand_order()
    if base_freq is None:
        base_freq = effective_width(base_probs)
    w = posterior_width(observed_open, observed_opps, base_freq, prior_strength)
    if mode == "tilt":
        return _tilted(base_probs, w, base_freq, order, lam)
    scale = max(taper * w, 0.004)
    out, cum = {}, 0.0
    for c in order:
        share = combos_of(c) / 1326
        mid = cum + share / 2
        cum += share
        z = (w - mid) / scale
        out[c] = share / (1 + math.exp(-max(min(z, 50), -50)))
    tot = sum(out.values())
    return {c: v / tot for c, v in out.items()}


def overlap(emp: dict, model: dict) -> float:
    """Sum of min(empirical share, model share); 1 means identical distributions."""
    n = sum(emp.values())
    return sum(min(v / n, model.get(c, 0.0)) for c, v in emp.items()) if n else 0.0


def loglik(counts: dict, model: dict, eps=0.005) -> float:
    """Mean log-likelihood per observed hand; the model is mixed with a combo-weighted uniform at weight eps."""
    n = sum(counts.values())
    if not n:
        return float("nan")
    tot = 0.0
    for c, v in counts.items():
        q = (1 - eps) * model.get(c, 0.0) + eps * combos_of(c) / 1326
        tot += v * math.log(q)
    return tot / n


def field_probs(counts: dict, prior=R.PRIOR_STRENGTH) -> dict:
    w = {c: counts.get(c, 0) + prior * combos_of(c) / 1326 for c in ALL_CLASSES}
    t = sum(w.values())
    return {c: v / t for c, v in w.items()}


def validate(prof_out: dict, order=None, **adj_kw) -> dict:
    """Leave-one-team-out: field range from the other teams vs seat-adjusted range from the team's own frequency."""
    acc, by_team = prof_out["_acc"], prof_out["_by_team"]
    order = order or hand_order()
    team_c = {t: sum((acc[s] for s in sks), Counter()) for t, sks in by_team.items()}
    field = sum(team_c.values(), Counter())
    res = {}
    for t, tc in team_c.items():
        rest = field - tc
        pooled = {"n": 0, "ll_field": 0.0, "ll_adj": 0.0, "ov_field": 0.0, "ov_adj": 0.0, "per_pos": {}}
        for pos in ("UTG", "HJ", "CO", "BTN", "SB"):
            opps = sum(tc[("fi", pos, o)] for o in ("open", "limp", "jam", "fold"))
            obs = tc[("fi", pos, "open")]
            counts = {k[2]: v for k, v in tc.items() if k[0] == "fiopen" and k[1] == pos}
            n = sum(counts.values())
            if n < 10:
                continue
            fo = {k[2]: v for k, v in rest.items() if k[0] == "fiopen" and k[1] == pos}
            f_opps = sum(rest[("fi", pos, o)] for o in ("open", "limp", "jam", "fold"))
            f_freq = sum(fo.values()) / f_opps if f_opps else 0.2
            fprobs = field_probs(fo)
            aprobs = adjusted_range(fprobs, obs, opps, order=order, base_freq=f_freq, **adj_kw)
            r = {"n": n, "opps": opps, "freq": obs / opps, "field_freq": f_freq,
                 "ll_field": loglik(counts, fprobs), "ll_adj": loglik(counts, aprobs),
                 "ov_field": overlap(counts, fprobs), "ov_adj": overlap(counts, aprobs)}
            pooled["per_pos"][pos] = r
            pooled["n"] += n
            for k in ("ll_field", "ll_adj", "ov_field", "ov_adj"):
                pooled[k] += r[k] * n
        if pooled["n"]:
            for k in ("ll_field", "ll_adj", "ov_field", "ov_adj"):
                pooled[k] /= pooled["n"]
            res[t] = pooled
    return res


def public(prof_out: dict) -> dict:
    return {"teams": prof_out["teams"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/arena/arena.db")
    ap.add_argument("--out", default="data/ranges/team_profiles.json")
    a = ap.parse_args()
    out = build_profiles(a.db)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(public(out), indent=1))
    order = hand_order()
    print(f"{'team':<16}{'place':>6}{'hands':>7}{'vpip':>6}{'pfr':>6}{'open_bb':>8}{'aggr':>6}{'f2bet':>6}")
    for t, d in sorted(out["teams"].items(), key=lambda kv: kv[1]["mean_place"] or 99):
        s = d["all"]
        print(f"{t:<16}{d['mean_place']:>6}{s['hands']:>7}{s['vpip']:>6}{s['pfr']:>6}{s['open_size_bb']!s:>8}"
              f"{s['postflop_aggr']!s:>6}{s['fold_to_bet']!s:>6}")
    for t, r in validate(out, order).items():
        print(t, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "per_pos"})


if __name__ == "__main__":
    main()
