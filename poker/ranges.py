"""Empirical preflop hand ranges of the LLM field, from the arena DB.

Counts are over the 169 hand classes. Each count is one dealt hand that took the action, so classes are already
weighted by how often they are dealt (pairs 6 combos, suited 4, offsuit 12). Probabilities are therefore
proportional to (count + prior) and are NOT multiplied by combos again; the prior is spread by combo weight.
Sampling a villain hand picks a class from probs, then a uniformly random combo of that class not blocked by the
hero or board cards. Only team entries are used, and LoRA (us) and house bots are excluded.

Situations: open (first raise, not all-in), limp, 3bet (raise at raises_before=1, not all-in), jam (all-in raise,
or any raise at raises_before>=2), call_vs_open (call at raises_before=1).
"""
import argparse
import itertools
import json
import random
import sqlite3
from collections import defaultdict
from pathlib import Path

RANKS = "23456789TJQKA"
SUITS = "cdhs"
BB = 2.0
MIN_SAMPLES = 30
PRIOR_STRENGTH = 10.0  # pseudo-samples spread over the 169 classes by combo weight
SIZE_BUCKETS = ("<=2.5x", "2.5-4x", ">4x")
SITUATIONS = ("open", "limp", "3bet", "jam", "call_vs_open")


def hand_class(c1: str, c2: str) -> str:
    a, b = sorted((c1, c2), key=lambda c: RANKS.index(c[0]), reverse=True)
    if a[0] == b[0]:
        return a[0] + b[0]
    return a[0] + b[0] + ("s" if a[1] == b[1] else "o")


def _all_classes() -> list[str]:
    out = []
    for i in range(12, -1, -1):
        for j in range(i, -1, -1):
            hi, lo = RANKS[i], RANKS[j]
            out += [hi + lo] if i == j else [hi + lo + "s", hi + lo + "o"]
    return out


ALL_CLASSES = _all_classes()


def combos_of(code: str) -> int:
    return 6 if len(code) == 2 else 4 if code[2] == "s" else 12


def expand(code: str) -> list[tuple[str, str]]:
    hi, lo = code[0], code[1]
    if len(code) == 2:
        return [(hi + a, lo + b) for a, b in itertools.combinations(SUITS, 2)]
    if code[2] == "s":
        return [(hi + s, lo + s) for s in SUITS]
    return [(hi + a, lo + b) for a in SUITS for b in SUITS if a != b]


def size_bucket(amount: float) -> str:
    x = amount / BB
    return "<=2.5x" if x <= 2.5 else "2.5-4x" if x <= 4 else ">4x"


def _bucket():
    return {"n": 0, "counts": defaultdict(int)}


def _situation(action, raises_before, allin):
    if action == "raise":
        if allin or raises_before >= 2:
            return "jam"
        return "open" if raises_before == 0 else "3bet"
    if action == "call":
        return "limp" if raises_before == 0 else "call_vs_open" if raises_before == 1 else None
    return None


def build(db_path: str) -> dict:
    con = sqlite3.connect(db_path)
    kinds = {sk: (k, n) for sk, n, k in con.execute("select sub_key, name, kind from entries")}
    rows = con.execute(
        "select match_id, hand_number, sub_key, action, amount, stack_after, card1, card2, hand_code, position, "
        "raises_before from actions where street='preflop' order by match_id, hand_number, idx")
    tbl = {s: defaultdict(lambda: defaultdict(_bucket)) for s in SITUATIONS}
    first_in = defaultdict(lambda: defaultdict(int))
    facing_open = defaultdict(lambda: defaultdict(int))
    seen_voluntary = set()
    for mid, hn, sk, action, amount, stack_after, c1, c2, code, pos, rb in rows:
        hand = (mid, hn)
        voluntary_before = hand in seen_voluntary
        if action in ("call", "raise"):
            seen_voluntary.add(hand)
        kind, name = kinds.get(sk, (None, None))
        if kind != "team" or name == "LoRA":
            continue
        rb = rb or 0
        allin = action == "raise" and stack_after is not None and stack_after <= 0
        code = code or (hand_class(c1, c2) if c1 and c2 else None)
        if code not in ALL_CLASSES:
            continue
        if rb == 0 and not voluntary_before and pos != "BB":
            first_in[pos]["jam" if allin else "open" if action == "raise" else "limp" if action == "call" else "fold"] += 1
        if rb == 1:
            facing_open[pos]["jam" if allin else "3bet" if action == "raise" else action] += 1
        sit = _situation(action, rb, allin)
        if sit is None:
            continue
        sizes = ["all"]
        if sit == "open":
            sizes.append(size_bucket(amount or 0))
        for p in (pos, "any"):
            for s in sizes:
                b = tbl[sit][p][s]
                b["n"] += 1
                b["counts"][code] += 1
    return {"meta": {"bb": BB, "min_samples": MIN_SAMPLES},
            **json.loads(json.dumps(tbl)), "first_in": json.loads(json.dumps(first_in)),
            "facing_open": json.loads(json.dumps(facing_open))}


def _bucket_of(table, situation, position, size):
    return table.get(situation, {}).get(position, {}).get(size)


def range_probs(table: dict, situation: str, position: str, size_bucket: str | None = None) -> dict[str, float]:
    """Probability per hand class. Fallback order: (position, size) -> (position, all) -> (any, size) -> (any, all),
    taking the first pool with >= MIN_SAMPLES. Smoothing adds PRIOR_STRENGTH pseudo-samples spread by combo weight."""
    order = []
    if size_bucket:
        order.append((position, size_bucket))
    order += [(position, "all")]
    if size_bucket:
        order.append(("any", size_bucket))
    order.append(("any", "all"))
    chosen = None
    for p, s in order:
        b = _bucket_of(table, situation, p, s)
        if b and b["n"] >= MIN_SAMPLES:
            chosen = b
            break
    counts = chosen["counts"] if chosen else {}
    w = {c: counts.get(c, 0) + PRIOR_STRENGTH * combos_of(c) / 1326 for c in ALL_CLASSES}
    tot = sum(w.values())
    return {c: v / tot for c, v in w.items()}


def equity_vs_range(hero_cards, probs, board=(), n=2000, seed=0) -> float:
    """Heads-up Monte Carlo equity of hero vs a villain hand drawn from `probs` (class -> weight)."""
    from pokerkit import Card, Deck, StandardHighHand

    rng = random.Random(seed)
    known = set(hero_cards) | set(board)
    classes, weights = zip(*[(c, p) for c, p in probs.items() if p > 0])
    deck = [repr(c) for c in Deck.STANDARD]
    hero = list(Card.parse("".join(hero_cards)))
    cache = {}
    total, done = 0.0, 0
    while done < n:
        code = rng.choices(classes, weights)[0]
        if code not in cache:
            cache[code] = [v for v in expand(code) if not (set(v) & known)]
        if not cache[code]:
            continue
        v = rng.choice(cache[code])
        rest = [c for c in deck if c not in known and c not in v]
        run = list(board) + rng.sample(rest, 5 - len(board))
        cards = list(Card.parse("".join(run)))
        mine = StandardHighHand.from_game(hero, cards)
        theirs = StandardHighHand.from_game(list(Card.parse("".join(v))), cards)
        total += 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0
        done += 1
    return total / n


def _width(table, situation, position, size=None):
    """Share of the 1326 combos whose class appears in the pool (raw count > 0), a rough range width."""
    b = _bucket_of(table, situation, position, size or "all")
    if not b:
        return 0.0
    return sum(combos_of(c) for c in b["counts"]) / 1326


def _top_width(table, situation, position, mass=0.8):
    """Share of combos in the smallest set of classes (by frequency per combo) covering `mass` of the actions."""
    b = _bucket_of(table, situation, position, "all")
    if not b or not b["n"]:
        return 0.0
    ranked = sorted(b["counts"], key=lambda c: b["counts"][c] / combos_of(c), reverse=True)
    cov = comb = 0
    for c in ranked:
        if cov >= mass * b["n"]:
            break
        cov += b["counts"][c]
        comb += combos_of(c)
    return comb / 1326


def summary(table: dict) -> str:
    out = ["pos   first-in n  open%  limp%  jam%  top80%-width  3bet% (vs open)  n_open  n_3bet"]
    for pos in ("UTG", "HJ", "CO", "BTN", "SB", "BB", "HU_SB", "HU_BB"):
        fi, fo = table["first_in"].get(pos, {}), table["facing_open"].get(pos, {})
        n, nf = sum(fi.values()), sum(fo.values())
        if not n and not nf:
            continue
        pct = lambda d, k, t: f"{100 * d.get(k, 0) / t:5.1f}" if t else "  n/a"
        no = (table["open"].get(pos, {}).get("all") or {"n": 0})["n"]
        n3 = (table["3bet"].get(pos, {}).get("all") or {"n": 0})["n"]
        out.append(f"{pos:<6}{n:>10}  {pct(fi, 'open', n)}  {pct(fi, 'limp', n)}  {pct(fi, 'jam', n)}  "
                   f"{100 * _top_width(table, 'open', pos):10.1f}  {pct(fo, '3bet', nf):>13}  {no:>6}  {n3:>6}")
    for sit in SITUATIONS:
        b = _bucket_of(table, sit, "any", "all") or {"n": 0}
        out.append(f"any {sit}: n={b['n']}")
    for sz in SIZE_BUCKETS:
        b = _bucket_of(table, "open", "any", sz) or {"n": 0}
        out.append(f"any open {sz}: n={b['n']}")
    out.append("equity (n=3000) of hero vs field range:")
    for hero in (("Jh", "Jd"), ("Ah", "Qd"), ("Kh", "Qd"), ("Ah", "5h"), ("7h", "7d")):
        e3 = equity_vs_range(hero, range_probs(table, "3bet", "any"), n=3000)
        ej = equity_vs_range(hero, range_probs(table, "jam", "any"), n=3000)
        out.append(f"  {hand_class(*hero):<4} vs 3bet {e3:.3f}   vs jam {ej:.3f}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/arena/arena.db")
    ap.add_argument("--out", default="data/ranges/field_ranges.json")
    a = ap.parse_args()
    table = build(a.db)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(table, indent=1))
    print(summary(table))


if __name__ == "__main__":
    main()
