"""Teacher labels for our arena decisions: flag clear mistakes, abstain everywhere else.

The teacher is deliberately conservative. It only judges spots where a simple, defensible rule gives a clear
answer, and it returns `preferred=None` (abstain) otherwise. Version 1 covers:

1. Calling a preflop all-in: our equity against the shover's *observed* shove range (from the arena database,
   blended with the population when the sample is small) versus the pot-odds price, plus a bust margin when the
   call risks our whole stack. Busting ranks last, so early in the game a coin flip for our stack is a fold.
2. Unopened preflop pots: open-raising or limping with hands outside a loose range for the seat is a fold, and
   folding a premium hand is a raise. Folding a marginal hand is never flagged.
3. Folding when checking is free is always a check.

Opponent hole cards inform the shove ranges only as aggregate statistics over past hands. They are never put in a
student prompt; SFT/OPSD inputs stay exactly what the model saw at decision time.
"""

from __future__ import annotations

import random
from functools import lru_cache

RANKS = "23456789TJQKA"
SUITS = "cdhs"
PREMIUM = {"AA", "KK", "QQ", "JJ", "TT", "AKs", "AQs", "AKo"}


def _parts(code: str) -> tuple[int, int, str]:
    """('AKs') -> (high rank index, low rank index, 's'|'o'|'')."""
    a, b = RANKS.index(code[0]), RANKS.index(code[1])
    kind = code[2] if len(code) == 3 else ""
    return max(a, b), min(a, b), kind


def combos(code: str) -> list[tuple[str, str]]:
    hi, lo, kind = _parts(code)
    r1, r2 = RANKS[hi], RANKS[lo]
    if hi == lo:
        return [(r1 + s1, r1 + s2) for i, s1 in enumerate(SUITS) for s2 in SUITS[i + 1 :]]
    if kind == "s":
        return [(r1 + s, r2 + s) for s in SUITS]
    return [(r1 + s1, r2 + s2) for s1 in SUITS for s2 in SUITS if s1 != s2]


def hand_code(cards: list[str]) -> str:
    a, b = sorted(cards, key=lambda c: RANKS.index(c[0]), reverse=True)
    if a[0] == b[0]:
        return a[0] + b[0]
    return a[0] + b[0] + ("s" if a[1] == b[1] else "o")


@lru_cache(maxsize=4096)
def _equity(hole: tuple[str, ...], codes: tuple[str, ...], board: tuple[str, ...], samples: int) -> float:
    from pokerkit import Card, Deck, StandardHighHand

    known = set(hole) | set(board)
    villain = [c for code in codes for c in combos(code) if not (set(c) & known)]
    if not villain:
        return 0.5
    rng = random.Random(repr((hole, codes, board, samples)))
    deck = [repr(c) for c in Deck.STANDARD]
    hero = list(Card.parse("".join(hole)))
    total = 0.0
    for _ in range(samples):
        v = rng.choice(villain)
        rest = [c for c in deck if c not in known and c not in v]
        runout = list(board) + rng.sample(rest, 5 - len(board))
        cards = list(Card.parse("".join(runout)))
        mine = StandardHighHand.from_game(hero, cards)
        theirs = StandardHighHand.from_game(list(Card.parse("".join(v))), cards)
        total += 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0
    return round(total / samples, 4)


def equity_vs_range(hole: list[str], codes: list[str], board: list[str] = (), samples: int = 600) -> float:
    """Heads-up pot share of `hole` against a uniform pick from the combos of `codes`, over random runouts."""
    return _equity(tuple(hole), tuple(sorted(set(codes))), tuple(board), samples)


def required_equity(price: float, risks_bust: bool, hand_number: int, total_hands: int = 30) -> float:
    """Pot-odds price plus a bust margin that shrinks as the game nears its end."""
    if not risks_bust:
        return price
    left = max(0.0, 1.0 - hand_number / total_hands)
    return round(price + 0.03 + 0.07 * left, 4)


def allin_call_verdict(hole, shove_range, *, price, risks_bust, hand_number, band=0.04, samples=600) -> dict:
    equity = equity_vs_range(hole, shove_range, samples=samples)
    required = required_equity(price, risks_bust, hand_number)
    preferred = "fold" if equity < required - band else "call" if equity > required + band else None
    return {"rule": "allin_call", "preferred": preferred, "equity": equity, "required": required}


def _in_loose_range(code: str, position: str) -> bool:
    """A loose but sane opening range per seat; anything outside it is a clear fold when the pot is unopened."""
    hi, lo, kind = _parts(code)
    H = lambda r: RANKS.index(r)  # noqa: E731
    gap = hi - lo
    if hi == lo:
        return True
    late = position in ("BTN", "SB", "HU_SB", "HU_BB")
    if kind == "s":
        if hi >= H("K"):
            return True
        if late:
            return (hi == H("Q") and lo >= H("5")) or (hi >= H("5") and gap <= 3)
        return (hi == H("Q") and lo >= H("9")) or (hi >= H("6") and gap <= 2 and lo >= H("5"))
    if hi == H("A"):
        return lo >= (H("2") if late else H("8"))
    if late:
        return (hi == H("K") and lo >= H("8")) or (hi >= H("9") and lo >= H("9") and gap <= 2) or (hi == H("Q") and lo >= H("9"))
    return lo >= H("T")


LATE = ("BTN", "SB", "HU_SB", "HU_BB")


def _trash(code: str) -> bool:
    """No pair, no card above 9, and not a suited hand with a gap of two or less."""
    hi, lo, kind = _parts(code)
    return hi != lo and hi < RANKS.index("T") and not (kind == "s" and hi - lo <= 2)


def open_verdict(code: str, position: str, action: str, to_call: float = 2) -> str | None:
    """Unopened preflop pot. Late-position raises are steals against a fold-heavy field, so only trash is flagged
    there; limps (calls) with hands outside the loose range are always flagged. A free check beats a fold."""
    if action == "fold":
        return "raise" if code in PREMIUM else None
    if action == "raise" and position in LATE:
        bad = _trash(code)
    else:
        bad = action in ("raise", "call") and not _in_loose_range(code, position)
    if not bad:
        return None
    return "check" if not to_call else "fold"


def free_fold_verdict(to_call: float, action: str) -> str | None:
    return "check" if action == "fold" and not to_call else None


POPULATION_MIN = 10  # below this many own shoves, add the population's shove hands to the range


def shove_range(con, sub_key: str, exclude: tuple[str, int] | None = None) -> tuple[list[str], int]:
    """Hand codes this submission went all-in with preflop, leaving out `exclude` = (match_id, hand_number).

    Returns (codes, own sample size). Small samples are widened with every player's shove hands, so a range is
    never built from one or two hands alone.
    """
    query = ("SELECT hand_code FROM actions WHERE street='preflop' AND action='raise' AND stack_after=0 "
             "AND hand_code IS NOT NULL AND NOT (match_id=? AND hand_number=?)")
    skip = exclude or ("", -1)
    own = [r[0] for r in con.execute(query + " AND sub_key=?", (*skip, sub_key))]
    codes = list(own)
    if len(own) < POPULATION_MIN:
        codes += [r[0] for r in con.execute(query, skip)]
    return sorted(set(codes)), len(own)


def _rows(con, match_id: str, hand: int) -> list:
    return list(con.execute("SELECT * FROM actions WHERE match_id=? AND hand_number=? ORDER BY idx", (match_id, hand)))


def _hole(con, match_id: str, hand: int, seat: int) -> list[str] | None:
    row = con.execute("SELECT card1, card2 FROM hole_cards WHERE match_id=? AND hand_number=? AND seat=?",
                      (match_id, hand, seat)).fetchone()
    return [row[0], row[1]] if row and row[0] and row[1] else None


def judge(con, match_id: str, hand: int, seat: int, action_index: int) -> dict:
    """Teacher verdict for one of our decisions, identified as in poker/arena/join_traces.py.

    `agree` is None when the teacher abstains, else whether our action matches the teacher's preference.
    """
    rows = _rows(con, match_id, hand)
    mine = [r for r in rows if r["seat"] == seat and not r["scripted"]]
    verdict = {"rule": None, "preferred": None, "actual": None, "agree": None}
    if action_index >= len(mine):
        return verdict | {"rule": "unmatched"}
    row = mine[action_index]
    actual = row["action"]
    verdict["actual"] = actual
    if (free := free_fold_verdict(row["to_call"], actual)):
        return verdict | {"rule": "free_fold", "preferred": free, "agree": False}
    if row["street"] != "preflop" or not row["hand_code"]:
        return verdict
    before = [r for r in rows if r["idx"] < row["idx"]]
    raises = [r for r in before if r["street"] == "preflop" and r["action"] in ("raise", "bet")]
    if not raises and row["to_call"] <= 2:
        preferred = open_verdict(row["hand_code"], row["position"], actual, to_call=row["to_call"])
        agree = None if preferred is None else preferred == actual
        return verdict | {"rule": "unopened", "preferred": preferred, "agree": agree}
    stack_before = row["stack_after"] - (row["chip_change"] or 0)
    callers_after_shove = 0
    if raises and raises[-1]["stack_after"] == 0 and actual in ("call", "fold"):
        shover = raises[-1]
        callers_after_shove = sum(1 for r in before if r["idx"] > shover["idx"] and r["action"] == "call")
        if callers_after_shove:
            return verdict | {"rule": "allin_call_multiway"}  # heads-up equity would mislead; abstain
        hole = _hole(con, match_id, hand, seat)
        if not hole:
            return verdict
        codes, own = shove_range(con, shover["sub_key"], exclude=(match_id, hand))
        price = row["to_call"] / (row["pot_before"] + row["to_call"])
        risks_bust = row["to_call"] >= 0.9 * stack_before
        result = allin_call_verdict(hole, codes, price=price, risks_bust=risks_bust, hand_number=hand)
        agree = None if result["preferred"] is None else result["preferred"] == actual
        return verdict | result | {"agree": agree, "shover": shover["sub_key"], "shover_n": own,
                                   "to_call": row["to_call"], "pot": row["pot_before"]}
    return verdict
