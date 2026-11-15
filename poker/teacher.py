"""Teacher labels for our arena decisions: flag clear mistakes, abstain everywhere else.

The teacher is deliberately conservative. It only judges spots where a simple, defensible rule gives a clear
answer, and it returns `preferred=None` (abstain) otherwise. Version 1 covers:

1. Calling a preflop all-in: our equity against the shover's *observed* shove range (from the arena database,
   blended with the population when the sample is small) versus the pot-odds price, plus a bust margin when the
   call risks our whole stack. Busting ranks last, so early in the game a coin flip for our stack is a fold.
2. Unopened preflop pots: open-raising or limping with hands outside a loose range for the seat is a fold, and
   folding a premium hand is a raise. Folding a marginal hand is never flagged.
3. Folding when checking is free is always a check.
4. `postflop_call` (version 2): a big call or fold heads-up on the flop, turn or river, judged by our equity on
   the current board against the strength classes (strong, top pair, pair, draw, air) of the hands the bettor held in other
   hands when betting the same street in the same size bucket, widened with the population when small.

Opponent hole cards inform the shove ranges only as aggregate statistics over past hands. They are never put in a
student prompt; SFT/OPSD inputs stay exactly what the model saw at decision time.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
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


def _simulate(hole, villain, board, samples, seed, weights=None) -> float:
    from pokerkit import Card, Deck, StandardHighHand

    known = set(hole) | set(board)
    rng = random.Random(seed)
    deck = [repr(c) for c in Deck.STANDARD]
    hero = list(Card.parse("".join(hole)))
    total = 0.0
    for _ in range(samples):
        v = rng.choice(villain) if weights is None else rng.choices(villain, weights)[0]
        rest = [c for c in deck if c not in known and c not in v]
        runout = list(board) + rng.sample(rest, 5 - len(board))
        cards = list(Card.parse("".join(runout)))
        mine = StandardHighHand.from_game(hero, cards)
        theirs = StandardHighHand.from_game(list(Card.parse("".join(v))), cards)
        total += 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0
    return round(total / samples, 4)


@lru_cache(maxsize=4096)
def _equity(hole: tuple[str, ...], codes: tuple[str, ...], board: tuple[str, ...], samples: int) -> float:
    known = set(hole) | set(board)
    villain = [c for code in codes for c in combos(code) if not (set(c) & known)]
    if not villain:
        return 0.5
    return _simulate(hole, villain, board, samples, repr((hole, codes, board, samples)))


def equity_vs_range(hole: list[str], codes: list[str], board: list[str] = (), samples: int = 600) -> float:
    """Heads-up pot share of `hole` against a uniform pick from the combos of `codes`, over random runouts."""
    return _equity(tuple(hole), tuple(sorted(set(codes))), tuple(board), samples)


# v5 exploit rules, each on only when named here, e.g. EXPLOITS = {"river_fold", "steal", "stab"}; also "oversize", "monster_call", "river_air_fold".
# v7: "rfi_gto", "iso_raise", "three_bet", "bb_defend", "vs_3bet", "postflop_sizing" (see the teacher v7 section).
EXPLOITS: set[str] = set()
STEAL_RANGES = {
    "CO": "22+ A2s+ K5s+ Q8s+ J8s+ T8s+ 97s+ 86s+ 76s 65s A7o+ K9o+ QTo+ JTo",
    "BTN": "22+ A2s+ K2s+ Q5s+ J7s+ T7s+ 96s+ 85s+ 75s+ 64s+ 54s A2o+ K8o+ Q9o+ J9o+ T9o 98o",
}
STEAL_RANGES["SB"] = STEAL_RANGES["BTN"]
STEAL_MIN_BB = 15
# "oversize": the cap on a non-all-in preflop raise-to (chips; BB = 2): open 7 + 2 per limper, 3-bet 3.0x, 4-bet+ 2.3x.
OVERSIZE_OPEN, OVERSIZE_PER_LIMPER, OVERSIZE_3BET, OVERSIZE_4BET = 7, 2, 3.0, 2.3
OVERSIZE_TOLERANCE = 1.5  # only raises above 1.5x the cap are clear oversizing; a 4-5bb open is not a mistake

EVIDENCE = False  # v4: act only on strong evidence (see the interval helpers and the verdicts below)
MARGIN_FOLD = 0.02
MARGIN_CALL = 0.02
K_BATCHES = 8


def _combine(batches: list[float], boots: list[float]) -> tuple[float, float, float]:
    """Mean of the batch estimates with a 95% interval, mean +/- 1.96 sd. The sd combines the Monte Carlo error of
    the mean (batch sd / sqrt(K)) with range uncertainty (sd of the bootstrap re-estimates)."""
    k = len(batches)
    mean = sum(batches) / k
    sd = lambda xs: (sum((x - sum(xs) / len(xs)) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5  # noqa: E731
    total = ((sd(batches) ** 2) / k + sd(boots) ** 2) ** 0.5
    return round(mean, 4), round(max(0.0, mean - 1.96 * total), 4), round(min(1.0, mean + 1.96 * total), 4)


@lru_cache(maxsize=4096)
def _equity_interval(hole, codes, board, samples):
    known = set(hole) | set(board)
    per = max(1, samples // K_BATCHES)
    base = repr((hole, codes, board, samples))

    def live(cs):
        return [c for code in cs for c in combos(code) if not (set(c) & known)]

    villain = live(codes)
    if not villain:
        return 0.5, 0.0, 1.0
    batches = [_simulate(hole, villain, board, per, f"{base}|b{i}") for i in range(K_BATCHES)]
    rng = random.Random(base + "|boot")
    boots = []
    for i in range(K_BATCHES):
        sample = live([rng.choice(codes) for _ in codes]) or villain
        boots.append(_simulate(hole, sample, board, per, f"{base}|r{i}"))
    return _combine(batches, boots)


def equity_interval(hole, codes, board=(), samples: int = 600) -> tuple[float, float, float]:
    """(equity, lo, hi) against a uniform pick from `codes`: K=8 seeded batches (MC error) plus 8 bootstrap
    resamples of the hand codes (range uncertainty); lo/hi = mean -/+ 1.96 sd of the combined spread."""
    return _equity_interval(tuple(hole), tuple(sorted(set(codes))), tuple(board), samples)


def required_equity(price: float, risks_bust: bool, hand_number: int, total_hands: int = 30) -> float:
    """Pot-odds price plus a bust margin that shrinks as the game nears its end."""
    if not risks_bust:
        return price
    left = max(0.0, 1.0 - hand_number / total_hands)
    return round(price + 0.03 + 0.07 * left, 4)


def allin_call_verdict(hole, shove_range, *, price, risks_bust, hand_number, band=0.04, samples=600) -> dict:
    if EVIDENCE:
        equity, lo, hi = equity_interval(hole, shove_range, samples=samples)
        required = required_equity(price, risks_bust, hand_number)
        preferred = "fold" if hi < required - MARGIN_FOLD else "call" if lo > required + MARGIN_CALL else None
        return {"rule": "allin_call", "preferred": preferred, "equity": equity, "required": required,
                "equity_lo": lo, "equity_hi": hi}
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


def _expand(token: str) -> set[str]:
    """One range token: '66+', '77', 'ATs+', 'KQo', 'KJo+' -> hand codes."""
    plus = token.endswith("+")
    t = token.rstrip("+")
    hi, lo = RANKS.index(t[0]), RANKS.index(t[1])
    if hi == lo:
        return {RANKS[r] * 2 for r in range(hi, len(RANKS))} if plus else {t}
    kind = t[2]
    kickers = range(lo, hi) if plus else [lo]
    return {RANKS[hi] + RANKS[k] + kind for k in kickers}


@lru_cache(maxsize=256)
def _range_set(spec: str) -> frozenset[str]:
    return frozenset(code for token in spec.split() for code in _expand(token))


def in_range(code: str, spec: str) -> bool:
    hi, lo, kind = _parts(code)
    norm = RANKS[hi] + RANKS[lo] + kind
    return norm in _range_set(spec)


# Standard-ish 6-max opening ranges for early seats (teacher v3). BTN/SB steals keep the trash-only check.
OPEN_RANGES = {
    "UTG": "66+ ATs+ KTs+ QTs+ JTs AJo+ KQo",
    "HJ": "55+ A8s+ K9s+ QTs+ JTs T9s ATo+ KJo+",
    "CO": "22+ A2s+ K8s+ Q9s+ J9s+ T9s 98s 87s 76s A9o+ KTo+ QTo+ JTo",
}
CONTINUE_VS_RAISE = "66+ A9s+ KTs+ QTs+ JTs T9s 98s AJo+ KQo"
CONTINUE_VS_RAISE_WIDE = "44+ A2s+ K9s+ Q9s+ J9s+ T9s 98s 87s ATo+ KJo+ QJo"  # EVIDENCE: clear mistakes only
CONTINUE_VS_3BET = "JJ+ AQs+ AKo"
SHOVE_RANGES = ((40, "TT+ AQs+ AKo"), (20, "77+ ATs+ AJo+ KQs"), (0, "22+ A2s+ A7o+ K9s+ KJo+ QTs+ JTs"))


def facing_raise_verdict(code: str, position: str, action: str, *, raises_before: int, to_call: float,
                         stack: float) -> str | None:
    """Preflop, facing a raise that is not all-in: continuing (call or re-raise) outside a sane range is a fold.
    Cheap big-blind defense (to call 4 or less) is not judged, and folds are never flagged."""
    if action not in ("call", "raise"):
        return None
    if position in ("BB", "HU_BB") and to_call <= 4 and raises_before == 1:
        return None
    big = stack and to_call >= 0.3 * stack
    spec = CONTINUE_VS_3BET if raises_before >= 2 or big else CONTINUE_VS_RAISE_WIDE if EVIDENCE else CONTINUE_VS_RAISE
    return None if in_range(code, spec) else "fold"


def open_shove_verdict(code: str, effective_bb: float) -> str | None:
    """Our preflop all-in raise: outside the shove range for this stack depth, fold (busting ranks last)."""
    spec = next(spec for floor, spec in SHOVE_RANGES if effective_bb > floor)
    return None if in_range(code, spec) else "fold"


def open_verdict(code: str, position: str, action: str, to_call: float = 2) -> str | None:
    """Unopened preflop pot. Late-position raises are steals against a fold-heavy field, so only trash is flagged
    there; limps (calls) with hands outside the loose range are always flagged. A free check beats a fold."""
    if action == "fold":
        return "raise" if code in PREMIUM else None
    if action == "raise" and position in LATE:
        bad = _trash(code)
    elif EVIDENCE and position in ("UTG", "HJ") and action in ("raise", "call"):
        bad = not in_range(code, OPEN_RANGES["CO"])
    elif EVIDENCE and position == "CO" and action in ("raise", "call"):
        bad = not _in_loose_range(code, position)
    elif position in OPEN_RANGES and action in ("raise", "call"):
        bad = not in_range(code, OPEN_RANGES[position])
    else:
        bad = action in ("raise", "call") and not _in_loose_range(code, position)
    if not bad:
        return None
    return "check" if not to_call else "fold"


def free_fold_verdict(to_call: float, action: str) -> str | None:
    return "check" if action == "fold" and not to_call else None


POPULATION_MIN = 10  # below this many own shoves, add the population's shove hands to the range
# Seats are anonymous to the student (seat0..seat5), so a per-team range is information it can never use at
# inference. By default every range is field-level: what all players did in this spot (current hand excluded).
PER_OPPONENT = False
# The final tournament has only other teams' models, no house bots. Field ranges therefore come from LLM-team
# opponents (entries.kind == 'team', our own team excluded), falling back to everyone when that set has no data.
OUR_TEAM = "LoRA"
FIELD_TEAMS_ONLY = True


def field_sub_keys(con) -> set[str]:
    return {r[0] for r in con.execute("SELECT sub_key FROM entries WHERE kind='team' AND name != ?", (OUR_TEAM,))}


def _field_clause(con, column: str = "sub_key") -> tuple[str, tuple]:
    keys = sorted(field_sub_keys(con)) if FIELD_TEAMS_ONLY else []
    if not keys:
        return "", ()
    return f" AND {column} IN ({','.join('?' * len(keys))})", tuple(keys)


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
    if not PER_OPPONENT or len(own) < POPULATION_MIN:
        clause, keys = _field_clause(con)
        field = [r[0] for r in con.execute(query + clause, (*skip, *keys))] if clause else []
        if not field:
            field = [r[0] for r in con.execute(query, skip)]
        codes = ([] if not PER_OPPONENT else codes) + field
    return sorted(set(codes)), len(own)


POSTFLOP_BAND = 0.08  # postflop ranges are rougher than preflop shove ranges
POSTFLOP_MIN = 8
BOARD_CARDS = {"flop": 3, "turn": 4, "river": 5}
CLASSES = ("strong", "top_pair", "pair", "draw", "air")
MIN_POSTFLOP_PRICE = 0.15  # a stack-share trigger alone does not make a tiny call worth judging


def _label_rank(label) -> int:
    from pokerkit import Label

    return list(Label).index(label)


def _board_category(board: list[str]) -> int:
    """Hand category the board makes on its own (pair, two pair, trips, full house, quads), by rank counts."""
    from pokerkit import Label

    counts = sorted((sum(1 for c in board if c[0] == r) for r in set(c[0] for c in board)), reverse=True)
    if counts[0] == 4:
        return _label_rank(Label.FOUR_OF_A_KIND)
    if counts[0] == 3:
        return _label_rank(Label.FULL_HOUSE if len(counts) > 1 and counts[1] == 2 else Label.THREE_OF_A_KIND)
    if counts[0] == 2:
        return _label_rank(Label.TWO_PAIR if counts.count(2) > 1 else Label.ONE_PAIR)
    return _label_rank(Label.HIGH_CARD)


def _has_draw(hole: list[str], board: list[str]) -> bool:
    cards = list(hole) + list(board)
    for suit in SUITS:
        if sum(1 for c in cards if c[1] == suit) == 4 and any(c[1] == suit for c in hole):
            return True
    held = {RANKS.index(c[0]) + 2 for c in hole}
    ranks = {RANKS.index(c[0]) + 2 for c in cards}
    for low in range(2, 11):  # windows of four ranks, open at both ends: 2-5 up to T-A is excluded at the ace
        window = set(range(low, low + 4))
        if low + 4 <= 14 and window <= ranks and window & held:
            return True
    return False


def strength_class(hole: list[str], board: list[str]) -> str:
    """strong / top_pair / pair / draw / air for `hole` on the cards dealt so far (3 to 5 board cards).

    strong = two pair or better made with the hole cards; top_pair = pair of the board's top card or an overpair;
    pair = any other pair using a hole card; draw = flush draw or open-ended straight draw (flop and turn only).
    A pair that exists only on the board does not count for the holder.
    """
    from pokerkit import Card, Label, StandardHighHand

    label = StandardHighHand.from_game(list(Card.parse("".join(hole))), list(Card.parse("".join(board)))).entry.label
    rank = _label_rank(label)
    if rank >= _label_rank(Label.TWO_PAIR) and rank > _board_category(board):
        return "strong"
    top = max(RANKS.index(c[0]) for c in board)
    h1, h2 = RANKS.index(hole[0][0]), RANKS.index(hole[1][0])
    board_ranks = {RANKS.index(c[0]) for c in board}
    if h1 == h2:
        return "top_pair" if h1 > top else "pair"
    if rank == _label_rank(Label.ONE_PAIR) and (h1 in board_ranks or h2 in board_ranks):
        return "top_pair" if top in (h1, h2) else "pair"
    if len(board) < 5 and _has_draw(hole, board):
        return "draw"
    return "air"


def _bucket_sql(row) -> str | None:
    """SQL condition for 'a bet of the same size bucket as `row`': all-in, then bet_frac <0.5, 0.5-1.0, >1.0."""
    if row["stack_after"] == 0:
        return "a.stack_after=0"
    frac = row["bet_frac"]
    if frac is None:
        return None
    if frac < 0.5:
        return "a.stack_after>0 AND a.bet_frac<0.5"
    if frac <= 1.0:
        return "a.stack_after>0 AND a.bet_frac>=0.5 AND a.bet_frac<=1.0"
    return "a.stack_after>0 AND a.bet_frac>1.0"


def postflop_class_counts(con, bettor, exclude: tuple[str, int] | None = None) -> tuple[dict, int]:
    """How often this bettor's past bets (same street, same size bucket) were each strength class, judged on the
    hand's own board at that street. `exclude` = (match_id, hand_number) is left out.

    Returns (class counts, own hands). Fewer than POSTFLOP_MIN own hands widens the counts with every player's
    hands. Aggregate use only: villain cards never reach a student prompt.
    """
    bucket = _bucket_sql(bettor)
    if bucket is None:
        return {}, 0
    need = BOARD_CARDS[bettor["street"]]
    query = ("SELECT DISTINCT a.match_id, a.hand_number, h.card1, h.card2, hd.board FROM actions a "
             "JOIN hole_cards h ON h.match_id=a.match_id AND h.hand_number=a.hand_number AND h.seat=a.seat "
             "JOIN hands hd ON hd.match_id=a.match_id AND hd.hand_number=a.hand_number "
             f"WHERE a.street=? AND a.action IN ('bet','raise') AND {bucket} "
             "AND h.card1 IS NOT NULL AND h.card2 IS NOT NULL AND NOT (a.match_id=? AND a.hand_number=?)")
    skip = exclude or ("", -1)
    args = (bettor["street"], *skip)

    def classify(rows):
        counts: dict = {}
        for r in rows:
            board = (r[4] or "").split()[:need]
            if len(board) == need:
                k = strength_class([r[2], r[3]], board)
                counts[k] = counts.get(k, 0) + 1
        return counts

    own = classify(con.execute(query + " AND a.sub_key=?", (*args, bettor["sub_key"])))
    n = sum(own.values())
    if PER_OPPONENT and n >= POSTFLOP_MIN:
        return own, n
    clause, keys = _field_clause(con, "a.sub_key")
    both = classify(con.execute(query + clause, (*args, *keys))) if clause else {}
    if sum(both.values()) < POSTFLOP_MIN:  # too little LLM-team data in this bucket: use everyone
        both = classify(con.execute(query, args))
    return both, n


def class_distribution(counts: dict) -> dict:
    """Add-one smoothed share of each class."""
    total = sum(counts.values()) + len(CLASSES)
    return {k: (counts.get(k, 0) + 1) / total for k in CLASSES}


@lru_cache(maxsize=1024)
def _class_groups(hole: tuple[str, ...], board: tuple[str, ...]) -> dict:
    known = set(hole) | set(board)
    by_class: dict = {k: [] for k in CLASSES}
    cards = [r + s for r in RANKS for s in SUITS if r + s not in known]
    for i, c1 in enumerate(cards):
        for c2 in cards[i + 1 :]:
            by_class[strength_class([c1, c2], list(board))].append((c1, c2))
    return by_class


def _weighted_villain(by_class: dict, dist) -> tuple[list, list] | None:
    live = [(by_class[k], p) for k, p in zip(CLASSES, dist) if by_class[k]]
    if not live:
        return None
    total = sum(p for _, p in live)
    villain, weights = [], []
    for group, p in live:
        villain += group
        weights += [p / total / len(group)] * len(group)
    return villain, weights


@lru_cache(maxsize=4096)
def _equity_classes(hole: tuple[str, ...], board: tuple[str, ...], dist: tuple[float, ...], samples: int) -> float:
    wv = _weighted_villain(_class_groups(hole, board), dist)
    if wv is None:
        return 0.5
    return _simulate(hole, wv[0], board, samples, repr((hole, board, dist, samples)), wv[1])


@lru_cache(maxsize=4096)
def equity_interval_classes(hole: tuple[str, ...], board: tuple[str, ...], counts: tuple, samples: int = 600):
    """(equity, lo, hi) against the class mixture given by `counts` = sorted (class, n) pairs. K=8 seeded batches
    (MC error) plus 8 bootstrap resamples of the observed class counts (range uncertainty); mean -/+ 1.96 sd."""
    by_class = _class_groups(hole, board)
    per = max(1, samples // K_BATCHES)
    base = repr((hole, board, counts, samples))

    def dist_of(cn: dict) -> tuple[float, ...]:
        d = class_distribution(cn)
        return tuple(d[k] for k in CLASSES)

    wv = _weighted_villain(by_class, dist_of(dict(counts)))
    if wv is None:
        return 0.5, 0.0, 1.0
    batches = [_simulate(hole, wv[0], board, per, f"{base}|b{i}", wv[1]) for i in range(K_BATCHES)]
    pool = [k for k, n in counts for _ in range(n)]
    rng = random.Random(base + "|boot")
    boots = []
    for i in range(K_BATCHES):
        cn: dict = {}
        for k in (rng.choice(pool) for _ in pool):
            cn[k] = cn.get(k, 0) + 1
        w = _weighted_villain(by_class, dist_of(cn))
        boots.append(_simulate(hole, w[0], board, per, f"{base}|r{i}", w[1]))
    return _combine(batches, boots)


CALL_EXTRA = 0.04  # extra margin for call corrections on the flop and turn


def postflop_call_verdict(hole, counts, board, *, price, risks_bust, hand_number, band=POSTFLOP_BAND,
                          samples=600) -> dict:
    dist = class_distribution(counts)
    if EVIDENCE:
        equity, lo, hi = equity_interval_classes(tuple(hole), tuple(board), tuple(sorted(counts.items())), samples)
        required = required_equity(price, risks_bust, hand_number)
        call_margin = MARGIN_CALL + (CALL_EXTRA if len(board) < 5 else 0.0)
        preferred = "fold" if hi < required - MARGIN_FOLD else "call" if lo > required + call_margin else None
        return {"rule": "postflop_call", "preferred": preferred, "equity": equity, "required": required,
                "equity_lo": lo, "equity_hi": hi, "dist": {k: round(v, 3) for k, v in dist.items()}}
    equity = _equity_classes(tuple(hole), tuple(board), tuple(dist[k] for k in CLASSES), samples)
    required = required_equity(price, risks_bust, hand_number)
    # Before the river the bettor can keep betting, so raw equity overstates what a call realizes: a "call"
    # correction needs CALL_EXTRA more margin there. Fold corrections and river spots keep the normal band.
    call_band = band + (CALL_EXTRA if len(board) < 5 else 0.0)
    preferred = "fold" if equity < required - band else "call" if equity > required + call_band else None
    return {"rule": "postflop_call", "preferred": preferred, "equity": equity, "required": required,
            "dist": {k: round(v, 3) for k, v in dist.items()}}


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
    if "monster_call" in EXPLOITS and (v := _monster_call(con, row, verdict, match_id, hand, seat)):
        return v
    if "river_air_fold" in EXPLOITS and (v := _river_air_fold(con, row, verdict, match_id, hand, seat)):
        return v
    if "river_fold" in EXPLOITS and (v := _river_fold(con, row, verdict, match_id, hand, seat)):
        return v
    if "stab" in EXPLOITS and (v := _stab(con, rows, row, verdict, match_id, hand, seat)):
        return v
    if row["street"] in BOARD_CARDS and (v := _v6_postflop(con, rows, row, verdict, match_id, hand, seat)):
        return v
    if "postflop_sizing" in EXPLOITS and (v := _postflop_sizing(con, row, verdict, match_id, hand, seat)):
        return v
    if row["street"] in BOARD_CARDS and row["to_call"] > 0 and actual in ("call", "fold"):
        return _judge_postflop(con, rows, row, verdict, match_id, hand, seat)
    if row["street"] != "preflop" or not row["hand_code"]:
        return verdict
    before = [r for r in rows if r["idx"] < row["idx"]]
    raises = [r for r in before if r["street"] == "preflop" and r["action"] in ("raise", "bet")]
    stack0 = row["stack_after"] - (row["chip_change"] or 0)
    # v6 rules first: a junk hand facing a big price is a fold, not a smaller raise (oversize).
    if "pad_read" in EXPLOITS and (v := _pad_read(con, row, before, raises, stack0, verdict, match_id, hand)):
        return v
    if "pot_odds_call" in EXPLOITS and (v := _pot_odds_call(row, before, stack0, verdict)):
        return v
    if "big_price_fold" in EXPLOITS and (v := _big_price_fold(row, stack0, verdict, before)):
        return v
    for name, rule in V7_PREFLOP:
        if name in EXPLOITS and (v := rule(row, before, raises, stack0, verdict)):
            return v
    if "raise_first_in" in EXPLOITS and (v := _raise_first_in(row, before, raises, verdict)):
        return v
    if "oversize" in EXPLOITS and (v := _oversize(row, before, raises, verdict)):
        return v
    if actual == "raise" and row["stack_after"] == 0 and not (raises and raises[-1]["stack_after"] == 0):
        preferred = open_shove_verdict(row["hand_code"], (row["effective_stack"] or stack0) / 2)
        agree = None if preferred is None else preferred == actual
        return verdict | {"rule": "own_shove", "preferred": preferred, "agree": agree}
    if not raises and row["to_call"] <= 2:
        preferred = open_verdict(row["hand_code"], row["position"], actual, to_call=row["to_call"])
        if preferred is None and "steal" in EXPLOITS and _steal(row, before, stack0):
            return verdict | {"rule": "steal", "preferred": "raise", "agree": False}
        agree = None if preferred is None else preferred == actual
        return verdict | {"rule": "unopened", "preferred": preferred, "agree": agree}
    stack_before = stack0
    if raises and raises[-1]["stack_after"] != 0:
        preferred = facing_raise_verdict(row["hand_code"], row["position"], actual, raises_before=len(raises),
                                         to_call=row["to_call"], stack=stack_before)
        agree = None if preferred is None else preferred == actual
        return verdict | {"rule": "facing_raise", "preferred": preferred, "agree": agree}
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


def _judge_postflop(con, rows, row, verdict, match_id, hand, seat) -> dict:
    """Rule `postflop_call` (version 2): a big call or fold heads-up on the flop, turn or river."""
    actual = row["action"]
    if row["players_remaining"] != 2:
        return verdict | {"rule": "postflop_multiway"}
    stack_before = row["stack_after"] - (row["chip_change"] or 0)
    price = row["to_call"] / (row["pot_before"] + row["to_call"])
    big = (row["facing_frac"] or 0) >= 0.5 or (row["to_call"] >= 0.25 * stack_before and price >= MIN_POSTFLOP_PRICE)
    if not big:
        return verdict | {"rule": "postflop_small_call"}
    bettors = [r for r in rows if r["idx"] < row["idx"] and r["street"] == row["street"]
               and r["action"] in ("bet", "raise") and r["seat"] != seat]
    hole = _hole(con, match_id, hand, seat)
    board_row = con.execute("SELECT board FROM hands WHERE match_id=? AND hand_number=?", (match_id, hand)).fetchone()
    if not bettors or not hole or not board_row or not board_row[0]:
        return verdict
    board = board_row[0].split()[: BOARD_CARDS[row["street"]]]
    if len(board) < BOARD_CARDS[row["street"]]:
        return verdict
    bettor = bettors[-1]
    counts, own = postflop_class_counts(con, bettor, exclude=(match_id, hand))
    risks_bust = row["to_call"] >= 0.9 * stack_before
    result = postflop_call_verdict(hole, counts, board, price=price, risks_bust=risks_bust, hand_number=hand)
    agree = None if result["preferred"] is None else result["preferred"] == actual
    return verdict | result | {"agree": agree, "bettor": bettor["sub_key"], "n": own, "street": row["street"],
                               "to_call": row["to_call"], "pot": row["pot_before"]}


def _board_for(con, match_id: str, hand: int, street: str) -> list[str] | None:
    board_row = con.execute("SELECT board FROM hands WHERE match_id=? AND hand_number=?", (match_id, hand)).fetchone()
    board = (board_row[0] or "").split()[: BOARD_CARDS[street]] if board_row else []
    return board if len(board) == BOARD_CARDS[street] else None


def _river_fold(con, row, verdict, match_id, hand, seat) -> dict | None:
    """Exploit `river_fold`: heads-up river, a 0.5-1.0 pot bet, and we call with a one-pair hand. Fold."""
    if (row["street"] != "river" or row["action"] != "call" or row["players_remaining"] != 2
            or not 0.5 <= (row["facing_frac"] or 0) <= 1.0):
        return None
    hole, board = _hole(con, match_id, hand, seat), _board_for(con, match_id, hand, "river")
    if not hole or not board or strength_class(hole, board) not in ("pair", "top_pair"):
        return None
    return verdict | {"rule": "river_fold", "preferred": "fold", "agree": False}


MONSTER_MAX_PRICE = 0.45


def _monster_call(con, row, verdict, match_id, hand, seat) -> dict | None:
    """Exploit `monster_call`: postflop, we fold a strong hand to a bet at a price of 0.45 or less. Call.

    Two pair counts only when the board is unpaired (both hole cards pair the board); otherwise trips or better.
    """
    from pokerkit import Card, Label, StandardHighHand

    if row["street"] not in BOARD_CARDS or row["action"] != "fold" or (row["to_call"] or 0) <= 0:
        return None
    if row["to_call"] / (row["pot_before"] + row["to_call"]) > MONSTER_MAX_PRICE:
        return None
    hole, board = _hole(con, match_id, hand, seat), _board_for(con, match_id, hand, row["street"])
    if not hole or not board or strength_class(hole, board) != "strong":
        return None
    rank = _label_rank(StandardHighHand.from_game(list(Card.parse("".join(hole))),
                                                  list(Card.parse("".join(board)))).entry.label)
    if rank < _label_rank(Label.THREE_OF_A_KIND) and _board_category(board) != _label_rank(Label.HIGH_CARD):
        return None
    return verdict | {"rule": "monster_call", "preferred": "call", "agree": False}


def _river_air_fold(con, row, verdict, match_id, hand, seat) -> dict | None:
    """Exploit `river_air_fold`: we call a river bet with pure air (no pair of our own, no made straight or better)."""
    if row["street"] != "river" or row["action"] != "call" or (row["to_call"] or 0) <= 0:
        return None
    hole, board = _hole(con, match_id, hand, seat), _board_for(con, match_id, hand, "river")
    if not hole or not board or strength_class(hole, board) != "air":
        return None
    return verdict | {"rule": "river_air_fold", "preferred": "fold", "agree": False}


def _steal(row, before, stack0) -> bool:
    """Exploit `steal`: unopened (no raise, no limp), CO/BTN/SB, we folded a hand in the steal range, 15bb or more."""
    spec = STEAL_RANGES.get(row["position"])
    if not spec or row["action"] != "fold" or not row["hand_code"] or not in_range(row["hand_code"], spec):
        return False
    if any(r["street"] == "preflop" and r["action"] in ("raise", "bet", "call") for r in before):
        return False
    return (row["effective_stack"] or stack0) / 2 >= STEAL_MIN_BB


def _stab(con, rows, row, verdict, match_id, hand, seat) -> dict | None:
    """Exploit `stab`: heads-up flop or turn, the opponent checked to us, we check with air or a draw. Bet."""
    if (row["street"] not in ("flop", "turn") or row["to_call"] != 0 or row["action"] != "check"
            or row["players_remaining"] != 2):
        return None
    if not any(r["idx"] < row["idx"] and r["street"] == row["street"] and r["seat"] != seat
               and r["action"] == "check" for r in rows):
        return None
    hole, board = _hole(con, match_id, hand, seat), _board_for(con, match_id, hand, row["street"])
    if not hole or not board or strength_class(hole, board) not in ("air", "draw"):
        return None
    return verdict | {"rule": "stab", "preferred": "bet", "agree": False}


def oversize_cap(before: list, raises: list) -> tuple[int, int]:
    """(cap, legal minimum) for a preflop raise-to, given the preflop rows before it and the raises among them."""
    last = raises[-1]["amount"] if raises else 2
    prev = raises[-2]["amount"] if len(raises) > 1 else 2
    if not raises:
        limpers = sum(1 for r in before if r["street"] == "preflop" and r["action"] == "call")
        return OVERSIZE_OPEN + OVERSIZE_PER_LIMPER * limpers, 2 * last
    factor = OVERSIZE_3BET if len(raises) == 1 else OVERSIZE_4BET
    return int(factor * last), int(last + max(last - prev, 2))


def _oversize(row, before, raises, verdict) -> dict | None:
    """Exploit `oversize`: a preflop raise, not all-in, above the cap for its spot. Raise, but smaller."""
    if (row["action"] != "raise" or not row["amount"] or row["stack_after"] == 0
            or any(r["amount"] is None for r in raises)):
        return None
    cap, minimum = oversize_cap(before, raises)
    if row["amount"] <= OVERSIZE_TOLERANCE * cap:
        return None
    return verdict | {"rule": "oversize", "preferred": "raise", "agree": False, "target_amount": max(cap, minimum)}


# ---- teacher v6 tier-1 rules (each on only when named in EXPLOITS) ----
BIG_PRICE_CHIPS, BIG_PRICE_STACK_FRAC, HUGE_PRICE_STACK_FRAC = 24, 0.20, 0.40
BIG_PRICE_CONTINUE = {"AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo", "AQs"}
HUGE_PRICE_CONTINUE = {"AA", "KK", "QQ", "JJ", "AKs", "AKo"}
FIRST_IN_RANGE = "22+ A2s+ A9o+ K9s+ KTo+ Q9s+ QTo+ J9s+ JTo T8s+ 98s 87s 76s 65s"
FIRST_IN_SEATS = ("CO", "BTN", "SB")
EARLY_OPEN_RANGE = "22+ ATs+ AJo+ KQs KQo"  # open-limps outside this from UTG/HJ are folds
# First-in raises from the early seats: c4-s6 folded 77 UTG, AdTd UTG, ATo and KJo in cheap unopened pots.
EARLY_RAISE_RANGE = {"UTG": "22+ A9s+ ATo+ KTs+ QTs+ JTs KQo", "HJ": "22+ A7s+ A5s ATo+ K9s+ QTs+ JTs T9s KJo+ QJo"}
PREMIUM_FIRST_IN = "TT+ AQs+ AQo+"
FIRST_IN_BASE = {"CO": 5, "BTN": 5, "SB": 6}
FIRST_IN_PER_LIMPER, FIRST_IN_CAP, FIRST_IN_CAP_PREMIUM, FIRST_IN_MAX_RAISE = 2, 8, 10, 8
MONSTER_BET_FRAC = 0.6
POT_CAP_STACK_FRAC = 0.5
POT_CAP_RAISED_MIN_PRICE = 0.25  # after our bet is raised, fold one pair only when the call needs more than this


def open_target(position: str, code: str, limpers: int) -> int:
    """Standard raise-to for an unopened pot: 5 (6 from the SB) plus 2 per limper, capped at 8 (10 for AA/KK)."""
    cap = FIRST_IN_CAP_PREMIUM if code in ("AA", "KK") else FIRST_IN_CAP
    return min(FIRST_IN_BASE.get(position, 5) + FIRST_IN_PER_LIMPER * limpers, cap)


POT_ODDS_HU, POT_ODDS_MULTI = 0.20, 0.12  # required equity at or below this is a call with any two cards


def _cheap_all_in(row, before, stack0) -> float | None:
    """The required equity when the call ends the betting for us (we are all-in for it, or the bettor is all-in) and
    the price is at or below the any-two-cards threshold: 20% against one opponent, 12% against more. Else None."""
    to_call = row["to_call"] or 0
    if row["street"] != "preflop" or to_call <= 0 or not stack0:
        return None
    if not (row["facing_allin"] or to_call >= stack0):
        return None
    required = to_call / ((row["pot_before"] or 0) + to_call)
    folded = {r["seat"] for r in before if r["action"] == "fold"}
    others = {r["seat"] for r in before if r["action"] in ("call", "raise", "bet") and r["seat"] != row["seat"]} - folded
    return required if required <= (POT_ODDS_HU if len(others) <= 1 else POT_ODDS_MULTI) else None


def _pot_odds_call(row, before, stack0, verdict) -> dict | None:
    """Rule `pot_odds_call`: folding our last chips (or to an all-in) at a tiny price gives the pot away. Call."""
    if row["action"] != "fold":
        return None
    required = _cheap_all_in(row, before, stack0)
    if required is None:
        return None
    return verdict | {"rule": "pot_odds_call", "preferred": "call", "agree": False, "required": round(required, 4)}


def _big_price_fold(row, stack0, verdict, before=()) -> dict | None:
    """Rule `big_price_fold`: a big price before the flop and we continue with a hand outside the strong set. Fold.
    A cheap all-in price (see `_cheap_all_in`) is never a fold."""
    to_call = row["to_call"] or 0
    if row["street"] != "preflop" or to_call <= 0 or row["action"] not in ("call", "raise"):
        return None
    if _cheap_all_in(row, before, stack0) is not None:
        return None
    allin = bool(row["facing_allin"])
    if not (to_call > BIG_PRICE_CHIPS or (stack0 and to_call > BIG_PRICE_STACK_FRAC * stack0) or allin):
        return None
    huge = allin or (stack0 and to_call > HUGE_PRICE_STACK_FRAC * stack0)
    if row["hand_code"] in (HUGE_PRICE_CONTINUE if huge else BIG_PRICE_CONTINUE):
        return None
    return verdict | {"rule": "big_price_fold", "preferred": "fold", "agree": False}


# ---- rule `pad_read`: continue range scaled by the raiser's profile as the teacher pad at that hand shows it ----
PAD_INDEX: dict | None = None  # poker.pad.rewrite.chain_index(joined rows); set by make_records when pad_read is on
PAD_BIG_CHIPS = 24            # 12 BB: facing a raise above this, or a shove, is a big price
PAD_WIDE_RANGE = "77+ ATs+ AJo+ KQs"
PAD_TIGHT_RANGE = "QQ+ AKs AKo"
PAD_WIDE_PRICE_FRAC = 0.45
PAD_WIDE_MIN_HANDS, PAD_WIDE_MIN_VPIP, PAD_WIDE_MIN_RAISES = 6, 0.60, 3
PAD_TIGHT_MIN_HANDS, PAD_TIGHT_MAX_VPIP = 8, 0.12


def pad_profile(match_id: str, seat: int, raiser: int, hand: int) -> dict | None:
    """What the teacher pad that `seat` sees at the start of `hand` says about `raiser`: hands observed, VPIP hands,
    raises. None when PAD_INDEX is unset, the pad cannot be rebuilt, or it does not list the raiser."""
    if PAD_INDEX is None:
        return None
    from poker.pad.rewrite import teacher_pad_at
    from poker.pad.writer import parse_pad

    pad = teacher_pad_at(PAD_INDEX, match_id, seat, hand)
    state = parse_pad(pad) if pad else None
    opp = state.opps.get(raiser) if state else None
    return None if opp is None else {"hands": opp.vb, "vpip": opp.va, "raises": opp.raises}


def _pad_read(con, row, before, raises, stack0, verdict, match_id, hand) -> dict | None:
    """Rule `pad_read`: heads-up preflop (only the raiser and us left), facing an all-in or a raise over 12 BB, we
    fold or call. The raiser's profile comes only from the teacher pad at this hand (VPIP a/b, raises). Against a wide
    aggressor (6+ hands seen, VPIP 60%+, 3+ raises), calls with 77+ ATs+ AJo+ KQs are right at a price of at most 45%
    of our stack; against a very tight raiser (8+ hands seen, VPIP 12% or less), only QQ+ AKs AKo should call."""
    to_call = row["to_call"] or 0
    if row["street"] != "preflop" or to_call <= 0 or row["action"] not in ("call", "fold") or not raises:
        return None
    if not (row["facing_allin"] or to_call > PAD_BIG_CHIPS):
        return None
    raiser = raises[-1]["seat"]
    if raiser == row["seat"]:
        return None
    dealt = {r[0] for r in con.execute("SELECT seat FROM hole_cards WHERE match_id=? AND hand_number=?", (match_id, hand))}
    live = dealt - {r["seat"] for r in before if r["action"] == "fold"}
    if live != {raiser, row["seat"]}:
        return None
    profile = pad_profile(match_id, row["seat"], raiser, hand)
    if profile is None or not profile["hands"]:
        return None
    rate, code = profile["vpip"] / profile["hands"], row["hand_code"]
    if (profile["hands"] >= PAD_WIDE_MIN_HANDS and rate >= PAD_WIDE_MIN_VPIP and profile["raises"] >= PAD_WIDE_MIN_RAISES
            and row["action"] == "fold" and stack0 and to_call <= PAD_WIDE_PRICE_FRAC * stack0
            and in_range(code, PAD_WIDE_RANGE)):
        return verdict | {"rule": "pad_read", "preferred": "call", "agree": False, "raiser": raiser, "profile": profile}
    if (profile["hands"] >= PAD_TIGHT_MIN_HANDS and rate <= PAD_TIGHT_MAX_VPIP
            and row["action"] == "call" and not in_range(code, PAD_TIGHT_RANGE)):
        return verdict | {"rule": "pad_read", "preferred": "fold", "agree": False, "raiser": raiser, "profile": profile}
    return None


def _raise_first_in(row, before, raises, verdict) -> dict | None:
    """Rule `raise_first_in`: unopened pot (limpers allowed). CO/BTN/SB hands in the steal range raise 5-8 instead of
    folding, limping or oversizing; UTG/HJ never open-limp (fold, or raise to 5 with a good hand)."""
    if raises or (row["to_call"] or 0) > 2 or row["position"] is None:
        return None
    code, action, position = row["hand_code"], row["action"], row["position"]
    limpers = sum(1 for r in before if r["street"] == "preflop" and r["action"] == "call")
    if position in FIRST_IN_SEATS and in_range(code, FIRST_IN_RANGE):
        target = open_target(position, code, limpers)
        if action in ("fold", "call"):
            return verdict | {"rule": "raise_first_in", "preferred": "raise", "agree": False, "target_amount": target}
        if (action == "raise" and row["amount"] and row["amount"] > FIRST_IN_MAX_RAISE and row["stack_after"] != 0
                and code not in ("AA", "KK")):
            return verdict | {"rule": "raise_first_in", "preferred": "raise", "agree": False, "target_amount": target}
        return None
    if position in ("UTG", "HJ") and action == "fold" and in_range(code, EARLY_RAISE_RANGE[position]):
        return verdict | {"rule": "raise_first_in", "preferred": "raise", "agree": False, "target_amount": 5 + 2 * limpers}
    if action == "fold" and position not in ("BB", "HU_BB") and in_range(code, PREMIUM_FIRST_IN):
        target = open_target(position, code, limpers) if position in FIRST_IN_BASE else 5 + 2 * limpers
        return verdict | {"rule": "raise_first_in", "preferred": "raise", "agree": False, "target_amount": target}
    if position in ("UTG", "HJ") and action == "call":
        if in_range(code, EARLY_OPEN_RANGE):
            return verdict | {"rule": "raise_first_in", "preferred": "raise", "agree": False, "target_amount": 5}
        return verdict | {"rule": "raise_first_in", "preferred": "fold", "agree": False}
    return None


def _is_monster(hole: list[str], board: list[str]) -> bool:
    """A set (pocket pair), straight, flush, full house or better that uses a hole card. Two pair and trips made
    with one hole card on a paired board are strong, not monsters."""
    from pokerkit import Card, Label, StandardHighHand

    if strength_class(hole, board) != "strong":
        return False
    mine = StandardHighHand.from_game(list(Card.parse("".join(hole))), list(Card.parse("".join(board))))
    rank = _label_rank(mine.entry.label)
    if rank < _label_rank(Label.THREE_OF_A_KIND):
        return False
    if rank == _label_rank(Label.THREE_OF_A_KIND):
        return hole[0][0] == hole[1][0]
    if len(board) == 5:  # the board alone may already make the hand
        return mine > StandardHighHand.from_game([], list(Card.parse("".join(board))))
    return True


def _v6_postflop(con, rows, row, verdict, match_id, hand, seat) -> dict | None:
    want_value, want_cap = "monster_value" in EXPLOITS, "one_pair_pot_cap" in EXPLOITS
    if not (want_value or want_cap):
        return None
    action, to_call = row["action"], row["to_call"] or 0
    hole, board = _hole(con, match_id, hand, seat), _board_for(con, match_id, hand, row["street"])
    if not hole or not board:
        return None
    if want_value and (action == "fold" and to_call > 0 or action == "check" and to_call == 0) and _is_monster(hole, board):
        if action == "fold":
            if (row["facing_frac"] or 0) <= 1.0:
                return verdict | {"rule": "monster_value", "preferred": "call", "agree": False}
        else:
            return verdict | {"rule": "monster_value", "preferred": "bet", "target_frac": MONSTER_BET_FRAC, "agree": False}
        return None
    if not want_cap or action not in ("call", "raise", "bet") or strength_class(hole, board) == "strong":
        return None
    stack0 = row["stack_after"] - (row["chip_change"] or 0)
    own_bets = [r["idx"] for r in rows if r["idx"] < row["idx"] and r["street"] == row["street"]
                and r["seat"] == seat and r["action"] in ("bet", "raise")]
    was_raised = row["street"] in ("flop", "turn") and bool(own_bets) and any(
        own_bets[0] < r["idx"] < row["idx"] and r["street"] == row["street"] and r["seat"] != seat
        and r["action"] == "raise" for r in rows)
    required = to_call / ((row["pot_before"] or 0) + to_call) if to_call > 0 else 0
    if was_raised and required <= POT_CAP_RAISED_MIN_PRICE:
        was_raised = False  # a cheap raise: one pair still has the odds to see the next card
    if action in ("call", "raise") and to_call > 0 and ((row["facing_frac"] or 0) > 1.0 or was_raised) \
            and not _has_draw(hole, board):
        return verdict | {"rule": "one_pair_pot_cap", "preferred": "fold", "agree": False}
    if action in ("bet", "raise") and stack0 and -(row["chip_change"] or 0) > POT_CAP_STACK_FRAC * stack0:
        return verdict | {"rule": "one_pair_pot_cap", "preferred": "check" if to_call == 0 else "fold", "agree": False}
    return None


# ---- teacher v7 rules (each on only when named in EXPLOITS): GTO-style preflop ranges and postflop sizing ----
V7_MIN_BB = 25  # shallower than this the chart is shove-or-fold territory: v7 preflop rules abstain
RFI_GTO = {
    "UTG": "44+ ATs+ A5s A4s KTs+ QTs+ JTs AQo+ KQo",
    "HJ": "33+ A2s+ K9s+ Q9s+ J9s+ T9s 98s 87s 76s ATo+ KJo+ QJo",
    "CO": "22+ A2s+ K6s+ Q8s+ J8s+ T8s+ 97s+ 86s+ 75s+ 65s 54s A9o+ KTo+ QTo+ JTo",
    "BTN": "22+ A2s+ K2s+ Q4s+ J6s+ T6s+ 96s+ 85s+ 75s+ 64s+ 53s+ 43s A2o+ K8o+ Q9o+ J9o+ T9o 98o",
    "SB": "22+ A2s+ K2s+ Q5s+ J7s+ T7s+ 97s+ 86s+ 76s 65s 54s A2o+ K8o+ Q9o+ J9o+ T9o",
}
RFI_SIZE = {"UTG": 5, "HJ": 5, "CO": 5, "BTN": 5, "SB": 6}
ISO_RANGE = "22+ A2s+ KTs+ QJs JTs AJo+ KQo"
ISO_BASE, ISO_PER_LIMPER = 4, 2
THREE_BET_VALUE = {"UTG": "QQ+ AKs AKo", "HJ": "QQ+ AKs AKo", "CO": "JJ+ AQs+ AKo",
                   "BTN": "TT+ AQs+ AQo+ KQs", "SB": "99+ ATs+ AJo+ KQs"}
THREE_BET_MAX_OPEN = 12  # bigger opens belong to the big-price rules
POSTFLOP_ORDER = ("SB", "BB", "UTG", "HJ", "CO", "BTN")  # who acts last after the flop is in position
DEFEND_LATE = "22+ A2s+ K5s+ Q7s+ J8s+ T8s+ 97s+ 86s+ 75s+ 65s 54s A8o+ KTo+ QTo+ JTo T9o"
DEFEND_EARLY = "22+ A2s+ K9s+ QTs+ JTs T9s 98s 87s ATo+ KJo+"
DEFEND_BIG = "22+ A2s+ K9s+ QTs+ J8s+ T8s+ 97s+ 86s+ 75s+ 65s 54s A9o+ KJo+ QJo T9o"  # a 3x+ open: trash dropped
DEFEND_MARGINAL_OFFSUIT = "K9o Q9o J9o T8o 98o 87o 76o"  # near break-even against a late open at a good price
DEFEND_MAX_REQUIRED = 0.28
VS3_TOP = {"AA", "KK", "QQ", "AKs", "AKo"}
VS3_IP_CALL = {"JJ", "TT", "AQs"}
VS3_MAX_PRICE_FRAC, VS3_SMALL_PRICE_FRAC, VS3_4BET_FACTOR = 0.20, 0.10, 2.3
SIZE_BANDS = {"dry": (0.0, 0.6, 0.33, 0.5), "wet": (0.5, 0.9, 0.7, 0.75), "river": (0.0, 1.1, 0.6, 0.75)}  # lo, hi, target, cap
WEAK_VALUE_MAX_RANK = RANKS.index("J")
WEAK_VALUE_HU_BET_FRAC = 0.5  # heads up, a bet of half the pot or less is a fine stab
MIN_BET_CHIPS = 2  # the big blind


def _v7_preflop(row, before, raises, stack0):
    """(position, hand code, limpers) when a v7 preflop rule may judge this row, else None: preflop, a fold or call
    (or a non-all-in raise), a known seat, a real stack, and no all-in anywhere in front of us."""
    if row["street"] != "preflop" or not row["hand_code"] or row["position"] is None or (row["to_call"] or 0) <= 0:
        return None
    if row["facing_allin"] or any(r["stack_after"] == 0 for r in raises) or (row["effective_stack"] or stack0) / 2 < V7_MIN_BB:
        return None
    limpers = sum(1 for r in before if r["street"] == "preflop" and r["action"] == "call" and not raises)
    return row["position"], row["hand_code"], limpers


def _v7_verdict(verdict, rule, preferred, **extra) -> dict:
    return verdict | {"rule": rule, "preferred": preferred, "agree": False} | extra


def _rfi_gto(row, before, raises, stack0, verdict) -> dict | None:
    """Rule `rfi_gto`: first in with no limper. A fold or limp of a hand in the seat's chart is a raise to 5 (6 from
    the SB); an open-limp of a hand outside the chart is a fold. A fold of an outside hand is never corrected."""
    ctx = _v7_preflop(row, before, raises, stack0)
    if ctx is None or raises or row["action"] not in ("fold", "call"):
        return None
    position, code, limpers = ctx
    if limpers or position not in RFI_GTO:
        return None
    if in_range(code, RFI_GTO[position]):
        return _v7_verdict(verdict, "rfi_gto", "raise", target_amount=RFI_SIZE[position])
    return _v7_verdict(verdict, "rfi_gto", "fold") if row["action"] == "call" else None


def _iso_raise(row, before, raises, stack0, verdict) -> dict | None:
    """Rule `iso_raise`: limpers in front and nobody raised. A fold or limp behind with an isolation hand is a raise to
    4 + 2 per limper; limping behind with anything else is a fold."""
    ctx = _v7_preflop(row, before, raises, stack0)
    if ctx is None or raises or row["action"] not in ("fold", "call"):
        return None
    position, code, limpers = ctx
    if not limpers or position not in RFI_GTO:
        return None
    hi, _, kind = _parts(code)
    playable = in_range(code, ISO_RANGE) or (position in ("CO", "BTN") and (kind == "s" or hi == RANKS.index("A")))
    if playable:
        return _v7_verdict(verdict, "iso_raise", "raise", target_amount=ISO_BASE + ISO_PER_LIMPER * limpers)
    return _v7_verdict(verdict, "iso_raise", "fold") if row["action"] == "call" else None


def _single_open(row, before, raises):
    """The one non-all-in raise we face with no caller after it, or None."""
    if len(raises) != 1 or raises[0]["seat"] == row["seat"] or not raises[0]["amount"]:
        return None
    if any(r["action"] == "call" and r["idx"] > raises[0]["idx"] for r in before):
        return None
    return raises[0]


def _in_position(hero: str, other: str) -> bool | None:
    if hero not in POSTFLOP_ORDER or other not in POSTFLOP_ORDER:
        return None
    return POSTFLOP_ORDER.index(hero) > POSTFLOP_ORDER.index(other)


def _three_bet(row, before, raises, stack0, verdict) -> dict | None:
    """Rule `three_bet`: one open in front of us and no caller. A flat call or fold of the value range for the opener's
    seat is a 3-bet to 3x the open in position, 4x out of it. Bluff 3-bets are never forced."""
    ctx = _v7_preflop(row, before, raises, stack0)
    opener = _single_open(row, before, raises) if ctx else None
    if opener is None or row["action"] not in ("fold", "call") or opener["amount"] > THREE_BET_MAX_OPEN:
        return None
    spec, ip = THREE_BET_VALUE.get(opener["position"]), _in_position(ctx[0], opener["position"])
    if not spec or ip is None or not in_range(ctx[1], spec):
        return None
    return _v7_verdict(verdict, "three_bet", "raise", target_amount=round(opener["amount"] * (3 if ip else 4)))


def _bb_defend(row, before, raises, stack0, verdict) -> dict | None:
    """Rule `bb_defend`: big blind facing one open and no caller. A fold of a hand in the defend range for the opener's
    seat is a call when the price is 28% or less; a call with an offsuit hand outside the range is a fold."""
    ctx = _v7_preflop(row, before, raises, stack0)
    opener = _single_open(row, before, raises) if ctx else None
    if opener is None or ctx[0] != "BB" or row["action"] not in ("fold", "call") or opener["position"] is None:
        return None
    code, to_call = ctx[1], row["to_call"]
    required = to_call / (row["pot_before"] + to_call)
    cheap, early = required <= DEFEND_MAX_REQUIRED, opener["position"] in ("UTG", "HJ")
    inside = in_range(code, DEFEND_EARLY if early else DEFEND_LATE if cheap else DEFEND_BIG)
    if row["action"] == "fold":
        return _v7_verdict(verdict, "bb_defend", "call", required=round(required, 4)) if inside and cheap else None
    hi, lo, kind = _parts(code)
    if inside or hi == lo or kind != "o" or (cheap and not early and in_range(code, DEFEND_MARGINAL_OFFSUIT)):
        return None
    return _v7_verdict(verdict, "bb_defend", "fold", required=round(required, 4))


def _vs_3bet(row, before, raises, stack0, verdict) -> dict | None:
    """Rule `vs_3bet`: we opened and face a 3-bet that is not a jam, for under 20% of our stack, with nobody calling
    behind it. QQ+ and AK continue (AA/KK 4-bet when the price is small), JJ/TT/AQs call in position; every other hand is
    left to `field_call`."""
    ctx = _v7_preflop(row, before, raises, stack0)
    if ctx is None or len(raises) != 2 or raises[0]["seat"] != row["seat"] or raises[1]["seat"] == row["seat"]:
        return None
    if any(r["action"] == "call" and r["idx"] > raises[1]["idx"] for r in before):
        return None  # a cold call after the 3-bet: squeeze spot, the price is not what it looks like
    if row["stack_after"] == 0 or not stack0 or row["to_call"] > VS3_MAX_PRICE_FRAC * stack0:
        return None
    position, code, action = ctx[0], ctx[1], row["action"]
    if code in VS3_TOP:
        if code in ("AA", "KK") and row["to_call"] <= VS3_SMALL_PRICE_FRAC * stack0 and action in ("fold", "call"):
            return _v7_verdict(verdict, "vs_3bet", "raise", target_amount=round(VS3_4BET_FACTOR * raises[1]["amount"]))
        return _v7_verdict(verdict, "vs_3bet", "call") if action == "fold" else None
    if code in VS3_IP_CALL:
        return _v7_verdict(verdict, "vs_3bet", "call") if action == "fold" and _in_position(position, raises[1]["position"]) else None
    return None  # every other hand goes to `field_call`, which prices the spot against the field's 3-bet range


def _board_is_wet(board: list[str]) -> bool:
    """Two-tone flop, three of a suit later, or three ranks inside a four-rank window. Paired boards count as dry."""
    ranks = [RANKS.index(c[0]) + 2 for c in board]
    if len(set(ranks)) < len(ranks):
        return False
    suit_most = max(sum(1 for c in board if c[1] == s) for s in SUITS)
    if suit_most >= (2 if len(board) == 3 else 3):
        return True
    values = sorted(set(ranks) | ({1} if 14 in ranks else set()))
    return any(values[i + 2] - values[i] <= 3 for i in range(len(values) - 2))


def _postflop_sizing(con, row, verdict, match_id, hand, seat) -> dict | None:
    """Rule `postflop_sizing`: our own non-all-in bet. J-high or worse with no pair and no draw on the flop or turn is a
    check. Otherwise (a monster may size freely) the bet should be about 33% of the pot on a dry board, 66-75% on a wet
    one, never above 1.1x pot; sizes outside the band get the target and a cap (and a floor on wet boards)."""
    if row["action"] != "bet" or row["street"] not in BOARD_CARDS or row["stack_after"] == 0 or row["bet_frac"] is None:
        return None
    hole, board = _hole(con, match_id, hand, seat), _board_for(con, match_id, hand, row["street"])
    if not hole or not board:
        return None
    pot = row["pot_before"]
    if (row["street"] != "river" and strength_class(hole, board) == "air"
            and max(RANKS.index(c[0]) for c in hole) <= WEAK_VALUE_MAX_RANK
            and ((row["players_remaining"] or 0) >= 3 or row["bet_frac"] > WEAK_VALUE_HU_BET_FRAC)
            and not _has_gutshot(hole, board)):
        return _v7_verdict(verdict, "postflop_sizing", "check", reason="weak_value")
    if _is_monster(hole, board):
        return None
    texture = "river" if row["street"] == "river" else "wet" if _board_is_wet(board) else "dry"
    lo, hi, target, cap = SIZE_BANDS[texture]
    if lo <= row["bet_frac"] <= hi:
        return None
    legal_min = max(MIN_BET_CHIPS, round(lo * pot))
    stack_before = row["stack_after"] - (row["chip_change"] or 0)
    if legal_min > round(cap * pot) or (lo and stack_before <= legal_min):
        return None  # the legal minimum is out of reach: no size can meet the band
    extra = {"min_amount": round(lo * pot)} if lo else {}
    return _v7_verdict(verdict, "postflop_sizing", "bet", reason="overbet" if row["bet_frac"] > hi else "mis_size",
                       texture=texture, target_frac=target, target_amount=round(cap * pot), **extra)


def _has_gutshot(hole: list[str], board: list[str]) -> bool:
    """Four of the five ranks in some straight window are held (hole cards must be among them): a 4-out draw."""
    held = {RANKS.index(c[0]) + 2 for c in hole}
    ranks = {RANKS.index(c[0]) + 2 for c in list(hole) + list(board)}
    held |= {1} if 14 in held else set()
    ranks |= {1} if 14 in ranks else set()
    return any(len(set(range(low, low + 5)) & ranks) >= 4 and set(range(low, low + 5)) & held for low in range(1, 11))


# ---- field_call: preflop price check against the empirical field range (poker/ranges.py) ----
FIELD_RANGES_PATH = Path(__file__).resolve().parent.parent / "data" / "ranges" / "field_ranges.json"
FIELD_SAMPLES = 1000
FIELD_FOLD_MARGIN, FIELD_CALL_MARGIN, FIELD_BIG_CALL_MARGIN = 0.04, 0.06, 0.10
FIELD_BIG_STACK_FRAC = 0.40
# Fold-to-call only: raw heads-up equity overstates what a hand wins when it plays the flop out of position, so the
# blinds realize less of it; offsuit hands outside broadways and big aces are never turned into calls.
FIELD_REALIZE_OOP, FIELD_REALIZE_IP = 0.80, 0.92
FIELD_OFFSUIT_CALLS = "A9o+ KTo+ QTo+ JTo"
_FIELD_TABLE: dict | None = None
_FIELD_LOADED = False
_FIELD_EQ: dict = {}


def _field_table() -> dict | None:
    """The field range table, loaded once; None when the file is missing or unreadable."""
    global _FIELD_TABLE, _FIELD_LOADED
    if not _FIELD_LOADED:
        _FIELD_LOADED = True
        try:
            _FIELD_TABLE = json.loads(FIELD_RANGES_PATH.read_text())
        except (OSError, ValueError):
            _FIELD_TABLE = None
    return _FIELD_TABLE


def _field_equity(table: dict, code: str, situation: str, position: str, size: str | None) -> float:
    key = (code, situation, position, size)
    if key not in _FIELD_EQ:
        from poker import ranges

        probs = ranges.range_probs(table, situation, position, size)
        _FIELD_EQ[key] = ranges.equity_vs_range(ranges.expand(code)[0], probs, n=FIELD_SAMPLES, seed=11)
    return _FIELD_EQ[key]


def _field_spot(row, before, raises) -> tuple[str, str, str | None] | None:
    """(situation, raiser position, size bucket) when we face exactly one live aggressor with nobody else in the pot,
    else None. One open -> `open`, a 3-bet -> `3bet`, an all-in or a 4-bet and up -> `jam`."""
    pre = [r for r in before if r["street"] == "preflop"]
    if not raises or raises[-1]["seat"] == row["seat"] or raises[-1]["position"] is None:
        return None
    if any(r["action"] == "call" for r in pre):
        return None  # a limper or a caller: multiway, the heads-up price is wrong
    last = raises[-1]
    if row["facing_allin"] or last["stack_after"] == 0 or len(raises) >= 3:
        return "jam", last["position"], None
    if len(raises) == 1:
        from poker.ranges import size_bucket

        return "open", last["position"], size_bucket(last["amount"] or 0)
    return "3bet", last["position"], None


def _field_call(row, before, raises, stack0, verdict) -> dict | None:
    """Rule `field_call`: preflop, one aggressor in front and nobody else in. Our equity against the field's range for
    that spot is compared with the pot odds. A call or raise well below the price is a fold; a fold well above it is a
    call (above 40% of our stack the fold needs 10 points of cushion, so huge prices stay folds)."""
    to_call = row["to_call"] or 0
    if (row["street"] != "preflop" or not row["hand_code"] or to_call <= 0 or not stack0
            or row["action"] not in ("fold", "call", "raise")):
        return None
    spot = _field_spot(row, before, raises)
    table = _field_table() if spot else None
    if not table:
        return None
    situation, position, size = spot
    equity = _field_equity(table, row["hand_code"], situation, position, size)
    required = to_call / (row["pot_before"] + to_call)
    extra = {"equity": round(equity, 4), "required": round(required, 4), "situation": situation, "raiser_position": position}
    if row["action"] in ("call", "raise"):
        return _v7_verdict(verdict, "field_call", "fold", **extra) if equity < required - FIELD_FOLD_MARGIN else None
    code = row["hand_code"]
    if code.endswith("o") and code not in _field_offsuit_calls():
        return None
    realize = FIELD_REALIZE_OOP if row["position"] in ("SB", "BB") else FIELD_REALIZE_IP
    margin = FIELD_BIG_CALL_MARGIN if to_call > FIELD_BIG_STACK_FRAC * stack0 else FIELD_CALL_MARGIN
    return _v7_verdict(verdict, "field_call", "call", **extra) if equity * realize > required + margin else None


_OFFSUIT_CALLS: set | None = None


def _field_offsuit_calls() -> set:
    global _OFFSUIT_CALLS
    if _OFFSUIT_CALLS is None:
        _OFFSUIT_CALLS = set(_range_set(FIELD_OFFSUIT_CALLS))
    return _OFFSUIT_CALLS


V7_PREFLOP = (("rfi_gto", _rfi_gto), ("iso_raise", _iso_raise), ("three_bet", _three_bet), ("bb_defend", _bb_defend),
              ("vs_3bet", _vs_3bet), ("field_call", _field_call))


def agrees(verdict: dict, action: str, amount=None, *, min_amount=None) -> bool:
    """Whether a (re)played action matches the verdict. With a `target_amount`, it must be a raise or bet no larger
    than that amount, and at least `min_amount` (the legal minimum) when that is known."""
    preferred = verdict.get("preferred")
    if min_amount is None:
        min_amount = verdict.get("min_amount")
    if not (action == preferred or (preferred == "raise" and action == "bet")):
        return False
    cap = verdict.get("target_amount")
    if cap is None:
        return True
    return action in ("raise", "bet") and amount is not None and amount <= cap and (min_amount is None or amount >= min_amount)
