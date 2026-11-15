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
    spec = CONTINUE_VS_3BET if raises_before >= 2 or big else CONTINUE_VS_RAISE
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
    if n >= POSTFLOP_MIN:
        return own, n
    both = classify(con.execute(query, args))
    return both, n


def class_distribution(counts: dict) -> dict:
    """Add-one smoothed share of each class."""
    total = sum(counts.values()) + len(CLASSES)
    return {k: (counts.get(k, 0) + 1) / total for k in CLASSES}


@lru_cache(maxsize=4096)
def _equity_classes(hole: tuple[str, ...], board: tuple[str, ...], dist: tuple[float, ...], samples: int) -> float:
    known = set(hole) | set(board)
    by_class: dict = {k: [] for k in CLASSES}
    cards = [r + s for r in RANKS for s in SUITS if r + s not in known]
    for i, c1 in enumerate(cards):
        for c2 in cards[i + 1 :]:
            by_class[strength_class([c1, c2], list(board))].append((c1, c2))
    live = [(k, by_class[k], p) for k, p in zip(CLASSES, dist) if by_class[k]]
    if not live:
        return 0.5
    total = sum(p for _, _, p in live)
    villain, weights = [], []
    for _, group, p in live:
        villain += group
        weights += [p / total / len(group)] * len(group)
    return _simulate(hole, villain, board, samples, repr((hole, board, dist, samples)), weights)


CALL_EXTRA = 0.04  # extra margin for call corrections on the flop and turn


def postflop_call_verdict(hole, counts, board, *, price, risks_bust, hand_number, band=POSTFLOP_BAND,
                          samples=600) -> dict:
    dist = class_distribution(counts)
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
    if row["street"] in BOARD_CARDS and row["to_call"] > 0 and actual in ("call", "fold"):
        return _judge_postflop(con, rows, row, verdict, match_id, hand, seat)
    if row["street"] != "preflop" or not row["hand_code"]:
        return verdict
    before = [r for r in rows if r["idx"] < row["idx"]]
    raises = [r for r in before if r["street"] == "preflop" and r["action"] in ("raise", "bet")]
    stack0 = row["stack_after"] - (row["chip_change"] or 0)
    if actual == "raise" and row["stack_after"] == 0 and not (raises and raises[-1]["stack_after"] == 0):
        preferred = open_shove_verdict(row["hand_code"], (row["effective_stack"] or stack0) / 2)
        agree = None if preferred is None else preferred == actual
        return verdict | {"rule": "own_shove", "preferred": preferred, "agree": agree}
    if not raises and row["to_call"] <= 2:
        preferred = open_verdict(row["hand_code"], row["position"], actual, to_call=row["to_call"])
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
