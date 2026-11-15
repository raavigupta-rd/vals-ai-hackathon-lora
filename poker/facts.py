"""Code-computed fact line for one action turn, and a conservative check of a reply's claims against it.

The arena plays the official harness unmodified, so nothing can be injected into its prompts. The model has to learn
to compute these facts itself; training targets get the line as a header (poker.sft.make_records --fact-header).

    fact_line(messages) -> "Facts: BTN (2 behind), AJs, board -, no made hand, pot 3, call 2 (40%), stack 98 BB,
                            rank 3/6 (+12 to 2nd, -40 to 4th), 18 hands left."
    contradictions(reasoning, facts) -> ["position: ...", ...]  (first-person claims the cards or seat disprove)

Position comes from the prompt's "Your seat: s (position p; 0 = small blind)" and the live ring (busted seats are
skipped), so it does not depend on the button marker. Chips are stack plus the bet already in front.
"""
from __future__ import annotations

import re

from poker.hints import parse_prompt, text_content
from poker.teacher import RANKS, SUITS, _board_category, _label_rank, hand_code

BB = 2


# ---- facts -----------------------------------------------------------------------------------------------------

def position_names(n: int) -> list[str]:
    """Names by position index (0 = small blind, 1 = big blind, then UTG ... button) for n live players."""
    if n <= 1:
        return ["BTN"][:n]
    if n == 2:
        return ["SB/BTN", "BB"]
    seats = {1: ["BTN"], 2: ["UTG", "BTN"], 3: ["UTG", "CO", "BTN"], 4: ["UTG", "HJ", "CO", "BTN"]}[n - 2]
    return ["SB", "BB"] + seats


def _ordinal(k: int) -> str:
    return f"{k}{'th' if 10 <= k % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(k % 10, 'th')}"


def _straight_windows(ranks: set[int]):
    ranks = set(ranks) | ({1} if 14 in ranks else set())
    for low in range(1, 11):
        window = set(range(low, low + 5))
        if window <= ranks:
            yield {14 if r == 1 else r for r in window}


def _val(card: str) -> int:
    return RANKS.index(card[0]) + 2


def _owns_flush(hole, board) -> bool:
    for suit in SUITS:
        mine = [c for c in hole if c[1] == suit]
        theirs = [c for c in board if c[1] == suit]
        if mine and len(mine) + len(theirs) >= 5:
            if len(theirs) < 5 or max(map(_val, mine)) > min(map(_val, theirs)):
                return True
    return False


def _owns_straight(hole, board) -> bool:
    allr, boardr = {_val(c) for c in hole + board}, {_val(c) for c in board}
    return any(not w <= boardr for w in _straight_windows(allr))


def _beats_board(hole: list[str], board: list[str]) -> bool:
    """On a five-card board: do our cards make a stronger hand than the board alone?"""
    from pokerkit import Card, StandardHighHand

    mine = StandardHighHand.from_game(list(Card.parse("".join(hole))), list(Card.parse("".join(board))))
    return mine > StandardHighHand.from_game([], list(Card.parse("".join(board))))


def made_hand(hole: list[str], board: list[str]) -> tuple[str | None, bool]:
    """(name, ambiguous). None = nothing made with our cards (a pair on the board is not ours)."""
    from pokerkit import Card, Label, StandardHighHand

    pocket = hole[0][0] == hole[1][0]
    if not board:
        return ("pocket pair" if pocket else None), False
    label = StandardHighHand.from_game(list(Card.parse("".join(hole))), list(Card.parse("".join(board)))).entry.label
    cat = _label_rank(label)
    board_cat = (_label_rank(StandardHighHand.from_game([], list(Card.parse("".join(board)))).entry.label)
                 if len(board) == 5 else _board_category(board))
    if cat <= board_cat and not (len(board) == 5 and cat == board_cat and cat > _label_rank(Label.ONE_PAIR)
                                 and _beats_board(hole, board)):
        if cat < board_cat or cat <= _label_rank(Label.HIGH_CARD):
            return None, False
        # equal category: our cards add nothing to the made hand the board already has
        if cat == _label_rank(Label.ONE_PAIR):
            return "board pair only", False
        return "playing the board", cat >= _label_rank(Label.TWO_PAIR)
    names = {Label.STRAIGHT_FLUSH: "straight flush", Label.FOUR_OF_A_KIND: "quads", Label.FULL_HOUSE: "full house",
             Label.FLUSH: "flush", Label.STRAIGHT: "straight", Label.TWO_PAIR: "two pair"}
    if label in names:
        return names[label], False
    if label == Label.THREE_OF_A_KIND:
        return ("set" if pocket else "trips"), False
    ranks = sorted({_val(c) for c in board}, reverse=True)
    h = [_val(c) for c in hole]
    if pocket:
        return ("overpair" if h[0] > ranks[0] else "pocket pair"), False
    matched = [v for v in h if v in ranks]
    if not matched:
        return None, False
    v = matched[0]
    return ("top pair" if v == ranks[0] else "bottom pair" if v == ranks[-1] else "second pair"), False


def draws(hole: list[str], board: list[str]) -> dict | None:
    """Flush and straight draws on the flop and turn, with outs counted by trying every unseen card."""
    if len(board) not in (3, 4):
        return None
    flush_now, straight_now = _owns_flush(hole, board), _owns_straight(hole, board)
    if flush_now or straight_now:
        return None
    seen = set(hole + board)
    flush, straight = set(), set()
    for r in RANKS:
        for s in SUITS:
            card = r + s
            if card in seen:
                continue
            if _owns_flush(hole, board + [card]):
                flush.add(card)
            if _owns_straight(hole, board + [card]):
                straight.add(card)
    if not flush and not straight:
        return None
    parts = (["flush draw"] if flush else []) + (
        ["open-ended straight draw" if len({c[0] for c in straight}) >= 2 else "gutshot"] if straight else [])
    return {"name": " + ".join(parts), "outs": len(flush | straight), "flush_outs": len(flush),
            "straight_outs": len(straight)}


def _hands(full: str) -> tuple[int, int] | None:
    found = re.findall(r"^## Hand (\d+) of (\d+)", full, re.M)
    return (int(found[-1][0]), int(found[-1][1])) if found else None


def _legal_line(prompt) -> str | None:
    """The prompt's last "Legal actions: ..." line, exactly as written."""
    texts = [text_content(m) for m in prompt]
    current = next((t for t in reversed(texts) if re.search(r"^Street: ", t, re.M)), "")
    found = re.findall(r"^Legal actions: (.+)$", current, re.M)
    return found[-1].strip() if found else None


def compute_facts(prompt) -> dict | None:
    state = parse_prompt(prompt)
    if not state:
        return None
    full = "\n".join(text_content(m) for m in prompt)
    seats = [s for s in state["seats"] if s["status"] != "out"]
    ring = sorted(s["seat"] for s in seats)
    n = len(ring)
    if state["seat"] not in ring or not 0 <= state["position"] < n:
        return None
    hero_idx = ring.index(state["seat"])
    sb_idx = (hero_idx - state["position"]) % n
    pos_of = {seat: (i - sb_idx) % n for i, seat in enumerate(ring)}
    names = position_names(n)
    if state["street"] == "preflop":
        order = (list(range(2, n)) + [0, 1]) if n > 2 else [0, 1]
    else:
        order = list(range(n)) if n > 2 else [1, 0]  # heads up the SB/button acts first preflop but last postflop
    by_seat = {s["seat"]: s for s in seats}
    hero_order = order.index(state["position"])
    behind = sum(1 for seat, p in pos_of.items() if seat != state["seat"] and order.index(p) > hero_order
                 and by_seat[seat]["status"] not in ("folded", "all-in"))
    hero = by_seat[state["seat"]]
    chips = hero["stack"] + hero["bet"]
    totals = sorted((s["stack"] + s["bet"] for s in seats), reverse=True)
    higher = sum(t > chips for t in totals)
    ties = sum(t == chips for t in totals)
    rank = higher + 1
    gap_up = totals[higher - 1] - chips if higher else None
    gap_down = chips - totals[higher + ties] if higher + ties < n else None
    made, ambiguous = made_hand(state["hole"], state["board"])
    hands = _hands(full)
    to_call, pot = state["to_call"], state["pot"]
    return dict(position=names[state["position"]], pos_num=state["position"], hand=hands[0] if hands else None,
                hands_total=hands[1] if hands else None, legal=_legal_line(prompt), behind=behind, hole_code=hand_code(state["hole"]),
                hole=state["hole"], board=state["board"], street=state["street"], made=made, ambiguous=ambiguous,
                draw=draws(state["hole"], state["board"]), pot=pot, to_call=to_call,
                required=to_call / (pot + to_call) if to_call > 0 else 0.0, chips=chips, stack_bb=chips / BB,
                rank=rank, n_live=n, gap_up=gap_up, gap_down=gap_down, ties=ties,
                hands_left=(hands[1] - hands[0]) if hands else None)


def render_facts(f: dict) -> str:
    made = f["made"] or ("no made hand")
    if f["draw"]:
        made += f", {f['draw']['name']} {f['draw']['outs']} outs"
    price = f"call {f['to_call']:g} ({f['required']:.0%})" if f["to_call"] > 0 else "no bet"
    rank = f"rank {f['rank']}/{f['n_live']}"
    gaps = ([f"+{f['gap_up']} to {_ordinal(f['rank'] - 1)}"] if f["gap_up"] else []) + (
        [f"-{f['gap_down']} to {_ordinal(f['rank'] + f['ties'])}"] if f["gap_down"] else [])
    if gaps:
        rank += f" ({', '.join(gaps)})"
    board = " ".join(f["board"]) or "-"
    parts = [f"{f['position']} ({f['behind']} behind)", f["hole_code"], f"board {board}", made, f"pot {f['pot']}", price,
             f"stack {round(f['stack_bb'])} BB", rank]
    if f["hands_left"] is not None:
        parts.append(f"{f['hands_left']} hands left")
    return "Facts: " + ", ".join(parts) + "."


def render_facts_v2(f: dict) -> str:
    """Facts v2: only fields the model can copy from the prompt or compute trivially. No rank, gap, position name or
    seats behind. "Facts: pos P of N (0=SB, 1=BB), c1 c2 (class), board ..., made, pot X, call Y (Z%) | no bet,
    legal <as listed, joined with ' | '>, stack S BB, hand H/30." """
    made = f["made"] or "no made hand"
    if f["draw"]:
        made += f", {f['draw']['name']} {f['draw']['outs']} outs"
    price = f"call {f['to_call']:g} ({f['required']:.0%})" if f["to_call"] > 0 else "no bet"
    parts = [f"pos {f['pos_num']} of {f['n_live']} (0=SB, 1=BB)", f"{' '.join(f['hole'])} ({f['hole_code']})",
             f"board {' '.join(f['board']) or '-'}", made, f"pot {f['pot']}", price]
    if f.get("legal"):
        parts.append("legal " + " | ".join(a.strip() for a in f["legal"].split(",")))
    parts.append(f"stack {round(f['stack_bb'])} BB")
    if f.get("hand") is not None:
        parts.append(f"hand {f['hand']}/{f['hands_total']}")
    return "Facts: " + ", ".join(parts) + "."


def fact_line_v2(prompt) -> str | None:
    f = compute_facts(prompt)
    return render_facts_v2(f) if f else None


def fact_line(prompt) -> str | None:
    f = compute_facts(prompt)
    return render_facts(f) if f else None


# ---- hallucination check ---------------------------------------------------------------------------------------

_POS = {
    "UTG": r"(?-i:UTG)(?!\s*\+)|under the gun", "HJ": r"(?-i:HJ)|hi-?jack", "CO": r"(?-i:CO)|cut-?off",
    "BTN": r"(?-i:BTN)|button|dealer seat", "SB": r"(?-i:SB)|small blind", "BB": r"(?-i:BB)|big blind",
    "EARLY": r"early position", "MIDDLE": r"middle position|mid position", "LATE": r"late position",
}
_POS_OK = {"UTG": {"UTG"}, "HJ": {"HJ"}, "CO": {"CO"}, "BTN": {"BTN"}, "SB": {"SB"}, "BB": {"BB"},
           "EARLY": {"UTG", "HJ"}, "MIDDLE": {"HJ", "CO"}, "LATE": {"CO", "BTN"}}
_POS_ANY = "|".join(f"(?P<{k}>{v})" for k, v in _POS.items())
_OTHERS = r"(?:facing|\bvs\b|versus|against|raise|bet|\bby\b|call|check|shove|jam|they|\bhe\b|she|opponent|villain|player|seat|others|everyone|behind|them|their|limp)"
_POS_PATTERNS = [
    re.compile(r"\b(?:I(?:'m|’m| am)|we(?:'re| are))\s+(?:currently |now |sitting |playing |acting |opening |first to act )*"
               r"(?:(?:on|in|at|from)\s+)?(?:the\s+)?(?:" + _POS_ANY + r")\b", re.I),
    re.compile(r"\bmy position(?: is|:| =)?\s*(?:the\s+)?(?:" + _POS_ANY + r")\b", re.I),
    re.compile(r"\bI\b(?:(?!" + _OTHERS + r")[^.;\n]){0,40}?\b(?:from|in|on|at)\s+(?:the\s+)?(?:" + _POS_ANY + r")\b", re.I),
    re.compile(r"\b(?:as|being)\s+(?:(?:on|in)\s+)?(?:the\s+)?(?:" + _POS_ANY + r")\s*,?\s+I\b", re.I),
]

_HAVE = (r"\bI(?:'ve|’ve|\s+(?:have|hold|made|hit|flopped|turned|rivered|got|now have|only have|just have|"
         r"actually have|do have|also have|am holding|'m holding)|'m holding)\s+(?:(?:got|made|hit|flopped)\s+)?")
_DET = r"(?:(?:a|an|the|my|one)\s+)?(?:(?:nut|nuts|made|strong|weak|small|big|medium|solid)\s+)*"
_RANKWORD = "aces|kings|queens|jacks|tens|nines|eights|sevens|sixes|fives|fours|threes|twos|deuces"
_CLAIMS = [
    ("full house", r"full house|a boat"), ("quads", r"quads|four of a kind"),
    ("set", r"set(?:\s+of\s+\w+)?\b(?!\s*up)"), ("trips", r"trips|three[- ]of[- ]a[- ]kind"),
    ("flush", r"flush(?!\s*(?:draw|possib|on the board))"),
    ("straight", r"straight(?!\s*(?:draw|flush draw|possib|on the board))"),
    ("two pair", r"two[- ]pair"), ("top pair", r"top[- ]pair|tptk"), ("overpair", r"over[- ]?pair"),
    ("pocket", r"pocket (?:pair|" + _RANKWORD + r")"),
    ("pair", r"(?:middle |second |bottom |weak |small )?pair\b"),
]
_CLAIM_RE = re.compile(_HAVE + _DET + "(?:" + "|".join(f"(?P<c{i}>{p})" for i, (_, p) in enumerate(_CLAIMS)) + ")", re.I)
_STRONG = {"top pair", "overpair", "two pair", "set", "trips", "straight", "flush", "full house", "quads", "straight flush"}
_ALLOWED = {
    "full house": {"full house", "quads", "straight flush"}, "quads": {"quads", "straight flush"},
    "set": {"set", "trips", "full house", "quads"}, "trips": {"set", "trips", "full house", "quads"},
    "flush": {"flush", "straight flush"}, "straight": {"straight", "straight flush"},
    "two pair": {"two pair", "full house", "quads"},
    "top pair": _STRONG, "overpair": _STRONG - {"top pair"},
}
_HYPO = re.compile(r"\b(?:if|when|unless|whether|hoping|hope|wish|need|want|should|could|would|might|maybe|suppose|"
                   r"assuming|until|not|no|never)\b|n't", re.I)
_CODE = re.compile(_HAVE + r"(?:(?:a|an|the)\s+)?(?-i:(?P<hi>[2-9TJQKA])(?P<lo>[2-9TJQKA])(?P<kind>[so])?)\b", re.I)
_PAIR_S = re.compile(r"\b(?-i:([2-9TJQKA])\1s)\b")
_OUTS = re.compile(r"\b(\d+)\s+(?:clean |live |direct |real )?outs?\b", re.I)
_SKIP_OUTS = re.compile(r"opponent|villain|\bthey\b|\bhe\b|\bshe\b|backdoor|overcard|\bpair\b|\bset\b|trips|two pair|full house|"
                        r"\bace\b|\bking\b|\bqueen\b", re.I)


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


_PAST = re.compile(r"\b(?:last|previous|earlier|prior|past)\s+(?:hand|round|orbit)s?\b|\bhand\s*#?\d+\b(?!\s+of)", re.I)


def _position_claims(sentence: str) -> list[str]:
    if _PAST.search(sentence):
        return []  # about an earlier hand
    found = []
    for pattern in _POS_PATTERNS:
        for match in pattern.finditer(sentence):
            found.append((match.start(), next(k for k in _POS if match.group(k))))
    return [claimed for _, claimed in sorted(found)]


def _check_hand(sentence: str, facts: dict, out: list[tuple[str, str | None]]) -> None:
    code = facts["hole_code"]
    for match in _CLAIM_RE.finditer(sentence):
        if _HYPO.search(sentence[:match.start()]):
            continue
        kind = next(_CLAIMS[i][0] for i in range(len(_CLAIMS)) if match.group(f"c{i}"))
        made = facts["made"]
        if kind == "pocket":
            bad = code[0] != code[1]
        elif kind == "pair":
            bad = made in (None, "board pair only", "playing the board")
        else:
            bad = made not in _ALLOWED[kind]
        out.append(("made", f"hand: claims {match.group(0).strip()!r}, cards make {made or 'no made hand'}"
                    if bad and not facts["ambiguous"] else None))
    for match in _CODE.finditer(sentence):
        if _HYPO.search(sentence[:match.start()]):
            continue
        hi, lo, kind = match["hi"], match["lo"], match["kind"]
        ranks = sorted([hi, lo], key=RANKS.index, reverse=True)
        claimed = "".join(ranks) + ("" if hi == lo else (kind or ""))
        if kind is None and hi != lo:
            continue  # a bare "AJ" says nothing about suits
        if re.search(r"off-?suit|unsuited", sentence, re.I):
            continue  # "JQs (offsuit)": the writer is contradicting their own suffix
        if hi == lo and kind == "s":
            out.append(("code", f"suited: {hi}{lo}s is a pair, which cannot be suited"))
        elif hi != lo and claimed != code and claimed[:2] == code[:2]:
            out.append(("code", f"suited: says {claimed} but the hand is {code}"))
        elif hi != lo and claimed[:2] != code[:2]:
            out.append(("code", f"hand: says {claimed} but the hand is {code}"))
        else:
            out.append(("code", None))


def _check_outs(sentence: str, facts: dict, out: list[tuple[str, str | None]]) -> None:
    if facts["street"] not in ("flop", "turn") or _SKIP_OUTS.search(sentence):
        return
    low = sentence.lower()
    flush = "flush" in low
    straight = bool(re.search(r"straight|gutshot|open-ended|oesd", low))
    if not (flush or straight or "draw" in low):
        return
    draw = facts["draw"] or {"outs": 0, "flush_outs": 0, "straight_outs": 0}
    truth = draw["flush_outs"] if flush and not straight else draw["straight_outs"] if straight and not flush else draw["outs"]
    for match in _OUTS.finditer(sentence):
        out.append(("outs", f"outs: says {match[1]} but the cards have {truth}" if abs(int(match[1]) - truth) > 2 else None))


_RANK_WORDS = {"A": "aces?", "K": "kings?", "Q": "queens?", "J": "jacks?", "T": "tens?", "9": "nines?", "8": "eights?",
               "7": "sevens?", "6": "sixes?", "5": "fives?", "4": "fours?", "3": "threes?|trey", "2": "twos?|deuces?"}
_SUIT_WORD = r"(?:(?P<off>off-?suit(?:ed)?|unsuited)|(?P<on>(?<![-\w])suited))\b"
_SUIT_GAP = r"[\s,:(\-\u2014]*(?:(?:is|are|a|an|the|hand|combo|here|it's|its|which is|but|only|just|as)\s+){0,3}"
_SUIT_FIRST = [
    re.compile(r"\b(?:my|our)\s+(?:starting\s+|hole\s+)?(?:hand|cards|holding)\b[^.;\n]{0,30}?\b" + _SUIT_WORD, re.I),
    re.compile(r"\bI(?:'ve|\u2019ve|\s+(?:have|hold|got|was dealt)|\s+am holding|'m holding)\s+(?:got\s+)?(?:(?:an?|the)\s+)?"
               r"(?:[\w-]+\s+){0,3}?\b" + _SUIT_WORD, re.I),
]
_SUIT_OPPONENT = re.compile(r"\b(?:opponents?|villains?|seat\s*\d|they|he|she|their|his|her|him|players?|others|everyone|shows?|showed)\b", re.I)
_SUIT_GENERIC = re.compile(r"\b(?:than|vs|versus|compared|better|worse|stronger|weaker|beats?|dominat\w*|range|connectors?)\b", re.I)


def _rank_alias(rank: str) -> str:
    return f"(?-i:{rank})|{_RANK_WORDS[rank]}"


def _suit_patterns(hole: list[str]) -> list[re.Pattern]:
    """Phrases that name our own two cards (exact cards, rank letters or words), each followed by a suited/offsuit word."""
    (a, b) = sorted(hole, key=lambda c: RANKS.index(c[0]), reverse=True)
    hi, lo = f"(?:{_rank_alias(a[0])})", f"(?:{_rank_alias(b[0])}" + ("|rags?|junk|trash" if RANKS.index(b[0]) < RANKS.index("T") else "") + ")"
    exact = rf"(?:\b{a}[\s,/&-]*{b}|\b{b}[\s,/&-]*{a})\b"
    ranks = rf"(?<!\w)(?:{hi}[- ]?{lo}|{lo}[- ]?{hi})(?!\w)"
    return [re.compile(f"(?:{exact}|{ranks}){_SUIT_GAP}{_SUIT_WORD}", re.I)]


def _check_suit(sentence: str, facts: dict, out: list[str]) -> None:
    """Claims that our own hole cards are suited/offsuit. Only the last such claim in the reply counts (see caller)."""
    hole = facts["hole"]
    if hole[0][0] == hole[1][0] or _SUIT_GENERIC.search(sentence):
        return
    matches = [m for p in _suit_patterns(hole) + _SUIT_FIRST for m in p.finditer(sentence)]
    matches = [m for m in matches if not _SUIT_OPPONENT.search(sentence[:m.start()]) and not _HYPO.search(sentence[:m.start()])]
    kinds = {"off" if m.group("off") else "on" for m in matches}
    if len(kinds) == 1:
        out.append(kinds.pop())


_CORRECTING = re.compile(r"\b(?:wait|actually|correction)\b|\bno,", re.I)


def contradictions(reasoning: str, facts: dict | None) -> list[str]:
    """First-person claims in `reasoning` that the seat or cards in `facts` disprove. Conservative: unsure means silent."""
    if not facts or not reasoning:
        return []
    checked: list[tuple[str, str | None]] = []
    claims: list[str] = []
    suit_claims: list[str] = []
    for sentence in _sentences(reasoning):
        if _CORRECTING.search(sentence):
            continue  # "wait", "actually", "no,": the writer is fixing an earlier claim
        claims += _position_claims(sentence)
        _check_suit(sentence, facts, suit_claims)
        _check_hand(sentence, facts, checked)
        _check_outs(sentence, facts, checked)
    # Like position, only the last stated claim of each kind counts.
    last = {kind: message for kind, message in checked}
    out = [m for m in last.values() if m]
    # Only the last stated position counts: "I'm in the SB... wait, I'm the BTN" is a self-correction.
    if claims and facts.get("position") and not _POS_OK[claims[-1]] & set(facts["position"].split("/")):
        out.append(f"position: says {claims[-1]} but we are {facts['position']}")
    truly_suited = facts["hole"][0][1] == facts["hole"][1][1]
    if suit_claims and (suit_claims[-1] == "on") != truly_suited:
        out.append(f"suited: says {'suited' if suit_claims[-1] == 'on' else 'offsuit'} but {' '.join(facts['hole'])} is "
                   f"{'suited' if truly_suited else 'offsuit'}")
    pair = re.fullmatch(r"([2-9TJQKA])\1", facts["hole_code"])
    if pair and re.search(r"\b" + pair[1] * 2 + r"s\b", reasoning) and not any(c.startswith("suited") for c in out):
        out.append("suited: pocket pair called suited")
    if pair and re.search(r"\b(?:pair suited|suited pair)\b", reasoning, re.I) and not any(c.startswith("suited") for c in out):
        out.append("suited: pocket pair called suited")
    return out


def main() -> None:
    """Count replies whose reasoning contradicts the facts, per deployment, with examples.

        uv run python -m poker.facts --joined data/arena/joined.jsonl --deployment infer_d3d2b6b92171 --show 5
    """
    import argparse
    import collections
    import json
    from pathlib import Path

    from poker.sft.make_records import _call

    parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--joined", type=Path, default=Path("data/arena/joined.jsonl"))
    parser.add_argument("--deployment", action="append", help="repeatable; default: every deployment")
    parser.add_argument("--show", type=int, default=0, help="print this many example contradictions per deployment")
    args = parser.parse_args()
    seen: dict = collections.defaultdict(lambda: collections.Counter())
    examples: dict = collections.defaultdict(list)
    for line in args.joined.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("source") != "arena" or r.get("kind") != "action" or r.get("cutoff") or _call(r) is None:
            continue
        if args.deployment and r.get("deployment") not in args.deployment:
            continue
        c = seen[r["deployment"]]
        c["replies"] += 1
        facts = compute_facts(r["messages"])
        if facts is None:
            c["no_facts"] += 1
            continue
        found = contradictions(r.get("reasoning") or "", facts)
        c["with_facts"] += 1
        if found:
            c["contradicted"] += 1
            for kind in {x.split(":")[0] for x in found}:
                c["kind_" + kind] += 1
            if len(examples[r["deployment"]]) < args.show:
                examples[r["deployment"]].append((found, (r.get("reasoning") or "")[:300]))
    for dep, c in seen.items():
        print(dep, json.dumps(dict(c)))
        for found, text in examples[dep]:
            print("  ", found, "|", text.replace("\n", " "))


if __name__ == "__main__":
    main()
