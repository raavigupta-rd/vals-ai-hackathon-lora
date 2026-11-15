"""Deterministic teacher scratchpad: parse a cleanup prompt, update counts, render a fixed-schema pad.

Pad schema (head-heavy: the event cuts the tail, so the facts that matter come first):

    H4/30 | me seat4 203 | rank 2/6
    seat0: 199 | VPIP 1/1 | raises 1 | showed: Kc Ah@H4 | talk-claims: 0 false
    ...one line per live opponent...
    out: seat5@H3                       (only when someone busted)
    plan: short: avoid marginal stack-offs

Counts accumulate across hands by parsing the previous pad back with `parse_pad`. A previous pad that is not in
this schema (the model's own free text, "(empty)"), from another seat, or from a hand not before this one is
ignored and the counts restart from this hand. VPIP is "voluntarily put chips in preflop" (call or raise; a
big-blind check is not) over hands the seat was dealt into. "raises" counts every bet or raise. A talk claim is
false when a seat claimed a made hand in table talk and then showed a weaker hand at showdown.

Pad v2 (opt-in: `teacher_pad(..., version=2)`) is head-heavy and exploit-driven, at most 1500 chars:

    H4/30 | me seat4 203 (101 BB) | rank 2/6 | gap up +3, down -3 | 26 left
    seat0: 199 | VPIP 7/9 | open 1/5 | 3bet 0/3 | jams 4 | folds-to-raise 2/6 | shown: Kc Ah@H4
    blinds: fold-to-steal 5/6
    out: seat5@H3
    > survive: 10 BB or less, shove or fold
    > seat0 jams wide (4/9): call jams with 77+ ATs+ AJo+ KQs

Every count comes from the cleanup prompt text (the other seats' actions, our own submit_action calls, the table
header and the showdown line), so it is observable at play time. "open o/p" is raises over chances to open (acting
with no raise and no limp before; isolation raises are not opens); "3bet t/p" is re-raises over the first decision
facing exactly one raise; "jams" are bets, raises or all-ins for the seat's whole hand-start stack (chips committed
on every street counted; never for a stack under 15 BB); "folds-to-raise f/g" is folds over the first decision
facing a raise. "all-in" events count as raises. The "blinds" line counts the two blind seats' replies to a steal
(an open from CO, BTN or SB; BTN and SB only when 3-handed). Lines starting "> " are exploit lines recomputed from
the counts each hand (never parsed back); they appear only with enough samples, and the survival line comes first so
trimming drops it last. `parse_pad` still reads v1 pads and v2 pads that carry an old "afk" field.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections import Counter
from dataclasses import dataclass, field

MAX_CHARS = 1500
SHOWED_KEEP = 3
EVENT = re.compile(r'^\s+(preflop|flop|turn|river): \S+ \(seat (\d+)\) ([a-z-]+)(?: to (\d+))?(?: and says: "(.*)")?\s*$')
TABLE = re.compile(r"^\s+seat (\d+):.*? stack (\d+), bet \d+(?:, (folded|out))?\s*$")
TABLE_FULL = re.compile(r"^\s+seat (\d+):.*? stack (\d+), bet (\d+)(?:, (folded|out))?\s*$")
STREET = re.compile(r"^Street: (\w+)", re.M)
CARD = r"[2-9TJQKA][cdhs]"
SHOWED_ITEM = re.compile(rf"({CARD} {CARD})@H(\d+)")
V2_HEAD = re.compile(r"H(\d+)/(\d+) \| me seat(\d+) (\d+) \(\d+ BB\) \| rank (?:\d+|out)/\d+ \| gap up \+\d+, down -\d+ \| \d+ left")
V2_OPP = re.compile(r"seat(\d+): (\d+) \| VPIP (\d+)/(\d+) \| open (\d+)/(\d+) \| 3bet (\d+)/(\d+) \| jams (\d+) \| "
                    r"folds-to-raise (\d+)/(\d+)(?: \| afk \d+)? \| shown: (.*?)(?: \| lies (\d+))?")
DELIMS = ("SCRATCHPAD>>>", "<<<SCRATCHPAD")
RANKS = "23456789TJQKA"

# (pattern after "I have/got/..."), minimum hand category (0 high card .. 8 straight flush) that backs the claim
CLAIMS = [
    (r"(?:the )?nuts|an? monster", 4),
    (r"(?:the )?boat|an? full house", 6),
    (r"an? flush(?! draw)", 5),
    (r"an? straight(?! draw)", 4),
    (r"(?:an? set|trips)", 3),
    (r"two pair", 2),
]
CLAIM_LEAD = r"\b(?:have|got|flopped|hit|made|rivered|turned|holding)\s+"


@dataclass
class Event:
    street: str
    seat: int
    action: str
    says: str | None = None
    amount: int | None = None


@dataclass
class HandSummary:
    hand: int
    total: int
    hero: int
    big_blind: int
    stacks: dict[int, int]
    out: set[int]
    seen: set[int]
    events: list[Event]
    showdown: dict[int, str]
    board: str
    hero_actions: list[str]
    prev_pad: str
    button: int | None = None  # seat with the button, from the table header
    totals: dict[int, int] = field(default_factory=dict)  # stack + bet per seat when the header was shown
    live: list[int] = field(default_factory=list)  # seats dealt in (not "out"), sorted
    full_events: list[Event] = field(default_factory=list)  # events in order, our own submit_action calls included

    @property
    def hero_turns(self) -> int:
        return len(self.hero_actions)


@dataclass
class Opp:
    stack: int = 0
    va: int = 0
    vb: int = 0
    raises: int = 0
    showed: list[tuple[str, int]] = field(default_factory=list)
    false_claims: int = 0
    opens: int = 0  # v2 counts from here on
    open_opp: int = 0
    three_bet: int = 0
    three_bet_opp: int = 0
    jams: int = 0
    ftr_f: int = 0
    ftr_g: int = 0
    afk: int = 0


@dataclass
class PadState:
    hand: int
    total: int
    hero: int
    stack: int
    opps: dict[int, Opp]
    out: dict[int, int]
    version: int = 1
    fts_f: int = 0  # blinds that folded to a steal
    fts_g: int = 0  # blind decisions facing a steal


def _text(item) -> str:
    content = item.get("content") if isinstance(item, dict) else None
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content if isinstance(content, str) else ""


def extract_pad(first_user: str) -> str:
    """The pad text shown at the top of the first user message ('' when empty or unreadable)."""
    start, end = first_user.find("<<<SCRATCHPAD\n"), first_user.rfind("\nSCRATCHPAD>>>")
    if start < 0 or end < start:
        return ""
    pad = first_user[start + len("<<<SCRATCHPAD\n"):end]
    return "" if pad.strip() == "(empty)" else pad


def parse_cleanup(messages: list[dict]) -> HandSummary:
    users = [(i, _text(m)) for i, m in enumerate(messages) if m.get("role") == "user" and m.get("type", "message") == "message"]
    header = next(((i, t) for i, t in users if re.match(r"## Hand \d+ of \d+", t)), None)
    over = users[-1][1] if users else ""
    if header is None or not re.match(r"## Hand \d+ is over", over):
        raise ValueError("not an after-hand cleanup prompt")
    hand, total = (int(x) for x in re.match(r"## Hand (\d+) of (\d+)", header[1]).groups())
    hero = int(re.search(r"^Your seat: (\d+)", header[1], re.M).group(1))
    seen = {int(m.group(1)) for ln in header[1].splitlines() if (m := TABLE.match(ln)) and m.group(3) != "out"}
    events: list[Event] = []
    for i, text in users:
        if i < header[0]:
            continue
        for ln in text.splitlines():
            if m := EVENT.match(ln):
                events.append(Event(m.group(1), int(m.group(2)), m.group(3), m.group(5), int(m.group(4)) if m.group(4) else None))
    stacks_line = re.search(r"^Stacks now: (.*?)\.?$", over, re.M)
    if not stacks_line:
        raise ValueError("no 'Stacks now' line")
    stacks, out = {}, set()
    for seat, value, tag in re.findall(r"seat(\d+) (\d+)( \(out\))?", stacks_line.group(1)):
        stacks[int(seat)] = int(value)
        if tag or int(value) == 0:
            out.add(int(seat))
    showdown = {}
    if m := re.search(r"^Showdown: (.*?)\. Mucked", over, re.M):
        showdown = {int(s): c for s, c in re.findall(rf"seat(\d+) showed ({CARD} {CARD})", m.group(1))}
    board = (re.search(r"^Final board: (.*)$", over, re.M) or [None, ""])[1].strip()
    hero_actions = []
    for m in messages:
        if m.get("type") == "function_call" and m.get("name") == "submit_action":
            try:
                hero_actions.append(str(json.loads(m["arguments"]).get("action")))
            except (ValueError, TypeError, AttributeError):
                hero_actions.append("?")
    blinds = re.search(r"[Bb]linds are (\d+)/(\d+)", _text(messages[0]) if messages else "")
    first_user = next((t for _, t in users if t.startswith("## Your scratchpad")), "")
    button, totals, live = None, {}, []
    for ln in header[1].splitlines():
        if m := TABLE_FULL.match(ln):
            totals[int(m.group(1))] = int(m.group(2)) + int(m.group(3))
            if m.group(4) != "out":
                live.append(int(m.group(1)))
            if "[button]" in ln:
                button = int(m.group(1))
    return HandSummary(hand, total, hero, int(blinds.group(2)) if blinds else 2, stacks, out, seen, events, showdown,
                       board, hero_actions, extract_pad(first_user), button, totals, sorted(live),
                       _full_events(messages, header[0], hero))


def _full_events(messages: list[dict], start: int, hero: int) -> list[Event]:
    """Every event in prompt order from the header on, with our own submit_action calls placed where we acted."""
    full: list[Event] = []
    street = "preflop"
    for i, m in enumerate(messages):
        if i < start:
            continue
        if m.get("type") == "function_call" and m.get("name") == "submit_action":
            try:
                args = json.loads(m["arguments"])
                action, amount = str(args.get("action")), args.get("amount")
            except (ValueError, TypeError, AttributeError):
                continue
            full.append(Event(street, hero, action, None, amount if isinstance(amount, int) else None))
        elif m.get("role") == "user" and m.get("type", "message") == "message":
            text = _text(m)
            for ln in text.splitlines():
                if e := EVENT.match(ln):
                    full.append(Event(e.group(1), int(e.group(2)), e.group(3), e.group(5), int(e.group(4)) if e.group(4) else None))
                elif sm := STREET.match(ln):
                    street = sm.group(1)
    return full


def hand_category(cards: list[str]) -> int:
    """Best 5-of-7 category: 0 high card, 1 pair, 2 two pair, 3 trips, 4 straight, 5 flush, 6 full house, 7 quads, 8 straight flush."""
    ranks = sorted((RANKS.index(c[0]) for c in cards), reverse=True)
    counts = Counter(ranks)
    by_suit: dict[str, list[int]] = {}
    for c in cards:
        by_suit.setdefault(c[1], []).append(RANKS.index(c[0]))

    def straight(rs) -> bool:
        s = set(rs) | ({-1} if 12 in rs else set())
        return any(all(x in s for x in range(lo, lo + 5)) for lo in range(-1, 9))

    flush = next((rs for rs in by_suit.values() if len(rs) >= 5), None)
    if flush and straight(flush):
        return 8
    shape = sorted(counts.values(), reverse=True)
    if shape[0] == 4:
        return 7
    if shape[0] == 3 and len(shape) > 1 and shape[1] >= 2:
        return 6
    if flush:
        return 5
    if straight(ranks):
        return 4
    return {3: 3}.get(shape[0], 2 if shape[0] == 2 and len(shape) > 1 and shape[1] == 2 else (1 if shape[0] == 2 else 0))


def claim_minimum(says: str) -> int | None:
    """Minimum hand category a table-talk line claims ('I flopped the nuts' -> 4), or None when it claims nothing."""
    text = says.lower().replace("'", "")
    if "draw" in text:
        return None
    for pattern, minimum in CLAIMS:
        if re.search(CLAIM_LEAD + r"(?:" + pattern + r")", text):
            return minimum
    return None


def parse_pad(text: str) -> PadState | None:
    """Read a pad in this module's schema back into counts; None for anything else."""
    lines = (text or "").split("\n")
    head = re.fullmatch(r"H(\d+)/(\d+) \| me seat(\d+) (\d+) \| rank (?:\d+|out)/\d+", lines[0]) if lines else None
    version = 1
    if not head and lines:
        head = V2_HEAD.fullmatch(lines[0])
        version = 2
    if not head:
        return None
    hand, total, hero, stack = (int(x) for x in head.groups())
    opps: dict[int, Opp] = {}
    out: dict[int, int] = {}
    fts = (0, 0)
    for ln in lines[1:]:
        if version == 2:
            if m := V2_OPP.fullmatch(ln):
                g = [int(x) for x in m.group(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)]
                opps[g[0]] = Opp(g[1], g[2], g[3], 0, [(c, int(h)) for c, h in SHOWED_ITEM.findall(m.group(12))],
                                 int(m.group(13) or 0), g[4], g[5], g[6], g[7], g[8], g[9], g[10])
            elif m := re.fullmatch(r"blinds: fold-to-steal (\d+)/(\d+)", ln):
                fts = (int(m.group(1)), int(m.group(2)))
            elif ln.startswith("out: "):
                out = {int(s): int(h) for s, h in re.findall(r"seat(\d+)@H(\d+)", ln)}
            continue
        m = re.fullmatch(r"seat(\d+): (\d+) \| VPIP (\d+)/(\d+) \| raises (\d+) \| showed: (.*?) \| talk-claims: (\d+) false", ln)
        if m:
            seat, st, va, vb, raises, showed, false = m.groups()
            opps[int(seat)] = Opp(int(st), int(va), int(vb), int(raises),
                                  [(c, int(h)) for c, h in SHOWED_ITEM.findall(showed)], int(false))
        elif ln.startswith("out: "):
            out = {int(s): int(h) for s, h in re.findall(r"seat(\d+)@H(\d+)", ln)}
    return PadState(hand, total, hero, stack, opps, out, version, *fts)


def preflop_stats(s: HandSummary) -> tuple[dict[int, Counter], Counter]:
    """Per-seat v2 counts for this hand from the ordered events, and the blinds' replies to a steal."""
    seats = s.live or sorted(s.seen | {s.hero})
    n = len(seats)
    off = {seat: (i - seats.index(s.button)) % n for i, seat in enumerate(seats)} if s.button in seats else {}
    steal_offsets = {0, 1} if n == 3 else {0, 1, n - 1}  # BTN, SB, CO (no CO, and never the BB, when 3-handed)
    steal_seats = {k for k, v in off.items() if n >= 3 and v in steal_offsets}
    blind_seats = {k for k, v in off.items() if n >= 3 and v in (1, 2)}
    per: dict[int, Counter] = {}
    blinds: Counter = Counter()
    raises, raiser, steal, acted, jammed, limped = 0, None, None, set(), set(), False
    tb_seen, ftr_seen = set(), set()
    committed: dict[int, int] = {}  # chips put in on the current street, by seat
    earlier: dict[int, int] = {}  # chips put in on earlier streets
    street, street_max = None, 0
    for e in s.full_events:
        c = per.setdefault(e.seat, Counter())
        if e.street != street:
            for seat, v in committed.items():
                earlier[seat] = earlier.get(seat, 0) + v
            committed, street_max = {}, 0
            street = e.street
        is_raise = e.action in ("raise", "bet", "all-in", "allin")
        if e.action in ("raise", "bet") and e.amount is not None:
            committed[e.seat] = street_max = e.amount
        elif e.action == "call":
            committed[e.seat] = street_max
        total = s.totals.get(e.seat, 1 << 30)
        if (e.seat != s.hero and total >= 15 * s.big_blind and e.seat not in jammed and (
                e.action in ("all-in", "allin") or (
                    e.action in ("raise", "bet") and e.amount is not None
                    and committed.get(e.seat, 0) + earlier.get(e.seat, 0) >= total))):
            jammed.add(e.seat)
            c["jams"] += 1
        if e.street != "preflop":
            continue
        if raises == 0 and e.seat not in acted and e.action != "check" and not limped:
            c["open_opp"] += 1
            c["opens"] += is_raise
        if raises == 1 and e.seat != raiser and e.seat not in tb_seen:
            tb_seen.add(e.seat)
            c["three_bet_opp"] += 1
            c["three_bet"] += is_raise
        if raises >= 1 and e.seat != raiser and e.seat not in ftr_seen:
            ftr_seen.add(e.seat)
            c["ftr_g"] += 1
            c["ftr_f"] += e.action == "fold"
        if steal is not None and raises == 1 and e.seat in blind_seats and e.seat != steal and e.seat not in acted:
            if e.seat != s.hero:
                blinds["g"] += 1
                blinds["f"] += e.action == "fold"
        acted.add(e.seat)
        if raises == 0 and e.action == "call":
            limped = True
        if is_raise:
            raises += 1
            raiser = e.seat
            steal = e.seat if raises == 1 and e.seat in steal_seats else None
    return per, blinds


def update_state(prev: PadState | None, s: HandSummary, v2: bool = False) -> PadState:
    if prev is not None and not (prev.hero == s.hero and prev.hand < s.hand):
        prev = None
    opps = {k: dataclasses.replace(v, showed=list(v.showed)) for k, v in (prev.opps if prev else {}).items()}
    per, blinds = preflop_stats(s) if v2 else ({}, Counter())
    out = dict(prev.out) if prev else {}
    first_claim: dict[int, int] = {}
    for e in s.events:
        if e.says and e.seat not in first_claim and (need := claim_minimum(e.says)) is not None:
            first_claim[e.seat] = need
    board = s.board.split() if s.board and s.board != "-" else []
    for seat in sorted(s.seen):
        if seat == s.hero:
            continue
        o = opps.setdefault(seat, Opp())
        mine = [e for e in s.events if e.seat == seat]
        o.vb += 1
        o.va += any(e.street == "preflop" and e.action in ("call", "raise", "bet", "all-in", "allin") for e in mine)
        o.raises += sum(e.action in ("raise", "bet") for e in mine)
        if v2:
            for key, value in per.get(seat, Counter()).items():
                setattr(o, key, getattr(o, key) + value)
        if seat in s.showdown:
            o.showed = (o.showed + [(s.showdown[seat], s.hand)])[-SHOWED_KEEP:]
            if seat in first_claim and hand_category(s.showdown[seat].split() + board) < first_claim[seat]:
                o.false_claims += 1
    for seat, stack in s.stacks.items():
        if seat == s.hero:
            continue
        if seat in s.out:
            out.setdefault(seat, s.hand)
            opps.pop(seat, None)
        else:
            opps.setdefault(seat, Opp()).stack = stack
    state = PadState(s.hand, s.total, s.hero, s.stacks.get(s.hero, 0), opps, out)
    if v2:
        state.version = 2
        state.fts_f = (prev.fts_f if prev else 0) + blinds["f"]
        state.fts_g = (prev.fts_g if prev else 0) + blinds["g"]
    return state


def rank_of(state: PadState) -> tuple[str, int]:
    alive = [o.stack for o in state.opps.values() if o.stack > 0]
    if state.stack <= 0:
        return "out", len(alive)
    return str(1 + sum(x > state.stack for x in alive)), len(alive) + 1


def plan_line(state: PadState, big_blind: int) -> str:
    stacks = [state.stack] + [o.stack for o in state.opps.values()]
    avg = sum(stacks) / len(stacks)
    rank, alive = rank_of(state)
    left = state.total - state.hand
    if state.stack > 0 and (state.stack < 15 * big_blind or state.stack < 0.5 * avg):
        return "plan: short: avoid marginal stack-offs; wait for a strong hand or a clean steal"
    if rank == "1" and state.stack >= 1.4 * avg:
        return "plan: chip lead: fold marginal hands, let the others bust each other"
    if alive == 2:
        return "plan: heads-up: widen opens, keep pressure"
    if left <= 5:
        return f"plan: endgame: protect rank {rank}/{alive}, avoid marginal stack-offs"
    rated = [(o.va / o.vb, seat) for seat, o in state.opps.items() if o.vb >= 3]
    loose = max(rated, default=None)
    tight = min(rated, default=None)
    bits = ["plan: mid: tight value, fold marginal spots"]
    if loose and loose[0] >= 0.5:
        bits.append(f"isolate loose seat{loose[1]}")
    if tight and tight[0] <= 0.2 and tight[1] != (loose or (0, -1))[1]:
        bits.append(f"respect seat{tight[1]} raises")
    return "; ".join(bits)


def gaps(state: PadState) -> tuple[int, int]:
    """Chips to the next stack above ours, and chips we lead the next stack below by (0 when none)."""
    alive = [o.stack for o in state.opps.values() if o.stack > 0]
    if state.stack <= 0:
        return 0, 0
    above = [x - state.stack for x in alive if x > state.stack]
    below = [state.stack - x for x in alive if x <= state.stack]
    return min(above, default=0), min(below, default=0)


def exploit_lines(state: PadState, big_blind: int = 2) -> list[str]:
    """Concrete exploit lines from the counts, each only with enough samples. Most valuable first."""
    lines = []
    rank, _ = rank_of(state)
    if 0 < state.stack <= 10 * big_blind:
        lines.append("> survive: 10 BB or less, shove or fold")
    elif rank == "1" and state.total - state.hand <= 8 and gaps(state)[1] >= 40:
        lines.append("> survive: leading late, avoid big pots with marginal hands")
    for seat, o in sorted(state.opps.items()):
        if o.vb >= 8 and o.jams >= 4 and o.jams / o.vb >= 0.25:
            lines.append(f"> seat{seat} jams wide ({o.jams}/{o.vb}): call jams with 77+ ATs+ AJo+ KQs")
    if state.fts_g >= 5 and state.fts_f / state.fts_g >= 0.6:
        lines.append(f"> blinds fold to steals ({state.fts_f}/{state.fts_g}): open BTN/CO wide")
    for seat, o in sorted(state.opps.items()):
        if o.ftr_g >= 5 and o.ftr_f / o.ftr_g >= 0.7:
            lines.append(f"> seat{seat} folds to raises ({o.ftr_f}/{o.ftr_g}): raise first-in and 3-bet light")
    for seat, o in sorted(state.opps.items()):
        if o.vb >= 8 and o.va / o.vb >= 0.5 and o.open_opp >= 0.8 * o.vb and o.opens <= 0.3 * o.va:
            lines.append(f"> seat{seat} calls a lot: value bet, do not bluff")
        elif o.vb >= 8 and o.va / o.vb <= 0.15:
            lines.append(f"> seat{seat} is tight (VPIP {o.va}/{o.vb}): respect its raises")
    return lines


def render_v2(state: PadState, big_blind: int = 2, showed_keep: int = SHOWED_KEEP, max_exploits: int = 99) -> str:
    rank, alive = rank_of(state)
    up, down = gaps(state)
    lines = [f"H{state.hand}/{state.total} | me seat{state.hero} {state.stack} ({state.stack // big_blind} BB) | rank {rank}/{alive}"
             f" | gap up +{up}, down -{down} | {state.total - state.hand} left"]
    for seat in sorted(state.opps):
        o = state.opps[seat]
        showed = ", ".join(f"{c}@H{h}" for c, h in o.showed[-showed_keep:] if showed_keep) or "-"
        lines.append(f"seat{seat}: {o.stack} | VPIP {o.va}/{o.vb} | open {o.opens}/{o.open_opp} | 3bet {o.three_bet}/{o.three_bet_opp}"
                     f" | jams {o.jams} | folds-to-raise {o.ftr_f}/{o.ftr_g} | shown: {showed}"
                     + (f" | lies {o.false_claims}" if o.false_claims else ""))
    if state.fts_g:
        lines.append(f"blinds: fold-to-steal {state.fts_f}/{state.fts_g}")
    if state.out:
        lines.append("out: " + ", ".join(f"seat{s}@H{h}" for s, h in sorted(state.out.items())))
    return "\n".join(lines + exploit_lines(state, big_blind)[:max_exploits])


def fit_v2(state: PadState, big_blind: int = 2) -> str:
    """The v2 pad inside MAX_CHARS: shed old showdown cards first, then exploit lines from the end."""
    pad = ""
    for exploits in (99, 4, 3, 2, 1, 0):
        for keep in range(SHOWED_KEEP, -1, -1):
            pad = render_v2(state, big_blind, keep, exploits)
            if len(pad) <= MAX_CHARS:
                return pad
    return "\n".join(pad[:MAX_CHARS].split("\n")[:-1])


def render(state: PadState, big_blind: int = 2, showed_keep: int = SHOWED_KEEP, version: int = 1) -> str:
    if version == 2:
        return render_v2(state, big_blind, showed_keep)
    rank, alive = rank_of(state)
    lines = [f"H{state.hand}/{state.total} | me seat{state.hero} {state.stack} | rank {rank}/{alive}"]
    for seat in sorted(state.opps):
        o = state.opps[seat]
        showed = ", ".join(f"{c}@H{h}" for c, h in o.showed[-showed_keep:] if showed_keep) or "-"
        lines.append(f"seat{seat}: {o.stack} | VPIP {o.va}/{o.vb} | raises {o.raises} | showed: {showed} | talk-claims: {o.false_claims} false")
    if state.out:
        lines.append("out: " + ", ".join(f"seat{s}@H{h}" for s, h in sorted(state.out.items())))
    lines.append(plan_line(state, big_blind))
    return "\n".join(lines)


def teacher_pad(messages: list[dict], prev_text: str | None = None, version: int = 1) -> tuple[str, PadState]:
    """Ideal pad for this cleanup prompt. `prev_text` overrides the pad parsed from the prompt (used when chaining).
    `version` 2 writes the v2 schema (see the module doc); a v1 pad shown as the previous pad is still read."""
    s = parse_cleanup(messages)
    state = update_state(parse_pad(s.prev_pad if prev_text is None else prev_text), s, version == 2)
    if version == 2:
        pad = fit_v2(state, s.big_blind)
    else:
        for keep in range(SHOWED_KEEP, -1, -1):
            pad = render(state, s.big_blind, keep)
            if len(pad) <= MAX_CHARS:
                break
        else:
            pad = "\n".join(pad[:MAX_CHARS].split("\n")[:-1])
    for d in DELIMS:  # table talk never enters the pad, but guard the invariant
        pad = pad.replace(d, "")
    return pad, parse_pad(pad) or state


def teacher_reasoning(s: HandSummary, state: PadState) -> str:
    rank, alive = rank_of(state)
    if s.showdown:
        shown = "; ".join(f"seat{k} showed {v}" for k, v in sorted(s.showdown.items()) if k != s.hero) or "only my hand was shown"
        first = f"Hand {s.hand} went to showdown ({shown})."
    else:
        first = f"Hand {s.hand} ended without a showdown."
    second = f"I hold {state.stack} chips, rank {rank} of {alive}."
    return f"{first} {second} I rewrite the pad in the fixed schema with updated VPIP, raise and showdown counts."
