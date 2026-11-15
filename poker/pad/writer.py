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
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field

MAX_CHARS = 1500
SHOWED_KEEP = 3
EVENT = re.compile(r'^\s+(preflop|flop|turn|river): \S+ \(seat (\d+)\) ([a-z-]+)(?: to (\d+))?(?: and says: "(.*)")?\s*$')
TABLE = re.compile(r"^\s+seat (\d+):.*? stack (\d+), bet \d+(?:, (folded|out))?\s*$")
CARD = r"[2-9TJQKA][cdhs]"
SHOWED_ITEM = re.compile(rf"({CARD} {CARD})@H(\d+)")
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


@dataclass
class PadState:
    hand: int
    total: int
    hero: int
    stack: int
    opps: dict[int, Opp]
    out: dict[int, int]


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
                events.append(Event(m.group(1), int(m.group(2)), m.group(3), m.group(5)))
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
    return HandSummary(hand, total, hero, int(blinds.group(2)) if blinds else 2, stacks, out, seen, events, showdown,
                       board, hero_actions, extract_pad(first_user))


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
    if not head:
        return None
    hand, total, hero, stack = (int(x) for x in head.groups())
    opps: dict[int, Opp] = {}
    out: dict[int, int] = {}
    for ln in lines[1:]:
        m = re.fullmatch(r"seat(\d+): (\d+) \| VPIP (\d+)/(\d+) \| raises (\d+) \| showed: (.*?) \| talk-claims: (\d+) false", ln)
        if m:
            seat, st, va, vb, raises, showed, false = m.groups()
            opps[int(seat)] = Opp(int(st), int(va), int(vb), int(raises),
                                  [(c, int(h)) for c, h in SHOWED_ITEM.findall(showed)], int(false))
        elif ln.startswith("out: "):
            out = {int(s): int(h) for s, h in re.findall(r"seat(\d+)@H(\d+)", ln)}
    return PadState(hand, total, hero, stack, opps, out)


def update_state(prev: PadState | None, s: HandSummary) -> PadState:
    if prev is not None and not (prev.hero == s.hero and prev.hand < s.hand):
        prev = None
    opps = {k: Opp(v.stack, v.va, v.vb, v.raises, list(v.showed), v.false_claims)
            for k, v in (prev.opps if prev else {}).items()}
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
    return PadState(s.hand, s.total, s.hero, s.stacks.get(s.hero, 0), opps, out)


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


def render(state: PadState, big_blind: int = 2, showed_keep: int = SHOWED_KEEP) -> str:
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


def teacher_pad(messages: list[dict], prev_text: str | None = None) -> tuple[str, PadState]:
    """Ideal pad for this cleanup prompt. `prev_text` overrides the pad parsed from the prompt (used when chaining)."""
    s = parse_cleanup(messages)
    state = update_state(parse_pad(s.prev_pad if prev_text is None else prev_text), s)
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
