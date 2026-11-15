"""Quality filters for SFT records, plus the "Pad:" read line (poker.sft.make_records --clean / --pad-read-line).

An audit of data/sft/c6a.jsonl found records that teach the wrong thing. clean(records) -> (records, stats) drops them,
one filter per problem. A record is counted under the first filter (in the order below) that drops it.

    no_v2_pad       action at hand > 1 whose prompt scratchpad has no "H{n}/{total} |" header line (hand 1 may be empty)
    pad_restart     pad_teacher: written pad's VPIP denominator is more than 1 below the hand number (counters restarted)
    pad_game_over   pad_teacher: pad written at or after the last hand (the game is over)
    suitedness      action reasoning says our cards are offsuit when suited, or suited when offsuit (poker.facts)
    illegal         target action not in the prompt's legal actions (fold with a check available counts as illegal)
    rfi_fold        first-in fold of a hand inside the v7 RFI chart for our seat (stack >= 25 BB, like the teacher)
    bad_fold        fold with top pair or better, or 8+ outs, at a required equity of 30% or less
    leakage         "suggested action", "hint", "teacher", "training", "LLM" in reasoning or content; analysis put in
                    content; a pasted JSON submit_action in reasoning
    confusion       3 or more "wait" / "re-read" in an action reasoning

add_pad_read_line(records) -> (records, stats) puts one line "Reads: ..." at the top of every action reasoning whose
prompt has a v2 pad (below a "Facts:" line when there is one). It keeps only the pad's exploit lines that matter for this
decision: the survive line; the steal line when we are first in from CO, BTN or SB; a seat line when that seat is still in
the hand or has acted in it. Everything is computed from the prompt alone.
"""
from __future__ import annotations

import re

from poker.hints import text_content
from poker.pad.writer import EVENT, V2_OPP, parse_pad
from poker.teacher import RFI_GTO, V7_MIN_BB, in_range

LAST_HAND = 30
CONFUSION_LIMIT = 3
BAD_FOLD_REQUIRED = 0.30
BAD_FOLD_OUTS = 8
FILTERS = ("no_v2_pad", "pad_restart", "pad_game_over", "suitedness", "illegal", "rfi_fold", "bad_fold", "leakage",
           "confusion")

_HAND = re.compile(r"^## Hand (\d+) of (\d+)", re.M)
_HAND_OVER = re.compile(r"^## Hand (\d+) is over", re.M)
_PAD_BLOCK = re.compile(r"<<<SCRATCHPAD\n?(.*?)\n?SCRATCHPAD>>>", re.S)
_PAD_HEAD = re.compile(r"^H\d+/\d+ \|", re.M)
_LEGAL = re.compile(r"^Legal actions: (.*)$", re.M)
_STRONG = {"top pair", "overpair", "two pair", "set", "trips", "straight", "flush", "full house", "quads", "straight flush"}
_LEAK_WORDS = re.compile(r"suggested action|\bhints?\b|\bteachers?\b|\btraining\b|(?-i:\bLLMs?\b)", re.I)
_JSON_ACTION = re.compile(r"\{\s*[\"']action[\"']\s*:|[\"']tool_call[\"']\s*:|[\"']name[\"']\s*:\s*[\"']submit_action[\"']")
_CONFUSED = re.compile(r"\bwait[,.!:;\u2014-]|\bwait\s+(?:no|let|that|I|actually|again)\b|\bre-?read|\bre-?check", re.I)
RFI_FOLD_MIN_HANDS_LEFT = 3  # in the last hands a fold can protect a place
_LATE_STEAL = {"CO", "BTN", "SB"}


def _is_action(record: dict) -> bool:
    return record["target"]["tool_call"]["name"] == "submit_action"


class Prompt:
    """What the prompt alone tells us: hand number, pad, legal actions, and this hand's events."""

    def __init__(self, messages):
        self.messages = messages
        users = [(i, text_content(m)) for i, m in enumerate(messages)]
        users = [(i, t) for i, t in users if t]
        self.full = "\n".join(t for _, t in users)
        found = _HAND.findall(self.full)
        self.hand = int(found[-1][0]) if found else None
        block = _PAD_BLOCK.search(self.full)
        self.pad = block[1].strip() if block else ""
        self.v2_pad = bool(_PAD_HEAD.search(self.pad))
        legal = _LEGAL.findall(users[-1][1]) if users else []
        self.legal = legal[-1] if legal else None
        start = next((i for i, t in reversed(users) if _HAND.search(t)), 0)
        self.acted_before = any(m.get("type") == "function_call" for m in messages[start:] if isinstance(m, dict))
        self.events = [(m[1], int(m[2]), m[3]) for i, t in users if i >= start for m in
                       (EVENT.match(line) for line in t.split("\n")) if m]

    def first_in(self) -> bool:
        """Preflop, nobody has put chips in beyond the blinds and we have not acted: every earlier event is a fold."""
        return not self.acted_before and all(street == "preflop" and act == "fold" for street, _, act in self.events)

    def acted_seats(self) -> set[int]:
        return {seat for _, seat, _ in self.events}


def _legal_ok(legal: str | None, args: dict) -> bool:
    if legal is None:
        return True  # nothing to check against
    action = args.get("action")
    amount = args.get("amount")
    if action == "fold":
        return not re.search(r"\bcheck\b", legal)
    if action == "check":
        return bool(re.search(r"\bcheck\b", legal))
    if action == "call":
        return bool(re.search(r"\bcall \d+", legal))
    if action in ("bet", "raise"):
        found = re.search(rf"\b{action} to (\d+)-(\d+)", legal)
        if not found or isinstance(amount, bool) or not isinstance(amount, (int, float)) or amount != int(amount):
            return False
        return int(found[1]) <= amount <= int(found[2])
    return False


def _pad_written(record: dict) -> str:
    return record["target"]["tool_call"]["arguments"].get("contents") or ""


def _pad_hand(record: dict, pad: str) -> tuple[int | None, int]:
    prompt = "\n".join(text_content(m) for m in record["messages"])
    over = _HAND_OVER.findall(prompt)
    state = parse_pad(pad)
    hand = int(over[-1]) if over else (state.hand if state else None)
    return hand, (state.total if state else LAST_HAND)


def _pad_filter(record: dict) -> str | None:
    pad = _pad_written(record)
    hand, total = _pad_hand(record, pad)
    if hand is None:
        return None
    if hand >= total:
        return "pad_game_over"
    for line in pad.split("\n")[1:]:
        if (m := V2_OPP.fullmatch(line)) and hand - int(m.group(4)) > 1:
            return "pad_restart"
    return None


def _first_in_fold(prompt: Prompt, facts: dict) -> bool:
    if facts["street"] != "preflop" or facts["position"] not in RFI_GTO or not prompt.first_in():
        return False
    if facts.get("hands_left") is not None and facts["hands_left"] < RFI_FOLD_MIN_HANDS_LEFT:
        return False
    return facts["stack_bb"] >= V7_MIN_BB and in_range(facts["hole_code"], RFI_GTO[facts["position"]])


def _bad_fold(facts: dict) -> bool:
    if facts["required"] > BAD_FOLD_REQUIRED:
        return False
    return facts["made"] in _STRONG or bool(facts["draw"] and facts["draw"]["outs"] >= BAD_FOLD_OUTS)


def _action_filter(record: dict) -> str | None:
    from poker.facts import compute_facts, contradictions

    reasoning = record["target"].get("reasoning") or ""
    prompt = Prompt(record["messages"])
    args = record["target"]["tool_call"]["arguments"]
    if prompt.hand is not None and prompt.hand > 1 and not prompt.v2_pad:
        return "no_v2_pad"
    facts = compute_facts(record["messages"])
    if facts and any(c.startswith("suited") for c in contradictions(reasoning, facts)):
        return "suitedness"
    if not _legal_ok(prompt.legal, args):
        return "illegal"
    if facts and args.get("action") == "fold":
        if _first_in_fold(prompt, facts):
            return "rfi_fold"
        if _bad_fold(facts):
            return "bad_fold"
    return None


def _leaks(record: dict) -> bool:
    target = record["target"]
    reasoning = target.get("reasoning") or ""
    return bool((target.get("content") or "").strip() or _LEAK_WORDS.search(reasoning) or _JSON_ACTION.search(reasoning))


def _first_failure(record: dict) -> str | None:
    action = _is_action(record)
    if action and (name := _action_filter(record)):
        return name
    if not action and record["source"] == "pad_teacher" and (name := _pad_filter(record)):
        return name
    if _leaks(record):
        return "leakage"
    if action and len(_CONFUSED.findall(record["target"].get("reasoning") or "")) >= CONFUSION_LIMIT:
        return "confusion"
    return None


def clean(records) -> tuple[list[dict], dict]:
    """Drop the records the filters above reject. stats: {"in", "out", "dropped_total", "dropped": {filter: {source: n}}}."""
    kept, dropped = [], {name: {} for name in FILTERS}
    records = list(records)
    for record in records:
        name = _first_failure(record)
        if name:
            by_source = dropped[name]
            by_source[record["source"]] = by_source.get(record["source"], 0) + 1
        else:
            kept.append(record)
    return kept, {"in": len(records), "out": len(kept), "dropped_total": len(records) - len(kept),
                  "dropped": {k: v for k, v in dropped.items() if v}}


# ---- pad read line ----------------------------------------------------------------------------------------------

def _relevant_pad_lines(prompt: Prompt, facts: dict | None) -> list[str]:
    exploit = [line[2:] for line in prompt.pad.split("\n") if line.startswith("> ")]
    live = {s for s, st in facts_seats(prompt) if st not in ("folded", "out")} if facts else set()
    present = live  # a folded seat cannot affect this decision
    out = []
    for line in exploit:
        if line.startswith("survive"):
            out.append(line)
        elif line.startswith("blinds fold to steals"):
            if facts and facts["position"] in _LATE_STEAL and facts["street"] == "preflop" and prompt.first_in():
                out.append(line)
        elif (m := re.match(r"seat(\d+) ", line)) and int(m[1]) in present:
            out.append(line)
    return out


def facts_seats(prompt: Prompt) -> list[tuple[int, str]]:
    """(seat, status) rows of the newest table in the prompt."""
    texts = [text_content(m) for m in prompt.messages]
    current = next((t for t in reversed(texts) if re.search(r"^Street: ", t, re.M)), "")
    return [(int(m[1]), m[2] or "") for m in
            re.finditer(r"^  seat (\d+): .*? stack \d+, bet \d+(?:, (folded|all-in|out))?$", current, re.M)]


def add_pad_read_line(records) -> tuple[list[dict], dict]:
    """Insert "Reads: <relevant exploit lines joined by '; '>" (or "Reads: nothing relevant") as the first line of each action
    reasoning whose prompt has a v2 pad, below an existing "Facts:" line. Idempotent."""
    from poker.facts import compute_facts

    out, stats = [], {"with_pad": 0, "relevant": 0, "nothing": 0, "no_pad": 0, "not_action": 0, "already": 0}
    for record in records:
        if not _is_action(record):
            stats["not_action"] += 1
            out.append(record)
            continue
        lines = (record["target"].get("reasoning") or "").split("\n")
        prompt = Prompt(record["messages"])
        if any(line.startswith("Reads: ") for line in lines[:2]):
            stats["already"] += 1
        elif not prompt.v2_pad:
            stats["no_pad"] += 1
        else:
            picked = _relevant_pad_lines(prompt, compute_facts(record["messages"]))
            stats["with_pad"] += 1
            stats["relevant" if picked else "nothing"] += 1
            line = "Reads: " + ("; ".join(picked) if picked else "nothing relevant")
            at = 1 if lines and lines[0].startswith("Facts: ") else 0
            record = record | {"target": record["target"] | {"reasoning": "\n".join(lines[:at] + [line] + lines[at:])}}
        out.append(record)
    return out, stats
