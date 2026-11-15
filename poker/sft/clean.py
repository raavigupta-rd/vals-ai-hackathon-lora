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

Strict target filters (clean(records, strict=True); poker.sft.make_records --clean-strict), for the c8 final model:

    tool_text            "<tool_call", "<function=", "</tool_response" or '"name": "submit_action"' in reasoning or content
    conclusion_mismatch  the last 250 chars of reasoning conclude one action ("I should fold", "I'll call", "folding is
                         best") and the tool call is another; skipped when two different verbs are concluded
    short_body           action reasoning without its Facts/Reads/Talk/Decision lines is under 120 chars
    long_body            the same body is over 1800 chars
    has_say              (only with no_say=True) the tool call has a say argument

legal_target(record) -> bool is the legality guard (verb legal, bet/raise amount inside the legal range).
add_decision_line(records) appends "Decision: <fold | check | call N | bet N | raise to N>" as the last reasoning line.

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
STRICT_FILTERS = ("tool_text", "junk_text", "conclusion_mismatch", "short_body", "long_body", "has_say")
MIN_BODY, MAX_BODY, CONCLUSION_TAIL = 120, 2600, 250
_JUNK_TEXT = re.compile(r"[^\x00-\x7f]|</?thinking>")

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


_TOOL_TEXT = re.compile(r"<tool_call|<function=|</tool_response|[\"']name[\"']\s*:\s*[\"']submit_action[\"']", re.I)
_HEADER_LINE = re.compile(r"^(?:Facts|Reads|Talk|Decision):")
_NOT_COND = r"(?!\s+(?:if|when|unless|to\b|against|once|only|after|but|or|and)\b)"
_WILL = r"(?:I(?:'ll|\u2019ll| will| should| would| must| need to| have to| had better|'m going to| am going to| think I should)|let'?s|we should)"
_CONCLUDE = {
    "fold": [rf"\b{_WILL}\s+(?:just\s+|simply\s+)?fold\b{_NOT_COND}", rf"\bfold(?:ing)? is (?:the )?(?:best|right|correct|clear|my)\b",
             r"\bthe (?:best|right|correct) (?:play|move|action|decision) is (?:to )?fold\b", r"(?:^|[.!?]\s+)(?:so,?\s+)?fold\.?\s*$"],
    "call": [rf"\b{_WILL}\s+(?:just\s+|simply\s+)?call\b{_NOT_COND}", rf"\bcall(?:ing)? is (?:the )?(?:best|right|correct|clear|my)\b",
             r"\bthe (?:best|right|correct) (?:play|move|action|decision) is (?:to )?call\b", r"(?:^|[.!?]\s+)(?:so,?\s+)?call(?: \d+)?\.?\s*$"],
    "check": [rf"\b{_WILL}\s+(?:just\s+|simply\s+)?check\b{_NOT_COND}", r"(?:^|[.!?]\s+)(?:so,?\s+)?check\.?\s*$"],
    "aggr": [rf"\b{_WILL}\s+(?:just\s+|simply\s+)?(?:raise|bet|jam|shove)\b{_NOT_COND}",
             r"(?:^|[.!?]\s+)(?:so,?\s+)?(?:raise|bet) (?:to )?\d+\.?\s*$"],
}
_CONCLUDE = {k: [re.compile(p, re.I) for p in v] for k, v in _CONCLUDE.items()}
_FAMILY = {"fold": "fold", "call": "call", "check": "check", "bet": "aggr", "raise": "aggr"}


def _body(reasoning: str) -> str:
    return "\n".join(line for line in reasoning.split("\n") if not _HEADER_LINE.match(line)).strip()


def _conclusion_mismatch(record: dict) -> bool:
    tail = _body(record["target"].get("reasoning") or "")[-CONCLUSION_TAIL:]
    said = {verb for verb, patterns in _CONCLUDE.items() if any(p.search(tail) for p in patterns)}
    action = _FAMILY.get(record["target"]["tool_call"]["arguments"].get("action"))
    return len(said) == 1 and action is not None and action not in said


def _strict_failure(record: dict, strict: bool, no_say: bool, before_leak: bool) -> str | None:
    target = record["target"]
    if before_leak:
        text = (target.get("reasoning") or "") + "\n" + (target.get("content") or "")
        if _TOOL_TEXT.search(text):
            return "tool_text"
        # non-ASCII fragments and think tags in a target teach the drift seen live (Chinese text, </thinking>)
        return "junk_text" if _is_action(record) and _JUNK_TEXT.search(text) else None
    if not _is_action(record):
        return None
    if strict:
        body = _body(target.get("reasoning") or "")
        if _conclusion_mismatch(record):
            return "conclusion_mismatch"
        if len(body) < MIN_BODY:
            return "short_body"
        if len(body) > MAX_BODY:
            return "long_body"
    if no_say and "say" in target["tool_call"]["arguments"]:
        return "has_say"
    return None


def _first_failure(record: dict, strict: bool = False, no_say: bool = False) -> str | None:
    action = _is_action(record)
    if action and (name := _action_filter(record)):
        return name
    if not action and record["source"] == "pad_teacher" and (name := _pad_filter(record)):
        return name
    if strict and (name := _strict_failure(record, strict, no_say, True)):
        return name
    if _leaks(record):
        return "leakage"
    if action and len(_CONFUSED.findall(record["target"].get("reasoning") or "")) >= CONFUSION_LIMIT:
        return "confusion"
    if strict or no_say:
        return _strict_failure(record, strict, no_say, False)
    return None


def legal_target(record: dict) -> bool:
    """Legality guard: an action turn's tool call uses a verb the prompt's legal line allows, and a bet or raise amount is an
    integer inside the legal range. Other turns (update_scratchpad) pass. A prompt without a legal line passes."""
    if not _is_action(record):
        return True
    return _legal_ok(Prompt(record["messages"]).legal, record["target"]["tool_call"]["arguments"])


def strip_say(records) -> tuple[list[dict], int]:
    """Copies of the records with the say argument removed from every tool call; also the number of records changed."""
    out, changed = [], 0
    for record in records:
        args = record["target"]["tool_call"]["arguments"]
        if isinstance(args, dict) and "say" in args:
            call = record["target"]["tool_call"] | {"arguments": {k: v for k, v in args.items() if k != "say"}}
            record = record | {"target": record["target"] | {"tool_call": call}}
            changed += 1
        out.append(record)
    return out, changed


def clean(records, strict: bool = False, no_say: bool = False) -> tuple[list[dict], dict]:
    """Drop the records the filters above reject. strict adds the c8 target filters; no_say adds has_say (drop any record
    with a say argument). stats: {"in", "out", "dropped_total", "dropped": {filter: {source: n}}}."""
    kept, dropped = [], {name: {} for name in FILTERS + STRICT_FILTERS}
    records = list(records)
    for record in records:
        name = _first_failure(record, strict, no_say)
        if name:
            by_source = dropped[name]
            by_source[record["source"]] = by_source.get(record["source"], 0) + 1
        else:
            kept.append(record)
    return kept, {"in": len(records), "out": len(kept), "dropped_total": len(records) - len(kept),
                  "dropped": {k: v for k, v in dropped.items() if v}}


# ---- decision line ---------------------------------------------------------------------------------------------

_DECISION_LINE = re.compile(r"Decision: (?:fold|check|call \d+|bet \d+|raise to \d+)")


def decision_text(args: dict, legal: str | None) -> str | None:
    action, amount = args.get("action"), args.get("amount")
    if action in ("fold", "check"):
        return action
    if action == "call":
        found = re.search(r"\bcall (\d+)", legal or "")
        return f"call {found[1]}" if found else None
    if action in ("bet", "raise") and isinstance(amount, (int, float)) and not isinstance(amount, bool) and amount == int(amount):
        return f"{action} {int(amount)}" if action == "bet" else f"raise to {int(amount)}"
    return None


def add_decision_line(records) -> tuple[list[dict], dict]:
    """Append "Decision: fold | check | call N | bet N | raise to N" as the last reasoning line of every action turn, built from
    the tool call and the prompt's legal line. Idempotent. Turns it cannot describe are left alone and counted."""
    out, stats = [], {"added": 0, "already": 0, "not_action": 0, "unbuildable": 0}
    for record in records:
        if not _is_action(record):
            stats["not_action"] += 1
        elif _DECISION_LINE.fullmatch((record["target"].get("reasoning") or "").rstrip().split("\n")[-1]):
            stats["already"] += 1
        elif (text := decision_text(record["target"]["tool_call"]["arguments"], Prompt(record["messages"]).legal)) is None:
            stats["unbuildable"] += 1
        else:
            reasoning = (record["target"].get("reasoning") or "").rstrip()
            record = record | {"target": record["target"] | {"reasoning": (reasoning + "\n" if reasoning else "") + "Decision: " + text}}
            stats["added"] += 1
        out.append(record)
    return out, stats


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


_HEADER_LINE = re.compile(r"^(Facts|Reads|Talk|Pad|Decision)\b[^\n]*:", re.I)


def strip_model_headers(records) -> tuple[list[dict], dict]:
    """Remove the header lines a model wrote itself (Facts / Reads / Talk / Pad at the top, Decision at the end) from
    action reasoning, so only code-computed headers reach training. Live Facts lines are often wrong."""
    out, n = [], 0
    for record in records:
        if not _is_action(record):
            out.append(record)
            continue
        lines = (record["target"].get("reasoning") or "").split("\n")
        start = 0
        while start < len(lines) and (not lines[start].strip() or _HEADER_LINE.match(lines[start].strip())):
            start += 1
        end = len(lines)
        while end > start and (not lines[end - 1].strip() or lines[end - 1].strip().lower().startswith("decision:")):
            end -= 1
        text = "\n".join(lines[start:end])
        if text != record["target"].get("reasoning"):
            n += 1
            record = record | {"target": record["target"] | {"reasoning": text}}
        out.append(record)
    return out, {"stripped": n}
