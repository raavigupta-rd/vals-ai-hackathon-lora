"""Turn teacher verdicts on our arena decisions into SFT records for poker/sft/build_sft_dataset.py.

Mix per cycle:
- teacher_correction: the teacher disagreed. Target = a short reasoning that states the numbers, then the
  teacher's action. Weighted up, since these are the behaviors we want to change.
- teacher_agree: the teacher judged the spot and agreed. Target = the model's own reply.
- self_clean: a sample of decisions the teacher abstained on, kept as the model wrote them. They hold the reply
  format and general play steady, so corrections do not spread to unrelated spots. Never selected by outcome.

Replies cut off at the token cap, or without exactly one submit_action call, are never targets.

    uv run python -m poker.sft.make_records --joined data/arena/joined.jsonl --cycle 1 --out data/sft/cycle1.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from poker.hints import parse_prompt

CORRECTION_WEIGHT = 2.0  # a correction in the model's own words (resampled)
BUST_RULES = ("allin_call", "own_shove")  # mistakes that bust us: placement drops ~2.4 places
BUST_WEIGHT = 3.0
TEMPLATE_WEIGHT = 1.0  # fixed-sentence fallback: cycle 1 showed these get memorized, so they count less
AGREE_WEIGHT = 1.0
SIMPLE_EXPLOITS = ("big_price_fold", "raise_first_in", "one_pair_pot_cap", "monster_value")  # teacher v6 tier 1
V7_EXPLOITS = ("big_price_fold", "monster_value", "one_pair_pot_cap", "rfi_gto", "iso_raise", "bb_defend", "three_bet",
               "vs_3bet", "field_call", "postflop_sizing")  # teacher v7; raise_first_in is replaced by rfi_gto and iso_raise
V7_RAISE_RULES = ("rfi_gto", "iso_raise", "three_bet", "vs_3bet")  # preflop raise-to corrections with a target_amount
ABSTAIN_WEIGHT = 1.0


def _call(record: dict) -> dict | None:
    calls = [c for c in record.get("tool_calls") or [] if c.get("name") == "submit_action"]
    if len(calls) != 1:
        return None
    args = calls[0].get("arguments")
    try:
        args = json.loads(args) if isinstance(args, str) else args
    except json.JSONDecodeError:
        return None
    return args if isinstance(args, dict) else None


def target_from_reply(record: dict) -> dict | None:
    args = _call(record)
    if args is None or not (record.get("reasoning") or "").strip():
        return None
    if args.get("action") not in ("bet", "raise"):
        args = {k: v for k, v in args.items() if k != "amount"}  # the harness ignores it; teach the clean form
    return {"reasoning": record["reasoning"], "content": record.get("content") or "",
            "tool_call": {"name": "submit_action", "arguments": args}}


def _v7_rfi(verdict: dict, hand_code: str) -> str:
    if verdict["preferred"] == "fold":
        return (f"It is folded to me and {hand_code} is not strong enough to open from this seat. Limping lets everyone "
                "in cheaply and gives up the initiative. Fold.")
    return (f"Everyone folded to me and {hand_code} is a good hand to open from this seat. A raise often takes the "
            "blinds right away and otherwise gives me the lead in the pot; folding or limping gives that up. "
            f"Raise to {verdict['target_amount']}.")


def _v7_iso(verdict: dict, hand_code: str) -> str:
    if verdict["preferred"] == "fold":
        return (f"Someone limped in front of me and {hand_code} is not a hand to play behind a limp. Calling just "
                "adds chips to a pot where I will often be behind. Fold.")
    return (f"Someone limped in front of me and {hand_code} plays well against a weak range. A raise isolates the limper "
            f"and gives me the lead in the pot, while calling lets the whole table in. Raise to {verdict['target_amount']}.")


def _v7_three_bet(verdict: dict, hand_code: str) -> str:
    return (f"{hand_code} is a strong hand against an open from this seat. Calling or folding gives up value; "
            f"a re-raise builds the pot while I am ahead and takes the lead. Raise to {verdict['target_amount']}.")


def _v7_bb_defend(verdict: dict, hand_code: str) -> str:
    if verdict["preferred"] == "fold":
        return (f"{hand_code} is an offsuit hand that rarely makes a strong hand, and I would play it out of position. "
                "Calling with it loses chips over time. Fold.")
    return (f"I am in the big blind and already have chips in, so I need only about {verdict['required']:.0%} equity to "
            f"call. {hand_code} has more than that against this opener. Call.")


def _v7_vs_3bet(verdict: dict, hand_code: str) -> str:
    if verdict["preferred"] == "fold":
        return (f"I raised and got re-raised. A re-raise usually shows a strong hand, and {hand_code} is not good enough "
                "to continue. Fold.")
    if verdict["preferred"] == "raise":
        return (f"I raised and got re-raised, and {hand_code} is one of the best hands I can have here. Re-raise "
                f"for value. Raise to {verdict['target_amount']}.")
    return (f"I raised and got re-raised. {hand_code} is strong enough to continue and should not fold to one "
            "re-raise at this price. Call.")


def _frac_words(frac: float) -> str:
    """A pot fraction in plain words: 0.33 -> 'a third of', 0.5 -> 'half', 0.7 -> 'two thirds of' ..."""
    names = ((1 / 3, "a third of"), (0.5, "half"), (0.6, "three fifths of"), (2 / 3, "two thirds of"), (0.75, "three quarters of"))
    return min(names, key=lambda n: abs(n[0] - frac))[1]


def _v7_field_call(verdict: dict, hand_code: str) -> str:
    spot = {"open": "a raise this size", "3bet": "a re-raise", "jam": "an all-in or a big re-raise"}.get(
        verdict.get("situation"), "a raise")
    eq, req = verdict.get("equity", 0.0), verdict.get("required", 0.0)
    mine = f"about {round(eq * 100)} in 100"
    price = f"The price needs about {round(req * 100)} in 100"
    if verdict["preferred"] == "fold":
        return (f"Against {spot}, players hold strong hands. {hand_code} wins {mine} against those hands. {price}, so "
                "continuing loses chips over time. Fold.")
    return (f"Against {spot}, players hold a wide mix, and many of those hands are weak. {hand_code} wins {mine} against "
            f"them. {price}, so it is a profitable spot. Call.")


def _v7_sizing(verdict: dict, hand_code: str) -> str:
    if verdict["preferred"] == "check":
        return ("I have no pair and no draw. Betting only folds out hands I beat and gets called or raised by hands that "
                "beat me. Check.")
    size = _frac_words(verdict["target_frac"])
    if verdict.get("texture") == "wet":
        why = "On a wet board I charge draws with a bet of about two thirds to three quarters of the pot"
    elif verdict.get("texture") == "river":
        why = "On the river a bet of about three fifths of the pot gets called by worse hands"
    else:
        why = "On a dry board a small bet does the job because there are few draws to charge"
    cost = "A bet this large risks too much" if verdict.get("reason") == "overbet" else "This size does not fit the board"
    return f"{cost} with a hand that is not a monster. {why}: bet about {size} the pot."


V7_TEXTS = {"rfi_gto": _v7_rfi, "iso_raise": _v7_iso, "three_bet": _v7_three_bet, "bb_defend": _v7_bb_defend,
            "vs_3bet": _v7_vs_3bet, "field_call": _v7_field_call, "postflop_sizing": _v7_sizing}


def correction_reasoning(verdict: dict, hand_code: str, pad_rewritten: bool = False) -> str:
    """pad_rewritten: the record shows a rewritten teacher pad, so text may point at it (rule pad_read only)."""
    rule, preferred = verdict["rule"], verdict["preferred"]
    if rule == "allin_call":
        price = verdict["to_call"] / (verdict["pot"] + verdict["to_call"])
        need = f"{verdict['required']:.0%}"
        have = f"{verdict['equity']:.0%}"
        if preferred == "fold":
            return (f"I face an all-in: {verdict['to_call']:g} more into a pot of {verdict['pot']:g}, so the price is "
                    f"{price:.0%} of the final pot. Calling risks my stack, and busting ranks last, so I need about "
                    f"{need} equity. Against this player's shoving range, {hand_code} has about {have}. Fold.")
        return (f"I face an all-in: {verdict['to_call']:g} more into a pot of {verdict['pot']:g}. I need about {need} "
                f"equity and {hand_code} has about {have} against this player's shoving range. This is a clear call.")
    if rule == "postflop_call":
        need, have = f"{verdict['required']:.0%}", f"{verdict['equity']:.0%}"
        facing = (f"On the {verdict['street']} I face a bet: {verdict['to_call']:g} more into a pot of "
                  f"{verdict['pot']:g}, so I need about {need} equity.")
        if preferred == "fold":
            return (f"{facing} Against the hands this player bets this way, {hand_code} has about {have} on this "
                    f"board. That is not enough. Fold.")
        return (f"{facing} Against the hands this player bets this way, {hand_code} has about {have} on this "
                f"board. That is enough, so call.")
    if rule == "unopened":
        if preferred == "raise":
            return f"{hand_code} is a premium hand and the pot is unopened. Raise for value."
        if preferred == "check":
            return f"{hand_code} is weak and checking is free. Check and see the flop."
        return (f"{hand_code} is too weak to play from this seat, and limping or raising it loses chips over time. "
                f"Fold and wait for a better spot.")
    if rule == "facing_raise":
        return (f"Someone has already raised. {hand_code} is not strong enough to continue against a raise, and "
                f"calling or re-raising with it loses chips and risks my stack. Fold.")
    if rule == "own_shove":
        return (f"Going all-in with {hand_code} at this stack depth risks my whole stack with a hand that is behind "
                f"most calling ranges. Busting ranks last. Fold and wait for a better spot.")
    if rule == "river_fold":
        return "On the river a bet this size is rarely a bluff; a pair is not enough. Fold."
    if rule == "monster_call":
        return "I have a very strong hand and the price is fine; folding here gives up a pot I am well ahead in. Call."
    if rule == "river_air_fold":
        return "On the river my hand has nothing; calling only beats a bluff, and bets here are rarely bluffs. Fold."
    if rule == "steal":
        return "Folded to me in late position: the blinds fold most of the time to a small raise. Raise."
    if rule == "stab":
        return "They checked, which usually means weakness; a half-pot bet often takes it down. Bet."
    if rule == "oversize":
        return ("A smaller raise does the same job here and keeps the pot manageable; there is no need to build a "
                f"huge pot before the flop. Raise to {verdict['target_amount']}.")
    if rule == "free_fold":
        return "Checking costs nothing, so folding can only lose. Check."
    if rule == "big_price_fold":
        return ("This is a big price before the flop. Only the strongest hands can continue against this much "
                "pressure; this hand is not one of them. Fold.")
    if rule == "pad_read":
        if preferred == "call":
            if not pad_rewritten:
                return f"{hand_code} is ahead of a wide shoving range at this price. Call."
            return ("My scratchpad shows this player plays most hands and raises often, so their range is wide. "
                    f"{hand_code} is ahead of it at this price. Call.")
        if not pad_rewritten:
            return f"A big raise from a player who plays almost no hands is very strong. {hand_code} is not good enough. Fold."
        return ("My scratchpad shows this player plays almost no hands, so a big raise from them is very strong. "
                f"{hand_code} is not good enough. Fold.")
    if rule == "raise_first_in":
        if preferred == "fold":
            return ("It is folded to me in an early seat. Limping with this hand loses chips over time. Fold.")
        return (f"Nobody has raised yet, and {hand_code} is good enough to open from here. A small raise often wins the "
                "blinds right away and otherwise builds a pot I start ahead in; folding or limping gives that up. "
                f"Raise to {verdict['target_amount']}.")
    if rule == "one_pair_pot_cap":
        action = "Check" if preferred == "check" else "Fold"
        return f"One pair is not strong enough to play a big pot; this bet or raise shows a strong range. {action}."
    if rule == "monster_value":
        if preferred == "call":
            return "I have a very strong hand. I should not fold to a normal-size bet. Call."
        return ("I have a very strong hand. I should not give the opponent a free card or fold for a normal-size bet; "
                "bet about two thirds of the pot for value.")
    if rule in V7_TEXTS:
        return V7_TEXTS[rule](verdict, hand_code)
    raise ValueError(f"no correction text for rule {rule!r}")


def _correction_args(verdict: dict, record: dict) -> dict | None:
    preferred = verdict["preferred"]
    if preferred in ("fold", "check", "call"):
        return {"action": preferred}
    if verdict["rule"] in ("raise_first_in", "monster_value"):
        state = parse_prompt(record["messages"])
        if not state or state["raise_min"] is None:
            return None
        if verdict["rule"] == "raise_first_in":
            if state["raise_type"] != "raise":
                return None
            target, action = verdict["target_amount"], "raise"
        else:
            if state["raise_type"] != "bet":
                return None
            target, action = round(verdict["target_frac"] * state["pot"]), "bet"
        return {"action": action, "amount": int(min(max(state["raise_min"], target), state["raise_max"]))}
    if verdict["rule"] in V7_RAISE_RULES or verdict["rule"] == "postflop_sizing":
        state = parse_prompt(record["messages"])
        if verdict["rule"] == "postflop_sizing":
            if not state or state["raise_min"] is None or state["raise_type"] != "bet":
                return None
            target, action = round(verdict["target_frac"] * state["pot"]), "bet"
        else:
            if not state or state["raise_min"] is None or state["raise_type"] != "raise":
                return None
            target, action = verdict["target_amount"], "raise"
        return {"action": action, "amount": int(min(max(state["raise_min"], target), state["raise_max"]))}
    if verdict["rule"] == "oversize":
        state = parse_prompt(record["messages"])
        if not state or state["raise_min"] is None or state["raise_type"] != "raise":
            return None
        return {"action": "raise", "amount": int(min(max(state["raise_min"], verdict["target_amount"]), state["raise_max"]))}
    state = parse_prompt(record["messages"])
    if not state or state["raise_min"] is None or state["raise_type"] not in ("raise", "bet"):
        return None
    if state["raise_type"] == "bet" and verdict["rule"] != "stab":
        return None
    if state["raise_type"] == "raise" and verdict["rule"] == "stab":
        return None
    if verdict["rule"] == "steal":
        amount = min(max(state["raise_min"], 5), state["raise_max"])  # 2.5bb
    elif verdict["rule"] == "stab":
        if state["raise_type"] != "bet":
            return None
        amount = min(max(state["raise_min"], round(0.5 * state["pot"])), state["raise_max"])
        return {"action": "bet", "amount": int(amount)}
    else:
        amount = min(max(state["raise_min"], 6), state["raise_max"])
    return {"action": "raise", "amount": int(amount)}


def select(rows, *, cycle: int, abstain_ratio: float = 2.0, seed: int = 0, resampled: dict | None = None,
           template_weight: float = TEMPLATE_WEIGHT, weight_scale: float = 1.0, rules: tuple | None = None,
           strip_say: bool = True, pad_rewritten=frozenset()):
    """rows: (joined record, teacher verdict, hand code). Returns (records, stats).

    pad_rewritten: span ids whose action prompt shows a rewritten teacher pad. Only those pad_read corrections may
    cite the scratchpad in their text.

    strip_say removes the table-talk "say" argument from every target's tool call: the best arena teams never talk."""
    resampled = resampled or {}
    stats = {"corrections": 0, "resampled": 0, "agreements": 0, "abstain_pool": 0, "abstain_kept": 0, "skipped": 0, "unvalidated_dropped": 0, "out_of_scope": 0,
             "say_stripped": 0}
    records, pool, stripped = [], [], set()

    def base(record, source, weight, target):
        args = target["tool_call"]["arguments"]
        if strip_say and "say" in args:
            target = target | {"tool_call": target["tool_call"] | {"arguments": {k: v for k, v in args.items() if k != "say"}}}
            stripped.add(record["span_id"])
        return {"decision_id": record["span_id"], "messages": record["messages"], "target": target,
                "weight": weight * weight_scale, "source": source, "cycle": cycle}

    for record, verdict, code in rows:
        if record.get("cutoff") or _call(record) is None:
            stats["skipped"] += 1
            continue
        if rules is not None and verdict.get("agree") is not None and verdict.get("rule") not in rules:
            # Ablation: this rule is switched off, so the spot is an anchor that keeps base's own action.
            stats["out_of_scope"] += verdict["agree"] is False
            verdict = {"rule": verdict.get("rule"), "preferred": None, "agree": None}
        if verdict.get("agree") is False:
            own = resampled.get(record["span_id"])
            if own and own.get("preferred", verdict["preferred"]) != verdict["preferred"]:
                own = None  # sampled for an older verdict: the teacher now prefers something else
            if own:
                target = {"reasoning": own["reasoning"], "content": own.get("content") or "",
                          "tool_call": {"name": "submit_action", "arguments": own["arguments"]}}
                weight = BUST_WEIGHT if verdict.get("rule") in BUST_RULES else CORRECTION_WEIGHT
                records.append(base(record, "teacher_resampled", weight, target))
                stats["resampled"] += 1
                stats["corrections"] += 1
                continue
            if template_weight <= 0:  # keep only corrections validated by a graded re-rollout
                stats["unvalidated_dropped"] += 1
                continue
            args = _correction_args(verdict, record)
            if args is None:
                stats["skipped"] += 1
                continue
            target = {"reasoning": correction_reasoning(verdict, code, record["span_id"] in pad_rewritten), "content": "",
                      "tool_call": {"name": "submit_action", "arguments": args}}
            records.append(base(record, "teacher_correction", template_weight, target))
            stats["corrections"] += 1
        elif verdict.get("agree") is True and (target := target_from_reply(record)):
            records.append(base(record, "teacher_agree", AGREE_WEIGHT, target))
            stats["agreements"] += 1
        elif verdict.get("agree") is None and (target := target_from_reply(record)):
            pool.append(base(record, "self_clean", ABSTAIN_WEIGHT, target))
    stats["abstain_pool"] = len(pool)
    keep = min(len(pool), round(abstain_ratio * (stats["corrections"] + stats["agreements"])))
    random.Random(seed).shuffle(pool)
    records += pool[:keep]
    stats["abstain_kept"] = keep
    stats["say_stripped"] = sum(r["decision_id"] in stripped for r in records)
    return records, stats


HALLUCINATION_DROP_SOURCES = ("teacher_agree", "self_clean", "teacher_resampled")  # the model's own words


def _is_action_target(record: dict) -> bool:
    return record["target"]["tool_call"]["name"] == "submit_action"


def drop_long(records, max_chars: int) -> tuple[list[dict], dict]:
    """Drop action-turn records whose target reasoning exceeds `max_chars`: long targets teach long replies, and the
    arena cuts replies off at the token cap. Cleanup and pad records are kept."""
    out, dropped = [], {}
    for record in records:
        if _is_action_target(record) and len(record["target"].get("reasoning") or "") > max_chars:
            dropped[record["source"]] = dropped.get(record["source"], 0) + 1
            continue
        out.append(record)
    return out, {"dropped": dropped}


def add_fact_headers(records) -> tuple[list[dict], dict]:
    """Prepend poker.facts.fact_line(prompt) and a newline to every action-turn target reasoning (any source).
    Cleanup-turn and pad records are left alone; so are records whose prompt cannot be parsed. Idempotent."""
    from poker.facts import fact_line

    out, stats = [], {"headed": 0, "no_facts": 0, "not_action": 0}
    for record in records:
        if not _is_action_target(record):
            stats["not_action"] += 1
        elif record["target"]["reasoning"].startswith("Facts: "):
            pass
        elif (line := fact_line(record["messages"])) is None:
            stats["no_facts"] += 1
        else:
            target = record["target"] | {"reasoning": line + "\n" + record["target"]["reasoning"]}
            record = record | {"target": target}
            stats["headed"] += 1
        out.append(record)
    return out, stats


def drop_hallucinated(records, sources=HALLUCINATION_DROP_SOURCES) -> tuple[list[dict], dict]:
    """Drop `sources` records whose reasoning contradicts the facts computed from their prompt (poker.facts.contradictions).
    Corrections are only counted: their reasoning is the lesson. Run before add_fact_headers."""
    from poker.facts import compute_facts, contradictions

    out, stats = [], {"checked": 0, "contradicted": {}, "dropped": 0, "no_facts": 0}
    for record in records:
        if _is_action_target(record):
            facts = compute_facts(record["messages"])
            if facts is None:
                stats["no_facts"] += 1
            else:
                stats["checked"] += 1
                if contradictions(record["target"]["reasoning"], facts):
                    stats["contradicted"][record["source"]] = stats["contradicted"].get(record["source"], 0) + 1
                    if record["source"] in sources:
                        stats["dropped"] += 1
                        continue
        out.append(record)
    return out, stats


_LEAK_MARKERS = ("SCRATCHPAD>>>", "<<<SCRATCHPAD")
_HAND_OVER = re.compile(r"## Hand (\d+) is over")
MULTI_TURN_MIN_ACTIONS = 3


def _mentions_hand(contents: str, hand: int) -> bool:
    """True when the pad names the hand: 'Hand 3', 'hand #3', 'H3', '#3' (not inside a longer number)."""
    return re.search(rf"(?:\bhand\s*#?\s*|\bh\s*#?\s*|#\s*){hand}(?!\d)", contents, re.IGNORECASE) is not None


def _cleanup_hand(record: dict) -> int | None:
    for message in reversed(record.get("messages") or []):
        found = _HAND_OVER.search(message.get("content") or "") if isinstance(message.get("content"), str) else None
        if found:
            return int(found.group(1))
    hand = (record.get("match") or {}).get("hand")
    return hand if isinstance(hand, int) else None


def _prompt_action_turns(record: dict) -> int:
    """Hero action turns in this hand, counted from the cleanup prompt: every user message that asks for submit_action."""
    return sum(1 for m in record.get("messages") or []
               if m.get("role") == "user" and isinstance(m.get("content"), str)
               and "Call submit_action with your decision" in m["content"])


def hero_action_index(rows) -> dict:
    """(match_id, hand, seat) -> hero action names in order, from joined action records."""
    found: dict = {}
    for r in rows:
        match = r.get("match") or {}
        if r.get("kind") != "action" or match.get("match_id") is None:
            continue
        args = _call(r) or {}
        key = (match["match_id"], match.get("hand"), match.get("seat"))
        position = match.get("action_index")
        if not isinstance(position, int):
            position = len(found.get(key, []))  # keep arrival order within this hand
        found.setdefault(key, []).append((position, args.get("action")))
    return {k: [a for _, a in sorted(v, key=lambda x: x[0])] for k, v in found.items()}


def is_multi_turn(record: dict, hero_actions: dict | None = None) -> bool:
    """Hand with >= 3 hero action turns, or one whose last hero action was a check."""
    match = record.get("match") or {}
    actions = (hero_actions or {}).get((match.get("match_id"), match.get("hand"), match.get("seat")))
    if actions:
        return len(actions) >= MULTI_TURN_MIN_ACTIONS or actions[-1] == "check"
    return _prompt_action_turns(record) >= MULTI_TURN_MIN_ACTIONS


def cleanup_anchors(records, *, n: int, cycle: int, seed: int = 0, hero_actions: dict | None = None) -> list[dict]:
    """Up to n after-hand cleanup turns answered with exactly one valid update_scratchpad call, as SFT anchors.
    SFT on action turns alone made fine-tunes answer cleanup turns with submit_action (c2b-s12: 67% correct).

    Dropped: pads that leak the SCRATCHPAD delimiters or never mention the current hand number (they teach stale
    or broken pads). Multi-turn hands (see is_multi_turn) are sampled first; single-action hands fill the rest."""
    multi, single = [], []
    for r in records:
        if r.get("kind") != "cleanup" or r.get("cutoff") or not (r.get("reasoning") or "").strip():
            continue
        calls = r.get("tool_calls") or []
        if len(calls) != 1 or calls[0].get("name") != "update_scratchpad":
            continue
        args = calls[0].get("arguments")
        try:
            args = json.loads(args) if isinstance(args, str) else args
        except json.JSONDecodeError:
            continue
        if not isinstance(args, dict) or set(args) != {"contents"} or not isinstance(args["contents"], str) \
                or len(args["contents"]) > 5000:
            continue
        if any(marker in args["contents"] for marker in _LEAK_MARKERS):
            continue
        hand = _cleanup_hand(r)
        if hand is None or not _mentions_hand(args["contents"], hand):
            continue
        item = {"decision_id": r["span_id"], "messages": r["messages"], "weight": 1.0, "source": "cleanup_anchor",
                "cycle": cycle, "target": {"reasoning": r["reasoning"], "content": r.get("content") or "",
                                           "tool_call": {"name": "update_scratchpad", "arguments": args}}}
        (multi if is_multi_turn(r, hero_actions) else single).append(item)
    rng = random.Random(seed)
    rng.shuffle(multi)
    rng.shuffle(single)
    return (multi + single)[:n]


def load_pad_records(path: Path, *, weight: float, cycle: int) -> list[dict]:
    """Teacher-pad SFT records (poker/pad/build_records.py). Each weight is multiplied by `weight`; 0 drops them all."""
    if weight <= 0:
        return []
    out = []
    for line in Path(path).open(encoding="utf-8"):
        if not line.strip():
            continue
        record = json.loads(line)
        out.append(record | {"weight": record.get("weight", 1.0) * weight, "source": record.get("source") or "teacher_pad",
                             "cycle": record.get("cycle", cycle)})
    return out


def apply_critic_weights(records, weights: dict) -> tuple[list[dict], int]:
    """Multiply each record's weight by weights[decision_id] (default 1). A zero weight drops the record.
    Returns (records, number dropped). Input records are not mutated."""
    out, dropped = [], 0
    for record in records:
        factor = float(weights.get(record["decision_id"], 1.0))
        if factor < 0:
            raise ValueError(f"negative critic weight for {record['decision_id']}: {factor}")
        if factor == 0:
            dropped += 1
            continue
        out.append(record | {"weight": record["weight"] * factor})
    return out, dropped


def main() -> None:
    from poker.arena import db
    from poker.teacher import judge

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--joined", type=Path, default=Path("data/arena/joined.jsonl"))
    parser.add_argument("--db", type=Path, default=Path("data/arena/arena.db"))
    parser.add_argument("--deployment", help="only decisions by this deployment id (the current entry)")
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--abstain-ratio", type=float, default=2.0)
    parser.add_argument("--template-weight", type=float, default=0.0,
                        help="weight of unvalidated fixed-sentence corrections; 0 drops them (default)")
    parser.add_argument("--exploits", help="comma list of teacher v5 exploit rules to enable: river_fold,steal,stab,oversize,monster_call,river_air_fold,"
                             "rfi_gto,iso_raise,bb_defend,three_bet,vs_3bet,postflop_sizing,pad_read (needs the teacher pad chain from --joined; pair with --rewrite-pads-strict)")
    parser.add_argument("--cleanup-ratio", type=float, default=0.3,
                        help="cleanup-turn anchors per action record (keeps update_scratchpad working)")
    parser.add_argument("--cleanup-deployment", default="infer_d3d2b6b92171",
                        help="whose cleanup replies to use as anchors (base answers 98.5%% correctly)")
    parser.add_argument("--rules", help="comma list of teacher rules whose corrections to keep (ablations); others become anchors")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--simple", action="store_true",
                      help="v6 tier-1 rules only: big_price_fold, raise_first_in, one_pair_pot_cap, monster_value, "
                           "free_fold; every older rule becomes an anchor (overrides --rules and --exploits)")
    mode.add_argument("--v7", action="store_true",
                      help="teacher v7 rules only: big_price_fold, monster_value, one_pair_pot_cap, free_fold plus rfi_gto, "
                           "iso_raise, bb_defend, three_bet, vs_3bet, postflop_sizing; every other rule becomes an anchor "
                           "(overrides --rules and --exploits)")
    parser.add_argument("--rewrite-pads", action="store_true",
                        help="show each action turn the teacher pad the deterministic writer would have left at the start "
                             "of that hand (poker/pad/rewrite.py); records with no gap-free chain keep their block")
    parser.add_argument("--rewrite-pads-strict", action="store_true",
                        help="implies --rewrite-pads; also drops action records whose pad could not be rewritten when they "
                             "carry a pad-dependent correction (rule pad_read), and reports rewritten/kept/dropped")
    parser.add_argument("--max-reasoning-chars", type=int, default=0,
                        help="drop action targets whose reasoning is longer than this (0 = keep all)")
    parser.add_argument("--pad-version", type=int, choices=(1, 2), default=1,
                        help="writer version for --rewrite-pads: 2 = head-heavy exploit pad (poker/pad/writer.py)")
    parser.add_argument("--fact-header", action="store_true",
                        help="prepend a code-computed 'Facts: position, hand, board, made hand, price, stack, rank' line "
                             "(poker/facts.py) to every action-turn target reasoning")
    parser.add_argument("--drop-hallucinated", action="store_true",
                        help="drop agreements and anchors whose reasoning contradicts those facts (wrong position, a set/"
                             "flush/straight/pair the cards do not make, wrong outs); reports counts, corrections are kept")
    parser.add_argument("--keep-say", action="store_true", help="keep the table-talk 'say' argument in targets (default: removed)")
    parser.add_argument("--weight-scale", type=float, default=1.0, help="multiply every weight (newer cycles weigh more)")
    parser.add_argument("--resampled", type=Path, help="poker/sft/resample.py output: own-words corrections")
    parser.add_argument("--pad-records", type=Path, help="teacher-pad SFT records (poker/pad/build_records.py) to merge in")
    parser.add_argument("--pad-weight", type=float, default=1.0, help="multiply the weight of every --pad-records record (0 drops them)")
    parser.add_argument("--critic-weights", type=Path,
                        help="JSON {span_id: weight} from poker/critic/merge.py: multiplies record weights, weight 0 drops the record")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    resampled = {}
    if args.resampled:
        for line in args.resampled.open(encoding="utf-8"):
            item = json.loads(line)
            if item.get("reply"):
                resampled[item["span_id"]] = item["reply"] | {"preferred": item.get("preferred")}
    rules = tuple(args.rules.split(",")) if args.rules else None
    if args.simple:
        import poker.teacher as teacher

        teacher.EXPLOITS = set(SIMPLE_EXPLOITS)
        rules = SIMPLE_EXPLOITS + ("free_fold",)
    elif args.v7:
        import poker.teacher as teacher

        teacher.EXPLOITS = set(V7_EXPLOITS)
        rules = V7_EXPLOITS + ("free_fold",)
    elif args.exploits:
        import poker.teacher as teacher

        teacher.EXPLOITS = {x for x in args.exploits.split(",") if x}
    import poker.teacher as teacher

    args.rewrite_pads = args.rewrite_pads or args.rewrite_pads_strict
    index = None
    if args.rewrite_pads or "pad_read" in teacher.EXPLOITS:
        from poker.pad.rewrite import chain_index

        index = chain_index(json.loads(line) for line in args.joined.open(encoding="utf-8"))
        if "pad_read" in teacher.EXPLOITS:
            teacher.PAD_INDEX = index  # pad_read reads the raiser's profile from the teacher pad at that hand
    con = db.connect(args.db)
    rows = []
    for line in args.joined.open(encoding="utf-8"):
        record = json.loads(line)
        match = record.get("match") or {}
        if record.get("source") != "arena" or record.get("kind") != "action" or not match.get("verified"):
            continue
        if args.deployment and record.get("deployment") != args.deployment:
            continue
        verdict = judge(con, match["match_id"], match["hand"], match["seat"], match["action_index"])
        row = con.execute("SELECT card1, card2 FROM hole_cards WHERE match_id=? AND hand_number=? AND seat=?",
                          (match["match_id"], match["hand"], match["seat"])).fetchone()
        from poker.teacher import hand_code
        rows.append((record, verdict, hand_code([row[0], row[1]]) if row else "this hand"))
    matches = {}
    for r, _, _ in rows:
        m = r["match"]
        matches[r["span_id"]] = (m.get("match_id"), m.get("seat"), m.get("hand"))
    pad_rewritten = set()
    if args.rewrite_pads:
        from poker.pad.rewrite import teacher_pad_at

        pad_rewritten = {span for span, key in matches.items()
                         if None not in key and teacher_pad_at(index, key[0], key[1], key[2]) is not None}
    records, stats = select(rows, cycle=args.cycle, abstain_ratio=args.abstain_ratio, resampled=resampled,
                            template_weight=args.template_weight, weight_scale=args.weight_scale,
                            rules=rules,
                            strip_say=not args.keep_say, pad_rewritten=pad_rewritten)
    if args.rewrite_pads:
        from poker.pad.rewrite import rewrite_records

        strict = {r["span_id"] for r, v, _ in rows if v.get("rule") == "pad_read"} if args.rewrite_pads_strict else set()
        records, rewrite_stats = rewrite_records(records, matches, index, strict_ids=strict, version=args.pad_version)
        stats["pad_rewrite"] = rewrite_stats
    if args.drop_hallucinated:
        records, stats["hallucination"] = drop_hallucinated(records)
    if args.max_reasoning_chars:
        records, stats["too_long"] = drop_long(records, args.max_reasoning_chars)
    if args.fact_header:
        records, stats["fact_header"] = add_fact_headers(records)
    if args.cleanup_ratio > 0:
        cleanups = [json.loads(line) for line in args.joined.open(encoding="utf-8")]
        cleanups = [r for r in cleanups if r.get("source") == "arena" and r.get("deployment") == args.cleanup_deployment]
        hero = hero_action_index(json.loads(line) for line in args.joined.open(encoding="utf-8"))
        extra = cleanup_anchors(cleanups, n=round(args.cleanup_ratio * len(records)), cycle=args.cycle, hero_actions=hero)
        records += extra
        stats["cleanup_anchors"] = len(extra)
    if args.pad_records:
        pads = load_pad_records(args.pad_records, weight=args.pad_weight, cycle=args.cycle)
        records += pads
        stats["pad_records"] = len(pads)
    if args.critic_weights:
        records, dropped = apply_critic_weights(records, json.loads(args.critic_weights.read_text(encoding="utf-8")))
        stats["critic_dropped"] = dropped
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    print(json.dumps(stats | {"records": len(records), "out": str(args.out)}))


if __name__ == "__main__":
    main()
