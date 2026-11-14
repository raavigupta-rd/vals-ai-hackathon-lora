"""Convert the actual PokerBench CSVs, retaining provenance and auditing engine compatibility.

Run from the repository root: python -m poker.pokerbench_to_records
Amounts in PokerBench are BB (SB=.5, BB=1); multiply by two for event chips.
Unknown or inconsistent state is retained explicitly and never certified as engine-verified.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import itertools
import copy
import json
from pathlib import Path
import re
import sys
from jsonschema import ValidationError

from poker.records import POSITIONS, STREETS, digest, validate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness/src"))
from pokerkit import Card, Deck, Mode, NoLimitTexasHoldem
from poker_harness.engine.table import AUTOMATIONS, Decision, legal_actions, resolve


def chips(value: str | float) -> float:
    return round(float(str(value).lower().replace("bb", "")) * 2, 6)


def cards(text: str) -> list[str]:
    found = re.findall(r"[2-9TJQKA][cdhs]", text)
    if "".join(found) != text:
        raise ValueError(f"invalid card string {text!r}")
    return found


def action(text: str, *, preflop: bool = False) -> dict:
    text = text.strip().lower()
    if text in ("fold", "check", "call"):
        return {"type": text, "to": None}
    if text == "allin":
        return {"type": "raise", "to": 200}
    if preflop:
        return {"type": "raise", "to": chips(text)}
    match = re.fullmatch(r"(bet|raise)\s+(\d+(?:\.\d+)?)", text)
    if not match:
        raise ValueError(f"unknown move {text!r}")
    return {"type": match[1], "to": chips(match[2])}


def preflop_history(text: str) -> list[dict]:
    tokens = text.split("/") if text else []
    if len(tokens) % 2:
        raise ValueError("unpaired preflop history")
    return [dict(street="preflop", seat=POSITIONS.index(pos), action=move["type"], to=move["to"])
            for pos, label in zip(tokens[::2], tokens[1::2])
            for move in [action(label, preflop=True)]]


def parse_row(row: dict, filename: str) -> dict:
    pre = filename.startswith("preflop")
    street = "preflop" if pre else row["evaluation_at"].lower()
    history = preflop_history(row["prev_line"] if pre else row["preflop_action"])
    board: list[str] = []
    if pre:
        position = row["hero_pos"]
    else:
        # Only the two preflop participants survive. Their order postflop is SB..BTN.
        participants = sorted({h["seat"] for h in history if h["action"] != "fold"})
        folded = {h["seat"] for h in history if h["action"] == "fold"}
        participants = [s for s in participants if s not in folded]
        if len(participants) != 2:
            raise ValueError("postflop participant mapping is not heads-up")
        seats = dict(zip(("OOP", "IP"), participants))
        position = POSITIONS[seats[row["hero_position"]]]
        board = cards(row["board_flop"])
        if street in ("turn", "river"):
            board += cards(row["board_turn"])
        if street == "river":
            board += cards(row["board_river"])
        current = 1
        for token in row["postflop_action"].split("/"):
            if token == "dealcards":
                current += 1
            elif "_" in token:
                if current > STREETS.index(street):
                    raise ValueError("history continues beyond evaluation street")
                pos, *parts = token.split("_")
                move = action(" ".join(parts))
                history.append(dict(street=STREETS[current], seat=seats[pos], action=move["type"], to=move["to"]))
    label = action(row["correct_decision"], preflop=pre)
    moves = [action(x, preflop=pre) for x in ast.literal_eval(row["available_moves"])]
    if label not in moves:
        raise ValueError("label absent from source available moves")
    aggressive = [m for m in moves if m["type"] in ("bet", "raise")]
    legal = dict(fold=any(m["type"] == "fold" for m in moves),
                 check=any(m["type"] == "check" for m in moves), call=None,
                 raise_type=aggressive[0]["type"] if aggressive else None,
                 raise_min=None, raise_max=None)
    state = dict(hand_index=None, street=street, hole=cards(row["hero_holding"] if pre else row["holding"]),
                 board=board, seat=POSITIONS.index(position), position=position, pot=chips(row["pot_size"]),
                 to_call=None, stacks=[None] * 6, effective_stack=None, legal=legal, history=history)
    # PokerBench does not expose hand IDs. A conservative scenario family groups all holdings/streets sharing
    # a preflop line and flop; this can over-group hands but cannot split their later decisions apart.
    family = [row["prev_line"] if pre else row["preflop_action"], "" if pre else row["board_flop"]]
    return dict(id=f"pb-{filename.split('_game_')[0]}-{row['']}", source="pokerbench",
                format=dict(players=6, sb=1, bb=2, starting_stack=200), state=state, action=label,
                reason=f"PokerBench solver label: {row['correct_decision']}.", solver_freqs=None,
                meta=dict(hand_id=digest(family), split="test" if "_test_" in filename else "train",
                          engine_verified=False, source_file=filename, source_row=int(row[""]),
                          source_unit="bb", chip_scale=2, available_moves=moves,
                          hand_id_kind="conservative_scenario_family"))


def _verify_engine(record: dict) -> None:
    """Replay known actions through the official engine. Only omitted preflop folds may be inferred.

    No opponent private cards are used for labels/equity. Arbitrary disjoint cards suffice for betting legality.
    A source pot mismatch fails verification rather than manufacturing a state to match the solver label.
    """
    data = record["state"]
    visible = data["hole"] + data["board"]
    available = iter(c for c in Deck.STANDARD if repr(c) not in visible)
    state = NoLimitTexasHoldem.create_state(AUTOMATIONS, True, 0, (1, 2), 2, (200,) * 6, 6, mode=Mode.CASH_GAME)
    # Deal in the same order as the official HandRunner (one card at a time around the table).
    holes = {s: iter(list(Card.parse("".join(data["hole"]))) if s == data["seat"] else [next(available), next(available)])
             for s in range(6)}
    while state.can_deal_hole():
        recipient = state.hole_dealee_index
        state.deal_hole(next(holes[recipient]))
    board = iter(Card.parse("".join(data["board"])))
    replayed = []

    def deal():
        while True:
            if state.can_burn_card():
                state.burn_card(next(available))
            elif state.can_deal_board():
                state.deal_board(tuple(next(board) for _ in range(state.board_dealing_count or 0)))
            else:
                break

    def apply(move):
        deal()
        if state.actor_index != move["seat"] or STREETS[state.street_index or 0] != move["street"]:
            raise ValueError("source action order differs from engine")
        requested = Decision(move["action"], move["to"])
        _, error = resolve(requested, legal_actions(state))
        if error:
            raise ValueError(error)
        if requested.type == "fold":
            state.fold()
        elif requested.type in ("check", "call"):
            state.check_or_call()
        else:
            state.complete_bet_or_raise_to(requested.amount)
        replayed.append(move)

    for move in data["history"]:
        deal()
        for _ in range(6):
            if state.actor_index == move["seat"] and STREETS[state.street_index or 0] == move["street"]:
                break
            if state.street_index != 0 or state.actor_index is None:
                raise ValueError("missing non-preflop action")
            apply(dict(street="preflop", seat=state.actor_index, action="fold", to=None))
        apply(move)
    deal()
    for _ in range(6):
        if state.actor_index == data["seat"] and STREETS[state.street_index or 0] == data["street"]:
            break
        if state.street_index != 0 or state.actor_index is None:
            raise ValueError("decision actor/street mismatch")
        apply(dict(street="preflop", seat=state.actor_index, action="fold", to=None))
        deal()
    if state.actor_index != data["seat"] or STREETS[state.street_index or 0] != data["street"]:
        raise ValueError("decision actor/street mismatch")
    if abs(float(state.total_pot_amount) - data["pot"]) > 1e-5:
        raise ValueError("source pot differs from engine replay")
    legal = legal_actions(state)
    label = record["action"]
    _, error = resolve(Decision(label["type"], label["to"]), legal)
    if error:
        raise ValueError("solver label illegal: " + error)
    data["history"] = replayed
    data["stacks"] = list(state.stacks)
    data["to_call"] = legal.call_amount
    data["legal"] = dict(fold=legal.can_fold, check=legal.can_check, call=legal.call_amount,
                         raise_type=legal.raise_type, raise_min=legal.min_raise_to if legal.raise_type else None,
                         raise_max=legal.max_raise_to if legal.raise_type else None)
    hero = data["seat"]
    live = [s for s in range(6) if s != hero and state.statuses[s]]
    data["effective_stack"] = min(state.stacks[hero], max((state.stacks[s] for s in live), default=0))
    record["meta"]["engine_verified"] = True


_REPLAY_CACHE: dict[str, dict | str] = {}


def verify_engine(record: dict) -> None:
    # Betting legality is independent of private card values. Solver exports repeat each betting scenario for
    # many holdings. Reuse the exact audited state, preserving this record's cards and source label.
    state = record["state"]
    key = digest([state["history"], state["seat"], state["street"], state["pot"]])
    cached = _REPLAY_CACHE.get(key)
    if isinstance(cached, str):
        raise ValueError(cached)
    if cached is None:
        trial = copy.deepcopy(record)
        try:
            # Check the state once, separately from the individual solver label.
            trial["action"] = {"type": "fold" if not state["legal"]["check"] else "check", "to": None}
            _verify_engine(trial)
        except (ValueError, StopIteration, TypeError, KeyError) as e:
            message = str(e) or type(e).__name__
            _REPLAY_CACHE[key] = message
            raise ValueError(message) from e
        cached = {k: trial["state"][k] for k in ("history", "stacks", "to_call", "legal", "effective_stack")}
        if len(_REPLAY_CACHE) < 100_000:
            _REPLAY_CACHE[key] = cached
    trial = copy.deepcopy(record)
    trial["state"].update(copy.deepcopy(cached))
    from poker.records import legal_action
    if not legal_action(trial):
        raise ValueError("solver label illegal under official engine")
    record["state"].update(copy.deepcopy(cached))
    record["meta"]["engine_verified"] = True


def convert(source: Path, output: Path, limit: int | None = None) -> dict:
    counts = Counter()
    output.mkdir(parents=True, exist_ok=True)
    paths = {s: output / ("eval" if s == "test" else "raw") / "pokerbench.jsonl" for s in ("train", "test")}
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    from contextlib import ExitStack
    with ExitStack() as stack:
        streams = {s: stack.enter_context(p.open("w", encoding="utf-8")) for s, p in paths.items()}
        rejects = stack.enter_context((output / "pokerbench_rejects.jsonl").open("w", encoding="utf-8"))
        for path in sorted(source.glob("*_game_scenario_information.csv")):
            with path.open(newline="", encoding="utf-8-sig") as f:
                for row in itertools.islice(csv.DictReader(f), limit):
                    counts["input"] += 1
                    if counts["input"] % 10_000 == 0:
                        print(f"converted {counts['input']} rows; {counts['engine_verified']} engine-verified", file=sys.stderr, flush=True)
                    try:
                        record = parse_row(row, path.name)
                        try:
                            verify_engine(record)
                        except (ValueError, StopIteration, TypeError, KeyError) as e:
                            record["meta"]["verification_error"] = str(e) or type(e).__name__
                            counts["unverified:" + record["meta"]["verification_error"]] += 1
                        validate(record)
                    except (ValueError, KeyError, TypeError, ValidationError) as e:
                        counts["rejected"] += 1
                        rejects.write(json.dumps(dict(file=path.name, row=row.get(""), error=str(e))) + "\n")
                        continue
                    split = record["meta"]["split"]
                    counts[split] += 1
                    counts["engine_verified"] += record["meta"]["engine_verified"]
                    streams[split].write(json.dumps(record, separators=(",", ":")) + "\n")
    report = dict(counts)
    (output / "pokerbench_stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/pokerbench"))
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--limit", type=int, help="rows per file for a smoke test")
    args = parser.parse_args()
    if not list(args.source.glob("*_game_scenario_information.csv")):
        parser.error("no PokerBench CSV files found")
    print(json.dumps(convert(args.source, args.out, args.limit), indent=2))


if __name__ == "__main__":
    main()
