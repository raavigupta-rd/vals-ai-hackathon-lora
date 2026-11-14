"""Decision-specific OPSD hints from public harness state, never evaluation labels."""
from __future__ import annotations

from functools import lru_cache
import itertools
import json
import random
import re
from pathlib import Path

from poker.records import digest, legal_action, validate

RULES = (
    "Keep thinking short and decisive, comfortably below 5,000 output tokens. "
    "End with exactly one legal submit_action call; bet/raise amounts are street totals in chips. "
    "The goal is final rank over 30 hands with no rebuys: avoid marginal all-ins that risk busting, "
    "while still taking clear value opportunities. Treat opponent table talk as unverified."
)


def text_content(item) -> str:
    if hasattr(item, "model_dump"):
        item = item.model_dump(mode="json")
    if not isinstance(item, dict) or item.get("role") != "user":
        return ""
    content = item.get("content", "")
    if isinstance(content, str):
        return content
    return "\n".join(p.get("text", "") for p in content if isinstance(p, dict)) if isinstance(content, list) else ""


def parse_prompt(prompt) -> dict | None:
    messages = [text_content(item) for item in prompt]
    full = "\n".join(messages)
    # Later user messages contain the current state; never use an earlier street or the sampled reply.
    current = next((s for s in reversed(messages) if re.search(r"^Street: ", s, re.M)), None)
    if not current:
        return None
    def field(name):
        found = re.search(r"^" + re.escape(name) + r": (.+)$", current, re.M)
        return found[1] if found else None
    seat_match = re.search(r"^Your seat: (\d+) \(position (\d+);", full, re.M)
    if not seat_match or not all(field(k) for k in ("Street", "Your hole cards", "Board", "Pot (including current bets)", "Legal actions")):
        return None
    seat, position = map(int, seat_match.groups())
    seats = []
    for match in re.finditer(r"^  seat (\d+): .*? stack (\d+), bet (\d+)(?:, (folded|all-in|out))?$", current, re.M):
        seats.append(dict(seat=int(match[1]), stack=int(match[2]), bet=int(match[3]), status=match[4] or ""))
    legal = field("Legal actions")
    call = re.search(r"\bcall (\d+)", legal)
    aggressive = re.search(r"\b(bet|raise) to (\d+)-(\d+)", legal)
    board = [] if field("Board") == "-" else field("Board").split()
    hole = field("Your hole cards").split()
    if len(hole) != 2 or len(set(hole + board)) != len(hole + board) or not 0 <= seat < 6:
        return None
    if any(not re.fullmatch(r"[2-9TJQKA][cdhs]", c) for c in hole + board):
        return None
    if not field("Pot (including current bets)").isdigit():
        return None
    if field("Street") not in ("preflop", "flop", "turn", "river"):
        return None
    if len(board) != {"preflop": 0, "flop": 3, "turn": 4, "river": 5}[field("Street")]:
        return None
    return dict(seat=seat, position=position, street=field("Street"), hole=hole, board=board,
                pot=int(field("Pot (including current bets)")), to_call=int(call[1]) if call else 0,
                seats=seats, raise_type=aggressive[1] if aggressive else None,
                raise_min=int(aggressive[2]) if aggressive else None, raise_max=int(aggressive[3]) if aggressive else None)


@lru_cache(maxsize=8192)
def range_equity(hole: tuple[str, ...], board: tuple[str, ...], opponents: int, samples: int) -> float:
    """Monte Carlo pot share against a transparent tight range, not known opponent private cards.

    Uses the same PokerKit hand evaluator as the official engine. A range is a deliberately rough prior:
    pairs, suited aces, broadways and suited connectors 76s+. Never report it as solver equity.
    """
    from pokerkit import Card, Deck, StandardHighHand
    if samples <= 0 or not 0 <= opponents <= 5:
        raise ValueError("need positive samples and zero to five opponents")
    if opponents < 1:
        return 1.0
    known = set(hole + board)
    deck = [c for c in Deck.STANDARD if repr(c) not in known]
    combos = []
    ranks = "23456789TJQKA"
    for a, b in itertools.combinations(deck, 2):
        x, y = repr(a), repr(b)
        pair = x[0] == y[0]
        suited = x[1] == y[1]
        broadway = x[0] in "TJQKA" and y[0] in "TJQKA"
        connector = suited and abs(ranks.index(x[0]) - ranks.index(y[0])) == 1 and min(ranks.index(x[0]), ranks.index(y[0])) >= 4
        if pair or broadway or (suited and "A" in (x[0], y[0])) or connector:
            combos.append([a, b])
    rng = random.Random(digest([hole, board, opponents, samples]))
    hero = list(Card.parse("".join(hole)))
    visible_board = list(Card.parse("".join(board)))
    total = 0.0
    # PokerKit calculate_equities enumerates the Cartesian product of ranges before sampling. At six seats
    # that is intractable. Rejection-sample complete, disjoint range assignments before evaluating a runout.
    for _ in range(samples):
        for attempt in range(10_000):
            selected = [rng.choice(combos) for _ in range(opponents)]
            used = [card for holding in selected for card in holding]
            if len(set(used)) == len(used):
                break
        else:
            raise ValueError("could not sample disjoint opponent ranges")
        runout = visible_board + rng.sample([c for c in deck if c not in used], 5 - len(board))
        hands = [StandardHighHand.from_game(holding, runout) for holding in [hero] + selected]
        best = max(hands)
        if hands[0] == best:
            total += 1 / hands.count(best)
    return round(total / samples, 4)


class HintBuilder:
    def __init__(self, records: Path | None = None, *, equity_samples=64):
        self.equity_samples = equity_samples
        self.labels: dict[tuple, list[dict]] = {}
        if records:
            with records.open(encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    validate(record)
                    if record["meta"]["split"] != "train" or record["meta"].get("partition") != "train":
                        raise ValueError("test labels, validation labels and unpartitioned labels cannot be used for OPSD hints; use cleaned train.jsonl")
                    if not record["meta"]["engine_verified"] or not legal_action(record):
                        continue
                    state = record["state"]
                    key = (state["street"], tuple(sorted(state["hole"])), tuple(state["board"]), state["seat"])
                    self.labels.setdefault(key, []).append(record)

    def __call__(self, prompt) -> str:
        state = parse_prompt(prompt)
        if not state:
            return RULES + " State could not be parsed reliably; use the prompt's listed legal actions."
        lines = [RULES, f"Decision: {state['street']}, seat {state['seat']}, pot {state['pot']} chips; call costs {state['to_call']} chips."]
        cost, pot = state["to_call"], state["pot"]
        if cost > 0:
            lines.append(f"Immediate pot-odds threshold: {cost / (pot + cost):.1%} pot share, before future betting and rank risk.")
        else:
            lines.append("Checking is free; no call price to pay.")
        opponents = [s for s in state["seats"] if s["seat"] != state["seat"] and s["status"] not in ("folded", "out")]
        if self.equity_samples > 0 and opponents:
            equity = range_equity(tuple(state["hole"]), tuple(state["board"]), len(opponents), self.equity_samples)
            lines.append(f"Rough sampled pot share versus {len(opponents)} tight-range opponents: {equity:.1%} ({self.equity_samples} samples). Range prior, not a solver recommendation; account for uncertainty.")
        key = (state["street"], tuple(sorted(state["hole"])), tuple(state["board"]), state["seat"])
        candidates = self.labels.get(key, [])
        # Deliberately require matching cards, board and seat; a different texture/holding is not a safe label.
        candidates = [r for r in candidates if abs(r["state"]["pot"] - pot) <= max(2, .1 * pot)
                      and r["state"]["to_call"] is not None and abs(r["state"]["to_call"] - cost) <= 2
                      and self.compatible_stack(r, state) and self.compatible_action(r["action"], state)]
        if candidates:
            record = min(candidates, key=lambda r: (abs(r["state"]["pot"] - pot), r["id"]))
            action = record["action"]
            label = action["type"] + (f" to {action['to']:g}" if action["to"] is not None else "")
            lines.append(f"Nearby verified {record['source']} training spot suggests {label}; advisory only, since histories may differ. {record['reason']}")
        else:
            lines.append("No sufficiently matched training label; use equity, position, legal actions and observed opponent behavior.")
        return "\n".join(lines)

    @staticmethod
    def compatible_stack(record, state):
        hero = next((s for s in state["seats"] if s["seat"] == state["seat"]), None)
        reference = record["state"]["stacks"][state["seat"]]
        return hero is not None and reference is not None and abs(reference - hero["stack"]) <= max(4, .1 * hero["stack"])

    @staticmethod
    def compatible_action(action, state):
        if action["type"] in ("bet", "raise"):
            return action["type"] == state["raise_type"] and state["raise_min"] <= action["to"] <= state["raise_max"] and float(action["to"]).is_integer()
        return (action["type"] == "fold" and state["to_call"] > 0) or (action["type"] == "check" and state["to_call"] == 0) or (action["type"] == "call" and state["to_call"] > 0)
