"""GRPO task generators for the self-hosted poker harness (BYOH). Scripted opponents, no LLM endpoint needed.

Contract with the harness (all keys opt-in):
- env_params.opponent_bots: one archetype per non-policy seat, in seat order.
- grader_params.reward: "spot" | "mix" | "cleanup"; weights w_ev, w_place, w_cleanup, w_fold_regret; ev_unit_bb;
  format_penalty.
"""

from __future__ import annotations

import random
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ac2.runtime import Message, Task

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "harness/src"))
from poker_harness.engine.scripted import ARCHETYPES  # noqa: E402
from poker_harness.spots import chen_score, hole_cards  # noqa: E402

from poker.game_tasks import game_datasets  # noqa: E402
from vals_tools.tasks import SEATS, SPLITS, TABLE_RULES, Opponents, datasets  # noqa: E402

__all__ = ["ARCHETYPES", "COUNTS", "MIN_CHEN", "build_all", "chen_score", "hole_cards", "screened"]

COUNTS = {"spot": 1200, "shove": 400, "short_game": 300, "cleanup": 150}
BASE_SEEDS = {"spot": 8_810_001, "shove": 8_820_002, "short_game": 8_830_003, "cleanup": 8_840_004}

# Seat 0 = SB, 1 = BB, 2 = UTG, 3 = HJ, 4 = CO, 5 = BTN (the button starts on the last seat). Minimum Chen score for
# hand 1: trash in early position facing no action is folded by every sample, so the group has zero variance.
MIN_CHEN = {0: 4, 1: 3, 2: 6, 3: 6, 4: 5, 5: 4}
MIN_CHEN_GAME = {0: 2, 1: 1, 2: 4, 3: 4, 4: 3, 5: 2}  # games run many hands, so hand 1 is screened loosely

# Tighter than the ladder: the final field is mostly each team's real model, which plays nit-ish to station-ish.
FIELD_WEIGHTS = {"nit": 3, "calling_station": 3, "maniac": 2, "random": 1, "shover": 1}

SPOT_PARAMS = {"reward": "spot", "w_fold_regret": 0.5, "ev_unit_bb": 2.0, "format_penalty": 2.0}
GAME_PARAMS = {"reward": "mix", "w_place": 1.0, "w_ev": 0.25, "w_cleanup": 0.5, "format_penalty": 1.0, "ev_unit_bb": 2.0}
CLEANUP_PARAMS = {"reward": "cleanup", "format_penalty": 1.0}


def screened(seed: int, seat: int, table: dict[int, int] = MIN_CHEN) -> bool:
    """True when the policy seat's hand-1 hole cards are worth playing from that seat."""
    return chen_score(hole_cards(seed, seat, seats=SEATS)) >= table[seat]


def _field(rng: random.Random, n: int) -> list[str]:
    return rng.choices(list(FIELD_WEIGHTS), weights=list(FIELD_WEIGHTS.values()), k=n)


def _with_shovers(rng: random.Random, n: int, count: int) -> list[str]:
    bots = _field(rng, n)
    for i in rng.sample(range(n), count):
        bots[i] = "shover"
    return bots


def _spot_bots(rng: random.Random, seat: int) -> list[str]:
    n = SEATS - 1
    if rng.random() < 0.3:  # bust-risk spot
        return _with_shovers(rng, n, rng.randint(2, 3))
    return _field(rng, n)


def _shove_bots(rng: random.Random, seat: int) -> list[str]:
    bots = [("shover" if rng.random() < 0.6 else rng.choice(("nit", "calling_station", "maniac"))) for _ in range(SEATS - 1)]
    if bots.count("shover") < 2:
        for i in rng.sample(range(SEATS - 1), 2):
            bots[i] = "shover"
    return bots


def _game_bots(rng: random.Random, seat: int) -> list[str]:
    return _with_shovers(rng, SEATS - 1, 1) if rng.random() < 0.25 else _field(rng, SEATS - 1)


def _cleanup_bots(rng: random.Random, seat: int) -> list[str]:
    return _field(rng, SEATS - 1)


def _env(seed: int, seat: int, hands: int, bots: list[str], **extra: Any) -> dict[str, Any]:
    return (
        TABLE_RULES
        | {
            "seed": seed,
            "seats": SEATS,
            "total_hands": hands,
            "reset_stacks": False,
            "policy_seats": [seat],
            "opponent_bots": bots,
        }
        | extra
    )


def _make(
    kind: str,
    used: set[int],
    bots: Callable[[random.Random, int], list[str]],
    hands: Callable[[random.Random], int],
    grader: dict[str, Any],
    prompt: Callable[[int, int], str],
    table: dict[int, int] | None,
    ev: bool,
) -> list[Task]:
    rng = random.Random(BASE_SEEDS[kind])
    out: list[Task] = []
    i = 0
    while len(out) < COUNTS[kind]:
        seat = i % SEATS
        seed = rng.getrandbits(48)
        if seed in used or (table is not None and not screened(seed, seat, table)):
            continue
        used.add(seed)
        n = hands(rng)
        extra = {"ev_labels": True, "ev_runouts": 128 if n == 1 else 64} if ev else {}
        out.append(
            Task(
                input=[Message(role="user", content=prompt(seat, n))],
                env_params=_env(seed, seat, n, bots(rng, seat), **extra),
                grader_params=dict(grader),
            )
        )
        i += 1
    return out


def _reserved_seeds() -> set[int]:
    """Every seed the starter splits and poker/game_tasks.py use."""
    seeds: set[int] = set()
    for count, base in SPLITS.values():
        rng = random.Random(base)
        seeds |= {rng.getrandbits(48) for _ in range(count)}
    for ts in (*game_datasets("x", Opponents("m")).values(), datasets("x", Opponents("m"))["x-table-smoke"]):
        seeds |= {t.env_params["seed"] for t in ts}
    return seeds


def _one_hand(seat: int, n: int) -> str:
    return f"Play one hand of {SEATS}-seat no-limit hold'em from seat {seat}."


def _game(seat: int, n: int) -> str:
    return f"Play a {SEATS}-seat no-limit hold'em game of {n} hands from seat {seat}."


def build_all() -> dict[str, list[Task]]:
    used = _reserved_seeds()
    return {
        "spot": _make("spot", used, _spot_bots, lambda r: 1, SPOT_PARAMS, _one_hand, MIN_CHEN, True),
        "shove": _make("shove", used, _shove_bots, lambda r: 1, SPOT_PARAMS, _one_hand, MIN_CHEN, True),
        "short_game": _make("short_game", used, _game_bots, lambda r: r.randint(6, 10), GAME_PARAMS, _game, MIN_CHEN_GAME, True),
        "cleanup": _make("cleanup", used, _cleanup_bots, lambda r: r.randint(3, 5), CLEANUP_PARAMS, _game, None, False),
    }
