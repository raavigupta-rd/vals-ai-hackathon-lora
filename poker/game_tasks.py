"""Multi-hand game tasks scored by final placement (grader_params reward "placement", see harness grading.py)."""

from __future__ import annotations

import random

from ac2.runtime import Message, Task

from vals_tools.tasks import SEATS, TABLE_RULES, Opponents

# (count, base seed, hands per game: fixed int or (low, high) inclusive range). Seeds differ from vals_tools SPLITS.
GAME_SPLITS = {
    "train-game": (400, 7_710_001, (6, 10)),
    "select-game": (100, 7_720_002, 30),
    "eval-game": (200, 7_730_003, 30),
}


def game_task(seed: int, policy_seat: int, opponents: Opponents, total_hands: int, starting_stack: int = 200) -> Task:
    """A `total_hands` game with chips carried over; the reward is the policy seat's final place."""
    return Task(
        input=[
            Message(
                role="user",
                content=f"Play a {SEATS}-seat no-limit hold'em game of {total_hands} hands from seat {policy_seat}.",
            )
        ],
        env_params=TABLE_RULES
        | {
            "seed": seed,
            "seats": SEATS,
            "total_hands": total_hands,
            "starting_stack": starting_stack,
            "reset_stacks": False,
            "policy_seats": [policy_seat],
            "ev_labels": True,
            "ev_runouts": 64,
            **opponents.env_params(),
        },
        grader_params={"reward": "placement"},
    )


def game_datasets(prefix: str, opponents: Opponents) -> dict[str, list[Task]]:
    out: dict[str, list[Task]] = {}
    for split, (count, base_seed, hands) in GAME_SPLITS.items():
        rng = random.Random(base_seed)
        tasks = []
        for i in range(count):
            n = rng.randint(*hands) if isinstance(hands, tuple) else hands
            tasks.append(game_task(rng.getrandbits(48), i % SEATS, opponents, n))
        out[f"{prefix}-{split}"] = tasks
    return out
