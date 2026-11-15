"""The `shover` archetype: open-shoves often, calls shoves only with premium hands, otherwise folds."""

import asyncio

import pytest

from poker_harness.engine import ScriptedPlayer
from poker_harness.engine.table import DecisionContext, LegalActions, SeatView


def ctx(hole, *, street="preflop", raised=False, can_check=False, call=2, raise_type="raise") -> DecisionContext:
    actions = [{"street": "preflop", "action": {"type": "raise", "amount": 200}}] if raised else []
    legal = LegalActions(can_fold=not can_check, can_check=can_check, call_amount=0 if can_check else call,
                         raise_type=raise_type, min_raise_to=4, max_raise_to=200 if raise_type else 0)
    seats = [SeatView(i, f"s{i}", 200, 0, "") for i in range(6)]
    board = [] if street == "preflop" else ["2c", "7d", "9h"]
    return DecisionContext(hand_number=1, total_hands=10, street=street, seat=0, position=2, button_seat=0,
                           hole_cards=hole, board=board, pot=3, seats=seats, hand_actions=actions, legal=legal)


def decide(player, c):
    d = asyncio.run(player.decide(c, ""))
    return d.type, d.amount


def test_registered() -> None:
    from poker_harness.engine.scripted import ARCHETYPES

    assert "shover" in ARCHETYPES
    assert ScriptedPlayer("x", "shover", 1).archetype == "shover"


def test_open_shove_rate_is_about_a_third_on_any_hand() -> None:
    player = ScriptedPlayer("s", "shover", 7)
    results = [decide(player, ctx(["7c", "2d"])) for _ in range(2000)]
    shoves = [r for r in results if r[0] == "raise"]
    assert 0.28 < len(shoves) / len(results) < 0.42
    assert all(amount == 200 for _, amount in shoves)  # max raise
    assert all(kind == "fold" for kind, _ in results if kind != "raise")


def test_big_blind_check_option_is_checked_when_not_shoving() -> None:
    player = ScriptedPlayer("s", "shover", 3)
    kinds = {decide(player, ctx(["7c", "2d"], can_check=True))[0] for _ in range(200)}
    assert kinds == {"raise", "check"}


@pytest.mark.parametrize("hole", [["9c", "9d"], ["Ah", "As"], ["Kc", "Kd"], ["Ac", "Kd"], ["Ac", "Kc"], ["Ac", "Qc"]])
def test_calls_a_shove_with_the_calling_range(hole) -> None:
    assert decide(ScriptedPlayer("s", "shover", 1), ctx(hole, raised=True)) == ("call", None)


@pytest.mark.parametrize("hole", [["8c", "8d"], ["Ac", "Qd"], ["Ac", "Jc"], ["Kc", "Qc"], ["7c", "2d"], ["Ac", "5c"]])
def test_folds_everything_else_to_a_shove(hole) -> None:
    assert decide(ScriptedPlayer("s", "shover", 1), ctx(hole, raised=True)) == ("fold", None)


def test_postflop_checks_or_folds() -> None:
    player = ScriptedPlayer("s", "shover", 1)
    assert decide(player, ctx(["Ac", "Ad"], street="flop", can_check=True))[0] == "check"
    assert decide(player, ctx(["Ac", "Ad"], street="flop", can_check=False))[0] == "fold"


def test_is_deterministic_for_a_seed() -> None:
    a, b = ScriptedPlayer("s", "shover", 9), ScriptedPlayer("s", "shover", 9)
    c = ctx(["7c", "2d"])
    assert [decide(a, c) for _ in range(50)] == [decide(b, c) for _ in range(50)]
