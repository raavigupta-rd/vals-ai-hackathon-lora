"""Scripted seats: a seeded random player and fixed archetypes, for tests and local games without a model.

Ported from Vals' poker-agent engine (poker-agent-benchmark-service 0.1.0) and used with Vals' permission; behavior is
unchanged, and tests/test_engine_parity.py checks it hand for hand.
"""

from __future__ import annotations

import random

from .table import Decision, DecisionContext

ARCHETYPES = ("random", "maniac", "nit", "calling_station", "shover")
SHOVE_RATE = 0.35


def _strong(hole_cards: list[str]) -> bool:
    """Pairs of tens or better, and AK/AQ."""
    ranks = {card[0] for card in hole_cards}
    return (len(ranks) == 1 and ranks <= set("AKQJT")) or ranks in ({"A", "K"}, {"A", "Q"})


def _shove_call(hole_cards: list[str]) -> bool:
    """Pairs of nines or better, AK, and AQ suited: what a `shover` calls a shove with."""
    ranks = {card[0] for card in hole_cards}
    suited = hole_cards[0][1] == hole_cards[1][1]
    return (len(ranks) == 1 and ranks <= set("AKQJT9")) or ranks == {"A", "K"} or (ranks == {"A", "Q"} and suited)


class ScriptedPlayer:
    """`random` mixes legal actions; `maniac` raises big; `nit` plays only strong hands; `calling_station` calls.

    `shover` (local training games only) open-shoves 35% of unraised preflop spots with any hand, calls a raise only
    with 99+, AK and AQs, and otherwise checks or folds, also after the flop."""

    def __init__(self, name: str, archetype: str, seed: int = 0) -> None:
        if archetype not in ARCHETYPES:
            raise ValueError(f"unknown archetype {archetype!r}")
        self.name = name
        self.archetype = archetype
        self.rng = random.Random(f"{name}:{seed}")

    def _choose(self, ctx: DecisionContext) -> Decision:
        legal = ctx.legal
        passive = Decision("check" if legal.can_check else "call")
        if self.archetype == "calling_station":
            return passive
        if self.archetype == "shover":
            free = Decision("check" if legal.can_check else "fold")
            if ctx.street != "preflop":
                return free
            raised = any(a["street"] == "preflop" and a["action"]["type"] in ("raise", "bet") for a in ctx.hand_actions)
            if raised:
                return passive if _shove_call(ctx.hole_cards) and legal.call_amount else free
            if legal.raise_type and self.rng.random() < SHOVE_RATE:
                return Decision(legal.raise_type, legal.max_raise_to)
            return free
        if self.archetype == "maniac":
            if legal.raise_type and self.rng.random() < 0.7:
                return Decision(legal.raise_type, min(legal.max_raise_to, max(legal.min_raise_to, ctx.pot * 2)))
            return passive
        if self.archetype == "nit":
            if not _strong(ctx.hole_cards):
                return Decision("check" if legal.can_check else "fold")
            if legal.raise_type and ctx.street == "preflop":
                return Decision(legal.raise_type, legal.min_raise_to)
            return passive
        roll = self.rng.random()
        if legal.raise_type and roll < 0.15:
            top = min(legal.max_raise_to, legal.min_raise_to * 3)
            return Decision(legal.raise_type, self.rng.randint(legal.min_raise_to, top))
        if legal.can_fold and roll > 0.6:
            return Decision("fold")
        return passive

    async def decide(self, ctx: DecisionContext, history: str) -> Decision:
        decision = self._choose(ctx)
        decision.model_response = {
            "model_name": self.name,
            "provider": "scripted",
            "latency_seconds": 0.0,
            "in_tokens": 0,
            "out_tokens": 0,
            "reasoning_tokens": 0,
            "cost": 0.0,
            "reasoning": "",
            "fallback_used": False,
            "error": None,
        }
        return decision
