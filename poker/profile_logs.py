"""Measure opponent tendencies from official-format complete game logs, with explicit opportunity counts.

This consumes exported logs, not an assumed arena replay API. Team/version metadata must be supplied explicitly.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path


def rate(successes: int, opportunities: int) -> dict:
    if not opportunities:
        return dict(successes=successes, opportunities=0, rate=None, ci95=None)
    p, n, z = successes / opportunities, opportunities, 1.96
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return dict(successes=successes, opportunities=opportunities, rate=p, ci95=[max(0, center - width), min(1, center + width)])


def profile(logs: list[dict], identities: dict) -> dict:
    counts = defaultdict(Counter)
    seen = set()
    skipped = Counter()
    versions = defaultdict(set)
    for log in logs:
        if not log.get("complete"):
            skipped["incomplete"] += 1
            continue
        game = log["game_id"]
        if game in seen:
            skipped["duplicate_game"] += 1
            continue
        seen.add(game)

        def team(player):
            identity = identities.get(player)
            if not identity or identity.get("probe"):
                return None
            versions[identity["team"]].add(identity["submission"])
            return identity["team"]

        for player, chips in log["final_stacks"].items():
            owner = team(player)
            if owner:
                counts[owner]["games"] += 1
                # Bust metrics only make sense in games without stack resets.
                if not log["session_info"].get("reset_stacks", True):
                    counts[owner]["bust_games"] += 1
                    counts[owner]["busts"] += chips == 0
        for hand in log["hands"]:
            vpip, pfr = set(), set()
            for player, start in hand["starting_state"].items():
                owner = team(player)
                if owner and start["starting_stack"] > 0:
                    counts[owner]["hands"] += 1
            street = None
            committed = {}
            pressure = None
            responded = set()
            for event in hand["actions"]:
                if event["street"] != street:
                    street = event["street"]
                    committed, pressure, responded = {}, None, set()
                    if street == "preflop":
                        for player, start in hand["starting_state"].items():
                            pos = start["position"]
                            blind = log["session_info"]["small_blind"] if pos == 0 else log["session_info"]["big_blind"] if pos == 1 else 0
                            committed[player] = hand.get("blinds", {}).get(player, min(blind, start["starting_stack"]))
                player, move = event["player"], event["action"]
                owner = team(player)
                response = event.get("model_response", {})
                failed = bool(response.get("error") or response.get("fallback_used"))
                if owner:
                    counts[owner]["decisions"] += 1
                    counts[owner]["failed"] += failed
                # Forced folds on malformed replies are failures, not evidence of strategic folding.
                if owner and not failed and event["call_amount"] > 0 and pressure and pressure["player"] != player and player not in responded:
                    for metric in ("shove", "pot_bet"):
                        if pressure[metric]:
                            counts[owner][metric + "_opportunities"] += 1
                            counts[owner][metric + "_folds"] += move["type"] == "fold"
                    responded.add(player)
                if not failed and street == "preflop" and owner:
                    if move["type"] in ("call", "bet", "raise"):
                        vpip.add(player)
                    if move["type"] in ("bet", "raise"):
                        pfr.add(player)
                if move["type"] in ("bet", "raise"):
                    prior = committed.get(player, 0)
                    # Pot-size pressure is the additional wager the raiser risks, compared with the pot before it.
                    added = move["amount"] - prior
                    pressure = dict(player=player, shove=event["stack_after"] == 0,
                                    pot_bet=added >= event["pot_before"])
                    responded = set()
                    committed[player] = move["amount"]
                elif move["type"] == "call":
                    committed[player] = committed.get(player, 0) + event["call_amount"]
            for player in vpip:
                counts[team(player)]["vpip"] += 1
            for player in pfr:
                counts[team(player)]["pfr"] += 1
    result = {}
    for name, c in sorted(counts.items()):
        result[name] = dict(submissions=sorted(versions[name]), counts=dict(c),
                            vpip=rate(c["vpip"], c["hands"]), pfr=rate(c["pfr"], c["hands"]),
                            fold_to_shove=rate(c["shove_folds"], c["shove_opportunities"]),
                            fold_to_pot_bet=rate(c["pot_bet_folds"], c["pot_bet_opportunities"]),
                            bust_rate=rate(c["busts"], c["bust_games"]), failure_rate=rate(c["failed"], c["decisions"]))
    return dict(teams=result, games_seen=len(seen), skipped=dict(skipped),
                notes=["95% Wilson intervals treat observations as independent; within-game correlation may widen uncertainty.",
                       "Unknown players and identities marked probe are excluded. No style is inferred without evidence.",
                       "Failed decisions are counted separately and excluded from strategic fold opportunities."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--identities", type=Path, required=True,
                        help='JSON mapping player names to {"team":..., "submission":..., "probe": false}')
    parser.add_argument("--out", type=Path, default=Path("out/scouting/profiles.json"))
    args = parser.parse_args()
    result = profile([json.loads(p.read_text(encoding="utf-8")) for p in args.logs], json.loads(args.identities.read_text(encoding="utf-8")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
