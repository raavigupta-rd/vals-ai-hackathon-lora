"""Sample grader: how the harness scores each rollout for AC2.

How it is wired
---------------
A run that names no grader is graded by the harness itself, through AC2's Grader API (routes in `server.py`):

    POST /submit_grading      start grading one rollout; AC2 may repeat it, so it is idempotent
    POST /get_grading_status  queued | running | completed | failed
    POST /collect_score       the reply `grade_log` builds below: {"score", "reasoning", "artifacts"}

`server.py` calls `check_grader_params` with the task's `grader_params`, then `grade_log` with the finished game log,
or `failure_score` when the rollout never produced one. Nothing in this module talks to the network.

What it scores (deliberately naive)
-----------------------------------
The trained seat's result in big blinds. With `env_params.ev_labels`, an all-in before the river counts its expected
share of the pot instead of the cards that came (`ev.py`); otherwise it is the chips actually won.

    score = 6 + clip(hand_reward, -5, +5),  hand_reward = clip(EV, -100 BB, +100 BB) / 1 BB

So every real hand scores in [1, 11]. Using 1 BB per reward point (not 20) makes typical single-hand pots
actually use that band, so GRPO sees real within-group gaps. Model-caused failures score exactly 0: a format
error (no usable `submit_action`, or a reply cut off at the token cap), or a rollout AC2 reports as failed or
truncated. Rewards are never negative, a failure is always worse than any real hand, and failed samples still
train instead of being dropped. The +6 is a shift, not a rescale, so group-relative advantages (GRPO) are unchanged.

Extension points (search for "EXTENSION POINT")
-----------------------------------------------
1. `hand_reward`: the raw reward for a played hand. Replace it with your own.
2. `GRADER_PARAMS`: per-task knobs your reward reads from the dataset's `grader_params`.
3. `artifacts`: anything you add to the reply is stored with each score row on AC2, for analysis.
Keep failures at 0 and real hands above it, or a model can learn that failing pays.

With `grader_params: {"reward": "placement"}` it scores the seat's final place in the game instead (1st = 11,
last = 1), using the event's ranking. Use it on multi-hand tasks (`total_hands` > 1, `reset_stacks: false`), where
busting can happen; on a single hand it is a coarse chip ranking. Placement uses the chips actually won, so it keeps
the card luck that EV removes.

The default reward scores chips only. On single-hand tasks the scratchpad starts blank and its after-hand update is never
read, so nothing here rewards or exercises scratchpad use; how to train that is up to you.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .engine import true_equity
from .game import placements
from .labels import fold_regret_bb

# Captured once when the process imports the grader. Editing files on disk does not change this value in a
# running process; report it in health and scores so future runs can attest the code that actually graded them.
GRADER_REVISION = hashlib.sha256(
    (
        Path(__file__).read_text(encoding="utf-8")
        + "\0"
        + Path(__file__).with_name("game.py").read_text(encoding="utf-8")
    ).encode()
).hexdigest()

REWARD_OFFSET = 6.0  # shifts the clipped reward [-5, +5] to [1, 11]
REWARD_FLOOR, REWARD_CEILING = -5.0, 5.0
FAILURE_SCORE = 0.0  # every model-caused failure
EV_CLIP_BB = 100.0  # one hand can't swing the reward by more than a full stack
EV_UNIT_BB = 1.0  # 1 BB of EV = 1 reward point so typical hands use most of [1, 11]
MAX_SCORE_BYTES = 1 << 20  # AC2 rejects collect_score replies over 1 MiB

# EXTENSION POINT 2: the `grader_params` keys your reward accepts. Unknown keys fail the grading, so a typo in a
# dataset can't silently change the reward.
GRADER_PARAMS: frozenset[str] = frozenset(
    {
        "reward",
        "format_penalty",  # float >= 0: each format error subtracts it instead of zeroing the game
        "ev_unit_bb",  # float > 0: BB of EV per reward point (default 1)
        "w_ev",
        "w_place",
        "w_cleanup",
        "w_fold_regret",
        "w_brier",  # float >= 0: weights ("mix"; w_fold_regret also "spot")
    }
)
# `grader_params.reward`: "ev" (default) scores chips in BB; "placement" scores the policy seat's final place in the
# game (event ranking: survivors by chips above busted seats, later busts above earlier ones), 1st = +5, last = -5.
# Opt-in modes: "spot" = EV (single-hand tasks) minus w_fold_regret * fold regret; "cleanup" = scratchpad quality;
# "mix" = w_ev*EV + w_place*placement + w_cleanup*cleanup - w_fold_regret*fold_regret - w_brier*Brier(p_win).
REWARDS = ("ev", "placement", "spot", "cleanup", "mix")
SCORE_FLOOR = 0.05  # opt-in modes: a game without a format error never scores below this (failures stay at 0)
WEIGHT_KEYS = ("w_ev", "w_place", "w_cleanup", "w_fold_regret", "w_brier")
MIX_DEFAULTS = {"w_ev": 1.0, "w_place": 0.0, "w_cleanup": 0.0, "w_fold_regret": 0.0, "w_brier": 0.0}
LEAK_MARKERS = ("SCRATCHPAD>>>", "<<<SCRATCHPAD")
# Cleanup quality per turn: first-try update 1, update after a retry 0.5, else 0; minus these penalties.
CLEANUP_PENALTY = {"cut": 0.25, "unchanged": 0.25, "leak": 0.3, "no_hand_number": 0.3}


class GradingError(ValueError):
    """The log can't be scored. AC2 drops the sample (GRADER_ERROR) instead of training on a made-up reward."""


@dataclass
class SeatResult:
    """What the trained seat did over one game log."""

    ev_bb: float  # EV in big blinds (all-in luck removed when the log has EV labels)
    realized_bb: float  # chips actually won, in big blinds
    method: str  # how ev_bb was computed: "realized", "allin", "showdown", ...
    decisions: int
    format_errors: int
    folded: bool
    hands: int
    place: int  # best final place among the policy seats, 1 = first (event ranking)
    seats: int  # seats at the table
    busted: bool = False


def check_grader_params(grader_params: dict[str, Any] | None) -> dict[str, Any]:
    """The task's `grader_params`, after refusing any key not in `GRADER_PARAMS`."""
    params = dict(grader_params or {})
    unknown = set(params) - GRADER_PARAMS
    if unknown:
        raise GradingError(f"unknown grader_params: {sorted(unknown)}")
    if params.get("reward", "ev") not in REWARDS:
        raise GradingError(f"unknown reward {params['reward']!r}; use one of {list(REWARDS)}")
    for key in ("format_penalty", "ev_unit_bb", *WEIGHT_KEYS):
        if key not in params:
            continue
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            raise GradingError(f"grader_params.{key} must be a finite number, got {value!r}")
        if (key == "ev_unit_bb" and value <= 0) or (key != "ev_unit_bb" and value < 0):
            raise GradingError(f"grader_params.{key} must be {'> 0' if key == 'ev_unit_bb' else '>= 0'}, got {value!r}")
    return params


def seat_result(log: dict[str, Any], policy_seats: list[str]) -> SeatResult:
    """Sum the trained seat's results over every hand in the log."""
    policy = set(policy_seats)
    if not log.get("hands") or not policy:
        raise GradingError("the rollout has no hands or no policy seats")
    info = log["session_info"]
    big_blind = info["big_blind"]
    start = dict(zip(info["player_names"], info["starting_stacks"], strict=True))
    seats = [(hand, seat) for hand in log["hands"] for seat in policy if seat in hand["starting_state"]]
    actions = [a for hand in log["hands"] for a in hand["actions"] if a["player"] in policy]
    labels = [hand.get("ev", {}).get(seat) for hand, seat in seats]
    places = placements(log, "elimination")
    if labels and all(label is not None for label in labels):
        ev_bb = sum(label["ev_bb"] for label in labels)
        realized_bb = sum(label["realized_bb"] for label in labels)
        method = labels[0]["method"]
    else:  # no EV labels on the task: score the chips actually won
        realized_bb = ev_bb = sum(log["final_stacks"][seat] - start[seat] for seat in policy) / big_blind
        method = "realized"
    return SeatResult(
        ev_bb=ev_bb,
        realized_bb=realized_bb,
        method=method,
        decisions=len(actions),
        format_errors=sum(bool(a["model_response"].get("format_error")) for a in actions),
        folded=any(a["action"]["type"] == "fold" for a in actions),
        hands=len(log["hands"]),
        place=min(places[seat] for seat in policy),
        seats=len(places),
        busted=any(log["final_stacks"][seat] == 0 for seat in policy),
    )


def hand_reward(result: SeatResult, params: dict[str, Any]) -> float:
    """EXTENSION POINT 1: the raw reward for a game the seat played without failing, before clipping and the +6.

    The default is EV in units of 1 BB, clipped to +-100 BB. `result` has everything the log says about the trained
    seat; read more from the log in `seat_result` if you need it. `params` is the task's checked `grader_params`.
    """
    if params.get("reward") == "placement":  # 1st place = +5, last = -5, evenly spaced
        return REWARD_CEILING - (REWARD_CEILING - REWARD_FLOOR) * (result.place - 1) / max(1, result.seats - 1)
    return max(-EV_CLIP_BB, min(EV_CLIP_BB, result.ev_bb)) / float(params.get("ev_unit_bb", EV_UNIT_BB))


def _clip(value: float) -> float:
    return max(REWARD_FLOOR, min(REWARD_CEILING, value))


def policy_fold_regret(log: dict[str, Any], policy_seats: list[str]) -> float:
    """Sum of `fold_regret_bb` over the policy seats' folds: the `labels.py` label when the log has it, else computed."""
    policy, total = set(policy_seats), 0.0
    big_blind = log["session_info"]["big_blind"]
    for hand in log["hands"]:
        starting = hand["starting_state"]
        live = set(starting)
        for a in hand["actions"]:
            name = a["player"]
            if name in policy and a["action"]["type"] == "fold":
                if "fold_regret_bb" in a:
                    total += a["fold_regret_bb"]
                else:
                    opponents = [starting[o]["hole_cards"] for o in sorted(live) if o != name]
                    total += fold_regret_bb(a, starting[name]["hole_cards"], opponents, big_blind)
            if a["action"]["type"] == "fold":
                live.discard(name)
    return total


def policy_brier(log: dict[str, Any], policy_seats: list[str]) -> tuple[float | None, int]:
    """Mean (p_win - true equity)^2 over policy actions that report p_win; (None, 0) when none do."""
    policy, errors = set(policy_seats), []
    for hand in log["hands"]:
        starting = hand["starting_state"]
        live = set(starting)
        for a in hand["actions"]:
            name = a["player"]
            p_win = a.get("model_response", {}).get("p_win")
            if name in policy and isinstance(p_win, int | float) and not isinstance(p_win, bool):
                opponents = [starting[o]["hole_cards"] for o in sorted(live) if o != name]
                equity = true_equity(starting[name]["hole_cards"], opponents, a.get("board", []))
                errors.append((float(p_win) - equity) ** 2)
            if a["action"]["type"] == "fold":
                live.discard(name)
    return (sum(errors) / len(errors), len(errors)) if errors else (None, 0)


def cleanup_stats(log: dict[str, Any], policy_seats: list[str]) -> dict[str, Any]:
    """Scratchpad quality of the policy seats' cleanup turns; `raw` in [0, 1]. Needs a scratchpad table's private log."""
    if "private" not in log:
        raise GradingError("the cleanup reward needs a scratchpad table: the log has no private section")
    stats = {"turns": 0, "first_try": 0, "retry": 0, "cut": 0, "unchanged": 0, "leaks": 0, "missing_hand_number": 0}
    quality = 0.0
    for seat in policy_seats:
        private = log["private"].get(seat, {}).get("hands", {})
        for hand in log["hands"]:
            if seat not in hand["starting_state"]:
                continue
            number = hand["hand_number"]
            entry = private.get(str(number), {})
            record = entry.get("cleanup") or {}
            attempts = record.get("attempts") or []
            result = record.get("result", "unchanged")
            pad = entry.get("scratchpad_after") or ""
            texts = [pad] + [a["contents"] for a in attempts if isinstance(a.get("contents"), str)]
            leak = any(marker in text for text in texts for marker in LEAK_MARKERS)
            mentions = re.search(rf"(?<!\d){number}(?!\d)", pad) is not None
            first = result == "updated" and bool(attempts) and attempts[0].get("outcome") == "ok"
            score = 1.0 if first else 0.5 if result == "updated" else 0.0
            stats["turns"] += 1
            stats["first_try"] += first
            stats["retry"] += result == "updated" and not first
            if result == "cut":
                stats["cut"] += 1
                score -= CLEANUP_PENALTY["cut"]
            elif result != "updated":
                stats["unchanged"] += 1
                score -= CLEANUP_PENALTY["unchanged"]
            if leak:
                stats["leaks"] += 1
                score -= CLEANUP_PENALTY["leak"]
            if not mentions:
                stats["missing_hand_number"] += 1
                score -= CLEANUP_PENALTY["no_hand_number"]
            quality += score
    turns = stats["turns"]
    stats["raw"] = max(0.0, min(1.0, quality / turns)) if turns else 0.0
    stats["first_try_share"] = stats["first_try"] / turns if turns else 0.0
    return stats


def reward_terms(
    log: dict[str, Any], policy_seats: list[str], result: SeatResult, params: dict[str, Any]
) -> dict[str, Any]:
    """Opt-in modes ("spot", "cleanup", "mix"): the per-term numbers and `raw`, the reward before the +6 and the floor."""
    mode = params["reward"]
    weights = MIX_DEFAULTS | {k: float(params[k]) for k in WEIGHT_KEYS if k in params}
    if mode == "spot":
        weights = dict.fromkeys(WEIGHT_KEYS, 0.0) | {
            "w_ev": 1.0,
            "w_fold_regret": float(params.get("w_fold_regret", 0.0)),
        }
    elif mode == "cleanup":
        weights = dict.fromkeys(WEIGHT_KEYS, 0.0) | {"w_cleanup": 1.0}
    terms: dict[str, Any] = {"weights": weights}
    ev_unit = float(params.get("ev_unit_bb", EV_UNIT_BB))
    terms["term_ev"] = _clip(max(-EV_CLIP_BB, min(EV_CLIP_BB, result.ev_bb)) / ev_unit)
    terms["term_place"] = hand_reward(result, {"reward": "placement"})
    raw = weights["w_ev"] * terms["term_ev"] + weights["w_place"] * terms["term_place"]
    terms["fold_regret_bb"] = None
    if weights["w_fold_regret"] > 0:
        terms["fold_regret_bb"] = policy_fold_regret(log, policy_seats)
        raw -= weights["w_fold_regret"] * terms["fold_regret_bb"]
    terms["term_cleanup"] = None
    if weights["w_cleanup"] > 0:
        stats = cleanup_stats(log, policy_seats)
        terms["term_cleanup"] = REWARD_FLOOR + (REWARD_CEILING - REWARD_FLOOR) * stats["raw"]
        terms["cleanup"] = stats
        raw += weights["w_cleanup"] * terms["term_cleanup"]
    terms["brier"], terms["brier_n"] = None, 0
    if weights["w_brier"] > 0:
        terms["brier"], terms["brier_n"] = policy_brier(log, policy_seats)
        raw -= weights["w_brier"] * (terms["brier"] or 0.0)
    terms["raw"] = raw
    return terms


def failure_score(reason: str) -> dict[str, Any]:
    """The reply for a rollout that failed before producing a log (AC2 reported it failed or truncated)."""
    return {
        "score": FAILURE_SCORE,
        "reasoning": reason,
        "artifacts": {"failure": True, "grader_revision": GRADER_REVISION},
    }


def grade_log(log: dict[str, Any], policy_seats: list[str], params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Score one finished game for its policy seats, as a `collect_score` reply."""
    params = params or {}
    result = seat_result(log, policy_seats)
    penalty = params.get("format_penalty")
    terms: dict[str, Any] = {}
    if result.format_errors and penalty is None:  # model-caused failure: exactly 0, below every real hand
        score = FAILURE_SCORE
    else:
        if params.get("reward") in ("spot", "cleanup", "mix"):
            terms = reward_terms(log, policy_seats, result, params)
            score = REWARD_OFFSET + min(REWARD_CEILING, terms["raw"])
        else:
            score = REWARD_OFFSET + _clip(hand_reward(result, params))
        if penalty is not None:  # opt-in: errors cost points but the game still beats a failure
            score -= float(penalty) * result.format_errors
            score = max(SCORE_FLOOR, score)
        elif terms:
            score = max(SCORE_FLOOR, score)
    return {
        "score": score,
        "reasoning": f"Policy seat {'/'.join(sorted(policy_seats))}: EV {result.ev_bb:+.1f} BB "
        f"(realized {result.realized_bb:+.1f}), {result.method}, {result.format_errors} format errors.",
        # EXTENSION POINT 3: add any per-rollout numbers you want to analyze later.
        "artifacts": {
            "ev_bb": result.ev_bb,
            "realized_bb": result.realized_bb,
            "ev_method": result.method,
            "policy_decisions": result.decisions,
            "policy_format_errors": result.format_errors,
            "folded": result.folded,
            "hands": result.hands,
            "place": result.place,
            "busted": result.busted,
            "grader_revision": GRADER_REVISION,
            **_term_artifacts(terms, result, penalty),
        },
    }


def _term_artifacts(terms: dict[str, Any], result: SeatResult, penalty: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if penalty is not None:
        out["format_penalty_total"] = float(penalty) * result.format_errors
    if not terms:
        return out
    out |= {k: terms[k] for k in ("term_ev", "term_place", "term_cleanup", "fold_regret_bb", "brier", "brier_n", "raw")}
    out["reward_weights"] = terms["weights"]
    if "cleanup" in terms:
        out |= {f"cleanup_{k}": v for k, v in terms["cleanup"].items() if k != "raw"}
        out["cleanup_raw"] = terms["cleanup"]["raw"]
    return out


def bounded(score: dict[str, Any]) -> dict[str, Any]:
    """Drop artifacts, then truncate reasoning, until the reply fits AC2's 1 MiB limit."""
    if len(json.dumps(score)) <= MAX_SCORE_BYTES:
        return score
    trimmed = {"score": score["score"], "reasoning": score.get("reasoning", ""), "artifacts": {"trimmed": True}}
    overflow = len(json.dumps(trimmed)) - MAX_SCORE_BYTES
    if overflow > 0:
        trimmed["reasoning"] = trimmed["reasoning"][: max(0, len(trimmed["reasoning"]) - overflow - 16)]
    return trimmed
