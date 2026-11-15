import random

from vals_tools.tasks import SPLITS, TABLE_RULES

from poker.game_tasks import game_datasets
from poker.grpo import tasks as T
from vals_tools.tasks import Opponents, datasets

BUILT = T.build_all()


def _starter_seeds() -> set[int]:
    seeds = set()
    for n, base in SPLITS.values():
        rng = random.Random(base)
        seeds |= {rng.getrandbits(48) for _ in range(n)}
    for ts in game_datasets("x", Opponents("m")).values():
        seeds |= {t.env_params["seed"] for t in ts}
    seeds |= {t.env_params["seed"] for t in datasets("x", Opponents("m"))["x-table-smoke"]}
    return seeds


def test_counts() -> None:
    assert {k: len(v) for k, v in BUILT.items()} == {
        "spot": T.COUNTS["spot"],
        "shove": T.COUNTS["shove"],
        "short_game": T.COUNTS["short_game"],
        "cleanup": T.COUNTS["cleanup"],
    }


def test_contract_keys() -> None:
    for kind, ts in BUILT.items():
        for t in ts:
            p = t.env_params
            assert p["preset"] == "event" and p["seats"] == 6
            assert len(p["opponent_bots"]) == 5
            assert set(p["opponent_bots"]) <= set(T.ARCHETYPES)
            assert "opponent_model" not in p
            assert len(p["policy_seats"]) == 1
            assert t.grader_params["reward"] == {"spot": "spot", "shove": "spot", "short_game": "mix", "cleanup": "cleanup"}[kind]
    s = BUILT["spot"][0]
    assert s.env_params["total_hands"] == 1 and s.env_params["reset_stacks"] is False
    assert s.env_params["ev_labels"] is True and s.env_params["ev_runouts"] == 128
    assert all(s.env_params[k] == v for k, v in TABLE_RULES.items() if k != "preset")
    assert "w_fold_regret" in s.grader_params and "ev_unit_bb" in s.grader_params
    g = BUILT["short_game"][0]
    assert 6 <= g.env_params["total_hands"] <= 10
    assert {"w_place", "w_ev", "w_cleanup"} <= set(g.grader_params) and "format_penalty" in g.grader_params
    assert g.grader_params["w_ev"] < g.grader_params["w_place"]
    c = BUILT["cleanup"][0]
    assert 3 <= c.env_params["total_hands"] <= 5


def test_seed_disjoint() -> None:
    starter = _starter_seeds()
    seen = set()
    for ts in BUILT.values():
        for t in ts:
            seed = t.env_params["seed"]
            assert seed not in starter and seed not in seen
            seen.add(seed)


def test_seat_coverage() -> None:
    for kind, ts in BUILT.items():
        seats = {t.env_params["policy_seats"][0] for t in ts}
        assert seats == set(range(6)), kind


def test_bots_per_nonpolicy_seat_and_shover_weight() -> None:
    shove = BUILT["shove"]
    avg = sum(t.env_params["opponent_bots"].count("shover") for t in shove) / len(shove)
    assert avg >= 2.5
    assert any("shover" in t.env_params["opponent_bots"] for t in BUILT["spot"])


def test_screening_skips_trash() -> None:
    for t in BUILT["spot"]:
        seat = t.env_params["policy_seats"][0]
        cards = T.hole_cards(t.env_params["seed"], seat, seats=6)
        assert T.chen_score(cards) >= T.MIN_CHEN[seat], (seat, cards)
    assert T.screened(1, 2) in (True, False)


def test_deterministic() -> None:
    again = T.build_all()
    for k in BUILT:
        assert [(t.env_params, t.grader_params) for t in again[k]] == [(t.env_params, t.grader_params) for t in BUILT[k]]
