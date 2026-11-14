from vals_tools.tasks import SPLITS, TABLE_RULES, Opponents, datasets

from poker.game_tasks import game_datasets, game_task

OPP = Opponents("my-project/opponent")
BUILT = game_datasets("p", OPP)


def test_counts_and_names() -> None:
    assert {k: len(v) for k, v in BUILT.items()} == {"p-train-game": 400, "p-select-game": 100, "p-eval-game": 200}


def test_hand_counts() -> None:
    assert {t.env_params["total_hands"] for t in BUILT["p-train-game"]} <= set(range(6, 11))
    assert len({t.env_params["total_hands"] for t in BUILT["p-train-game"]}) > 1
    for name in ("p-select-game", "p-eval-game"):
        assert {t.env_params["total_hands"] for t in BUILT[name]} == {30}


def test_game_task_fields() -> None:
    t = game_task(123, 2, OPP, 8)
    p = t.env_params
    assert t.grader_params == {"reward": "placement"}
    assert p["reset_stacks"] is False and p["total_hands"] == 8 and p["seats"] == 6
    assert p["policy_seats"] == [2] and p["ev_labels"] is True and p["starting_stack"] == 200
    assert p["opponent_model"] == "my-project/opponent"
    assert all(p[k] == v for k, v in TABLE_RULES.items())


def test_seat_cycling() -> None:
    for tasks in BUILT.values():
        assert [t.env_params["policy_seats"][0] for t in tasks[:12]] == [i % 6 for i in range(12)]
        assert all(t.grader_params == {"reward": "placement"} for t in tasks)


def test_no_seed_overlap() -> None:
    new = [t.env_params["seed"] for tasks in BUILT.values() for t in tasks]
    assert len(new) == len(set(new)) == 700
    starter = {t.env_params["seed"] for tasks in datasets("s", OPP).values() for t in tasks}
    assert not set(new) & starter
