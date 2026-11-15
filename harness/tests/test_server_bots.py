import httpx
import pytest

from poker_harness.server import create_app

from test_server import fake_relay, play

BOTS = ["shover", "nit", "calling_station", "maniac", "random"]


def task(hands: int, **extra: object) -> dict:
    env = {"seed": 7, "total_hands": hands, "seats": 6, "policy_seats": [0], "opponent_bots": BOTS} | extra
    return {"input": [{"role": "user", "content": "play"}], "env_params": env}


@pytest.fixture(autouse=True)
def no_opponent_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POKER_OPPONENT_API_KEY", raising=False)
    monkeypatch.delenv("AC2_API_KEY", raising=False)


@pytest.mark.parametrize("hands", [1, 3])
async def test_bots_fill_non_policy_seats_and_never_call_the_relay(tmp_path, hands: int) -> None:
    requests: list[dict] = []
    app = create_app(relay_transport=httpx.ASGITransport(app=fake_relay(requests)), runs_dir=tmp_path)
    status, outputs = await play(app, task(hands))
    log = outputs["log"]
    actions = [a for h in log["hands"] for a in h["actions"]]

    assert status["status"] == "completed" and len(log["hands"]) == hands
    assert log["session_info"]["player_names"] == [f"seat{i}" for i in range(6)]
    assert outputs["policy_seats"] == ["seat0"]
    policy_actions = [a for a in actions if a["player"] == "seat0"]
    assert len(requests) == len(policy_actions) > 0  # only the policy seat reaches the relay
    bot_actions = [a for a in actions if a["player"] != "seat0"]
    assert bot_actions and all(a["model_response"]["provider"] == "scripted" for a in bot_actions)


async def test_bots_are_deterministic_for_a_seed(tmp_path) -> None:
    runs = []
    for _ in range(2):
        app = create_app(relay_transport=httpx.ASGITransport(app=fake_relay([])), runs_dir=tmp_path)
        _, outputs = await play(app, task(3))
        runs.append([(a["player"], a["action"]) for h in outputs["log"]["hands"] for a in h["actions"]])
    assert runs[0] == runs[1]


@pytest.mark.parametrize(
    ("bots", "message"),
    [(["shover"], "opponent_bots"), (["shover", "nit", "calling_station", "maniac", "bogus"], "bogus")],
)
async def test_bad_opponent_bots_fail_the_rollout_clearly(tmp_path, bots, message) -> None:
    app = create_app(relay_transport=httpx.ASGITransport(app=fake_relay([])), runs_dir=tmp_path)
    status, _ = await play(app, task(1, opponent_bots=bots))
    assert status["status"] == "failed"
    assert message in str(status)


async def test_opponent_model_is_still_required_for_non_policy_seats_without_bots(tmp_path) -> None:
    # Omitting the key keeps the old behavior: non-policy seats use the relay.
    requests: list[dict] = []
    app = create_app(relay_transport=httpx.ASGITransport(app=fake_relay(requests)), runs_dir=tmp_path)
    t = task(1)
    del t["env_params"]["opponent_bots"]
    status, outputs = await play(app, t)
    actions = [a for h in outputs["log"]["hands"] for a in h["actions"]]
    assert status["status"] == "completed" and len(requests) == len(actions)
