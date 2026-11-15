"""env_params.scratchpad_init seeds a seat's hand-1 scratchpad (opt-in; training tasks)."""

from pathlib import Path

import httpx
import pytest
from test_event_rules import recording_relay
from test_server import play

from poker_harness.prompts import scratchpad_block
from poker_harness.server import create_app

PAD = "H1/1 | me seat1 200 | rank 1/6\nseat2: VPIP 9/10 | raises 6 | shoves 4"


def first_pad_messages(requests: list[dict]) -> list[str]:
    return [b["messages"][1]["content"] for b in requests if "scratchpad" in b["messages"][1]["content"]]


async def run(tmp_path: Path, **extra: object) -> tuple[dict, dict, list[dict]]:
    requests: list[dict] = []
    app = create_app(relay_transport=httpx.ASGITransport(app=recording_relay(requests)), runs_dir=tmp_path)
    params = {"preset": "event", "seed": 21, "total_hands": 1, "policy_seats": [1], "opponent_bots": ["nit"] * 5} | extra
    status, outputs = await play(app, {"input": [], "env_params": params})
    return status, outputs, requests


async def test_a_string_seeds_every_policy_seat(tmp_path: Path) -> None:
    status, _, requests = await run(tmp_path, scratchpad_init=PAD)
    assert status["status"] == "completed"
    assert requests[0]["messages"][1]["content"] == scratchpad_block(PAD, 5000)


async def test_a_dict_seeds_by_seat_and_accepts_string_keys(tmp_path: Path) -> None:
    status, _, requests = await run(tmp_path, scratchpad_init={"1": PAD})
    assert status["status"] == "completed"
    assert requests[0]["messages"][1]["content"] == scratchpad_block(PAD, 5000)


async def test_default_is_unchanged(tmp_path: Path) -> None:
    status, _, requests = await run(tmp_path)
    assert status["status"] == "completed"
    assert requests[0]["messages"][1]["content"] == scratchpad_block("", 5000)


async def test_ignored_when_the_scratchpad_is_off(tmp_path: Path) -> None:
    status, _, requests = await run(tmp_path, scratchpad=False, scratchpad_init=PAD)
    assert status["status"] == "completed"
    assert all("scratchpad" not in str(b["messages"]) for b in requests)


@pytest.mark.parametrize(
    "bad",
    ["x" * 5001, {"1": "x" * 5001}, {"9": "hi"}, {"1": 5}, 7],
)
async def test_invalid_values_fail_the_rollout(tmp_path: Path, bad: object) -> None:
    status, _, requests = await run(tmp_path, scratchpad_init=bad)
    assert status["status"] == "failed" and "scratchpad_init" in status["error_message"]
    assert not requests
