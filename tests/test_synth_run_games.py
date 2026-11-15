import json
from pathlib import Path

from poker.synth.run_games import game_command, parse_summary, run_games

OK = "\n10 hands, 31 decisions by vals-hackathon/base, 0 failed\nseat0 net: +1.0 BB; log: x"


def test_command_selects_bots_and_seed():
    cmd = game_command("vals-hackathon/base", 7003, 10, "shover,nit", Path("/tmp/o"))
    assert cmd[:5] == ["uv", "run", "python", "play_local.py", "--model"]
    assert cmd[cmd.index("--bots") + 1] == "shover,nit" and cmd[cmd.index("--seed") + 1] == "7003"
    assert cmd[cmd.index("--opponents") + 1] == "bots" and cmd[cmd.index("--hands") + 1] == "10"


def test_parse_summary():
    assert parse_summary(OK) == {"hands": 10, "decisions": 31, "failed": 0}
    assert parse_summary("boom") == {}


def test_runs_every_game_with_distinct_seeds_and_hides_the_key(tmp_path):
    seen = []

    def runner(cmd, env):
        seen.append((cmd[cmd.index("--seed") + 1], env["AC2_API_KEY"]))
        return 0, OK

    m = run_games("m", games=5, hands=10, concurrency=3, bots="shover", seed_base=100, out_dir=tmp_path,
                  runner=runner, key="SECRET")
    assert sorted(s for s, _ in seen) == ["100", "101", "102", "103", "104"]
    assert m["games_played"] == 5 and "SECRET" not in json.dumps(m)
    assert m["since"] <= m["until"]


def test_retries_after_a_429_with_a_longer_pause(tmp_path):
    calls, pauses = {"n": 0}, []

    def runner(cmd, env):
        calls["n"] += 1
        return (1, "HTTP 429 too many") if calls["n"] == 1 else (0, OK)

    m = run_games("m", games=1, hands=10, concurrency=1, bots="shover", seed_base=1, out_dir=tmp_path,
                  runner=runner, sleep=pauses.append, key="k")
    assert m["games_played"] == 1 and m["games"][0]["attempts"] == 2 and pauses == [10.0]


def test_a_game_that_keeps_failing_is_reported(tmp_path):
    m = run_games("m", games=1, hands=10, concurrency=1, bots="shover", seed_base=1, out_dir=tmp_path,
                  runner=lambda c, e: (1, "bad"), sleep=lambda s: None, key="k", retries=1)
    assert m["games_played"] == 0 and m["games"][0]["rc"] == 1
