"""poker/evalbots: paired local evaluation against a seeded scripted field, with a fake completion client only."""

import asyncio
import json
import math

import pytest

from poker.evalbots import run as eb
from poker_harness.seat import Completion


def call(name, args):
    return {"id": "c", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def fake_factory(action="call", cleanup="good", counter=None):
    """A (deployment, params) -> (complete, cleanup_complete) factory that never touches the network."""

    def make(deployment, params):
        async def complete(messages):
            if counter is not None:
                counter.append(deployment)
            last = str(messages[-1].get("content", ""))
            if "is over" in last:
                if cleanup == "good":
                    c = call("update_scratchpad", {"contents": "hand notes"})
                else:  # answers cleanup with submit_action: no pad update
                    c = call("submit_action", {"action": "check"})
            else:
                c = call("submit_action", {"action": action})
            return Completion(text="", tool_arguments=c["function"]["arguments"] if c["function"]["name"] == "submit_action" else None,
                              tool_calls=[c], in_tokens=10, out_tokens=5)

        return complete, complete

    return make


def test_plan_pairs_seeds_across_deployments_and_rotates_field():
    plan = eb.build_plan(["a", "b"], games=3, hands=30, seed_base=100, field=["nit", "shover", "maniac"], rotate=True)
    assert len(plan) == 6
    by_dep = {d: [(p["seed"], tuple(p["bots"])) for p in plan if p["deployment"] == d] for d in ("a", "b")}
    assert by_dep["a"] == by_dep["b"]
    assert [s for s, _ in by_dep["a"]] == [100, 101, 102]
    assert by_dep["a"][1][1] == ("shover", "maniac", "nit", "shover", "maniac")
    flat = eb.build_plan(["a"], 2, 30, 1, ["nit", "shover"], rotate=False)
    assert flat[0]["bots"] == flat[1]["bots"] == ["nit", "shover", "nit", "shover", "nit"]  # cycled to 5 seats


def test_default_field_and_validation():
    assert set(eb.DEFAULT_FIELD) == {"nit", "calling_station", "shover", "maniac", "random"}
    with pytest.raises(ValueError):
        eb.build_plan(["a"], 1, 30, 1, ["nit", "wizard"])


def test_play_game_and_metrics_with_a_fake_client():
    plan = eb.build_plan(["a"], 1, 6, 7, eb.DEFAULT_FIELD)[0]
    log = asyncio.run(eb.play_game(plan, fake_factory()))
    m = eb.game_metrics(log)
    assert 1 <= m["place"] <= 6 and m["bust"] in (0, 1)
    assert m["net_chips"] == log["net_chips"]["seat0"]
    assert m["decisions"] > 0 and m["format_errors"] == 0 and m["in_tokens"] >= 10 * m["decisions"]
    assert m["cleanup_turns"] > 0 and m["cleanup_first_ok"] == m["cleanup_turns"]
    assert m["allin_calls"] <= m["allin_faced"]


def test_bad_cleanup_shows_up_as_invalid():
    plan = eb.build_plan(["a"], 1, 4, 7, eb.DEFAULT_FIELD)[0]
    m = eb.game_metrics(asyncio.run(eb.play_game(plan, fake_factory(cleanup="bad"))))
    assert m["cleanup_turns"] > 0 and m["cleanup_first_ok"] == 0


def test_same_seed_same_result():
    plan = eb.build_plan(["a"], 1, 6, 9, eb.DEFAULT_FIELD)[0]
    m1 = eb.game_metrics(asyncio.run(eb.play_game(plan, fake_factory())))
    m2 = eb.game_metrics(asyncio.run(eb.play_game(plan, fake_factory())))
    assert {k: m1[k] for k in ("place", "net_chips", "allin_faced")} == {k: m2[k] for k in ("place", "net_chips", "allin_faced")}


def test_evaluate_runs_every_pair_and_reports_paired_differences():
    seen = []
    res = asyncio.run(eb.evaluate(["call", "fold"], games=4, hands=6, seed_base=5, field=eb.DEFAULT_FIELD,
                                  factory_for=lambda dep: fake_factory(action=dep, counter=seen), concurrency=3))
    assert len(res["games"]["call"]) == 4 and len(res["games"]["fold"]) == 4
    assert set(seen) == {"call", "fold"}
    rep = eb.report(res, baseline="call", n_boot=200, seed=1)
    assert rep["baseline"] == "call" and rep["games"] == 4
    d = rep["deployments"]["fold"]["paired_vs_baseline"]["net_chips"]
    assert d["n"] == 4 and d["ci_low"] <= d["mean"] <= d["ci_high"]
    assert rep["deployments"]["call"]["paired_vs_baseline"] is None
    text = eb.format_report(rep)
    assert "fold" in text and "net_chips" in text


def test_bootstrap_ci_brackets_a_constant_difference():
    lo, hi = eb.bootstrap_ci([2.0] * 10, lambda idx: sum(2.0 for _ in idx) / len(idx), n_boot=100, seed=3)
    assert lo == hi == 2.0


def test_ratio_metric_pools_numerators_and_handles_zero_denominators():
    zero = dict.fromkeys(("place", "bust", "net_chips", "decisions", "format_errors", "cleanup_turns",
                          "cleanup_first_ok", "in_tokens", "out_tokens"), 0)
    s = eb.summarize([zero | {"allin_calls": 1, "allin_faced": 2}, zero | {"allin_calls": 0, "allin_faced": 0}],
                     n_boot=50, seed=1)
    assert s["allin_call_rate"]["mean"] == 0.5
    s0 = eb.summarize([zero | {"allin_calls": 0, "allin_faced": 0}], n_boot=50, seed=1)
    assert math.isnan(s0["allin_call_rate"]["mean"])


def test_dry_run_lists_plan_without_a_key(capsys, monkeypatch):
    monkeypatch.delenv("AC2_API_KEY", raising=False)
    rc = eb.main(["--deployments", "p/a", "p/b", "--games", "2", "--hands", "30", "--seed-base", "10", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and "p/a" in out and "p/b" in out and "seed 10" in out and "4 games" in out
