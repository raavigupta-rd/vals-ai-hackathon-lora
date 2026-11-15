import copy
import json
from pathlib import Path

import pytest

from poker.build_records import build
from poker.hints import HintBuilder, parse_prompt, range_equity
from poker.pokerbench_to_records import parse_row, verify_engine
from poker.records import legal_action, validate


def source_row(**updates):
    return {"": "3", "prev_line": "", "hero_pos": "SB", "hero_holding": "QdJc",
            "correct_decision": "3.0bb", "num_players": "1", "num_bets": "0",
            "available_moves": "['3.0bb', 'call', 'fold']", "pot_size": "1.5"} | updates


def record(**updates):
    value = parse_row(source_row(**updates), "preflop_60k_train_set_game_scenario_information.csv")
    verify_engine(value)
    validate(value)
    return value


def test_preflop_engine_replay_fills_omitted_folds_and_normalizes_chips():
    value = record()
    assert value["action"] == {"type": "raise", "to": 6}
    assert value["state"]["pot"] == 3
    assert value["state"]["stacks"] == [199, 198, 200, 200, 200, 200]
    assert len(value["state"]["history"]) == 4
    assert value["meta"]["engine_verified"] and legal_action(value)
    bad = copy.deepcopy(value)
    bad["action"]["to"] = 201
    assert not legal_action(bad)


def test_source_inconsistency_is_not_certified():
    value = parse_row(source_row(pot_size="19"), "preflop_60k_train_set_game_scenario_information.csv")
    with pytest.raises(ValueError, match="pot differs"):
        verify_engine(value)
    assert not value["meta"]["engine_verified"]


def test_postflop_uses_decision_street_only_and_maps_relative_position():
    row = {"": "1", "preflop_action": "SB/3.0bb/BB/10.0bb/SB/call", "board_flop": "Ks7h2d",
           "board_turn": "Jc", "board_river": "7c", "aggressor_position": "IP",
           "postflop_action": "OOP_CHECK/IP_BET_10/OOP_CALL/dealcards/Jc/OOP_CHECK/IP_BET_44/OOP_RAISE_80",
           "evaluation_at": "Turn", "available_moves": "['Fold', 'Call']", "pot_size": "164",
           "hero_position": "IP", "holding": "AhQd", "correct_decision": "Call"}
    value = parse_row(row, "postflop_10k_test_set_game_scenario_information.csv")
    validate(value)
    assert value["state"]["board"] == ["Ks", "7h", "2d", "Jc"]
    assert value["state"]["seat"] == 1
    assert value["state"]["position"] == "BB"
    assert value["meta"]["split"] == "test"


def write(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_cleaner_excludes_test_families_dedupes_and_never_splits_a_hand(tmp_path):
    a = record()
    b = record(hero_holding="AsKd")
    b["meta"]["hand_id"] = "another-hand"
    b["id"] += "b"
    c = record(hero_holding="AcKh")
    c["meta"]["hand_id"] = "another-hand"
    c["id"] += "c"
    held = copy.deepcopy(a)
    held["meta"]["split"] = "test"
    inputs, test = tmp_path / "input.jsonl", tmp_path / "test.jsonl"
    write(inputs, [a, b, b, c])
    write(test, [held])
    stats = build([inputs], [test], tmp_path / "result")
    assert stats["counts"]["drop_test_overlap"] == 1
    assert stats["counts"]["drop_duplicate"] == 1
    splits = [list((tmp_path / "result" / f"{s}.jsonl").open()) for s in ("train", "val")]
    assert sorted(map(len, splits)) == [0, 2]


def test_fold_cap_cannot_be_evaded_by_all_fold_input(tmp_path):
    value = record(correct_decision="fold")
    path = tmp_path / "input.jsonl"
    write(path, [value])
    stats = build([path], [], tmp_path / "result")
    assert stats["counts"]["drop_fold_cap"] == 1


def prompt(street="preflop", board="-", pot=3, legal="fold, call 1, raise to 4-200"):
    return {"role": "user", "content": f"""## Hand 1 of 30
Your seat: 0 (position 0; 0 = small blind)
Street: {street}
Your hole cards: Qd Jc
Board: {board}
Pot (including current bets): {pot}
Table:
  seat 0: hero (you) stack 199, bet 1
  seat 1: villain stack 198, bet 2
  seat 2: other stack 200, bet 0, folded

Legal actions: {legal}
Call submit_action with your decision."""}


def test_hint_uses_latest_state_not_old_prompt_and_computes_pot_odds():
    later = prompt("flop", "As 7h 2d", 30, "fold, call 10, raise to 20-190")
    later["content"] = later["content"].split("Street:", 1)[-1]
    later["content"] = "Street:" + later["content"]
    messages = [prompt(), later]
    parsed = parse_prompt(messages)
    assert parsed["street"] == "flop" and parsed["board"] == ["As", "7h", "2d"]
    hint = HintBuilder(equity_samples=0)(messages)
    assert "25.0%" in hint and "exactly one legal submit_action" in hint
    assert "No sufficiently matched" in hint


def test_hint_rejects_test_labels_and_matches_only_safe_actions(tmp_path):
    path = tmp_path / "records.jsonl"
    value = record()
    value["meta"]["partition"] = "train"
    write(path, [value])
    builder = HintBuilder(path, equity_samples=0)
    assert "suggests raise to 6" in builder([prompt()])
    assert "suggests" not in builder([prompt(legal="fold, call 1")])
    value["meta"]["split"] = "test"
    write(path, [value])
    with pytest.raises(ValueError, match="test labels"):
        HintBuilder(path)
    value["meta"].update(split="train", partition="val")
    write(path, [value])
    with pytest.raises(ValueError, match="validation labels"):
        HintBuilder(path)


def test_range_equity_preserves_random_state_and_uses_only_visible_cards():
    import random
    saved = random.getstate()
    # A royal flush has complete equity irrespective of the range or random runout.
    assert range_equity(("As", "Ks"), ("Qs", "Js", "Ts", "2d", "3c"), 1, 16) == 1
    assert random.getstate() == saved
    estimate = range_equity(("As", "Kd"), (), 5, 16)
    assert 0 <= estimate <= 1


def test_hints_on_twenty_real_official_harness_decisions(tmp_path):
    import asyncio
    from poker_harness.engine import HandRunner, ScriptedPlayer
    from poker_harness.prompts import decision_prompt

    contexts = []

    class Recorder(ScriptedPlayer):
        async def decide(self, ctx, history):
            contexts.append(ctx)
            return await super().decide(ctx, history)

    players = {s: Recorder(f"seat{s}", "calling_station", seed=0) for s in range(6)}
    runner = HandRunner(players=players, stacks=dict.fromkeys(range(6), 200), button_seat=5,
                        hand_number=1, total_hands=1, seed=0, small_blind=1, big_blind=2)
    asyncio.run(runner.play(dict.fromkeys(range(6), ""), 0))
    assert len(contexts) >= 20
    builder = HintBuilder(equity_samples=16)
    reports = []
    for ctx in contexts[:20]:
        messages = [{"role": "user", "content": decision_prompt(ctx)}]
        parsed = parse_prompt(messages)
        assert parsed is not None and parsed["pot"] == ctx.pot
        assert parsed["hole"] == ctx.hole_cards and parsed["seat"] == ctx.seat
        hint = builder(messages)
        assert "Rough sampled pot share" in hint and len(hint) < 1500
        reports.append({"source": "official engine scripted smoke, not base traces", "prompt": messages[0]["content"], "hint": hint})
    (tmp_path / "hint_smoke_report.json").write_text(json.dumps(reports, indent=2))


def test_cleaner_rejects_fractional_event_chip_states(tmp_path):
    value = record()
    value["state"]["stacks"][2] = 199.5
    path = tmp_path / "input.jsonl"
    write(path, [value])
    stats = build([path], [], tmp_path / "result")
    assert stats["counts"]["drop_fractional_event_chips"] == 1


def test_profiles_count_opportunities_and_separate_forced_folds():
    from poker.profile_logs import profile

    def event(player, kind, call=0, amount=None, stack=100, failed=False):
        return {"player": player, "street": "preflop", "action": {"type": kind, **({"amount": amount} if amount is not None else {})},
                "call_amount": call, "pot_before": 3, "stack_after": stack,
                "model_response": {"error": "reply hit cap" if failed else None}}

    log = {"game_id": "g1", "complete": True, "final_stacks": {"a": 200, "b": 199, "c": 0},
           "session_info": {"reset_stacks": False, "small_blind": 1, "big_blind": 2},
           "hands": [{"starting_state": {p: {"starting_stack": 200, "position": i} for i, p in enumerate(("a", "b", "c"))},
                      "actions": [event("a", "raise", amount=200, stack=0),
                                  event("b", "fold", call=198), event("c", "fold", call=200, failed=True)]}]}
    identities = {p: {"team": p, "submission": "v1", "probe": False} for p in ("a", "b", "c")}
    report = profile([log, log], identities)
    assert report["skipped"]["duplicate_game"] == 1
    assert report["teams"]["a"]["pfr"]["rate"] == 1
    assert report["teams"]["b"]["fold_to_shove"]["rate"] == 1
    assert report["teams"]["b"]["fold_to_pot_bet"]["opportunities"] == 1
    assert report["teams"]["c"]["failure_rate"]["rate"] == 1
    assert report["teams"]["c"]["fold_to_shove"]["rate"] is None
    identities["a"]["probe"] = True
    assert "a" not in profile([log], identities)["teams"]


def test_observed_arena_schema_adapter_preserves_failures_and_blinds():
    from poker.arena_replays import adapt
    from poker.profile_logs import profile

    replay = {"schema": "vals-arena.match/1", "id": "m1", "status": "complete",
              "format": {"small_blind": 1, "big_blind": 2, "reset_stacks": False},
              "seats": [{"seat": 0, "entry": {"id": "team", "kind": "team", "name": "Team", "submission_name": "base", "deployment_id": "d", "deployment": {"version": 1}}},
                        {"seat": 1, "entry": {"id": "house", "kind": "house", "model": "bot"}}],
              "results": [{"seat": 0, "final_chips": 199}, {"seat": 1, "final_chips": 201}],
              "hands": [{"hand_number": 1, "button_seat": 0, "players": [{"seat": 0, "starting_stack": 200}, {"seat": 1, "starting_stack": 200}],
                         "blinds": {"0": 1, "1": 2}, "actions": [{"seat": 0, "street": "preflop", "action": "fold", "amount": None,
                          "pot_before": 3, "to_call": 1, "stack_after": 199, "token_cap": True}]}]}
    log, identities = adapt(replay)
    assert log["hands"][0]["blinds"]["team:d:1"] == 1
    assert len(identities) == 1
    result = profile([log], identities)
    assert result["teams"]["team"]["failure_rate"]["rate"] == 1
