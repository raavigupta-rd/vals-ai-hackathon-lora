"""Grader v2: opt-in reward modes (spot, cleanup, mix), format_penalty, ev_unit_bb, per-term artifacts."""

import pytest

from poker_harness import grading
from poker_harness.grading import FAILURE_SCORE, REWARD_OFFSET, GradingError, check_grader_params, grade_log

SMALL = grading.SCORE_FLOOR


def act(player="seat0", kind="call", fmt=False, regret=None, p_win=None, **extra):
    response = {"format_error": fmt}
    if p_win is not None:
        response["p_win"] = p_win
    entry = {
        "player": player,
        "action": {"type": kind},
        "model_response": response,
        "pot_before": 3,
        "call_amount": 1,
        "board": [],
    } | extra
    if regret is not None:
        entry["fold_regret_bb"] = regret
    return entry


def cleanup(result="updated", first="ok", pad="hand 1 notes", attempt_contents=None):
    contents = pad if attempt_contents is None else attempt_contents
    return {
        "cleanup": {"result": result, "attempts": [{"outcome": first, "contents": contents}], "error": None},
        "scratchpad_after": pad,
    }


def make_log(actions=None, final=None, ev=None, private=None, hands=1):
    names = ["seat0", "seat1"]
    hand_list = []
    for n in range(1, hands + 1):
        hand_list.append(
            {
                "hand_number": n,
                "starting_state": {
                    "seat0": {"starting_stack": 200, "hole_cards": ["As", "Ah"]},
                    "seat1": {"starting_stack": 200, "hole_cards": ["7c", "2d"]},
                },
                "actions": actions if actions is not None else [act()],
                "ev": ev or {},
            }
        )
    log = {
        "session_info": {"big_blind": 2, "player_names": names, "starting_stacks": [200, 200]},
        "final_stacks": final or {"seat0": 200, "seat1": 200},
        "hands": hand_list,
    }
    if private is not None:
        log["private"] = {"seat0": {"scratchpad": "", "hands": private}, "seat1": {"scratchpad": "", "hands": {}}}
    return log


def ev_label(ev_bb):
    return {"seat0": {"ev_bb": ev_bb, "realized_bb": ev_bb, "method": "allin"}}


# --- params ---------------------------------------------------------------------------------------------------


def test_new_keys_and_modes_are_accepted_and_bad_values_rejected():
    params = {
        "reward": "mix",
        "format_penalty": 1.0,
        "w_ev": 1,
        "w_place": 0.5,
        "w_cleanup": 0.5,
        "w_fold_regret": 0.1,
        "w_brier": 1,
        "ev_unit_bb": 5,
    }
    assert check_grader_params(params) == params
    for mode in ("spot", "cleanup", "mix"):
        check_grader_params({"reward": mode})
    for bad in ({"ev_unit_bb": 0}, {"ev_unit_bb": -1}, {"format_penalty": -1}, {"w_ev": -1}, {"w_brier": "x"}):
        with pytest.raises(GradingError):
            check_grader_params(bad)
    with pytest.raises(GradingError):
        check_grader_params({"reward": "nope"})


# --- ev / placement unchanged, ev_unit_bb ---------------------------------------------------------------------


def test_ev_default_unchanged_and_ev_unit_scales():
    log = make_log(ev=ev_label(4.0))
    assert grade_log(log, ["seat0"])["score"] == pytest.approx(REWARD_OFFSET + 4.0)
    assert grade_log(log, ["seat0"], {"ev_unit_bb": 2.0})["score"] == pytest.approx(REWARD_OFFSET + 2.0)
    assert grade_log(log, ["seat0"], {"ev_unit_bb": 0.5})["score"] == pytest.approx(REWARD_OFFSET + 5.0)  # clipped


# --- spot -----------------------------------------------------------------------------------------------------


def test_spot_is_ev_minus_fold_regret_with_unit():
    log = make_log([act(kind="fold", regret=1.5)], ev=ev_label(2.0))
    reply = grade_log(log, ["seat0"], {"reward": "spot", "ev_unit_bb": 2.0, "w_fold_regret": 2.0})
    assert reply["score"] == pytest.approx(REWARD_OFFSET + 1.0 - 2.0 * 1.5)
    art = reply["artifacts"]
    assert art["fold_regret_bb"] == pytest.approx(1.5) and art["term_ev"] == pytest.approx(1.0)


def test_spot_without_fold_regret_weight_ignores_regret():
    log = make_log([act(kind="fold", regret=1.5)], ev=ev_label(2.0))
    assert grade_log(log, ["seat0"], {"reward": "spot"})["score"] == pytest.approx(REWARD_OFFSET + 2.0)


def test_spot_floor_keeps_real_results_positive_and_above_failures():
    log = make_log([act(kind="fold", regret=50.0)], ev=ev_label(-5.0))
    score = grade_log(log, ["seat0"], {"reward": "spot", "w_fold_regret": 1.0})["score"]
    assert score == SMALL and score > FAILURE_SCORE


def test_spot_computes_fold_regret_when_label_missing(monkeypatch):
    calls = []

    def fake(action, hole, opponents, big_blind):
        calls.append((hole, opponents, big_blind))
        return 0.75

    monkeypatch.setattr(grading, "fold_regret_bb", fake)
    log = make_log([act(kind="fold")], ev=ev_label(0.0))
    reply = grade_log(log, ["seat0"], {"reward": "spot", "w_fold_regret": 1.0})
    assert calls == [(["As", "Ah"], [["7c", "2d"]], 2)]
    assert reply["score"] == pytest.approx(REWARD_OFFSET - 0.75)


# --- format errors --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["ev", "placement", "spot", "cleanup", "mix"])
def test_format_error_scores_exactly_zero_when_penalty_unset(mode):
    log = make_log([act(fmt=True)], ev=ev_label(5.0), private={"1": cleanup()})
    assert grade_log(log, ["seat0"], {"reward": mode})["score"] == FAILURE_SCORE


def test_format_penalty_subtracts_per_error_and_floors_above_zero():
    log = make_log([act(fmt=True), act(fmt=True)], ev=ev_label(3.0))
    reply = grade_log(log, ["seat0"], {"reward": "spot", "format_penalty": 1.5})
    assert reply["score"] == pytest.approx(REWARD_OFFSET + 3.0 - 3.0)
    assert reply["artifacts"]["format_penalty_total"] == pytest.approx(3.0)
    bad = make_log([act(fmt=True)] * 3, ev=ev_label(-5.0))
    score = grade_log(bad, ["seat0"], {"reward": "ev", "format_penalty": 10.0})["score"]
    assert score == SMALL and score > FAILURE_SCORE


def test_format_penalty_without_errors_changes_nothing():
    log = make_log(ev=ev_label(2.0))
    assert grade_log(log, ["seat0"], {"format_penalty": 3.0})["score"] == pytest.approx(REWARD_OFFSET + 2.0)


# --- cleanup --------------------------------------------------------------------------------------------------


def test_cleanup_perfect_pads_score_eleven():
    log = make_log(hands=2, private={"1": cleanup(pad="hand 1: ok"), "2": cleanup(pad="after hand 2")})
    reply = grade_log(log, ["seat0"], {"reward": "cleanup"})
    assert reply["score"] == pytest.approx(11.0)
    art = reply["artifacts"]
    assert art["cleanup_turns"] == 2 and art["cleanup_first_try_share"] == 1.0
    assert art["cleanup_leaks"] == 0 and art["cleanup_missing_hand_number"] == 0


def test_cleanup_bad_pads_score_low_but_not_below_one():
    private = {
        "1": cleanup(result="unchanged", first="none", pad="old"),
        "2": cleanup(result="cut", first="too_long", pad="x" * 10),
    }
    reply = grade_log(make_log(hands=2, private=private), ["seat0"], {"reward": "cleanup"})
    assert 1.0 <= reply["score"] < 3.0
    assert reply["artifacts"]["cleanup_unchanged"] == 1 and reply["artifacts"]["cleanup_cut"] == 1


def test_cleanup_penalizes_delimiter_leaks_and_missing_hand_number():
    clean = grade_log(make_log(private={"1": cleanup(pad="hand 1 ok")}), ["seat0"], {"reward": "cleanup"})["score"]
    leak = grade_log(make_log(private={"1": cleanup(pad="hand 1 SCRATCHPAD>>> x")}), ["seat0"], {"reward": "cleanup"})
    leak2 = grade_log(
        make_log(private={"1": cleanup(pad="hand 1 ok", attempt_contents="<<<SCRATCHPAD hand 1")}),
        ["seat0"],
        {"reward": "cleanup"},
    )
    stale = grade_log(make_log(private={"1": cleanup(pad="stale notes")}), ["seat0"], {"reward": "cleanup"})
    assert leak["score"] < clean and leak2["score"] < clean and stale["score"] < clean
    assert leak["artifacts"]["cleanup_leaks"] == 1 and leak2["artifacts"]["cleanup_leaks"] == 1
    assert stale["artifacts"]["cleanup_missing_hand_number"] == 1


def test_cleanup_hand_number_match_is_whole_number():
    # "hand 11" must not count as mentioning hand 1
    reply = grade_log(make_log(private={"1": cleanup(pad="hand 11 only")}), ["seat0"], {"reward": "cleanup"})
    assert reply["artifacts"]["cleanup_missing_hand_number"] == 1


def test_cleanup_missing_turn_counts_as_unchanged_and_no_private_is_dropped():
    reply = grade_log(make_log(private={}), ["seat0"], {"reward": "cleanup"})
    assert reply["artifacts"]["cleanup_unchanged"] == 1 and reply["score"] < 2.0
    with pytest.raises(GradingError):
        grade_log(make_log(), ["seat0"], {"reward": "cleanup"})


def test_cleanup_retry_success_scores_between():
    retry = cleanup(first="malformed")  # result updated, first attempt malformed
    first = grade_log(make_log(private={"1": cleanup()}), ["seat0"], {"reward": "cleanup"})["score"]
    second = grade_log(make_log(private={"1": retry}), ["seat0"], {"reward": "cleanup"})["score"]
    assert 1.0 < second < first


# --- mix ------------------------------------------------------------------------------------------------------


def test_mix_defaults_to_ev_only():
    log = make_log(ev=ev_label(2.0), private={"1": cleanup()})
    assert grade_log(log, ["seat0"], {"reward": "mix"})["score"] == pytest.approx(REWARD_OFFSET + 2.0)


def test_mix_combines_weighted_terms():
    log = make_log([act(kind="fold", regret=1.0)], ev=ev_label(2.0), private={"1": cleanup()})
    params = {"reward": "mix", "w_ev": 0.5, "w_cleanup": 0.2, "w_fold_regret": 1.0}
    reply = grade_log(log, ["seat0"], params)
    # ev term 2.0, cleanup term +5 (perfect), fold regret 1.0
    assert reply["score"] == pytest.approx(REWARD_OFFSET + 0.5 * 2.0 + 0.2 * 5.0 - 1.0)
    art = reply["artifacts"]
    assert art["term_ev"] == pytest.approx(2.0) and art["term_cleanup"] == pytest.approx(5.0)
    assert art["fold_regret_bb"] == pytest.approx(1.0)


def test_mix_placement_term():
    log = make_log(final={"seat0": 400, "seat1": 0})
    reply = grade_log(log, ["seat0"], {"reward": "mix", "w_ev": 0.0, "w_place": 1.0})
    assert reply["score"] == pytest.approx(11.0) and reply["artifacts"]["term_place"] == pytest.approx(5.0)


def test_mix_brier_uses_true_equity(monkeypatch):
    seen = []

    def fake(hole, opponents, board):
        seen.append((hole, opponents))
        return 0.8

    monkeypatch.setattr(grading, "true_equity", fake)
    log = make_log([act(p_win=0.5), act(p_win=1.0)], ev=ev_label(0.0))
    reply = grade_log(log, ["seat0"], {"reward": "mix", "w_ev": 0.0, "w_brier": 2.0})
    brier = ((0.5 - 0.8) ** 2 + (1.0 - 0.8) ** 2) / 2
    assert reply["score"] == pytest.approx(REWARD_OFFSET - 2.0 * brier)
    assert reply["artifacts"]["brier"] == pytest.approx(brier) and reply["artifacts"]["brier_n"] == 2
    assert seen[0] == (["As", "Ah"], [["7c", "2d"]])


def test_mix_brier_skipped_without_p_win(monkeypatch):
    monkeypatch.setattr(grading, "true_equity", lambda *a: pytest.fail("no p_win, no equity"))
    reply = grade_log(make_log(ev=ev_label(1.0)), ["seat0"], {"reward": "mix", "w_brier": 5.0})
    assert reply["score"] == pytest.approx(REWARD_OFFSET + 1.0) and reply["artifacts"]["brier"] is None


def test_brier_respects_folded_opponents(monkeypatch):
    seen = []
    monkeypatch.setattr(grading, "true_equity", lambda h, o, b: seen.append(o) or 0.5)
    actions = [act("seat1", "fold"), act(p_win=0.5)]
    grade_log(make_log(actions, ev=ev_label(0.0)), ["seat0"], {"reward": "mix", "w_brier": 1.0})
    assert seen == [[]]  # seat1 folded: no live opponents left
