from poker.teacher import (
    allin_call_verdict,
    combos,
    equity_vs_range,
    free_fold_verdict,
    open_verdict,
    required_equity,
)


def test_combos_counts_pairs_suited_and_offsuit():
    assert len(combos("77")) == 6
    assert len(combos("AKs")) == 4
    assert len(combos("96o")) == 12
    assert all(a[1] == b[1] for a, b in combos("AKs"))
    assert all(a[1] != b[1] for a, b in combos("96o"))


def test_equity_vs_range_orders_hands_and_removes_blocked_combos():
    tight = ["QQ", "KK", "AA", "AKs", "AKo"]
    junk = equity_vs_range(["9h", "6c"], tight, samples=400)
    good = equity_vs_range(["Ah", "Ad"], tight, samples=400)
    assert junk < 0.30 < 0.70 < good
    # Holding both remaining aces' blockers must not crash, and AA vs a range of only AA is a near-coin chop.
    assert 0.4 < equity_vs_range(["Ah", "Ad"], ["AA"], samples=200) < 0.6


def test_required_equity_adds_a_bust_margin_only_when_the_call_risks_our_stack():
    assert required_equity(price=0.40, risks_bust=False, hand_number=5) == 0.40
    assert required_equity(price=0.40, risks_bust=True, hand_number=5) > required_equity(price=0.40, risks_bust=True, hand_number=29)
    assert required_equity(price=0.40, risks_bust=True, hand_number=29) > 0.40


def test_allin_call_verdict_folds_junk_and_calls_premiums_against_a_shove_range():
    shoves = ["22", "55", "77", "TT", "QQ", "AA", "A5s", "AJo", "KQs", "K9s", "T9s", "QJo"]
    fold = allin_call_verdict(["9h", "6c"], shoves, price=0.48, risks_bust=True, hand_number=1)
    call = allin_call_verdict(["Kh", "Kd"], shoves, price=0.48, risks_bust=True, hand_number=1)
    assert fold["preferred"] == "fold" and call["preferred"] == "call"
    assert fold["equity"] < fold["required"] < call["equity"]


def test_allin_call_verdict_abstains_near_the_threshold():
    shoves = ["AA", "KK", "QQ", "AKs", "AKo"]
    verdict = allin_call_verdict(["Jh", "Jd"], shoves, price=0.45, risks_bust=False, hand_number=10)
    assert verdict["preferred"] in (None, "call", "fold")
    tiny = allin_call_verdict(["Jh", "Jd"], shoves, price=0.45, risks_bust=False, hand_number=10, band=0.5)
    assert tiny["preferred"] is None


def test_open_verdict_flags_only_clear_preflop_mistakes():
    assert open_verdict("72o", "UTG", "raise") == "fold"
    assert open_verdict("72o", "UTG", "call") == "fold"
    assert open_verdict("AA", "UTG", "fold") == "raise"
    assert open_verdict("AKo", "CO", "fold") == "raise"
    assert open_verdict("A5s", "BTN", "raise") is None  # a fine button open
    assert open_verdict("K9o", "BTN", "raise") is None
    assert open_verdict("98s", "UTG", "fold") is None  # folding marginal hands is never flagged


def test_free_fold_is_always_wrong():
    assert free_fold_verdict(to_call=0, action="fold") == "check"
    assert free_fold_verdict(to_call=4, action="fold") is None
    assert free_fold_verdict(to_call=0, action="check") is None


from poker.arena import db
from poker.teacher import judge, shove_range


def _match(mid, hands):
    seats = [{"seat": s, "entry": {"id": f"e{s}", "kind": "team", "name": f"t{s}", "submission_name": "v1",
                                   "deployment_id": f"d{s}"}} for s in (0, 1)]
    return {"id": mid, "kind": "ranked", "status": "complete", "format": {"total_hands": 30}, "seats": seats,
            "results": [], "hands": hands}


def _hand(n, cards0, cards1, actions):
    players = [{"seat": 0, "hole_cards": cards0, "starting_stack": 200},
               {"seat": 1, "hole_cards": cards1, "starting_stack": 200}]
    return {"hand_number": n, "button_seat": 0, "blinds": {"0": 1, "1": 2}, "players": players, "board": [],
            "showdown": True, "shown": {}, "actions": actions}


def _act(seat, action, amount, pot, to_call, change, stack_after):
    return {"seat": seat, "street": "preflop", "action": action, "amount": amount, "pot_before": pot,
            "to_call": to_call, "chip_change": change, "stack_after": stack_after}


def _db():
    con = db.connect(":memory:")
    shove_then_fold = [_act(0, "raise", 200, 3, 1, -199, 0), _act(1, "fold", None, 202, 198, 0, 198)]
    shove_then_call = [_act(0, "raise", 200, 3, 1, -199, 0), _act(1, "call", None, 202, 198, -198, 0)]
    db.load_match(con, _match("m_a", [_hand(1, ["Qh", "Qd"], ["7c", "2d"], shove_then_fold),
                                      _hand(2, ["7h", "2s"], ["9h", "6c"], shove_then_call)]))
    con.commit()
    return con


def test_shove_range_leaves_out_the_hand_being_judged():
    con = _db()
    codes, own = shove_range(con, "e0|v1|d0", exclude=("m_a", 2))
    assert own == 1 and "QQ" in codes and "72o" not in codes


def test_judge_flags_a_junk_call_of_an_early_all_in():
    verdict = judge(_db(), "m_a", 2, seat=1, action_index=0)
    assert verdict["rule"] == "allin_call" and verdict["actual"] == "call" and verdict["preferred"] == "fold"
    assert verdict["agree"] is False


def test_judge_abstains_without_a_matching_rule():
    con = _db()
    # Seat 0's shove in hand 1 is an unopened raise with QQ: in range, nothing to flag.
    verdict = judge(con, "m_a", 1, seat=0, action_index=0)
    assert verdict["preferred"] is None and verdict["agree"] is None


def test_open_verdict_never_prefers_fold_when_checking_is_free():
    assert open_verdict("J6o", "BB", "raise", to_call=0) == "check"


def test_late_position_steals_are_only_flagged_with_trash():
    for code in ("K6o", "Q8o", "J6o", "T5s", "Q4s"):
        assert open_verdict(code, "BTN", "raise") is None, code
    assert open_verdict("94o", "BTN", "raise") == "fold"
    assert open_verdict("J2o", "CO", "call") == "fold"  # limping junk is still flagged
