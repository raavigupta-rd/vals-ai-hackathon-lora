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


# ---- postflop_call (version 2) ----

from poker.teacher import postflop_class_counts, strength_class

VILLAIN_HANDS = [["Ac", "Ad"], ["Qc", "Qd"], ["Jc", "Jd"], ["Th", "Td"], ["Ah", "As"], ["Qh", "Qs"],
                 ["Jh", "Js"], ["8c", "8d"], ["As", "Ac"]]


def _pa(seat, street, action, amount, pot, to_call, change, stack_after):
    return {"seat": seat, "street": street, "action": action, "amount": amount, "pot_before": pot,
            "to_call": to_call, "chip_change": change, "stack_after": stack_after}


def _phand(n, cards, actions, board=("Ks", "9d", "4h", "2c", "3d"), seats=(0, 1)):
    players = [{"seat": s, "hole_cards": cards[s], "starting_stack": 200} for s in seats]
    return {"hand_number": n, "button_seat": 0, "blinds": {"0": 1, "1": 2}, "players": players,
            "board": list(board), "showdown": True, "shown": {}, "actions": actions}


def _bet(frac_chips, pot=100, street="flop"):
    return _pa(0, street, "bet", frac_chips, pot, 0, -frac_chips, 100)


def _postflop_db(our_cards, our_action, bet=80, street="flop", multiway=False):
    hands = [_phand(i + 1, {0: v, 1: ["7d", "3s"]}, [_bet(80), _pa(1, "flop", "fold", None, 180, 80, 0, 100)])
             for i, v in enumerate(VILLAIN_HANDS)]
    pot = 100 + bet
    cur = [_bet(bet, street=street),
           _pa(1, street, our_action, None, pot, bet, -bet if our_action == "call" else 0,
               100 - bet if our_action == "call" else 100)]
    cards = {0: ["5h", "6h"], 1: our_cards}
    seats = (0, 1)
    if multiway:
        cards[2] = ["8h", "8s"]
        seats = (0, 1, 2)
    hands.append(_phand(50, cards, cur, board=("Ks", "9d", "4h", "2c", "3d"), seats=seats))
    con = db.connect(":memory:")
    m = _match("m_p", hands)
    if multiway:
        m["seats"].append({"seat": 2, "entry": {"id": "e2", "kind": "team", "name": "t2", "submission_name": "v1",
                                                 "deployment_id": "d2"}})
    db.load_match(con, m)
    con.commit()
    return con


def test_postflop_call_flags_a_big_junk_call_on_the_flop():
    v = judge(_postflop_db(["7h", "2s"], "call"), "m_p", 50, seat=1, action_index=0)
    assert v["rule"] == "postflop_call" and v["preferred"] == "fold" and v["agree"] is False
    assert v["actual"] == "call" and v["to_call"] == 80 and v["pot"] == 180 and v["n"] == 9
    assert v["equity"] < v["required"] and v["bettor"] == "e0|v1|d0"


def test_postflop_call_prefers_calling_with_a_set_and_agrees():
    v = judge(_postflop_db(["Kh", "Kd"], "call"), "m_p", 50, seat=1, action_index=0)
    assert v["rule"] == "postflop_call" and v["preferred"] == "call" and v["agree"] is True
    folded = judge(_postflop_db(["Kh", "Kd"], "fold"), "m_p", 50, seat=1, action_index=0)
    assert folded["preferred"] == "call" and folded["agree"] is False


def test_postflop_call_abstains_on_small_bets_and_multiway_pots():
    small = judge(_postflop_db(["7h", "2s"], "call", bet=20), "m_p", 50, seat=1, action_index=0)
    assert small["preferred"] is None and small["agree"] is None
    multi = judge(_postflop_db(["7h", "2s"], "call", multiway=True), "m_p", 50, seat=1, action_index=0)
    assert multi["rule"] == "postflop_multiway" and multi["preferred"] is None


def test_postflop_call_does_not_judge_raises_and_free_folds_still_win():
    con = _postflop_db(["7h", "2s"], "call")
    con.execute("UPDATE actions SET action='raise' WHERE match_id='m_p' AND hand_number=50 AND seat=1")
    v = judge(con, "m_p", 50, seat=1, action_index=0)
    assert v["rule"] != "postflop_call" and v["preferred"] is None


def test_strength_class_fixtures():
    cases = [
        (["Kh", "Kd"], ["Ks", "9d", "4h"], "strong"),        # set
        (["Ah", "9h"], ["Ks", "9d", "4h"], "pair"),          # middle pair, ace kicker
        (["Kh", "Qd"], ["Ks", "9d", "4h"], "top_pair"),
        (["Ah", "Ad"], ["Ks", "9d", "4h"], "top_pair"),      # overpair counts as top pair
        (["Qh", "Qd"], ["Ks", "9d", "4h"], "pair"),          # pocket pair below the top card
        (["9h", "4d"], ["Ks", "9d", "4c"], "strong"),        # two pair
        (["Ah", "Qd"], ["Ks", "Kd", "4h"], "air"),           # a pair only on the board is no pair
        (["Ah", "2d"], ["Ks", "Kd", "9h", "9c"], "air"),     # board two pair, nothing from the hole
        (["Ah", "5h"], ["Kh", "9h", "4c"], "draw"),          # flush draw (ace-high, no pair)
        (["Jh", "Td"], ["9s", "8d", "2c"], "draw"),          # open-ended straight draw
        (["Jh", "Td"], ["9s", "8d", "2c", "Ks", "4h"], "air"),  # the same hand on the river is air
        (["Ah", "5h"], ["Kh", "9h", "4h", "2c", "3s"], "strong"),  # made flush
        (["Jh", "Td"], ["9s", "8d", "7c"], "strong"),        # straight
        (["Ah", "Kd"], ["Js", "9d", "4c"], "air"),
        (["7h", "2s"], ["Ks", "9d", "4h"], "air"),
    ]
    for hole, board, want in cases:
        assert strength_class(hole, board) == want, (hole, board)


def test_postflop_class_counts_classify_on_each_hands_own_board():
    con = _postflop_db(["7h", "2s"], "call")
    bettor = con.execute("SELECT * FROM actions WHERE match_id='m_p' AND hand_number=50 AND seat=0").fetchone()
    counts, own = postflop_class_counts(con, bettor, exclude=("m_p", 50))
    assert own == 9 and sum(counts.values()) == 9
    assert counts["top_pair"] == 3 and counts["pair"] == 6  # AA x3 are overpairs; QQ, JJ, TT, 88 are pairs
    # A different size bucket finds nothing from this bettor.
    con.execute("UPDATE actions SET bet_frac=1.5 WHERE match_id='m_p' AND hand_number=50 AND seat=0")
    big = con.execute("SELECT * FROM actions WHERE match_id='m_p' AND hand_number=50 AND seat=0").fetchone()
    assert postflop_class_counts(con, big, exclude=("m_p", 50)) == ({}, 0)


def test_postflop_call_ignores_tiny_calls_that_are_only_big_against_our_stack():
    con = _postflop_db(["7h", "2s"], "call", bet=10)  # price 10/120 = 0.08 < 0.15, 10% of stack anyway
    v = judge(con, "m_p", 50, seat=1, action_index=0)
    assert v["preferred"] is None


def test_postflop_call_corrections_need_extra_margin_before_the_river(monkeypatch):
    import poker.teacher as t
    monkeypatch.setattr(t, "_equity_classes", lambda *a, **k: 0.40)  # required 0.29: +0.11 over the price
    counts = {c: 1 for c in t.CLASSES}
    flop = t.postflop_call_verdict(["Ah", "Kd"], counts, ["2c", "7d", "9s"], price=0.29, risks_bust=False, hand_number=10)
    river = t.postflop_call_verdict(["Ah", "Kd"], counts, ["2c", "7d", "9s", "Th", "3c"], price=0.29, risks_bust=False,
                                    hand_number=10)
    assert flop["preferred"] is None  # 0.11 above the price is not enough when more betting can follow
    assert river["preferred"] == "call"


from poker.teacher import facing_raise_verdict, in_range, open_shove_verdict


def test_in_range_parses_compact_range_notation():
    spec = "66+ ATs+ KQs AJo+ KQo"
    assert in_range("77", spec) and in_range("AA", spec) and not in_range("55", spec)
    assert in_range("AJs", spec) and not in_range("A9s", spec)
    assert in_range("AQo", spec) and not in_range("ATo", spec)
    assert in_range("KQs", spec) and in_range("KQo", spec) and not in_range("KJs", spec)


def test_tighter_early_position_opens():
    assert open_verdict("K9o", "UTG", "raise") == "fold"
    assert open_verdict("A8o", "HJ", "raise") == "fold"
    assert open_verdict("ATs", "UTG", "raise") is None
    assert open_verdict("A5s", "CO", "raise") is None


def test_facing_a_raise_flags_weak_continues_but_not_cheap_big_blind_defense():
    assert facing_raise_verdict("K7o", "CO", "call", raises_before=1, to_call=4, stack=200) == "fold"
    assert facing_raise_verdict("JTs", "BTN", "call", raises_before=1, to_call=4, stack=200) is None
    assert facing_raise_verdict("A9o", "HJ", "raise", raises_before=1, to_call=4, stack=200) == "fold"
    assert facing_raise_verdict("K7o", "BB", "call", raises_before=1, to_call=3, stack=200) is None  # cheap defense
    assert facing_raise_verdict("99", "CO", "call", raises_before=2, to_call=14, stack=200) == "fold"  # vs a 3-bet
    assert facing_raise_verdict("QQ", "CO", "call", raises_before=2, to_call=14, stack=200) is None
    assert facing_raise_verdict("K7o", "CO", "fold", raises_before=1, to_call=4, stack=200) is None


def test_open_shoves_need_a_shove_range_for_the_stack_depth():
    assert open_shove_verdict("J4o", effective_bb=100) == "fold"
    assert open_shove_verdict("QQ", effective_bb=100) is None
    assert open_shove_verdict("A9s", effective_bb=100) == "fold"
    assert open_shove_verdict("A9s", effective_bb=12) is None  # short stacks shove wider
