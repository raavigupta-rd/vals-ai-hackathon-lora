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
from poker.teacher import agrees, judge, shove_range


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


def test_ranges_are_field_level_by_default_because_seats_are_anonymous(monkeypatch):
    import poker.teacher as t
    con = _db()
    monkeypatch.setattr(t, "POPULATION_MIN", 1)  # seat 0 has one own shove (QQ): enough to use it per-opponent
    monkeypatch.setattr(t, "PER_OPPONENT", True)
    per_opp, _ = t.shove_range(con, "e0|v1|d0", exclude=("m_a", 2))
    monkeypatch.setattr(t, "PER_OPPONENT", False)
    field, own = t.shove_range(con, "e0|v1|d0", exclude=("m_a", 2))
    assert own == 1 and per_opp == ["QQ"]
    assert set(field) >= {"QQ"} and field == sorted(set(field))  # the field's shoves, not seat 0's own record


# ---- teacher v4: evidence test ----


def test_equity_interval_brackets_the_point_estimate_and_orders_hands():
    import poker.teacher as t
    tight = ["QQ", "KK", "AA", "AKs", "AKo"]
    eq, lo, hi = t.equity_interval(["9h", "6c"], tight, samples=400)
    assert 0.0 <= lo <= eq <= hi <= 1.0 and hi - lo > 0
    eq2, lo2, hi2 = t.equity_interval(["Ah", "Ad"], tight, samples=400)
    assert lo2 > hi and eq2 > eq
    assert t.equity_interval(["9h", "6c"], tight, samples=400) == (eq, lo, hi)  # seeded and cached


def test_equity_interval_classes_is_deterministic_and_ordered():
    import poker.teacher as t
    counts = {"strong": 1, "top_pair": 3, "pair": 6}
    board = ("Ks", "9d", "4h")
    junk = t.equity_interval_classes(("7h", "2s"), board, tuple(sorted(counts.items())), samples=400)
    set_ = t.equity_interval_classes(("Kh", "Kd"), board, tuple(sorted(counts.items())), samples=400)
    assert junk[1] <= junk[0] <= junk[2] and set_[1] > junk[2]
    assert junk == t.equity_interval_classes(("7h", "2s"), board, tuple(sorted(counts.items())), samples=400)


def test_evidence_flag_is_off_by_default():
    import poker.teacher as t
    assert t.EVIDENCE is False and t.MARGIN_FOLD == 0.02 and t.MARGIN_CALL == 0.02


def test_evidence_allin_call_needs_the_whole_interval_past_the_threshold(monkeypatch):
    import poker.teacher as t
    monkeypatch.setattr(t, "EVIDENCE", True)
    required = t.required_equity(0.40, False, 10)  # 0.40
    def at(eq, lo, hi):
        monkeypatch.setattr(t, "equity_interval", lambda *a, **k: (eq, lo, hi))
        return t.allin_call_verdict(["9h", "6c"], ["AA"], price=0.40, risks_bust=False, hand_number=10)
    assert at(0.30, 0.25, 0.35)["preferred"] == "fold"  # hi 0.35 < 0.40 - 0.02
    assert at(0.34, 0.25, 0.39)["preferred"] is None  # point estimate is low but the interval reaches 0.39
    assert at(0.50, 0.43, 0.57)["preferred"] == "call"  # lo 0.43 > 0.42
    assert at(0.50, 0.41, 0.57)["preferred"] is None
    v = at(0.30, 0.25, 0.35)
    assert v["equity_lo"] == 0.25 and v["equity_hi"] == 0.35 and v["required"] == required


def test_evidence_postflop_call_keeps_the_extra_margin_before_the_river(monkeypatch):
    import poker.teacher as t
    monkeypatch.setattr(t, "EVIDENCE", True)
    monkeypatch.setattr(t, "equity_interval_classes", lambda *a, **k: (0.5, 0.45, 0.55))
    counts = {c: 1 for c in t.CLASSES}
    flop = t.postflop_call_verdict(["Ah", "Kd"], counts, ["2c", "7d", "9s"], price=0.40, risks_bust=False, hand_number=10)
    river = t.postflop_call_verdict(["Ah", "Kd"], counts, ["2c", "7d", "9s", "Th", "3c"], price=0.40, risks_bust=False,
                                    hand_number=10)
    assert flop["preferred"] is None  # lo 0.45 <= 0.40 + 0.02 + 0.04
    assert river["preferred"] == "call" and river["equity_lo"] == 0.45 and river["equity_hi"] == 0.55


def test_evidence_mode_widens_preflop_ranges(monkeypatch):
    import poker.teacher as t
    assert t.open_verdict("KTo", "UTG", "raise") == "fold"
    assert t.facing_raise_verdict("A8o", "CO", "call", raises_before=1, to_call=4, stack=200) == "fold"
    monkeypatch.setattr(t, "EVIDENCE", True)
    assert t.open_verdict("KTo", "UTG", "raise") is None  # inside the CO range
    assert t.open_verdict("A9o", "HJ", "raise") is None
    assert t.open_verdict("72o", "UTG", "raise") == "fold"  # still a clear mistake
    assert t.open_verdict("AA", "UTG", "fold") == "raise"
    assert t.facing_raise_verdict("A8s", "CO", "call", raises_before=1, to_call=4, stack=200) is None
    assert t.facing_raise_verdict("K7o", "CO", "call", raises_before=1, to_call=4, stack=200) == "fold"
    assert t.facing_raise_verdict("99", "CO", "call", raises_before=2, to_call=14, stack=200) == "fold"  # 3-bet spec same
    assert t.open_shove_verdict("A9s", effective_bb=100) == "fold"  # own_shove unchanged


# ---- exploit rules (v5): river_fold, steal, stab; all OFF unless named in EXPLOITS ----

import pytest

from poker import teacher as _t

FLOP_BOARD = ("Ks", "9d", "4h", "2c", "3d")


@pytest.fixture
def exploits(monkeypatch):
    def enable(*names):
        monkeypatch.setattr(_t, "EXPLOITS", set(names))
    return enable


def _xdb(hands, nseats=2):
    con = db.connect(":memory:")
    m = _match("m_x", hands)
    for s in range(2, nseats):
        m["seats"].append({"seat": s, "entry": {"id": f"e{s}", "kind": "team", "name": f"t{s}",
                                                 "submission_name": "v1", "deployment_id": f"d{s}"}})
    db.load_match(con, m)
    con.commit()
    return con


def _river_db(our_cards, to_call=80, pot=180, board=FLOP_BOARD, action="call"):
    acts = [_pa(0, "river", "bet", to_call, pot - to_call, 0, -to_call, 100),
            _pa(1, "river", action, None, pot, to_call, -to_call if action == "call" else 0, 100)]
    return _xdb([_phand(1, {0: ["5h", "6h"], 1: our_cards}, acts, board=board)])


def test_exploits_default_to_an_empty_set():
    assert _t.EXPLOITS == set()


def test_river_fold_is_off_by_default():
    v = judge(_river_db(["Kh", "7d"]), "m_x", 1, seat=1, action_index=0)
    assert v["rule"] != "river_fold"


def test_river_fold_flags_a_pair_call_of_a_big_river_bet(exploits):
    exploits("river_fold")
    for cards in (["Kh", "7d"], ["9h", "7d"]):  # top pair, second pair
        v = judge(_river_db(cards), "m_x", 1, seat=1, action_index=0)
        assert v["rule"] == "river_fold" and v["preferred"] == "fold" and v["agree"] is False, cards
    folded = judge(_river_db(["Kh", "7d"], action="fold"), "m_x", 1, seat=1, action_index=0)
    assert folded["rule"] != "river_fold"  # only calls are corrected


def test_river_fold_leaves_strong_hands_small_bets_and_air_alone(exploits):
    exploits("river_fold")
    strong = judge(_river_db(["Kh", "Kd"]), "m_x", 1, seat=1, action_index=0)
    assert strong["rule"] != "river_fold"
    small = judge(_river_db(["Kh", "7d"], to_call=30, pot=130), "m_x", 1, seat=1, action_index=0)  # 0.3 pot
    assert small["rule"] != "river_fold"
    huge = judge(_river_db(["Kh", "7d"], to_call=200, pot=300), "m_x", 1, seat=1, action_index=0)  # 2.0 pot
    assert huge["rule"] != "river_fold"
    air = judge(_river_db(["7h", "6s"]), "m_x", 1, seat=1, action_index=0)
    assert air["rule"] != "river_fold"


def test_river_fold_needs_a_heads_up_river(exploits):
    exploits("river_fold")
    acts = [_pa(0, "river", "bet", 80, 100, 0, -80, 100), _pa(1, "river", "call", None, 180, 80, -80, 100)]
    con = _xdb([_phand(1, {0: ["5h", "6h"], 1: ["Kh", "7d"], 2: ["8h", "8s"]}, acts, seats=(0, 1, 2))], nseats=3)
    assert judge(con, "m_x", 1, seat=1, action_index=0)["rule"] != "river_fold"
    flop = [_pa(0, "flop", "bet", 80, 100, 0, -80, 100), _pa(1, "flop", "call", None, 180, 80, -80, 100)]
    con = _xdb([_phand(1, {0: ["5h", "6h"], 1: ["Kh", "7d"]}, flop)])
    assert judge(con, "m_x", 1, seat=1, action_index=0)["rule"] != "river_fold"


def _steal_db(position_seat, cards, action="fold", limp=False, raised=False, stack=200):
    # 4 seats, button 0: SB=1, BB=2, CO=3, BTN=0 (see position_labels)
    acts = []
    if limp:
        acts.append(_act(2, "call", None, 3, 2, -2, stack - 2))
    if raised:
        acts.append(_act(2, "raise", 6, 3, 2, -6, stack - 6))
    to_call = 1 if position_seat == 1 else 0 if position_seat == 2 else 2
    acts.append(_act(position_seat, action, None, 3, to_call, 0, stack))
    players = [{"seat": s, "hole_cards": cards if s == position_seat else ["2c", "3d"], "starting_stack": stack}
               for s in range(4)]
    hand = {"hand_number": 1, "button_seat": 0, "blinds": {"1": 1, "2": 2}, "players": players, "board": [],
            "showdown": False, "shown": {}, "actions": acts}
    return _xdb([hand], nseats=4)


def test_steal_is_off_by_default():
    v = judge(_steal_db(3, ["Ah", "5s"]), "m_x", 1, seat=3, action_index=0)
    assert v["preferred"] is None


def test_steal_prefers_raise_for_folded_steal_hands_in_late_position(exploits):
    exploits("steal")
    co = judge(_steal_db(3, ["Ah", "5h"]), "m_x", 1, seat=3, action_index=0)
    assert co["rule"] == "steal" and co["preferred"] == "raise" and co["agree"] is False
    btn = judge(_steal_db(0, ["Kh", "3h"]), "m_x", 1, seat=0, action_index=0)
    assert btn["rule"] == "steal" and btn["preferred"] == "raise"
    sb = judge(_steal_db(1, ["Kh", "8c"]), "m_x", 1, seat=1, action_index=0)
    assert sb["rule"] == "steal" and sb["preferred"] == "raise"


def test_steal_ranges_differ_by_seat_and_exclude_trash(exploits):
    exploits("steal")
    assert judge(_steal_db(3, ["Kh", "3h"]), "m_x", 1, seat=3, action_index=0)["preferred"] is None  # K3s: BTN only
    assert judge(_steal_db(0, ["7h", "2c"]), "m_x", 1, seat=0, action_index=0)["preferred"] is None
    assert _t.STEAL_RANGES["CO"] != _t.STEAL_RANGES["BTN"] and _t.STEAL_RANGES["BTN"] == _t.STEAL_RANGES["SB"]


def test_steal_needs_an_unopened_pot_and_a_deep_stack_and_a_fold(exploits):
    exploits("steal")
    assert judge(_steal_db(3, ["Ah", "5h"], limp=True), "m_x", 1, seat=3, action_index=0)["rule"] != "steal"
    assert judge(_steal_db(3, ["Ah", "5h"], raised=True), "m_x", 1, seat=3, action_index=0)["rule"] != "steal"
    short = judge(_steal_db(3, ["Ah", "5h"], stack=28), "m_x", 1, seat=3, action_index=0)  # 14bb
    assert short["rule"] != "steal"
    deep = judge(_steal_db(3, ["Ah", "5h"], stack=30), "m_x", 1, seat=3, action_index=0)  # 15bb
    assert deep["rule"] == "steal"
    raised = judge(_steal_db(3, ["Ah", "5h"], action="raise"), "m_x", 1, seat=3, action_index=0)
    assert raised["rule"] != "steal"


def test_steal_does_not_apply_in_the_big_blind(exploits):
    exploits("steal")
    v = judge(_steal_db(2, ["Ah", "5h"], action="fold"), "m_x", 1, seat=2, action_index=0)
    assert v["rule"] != "steal"


def _stab_db(our_cards, our_action="check", opp_checked=True, board=FLOP_BOARD, street="flop", seats=2):
    acts = []
    if opp_checked:
        acts.append(_pa(0, street, "check", None, 10, 0, 0, 100))
    acts.append(_pa(1, street, our_action, None, 10, 0, 0, 100))
    cards = {0: ["5h", "6c"], 1: our_cards} | ({2: ["8h", "8s"]} if seats == 3 else {})
    return _xdb([_phand(1, cards, acts, board=board, seats=tuple(range(seats)))], nseats=seats)


def test_stab_is_off_by_default():
    v = judge(_stab_db(["7h", "2s"]), "m_x", 1, seat=1, action_index=0)
    assert v["preferred"] is None


def test_stab_prefers_a_bet_after_a_check_with_air_or_a_draw(exploits):
    exploits("stab")
    air = judge(_stab_db(["7h", "2s"]), "m_x", 1, seat=1, action_index=0)
    assert air["rule"] == "stab" and air["preferred"] == "bet" and air["agree"] is False
    draw = judge(_stab_db(["Ah", "6h"], board=("Ks", "9h", "4h", "2c", "3d")), "m_x", 1, seat=1, action_index=0)
    assert draw["rule"] == "stab" and draw["preferred"] == "bet"
    turn = judge(_stab_db(["7h", "6s"], street="turn"), "m_x", 1, seat=1, action_index=0)
    assert turn["rule"] == "stab"


def test_stab_leaves_made_hands_first_to_act_river_and_multiway_alone(exploits):
    exploits("stab")
    assert judge(_stab_db(["Kh", "Kd"]), "m_x", 1, seat=1, action_index=0)["rule"] != "stab"
    assert judge(_stab_db(["7h", "2s"], opp_checked=False), "m_x", 1, seat=1, action_index=0)["rule"] != "stab"
    assert judge(_stab_db(["7h", "2s"], street="river"), "m_x", 1, seat=1, action_index=0)["rule"] != "stab"
    assert judge(_stab_db(["7h", "2s"], seats=3), "m_x", 1, seat=1, action_index=0)["rule"] != "stab"
    bet = judge(_stab_db(["7h", "2s"], our_action="bet"), "m_x", 1, seat=1, action_index=0)
    assert bet["rule"] != "stab"


def test_field_ranges_use_llm_team_opponents_only_when_any_exist(monkeypatch):
    import poker.teacher as t
    con = _db()
    # The fixture's only shover is seat 0 (team e0). Make it a house bot: the team-only field is then empty,
    # so the range falls back to everyone rather than coming back empty.
    con.execute("UPDATE entries SET kind='house' WHERE sub_key='e0|v1|d0'")
    codes, _ = t.shove_range(con, "e0|v1|d0", exclude=("m_a", 2))
    assert codes == ["QQ"]
    con.execute("UPDATE entries SET kind='team' WHERE sub_key='e0|v1|d0'")
    con.execute("UPDATE entries SET name='LoRA' WHERE sub_key='e0|v1|d0'")  # our own seats are never the field
    assert t.field_sub_keys(con) == {"e1|v1|d1"}


# ---- exploit rule "oversize": preflop non-all-in raises above a cap; OFF unless named in EXPLOITS ----

def _over_db(our_seat, acts, stack=200):
    players = [{"seat": s, "hole_cards": ["Ah", "Kd"] if s == our_seat else ["2c", "3d"], "starting_stack": stack}
               for s in range(4)]
    hand = {"hand_number": 1, "button_seat": 0, "blinds": {"1": 1, "2": 2}, "players": players, "board": [],
            "showdown": False, "shown": {}, "actions": acts}
    return _xdb([hand], nseats=4)


def _oversize(acts, exploits_on=True, our_seat=3, index=None):
    con = _over_db(our_seat, acts)
    mine = [a for a in acts if a["seat"] == our_seat]
    return judge(con, "m_x", 1, seat=our_seat, action_index=len(mine) - 1 if index is None else index)


def test_oversize_is_off_by_default():
    v = _oversize([_act(3, "raise", 40, 3, 2, -40, 160)])
    assert v["rule"] != "oversize"


def test_oversize_flags_big_opens_and_sets_the_cap(exploits):
    exploits("oversize")
    v = _oversize([_act(3, "raise", 12, 3, 2, -12, 188)])
    assert v["rule"] == "oversize" and v["preferred"] == "raise" and v["agree"] is False
    assert v["target_amount"] == 7 and v["actual"] == "raise"
    assert _oversize([_act(3, "raise", 7, 3, 2, -7, 193)])["rule"] != "oversize"  # at the cap


def test_oversize_cap_grows_by_two_chips_per_limper(exploits):
    exploits("oversize")
    limp = _act(2, "call", None, 3, 2, -2, 198)
    assert _oversize([limp, _act(3, "raise", 9, 5, 2, -9, 191)])["rule"] != "oversize"
    v = _oversize([limp, _act(3, "raise", 14, 5, 2, -14, 186)])  # above 1.5 x cap 9
    assert v["rule"] == "oversize" and v["target_amount"] == 9


def test_oversize_three_bets_cap_at_three_times_the_raise(exploits):
    exploits("oversize")
    first = _act(2, "raise", 6, 3, 2, -6, 194)
    assert _oversize([first, _act(3, "raise", 18, 9, 6, -18, 182)])["rule"] != "oversize"
    v = _oversize([first, _act(3, "raise", 28, 9, 6, -28, 172)])  # above 1.5 x cap 18
    assert v["rule"] == "oversize" and v["target_amount"] == 18 and v["agree"] is False


def test_oversize_four_bets_cap_at_two_point_three_times(exploits):
    exploits("oversize")
    acts = [_act(2, "raise", 6, 3, 2, -6, 194), _act(3, "raise", 20, 9, 6, -20, 180)]
    ours = _act(0, "raise", 46, 29, 20, -46, 154)
    assert _oversize(acts + [ours], our_seat=0)["rule"] != "oversize"
    v = _oversize(acts + [_act(0, "raise", 70, 29, 20, -70, 130)], our_seat=0)  # above 1.5 x cap 46
    assert v["rule"] == "oversize" and v["target_amount"] == 46


def test_oversize_ignores_all_ins_calls_and_folds(exploits):
    exploits("oversize")
    assert _oversize([_act(3, "raise", 200, 3, 2, -200, 0)])["rule"] != "oversize"
    assert _oversize([_act(3, "call", None, 3, 2, -2, 198)])["rule"] != "oversize"
    assert _oversize([_act(3, "fold", None, 3, 2, 0, 200)])["rule"] != "oversize"


def test_agrees_checks_the_action_and_the_amount_cap():
    plain = {"preferred": "fold"}
    assert _t.agrees(plain, "fold", None) and not _t.agrees(plain, "call", None)
    v = {"preferred": "raise", "target_amount": 7}
    assert _t.agrees(v, "raise", 7) and _t.agrees(v, "raise", 5)
    assert not _t.agrees(v, "raise", 8) and not _t.agrees(v, "call", None) and not _t.agrees(v, "raise", None)
    assert _t.agrees(v, "raise", 5, min_amount=4) and not _t.agrees(v, "raise", 3, min_amount=4)
    assert _t.agrees({"preferred": "raise"}, "bet", 30)  # no cap: any raise or bet matches, as before


def test_oversize_ignores_mild_overshoots_below_the_tolerance(exploits):
    exploits("oversize")
    assert _oversize([_act(3, "raise", 10, 3, 2, -10, 190)])["rule"] != "oversize"  # 5bb open: 10 <= 1.5 x 7
    v = _oversize([_act(3, "raise", 11, 3, 2, -11, 189)])
    assert v["rule"] == "oversize" and v["target_amount"] == 7


# ---- exploit rules: monster_call and river_air_fold; OFF unless named in EXPLOITS ----

PAIRED_BOARD = ("Ks", "Kd", "4h", "2c", "3d")


def _street_db(street, our_cards, to_call=80, pot=180, board=FLOP_BOARD, action="fold", seats=2):
    acts = [_pa(0, street, "bet", to_call, pot - to_call, 0, -to_call, 100),
            _pa(1, street, action, None, pot, to_call, -to_call if action == "call" else 0, 100)]
    cards = {0: ["5h", "6h"], 1: our_cards, 2: ["8h", "8s"]}
    return _xdb([_phand(1, {s: cards[s] for s in range(seats)}, acts, board=board, seats=tuple(range(seats)))],
                nseats=seats)


def test_monster_call_is_off_by_default():
    v = judge(_street_db("river", ["Kh", "Kd"]), "m_x", 1, seat=1, action_index=0)
    assert v["rule"] != "monster_call"


def test_monster_call_prefers_a_call_when_we_fold_a_strong_hand(exploits):
    exploits("monster_call")
    for street in ("flop", "turn", "river"):
        for cards in (["Kh", "Kd"], ["Kh", "9h"]):  # a set, two pair with both hole cards
            v = judge(_street_db(street, cards), "m_x", 1, seat=1, action_index=0)
            assert v["rule"] == "monster_call" and v["preferred"] == "call" and v["agree"] is False, (street, cards)
            assert v["actual"] == "fold"


def test_monster_call_works_multiway(exploits):
    exploits("monster_call")
    v = judge(_street_db("flop", ["Kh", "Kd"], seats=3), "m_x", 1, seat=1, action_index=0)
    assert v["rule"] == "monster_call" and v["preferred"] == "call"


def test_monster_call_price_gate(exploits):
    exploits("monster_call")
    ok = judge(_street_db("river", ["Kh", "Kd"], to_call=100, pot=200), "m_x", 1, seat=1, action_index=0)  # 0.33
    assert ok["rule"] == "monster_call"
    huge = judge(_street_db("river", ["Kh", "Kd"], to_call=300, pot=350), "m_x", 1, seat=1, action_index=0)  # 0.46
    assert huge["rule"] != "monster_call"


def test_monster_call_abstains_on_weak_two_pair_and_non_strong_hands(exploits):
    exploits("monster_call")
    board_pair_two_pair = judge(_street_db("river", ["4c", "7d"], board=PAIRED_BOARD), "m_x", 1, seat=1, action_index=0)
    assert board_pair_two_pair["rule"] != "monster_call"
    pocket_plus_board_pair = judge(_street_db("river", ["9h", "9c"], board=PAIRED_BOARD), "m_x", 1, seat=1,
                                   action_index=0)
    assert pocket_plus_board_pair["rule"] != "monster_call"
    trips_on_paired_board = judge(_street_db("river", ["Kh", "7d"], board=PAIRED_BOARD), "m_x", 1, seat=1,
                                  action_index=0)
    assert trips_on_paired_board["rule"] == "monster_call"  # trips or better always counts
    top_pair = judge(_street_db("river", ["Kh", "7d"]), "m_x", 1, seat=1, action_index=0)
    assert top_pair["rule"] != "monster_call"
    called = judge(_street_db("river", ["Kh", "Kd"], action="call"), "m_x", 1, seat=1, action_index=0)
    assert called["rule"] != "monster_call"  # only folds are corrected


def test_monster_call_beats_postflop_call_when_enabled(exploits):
    base = judge(_postflop_db(["Kh", "Kd"], "fold"), "m_p", 50, seat=1, action_index=0)
    assert base["rule"] == "postflop_call"
    exploits("monster_call")
    v = judge(_postflop_db(["Kh", "Kd"], "fold"), "m_p", 50, seat=1, action_index=0)
    assert v["rule"] == "monster_call" and v["preferred"] == "call"
    junk = judge(_postflop_db(["7h", "2s"], "call"), "m_p", 50, seat=1, action_index=0)
    assert junk["rule"] == "postflop_call"  # falls through when the rule does not fire


def test_river_air_fold_is_off_by_default():
    v = judge(_street_db("river", ["7h", "6s"], action="call"), "m_x", 1, seat=1, action_index=0)
    assert v["rule"] != "river_air_fold"


def test_river_air_fold_prefers_fold_for_a_pure_air_call(exploits):
    exploits("river_air_fold")
    for board in (FLOP_BOARD, PAIRED_BOARD):  # a pair only on the board is still air
        v = judge(_street_db("river", ["7h", "6s"], board=board, action="call"), "m_x", 1, seat=1, action_index=0)
        assert v["rule"] == "river_air_fold" and v["preferred"] == "fold" and v["agree"] is False, board
        assert v["actual"] == "call"
    multi = judge(_street_db("river", ["7h", "6s"], action="call", seats=3), "m_x", 1, seat=1, action_index=0)
    assert multi["rule"] == "river_air_fold"


def test_river_air_fold_leaves_pairs_made_hands_folds_and_earlier_streets_alone(exploits):
    exploits("river_air_fold")
    for cards in (["Kh", "7d"], ["9h", "7d"], ["Kh", "Kd"], ["5s", "6s"]):  # top pair, pair, set, straight
        v = judge(_street_db("river", cards, action="call"), "m_x", 1, seat=1, action_index=0)
        assert v["rule"] != "river_air_fold", cards
    folded = judge(_street_db("river", ["7h", "6s"], action="fold"), "m_x", 1, seat=1, action_index=0)
    assert folded["rule"] != "river_air_fold"
    turn = judge(_street_db("turn", ["7h", "6s"], action="call"), "m_x", 1, seat=1, action_index=0)
    assert turn["rule"] != "river_air_fold"


# ---- teacher v6 tier-1 rules: big_price_fold, raise_first_in, one_pair_pot_cap, monster_value; OFF unless named ----

def _pre_db(position_seat, cards, action, amount=None, before=(), stack=200, nseats=4):
    # blinds: seat 1 = SB (1), seat 2 = BB (2), button seat 0; 4 seats: BTN 0, SB 1, BB 2, CO 3;
    # 6 seats: BTN 0, SB 1, BB 2, UTG 3, HJ 4, CO 5
    acts = list(before)
    last = acts[-1] if acts else None
    pot = 3 + sum(-a["chip_change"] for a in acts)
    to_call = (last["amount"] if last and last["action"] == "raise" else 2)
    if position_seat == 1 and not any(a["action"] == "raise" for a in acts):
        to_call = 1
    change = 0 if action == "fold" else -(amount or to_call)
    after = stack + change if not (action == "raise" and amount == stack) else 0
    acts.append(_act(position_seat, action, amount, pot, to_call, change, after))
    players = [{"seat": s, "hole_cards": cards if s == position_seat else ["2c", "3d"], "starting_stack": stack}
               for s in range(nseats)]
    hand = {"hand_number": 1, "button_seat": 0, "blinds": {"1": 1, "2": 2}, "players": players, "board": [],
            "showdown": False, "shown": {}, "actions": acts}
    return _xdb([hand], nseats=nseats)


def _raise_act(seat, amount, stack=200):
    return _act(seat, "raise", amount, 3, 2, -amount, stack - amount if amount < stack else 0)


def _judge1(con, seat):
    return judge(con, "m_x", 1, seat=seat, action_index=0)


def test_v6_rules_are_off_by_default():
    v = _judge1(_pre_db(0, ["7h", "2d"], "call", before=[_raise_act(3, 30)]), 0)
    assert v["rule"] != "big_price_fold"
    assert _judge1(_pre_db(0, ["Ah", "5h"], "fold"), 0)["rule"] != "raise_first_in"


def test_big_price_fold_prefers_fold_for_weak_hands_facing_a_big_raise(exploits):
    exploits("big_price_fold")
    for action, amount in (("call", None), ("raise", 90)):
        v = _judge1(_pre_db(0, ["7h", "2d"], action, amount, before=[_raise_act(3, 30)]), 0)
        assert v["rule"] == "big_price_fold" and v["preferred"] == "fold" and v["agree"] is False, action
    # 21 to call is above 20% of a 100 stack
    v = _judge1(_pre_db(0, ["7h", "2d"], "call", before=[_raise_act(3, 21, 100)], stack=100), 0)
    assert v["rule"] == "big_price_fold" and v["preferred"] == "fold"


def test_big_price_fold_allows_the_continue_set_and_never_forces_calls(exploits):
    exploits("big_price_fold")
    for cards in (["Ah", "Ad"], ["Th", "Td"], ["Ah", "Qh"], ["Ah", "Kd"]):
        v = _judge1(_pre_db(0, cards, "call", before=[_raise_act(3, 30)]), 0)
        assert v["rule"] != "big_price_fold", cards
    v = _judge1(_pre_db(0, ["Ah", "Ad"], "fold", before=[_raise_act(3, 30)]), 0)
    assert v["rule"] != "big_price_fold" and v["preferred"] != "call"
    v = _judge1(_pre_db(0, ["7h", "2d"], "fold", before=[_raise_act(3, 30)]), 0)
    assert v["rule"] != "big_price_fold"


def test_big_price_fold_is_tighter_against_all_ins_and_huge_prices(exploits):
    exploits("big_price_fold")
    shove = _raise_act(3, 200)
    for cards in (["9h", "9d"], ["Ah", "Qh"]):  # 99 and AQs are not enough against an all-in
        v = _judge1(_pre_db(0, cards, "call", before=[shove]), 0)
        assert v["rule"] == "big_price_fold" and v["preferred"] == "fold", cards
    for cards in (["Th", "Td"], ["Qh", "Qd"], ["Jh", "Jd"], ["Ah", "Kh"], ["Ah", "Kd"]):  # TT is in the huge-price set
        assert _judge1(_pre_db(0, cards, "call", before=[shove]), 0)["rule"] != "big_price_fold", cards
    # 0.45 of the stack is above 0.40, so AQs is out; 0.3 of the stack lets AQs continue
    big = _judge1(_pre_db(0, ["Ah", "Qh"], "call", before=[_raise_act(3, 90)]), 0)
    assert big["rule"] == "big_price_fold" and big["preferred"] == "fold"
    ok = _judge1(_pre_db(0, ["Ah", "Qh"], "call", before=[_raise_act(3, 60)]), 0)
    assert ok["rule"] != "big_price_fold"


def test_big_price_fold_ignores_small_prices(exploits):
    exploits("big_price_fold")
    v = _judge1(_pre_db(0, ["7h", "2d"], "call", before=[_raise_act(3, 8)]), 0)
    assert v["rule"] != "big_price_fold"


def test_open_target_sizes():
    from poker.teacher import open_target

    assert open_target("BTN", "A5s", 0) == 5 and open_target("CO", "A5s", 0) == 5
    assert open_target("SB", "A5s", 0) == 6
    assert open_target("BTN", "A5s", 1) == 7 and open_target("BTN", "A5s", 3) == 8
    assert open_target("SB", "AKs", 2) == 8
    assert open_target("BTN", "AA", 3) == 10 and open_target("BTN", "KK", 5) == 10


def test_raise_first_in_prefers_a_sized_raise_for_folds_and_limps(exploits):
    exploits("raise_first_in")
    for seat, pos, size in ((0, "BTN", 5), (3, "CO", 5), (1, "SB", 6)):
        for action in ("fold", "call"):
            v = _judge1(_pre_db(seat, ["Ah", "5h"], action), seat)
            assert (v["rule"], v["preferred"], v["agree"], v["target_amount"]) == ("raise_first_in", "raise", False, size), (pos, action)
    limp = _judge1(_pre_db(0, ["Kh", "Qd"], "fold", before=[_act(3, "call", None, 3, 2, -2, 198)]), 0)
    assert limp["preferred"] == "raise" and limp["target_amount"] == 7


def test_raise_first_in_needs_range_late_seat_and_an_unopened_pot(exploits):
    exploits("raise_first_in")
    assert _judge1(_pre_db(0, ["7h", "2d"], "fold"), 0)["rule"] != "raise_first_in"
    assert _judge1(_pre_db(2, ["Ah", "5h"], "check"), 2)["rule"] != "raise_first_in"  # big blind
    assert _judge1(_pre_db(0, ["Ah", "5h"], "fold", before=[_raise_act(3, 6)]), 0)["rule"] != "raise_first_in"


def test_raise_first_in_trims_oversized_opens(exploits):
    exploits("raise_first_in")
    v = _judge1(_pre_db(0, ["Kh", "Qd"], "raise", 12), 0)
    assert v["rule"] == "raise_first_in" and v["preferred"] == "raise" and v["target_amount"] == 5 and v["agree"] is False
    for cards, amount in ((["Kh", "Qd"], 8), (["Kh", "Kd"], 12), (["Kh", "Qd"], 200)):
        assert _judge1(_pre_db(0, cards, "raise", amount), 0)["rule"] != "raise_first_in", (cards, amount)


def test_raise_first_in_early_seats_never_open_limp(exploits):
    exploits("raise_first_in")
    weak = _judge1(_pre_db(3, ["7h", "2d"], "call", nseats=6), 3)  # UTG
    assert weak["rule"] == "raise_first_in" and weak["preferred"] == "fold" and weak["agree"] is False
    assert _judge1(_pre_db(3, ["Ah", "5h"], "call", nseats=6), 3)["preferred"] == "fold"
    good = _judge1(_pre_db(4, ["Kh", "Qd"], "call", nseats=6), 4)  # HJ
    assert good["preferred"] == "raise" and good["target_amount"] == 5
    assert _judge1(_pre_db(3, ["7h", "2d"], "fold", nseats=6), 3)["rule"] != "raise_first_in"
    assert _judge1(_pre_db(3, ["7h", "2d"], "raise", 6, nseats=6), 3)["rule"] != "raise_first_in"


def _bet_db(our_cards, our_acts, street="flop", board=FLOP_BOARD, opp_acts=(), stack=200):
    return _xdb([_phand(1, {0: ["5h", "6h"], 1: our_cards}, list(our_acts), board=board)])


def test_one_pair_pot_cap_folds_pairs_and_air_facing_an_overbet(exploits):
    exploits("one_pair_pot_cap")
    for cards in (["Kh", "7d"], ["9h", "7d"], ["7h", "6s"], ["Ah", "Qd"]):
        for street in ("flop", "river"):
            v = _judge1(_street_db(street, cards, to_call=150, pot=250, action="call"), 1)
            assert v["rule"] == "one_pair_pot_cap" and v["preferred"] == "fold" and v["agree"] is False, (cards, street)
    ok = _judge1(_street_db("flop", ["Kh", "7d"], to_call=80, pot=180, action="call"), 1)  # 0.8 pot
    assert ok["rule"] != "one_pair_pot_cap"
    strong = _judge1(_street_db("flop", ["Kh", "Kd"], to_call=150, pot=250, action="call"), 1)
    assert strong["rule"] != "one_pair_pot_cap"
    folded = _judge1(_street_db("flop", ["Kh", "7d"], to_call=150, pot=250, action="fold"), 1)
    assert folded["rule"] != "one_pair_pot_cap"


def _raised_db(our_cards, street, raise_to=100, board=("Ks", "9d", "4h", "2c", "3d")):
    """We bet 20 into 100, seat 0 raises to `raise_to`, we call: to_call = raise_to - 20 into 120 + raise_to."""
    to_call = raise_to - 20
    acts = [_pa(1, street, "bet", 20, 100, 0, -20, 180), _pa(0, street, "raise", raise_to, 120, 20, -raise_to, 200 - raise_to),
            _pa(1, street, "call", None, 120 + raise_to, to_call, -to_call, 180 - to_call)]
    return _xdb([_phand(1, {0: ["5h", "6h"], 1: our_cards}, acts, board=board)])


def test_one_pair_pot_cap_folds_one_pair_facing_a_raise_on_the_flop_or_turn(exploits):
    exploits("one_pair_pot_cap")
    for street in ("flop", "turn"):
        v = judge(_raised_db(["Kh", "7d"], street), "m_x", 1, seat=1, action_index=1)
        assert v["rule"] == "one_pair_pot_cap" and v["preferred"] == "fold", street
    river = judge(_raised_db(["Kh", "7d"], "river"), "m_x", 1, seat=1, action_index=1)
    assert river["rule"] != "one_pair_pot_cap"
    strong = judge(_raised_db(["Kh", "Kd"], "flop"), "m_x", 1, seat=1, action_index=1)
    assert strong["rule"] != "one_pair_pot_cap"


def test_one_pair_pot_cap_calls_a_cheap_raise_and_keeps_strong_draws(exploits):
    exploits("one_pair_pot_cap")
    cheap = judge(_raised_db(["Kh", "7d"], "flop", raise_to=60), "m_x", 1, seat=1, action_index=1)  # 40 into 220: 18%
    assert cheap["rule"] != "one_pair_pot_cap"
    draw_board = ("Kc", "3c", "Qc", "Jh", "2d")  # AcAd: overpair + nut flush draw + gutshot
    for street in ("flop", "turn"):
        v = judge(_raised_db(["Ac", "Ad"], street, raise_to=100, board=draw_board), "m_x", 1, seat=1, action_index=1)
        assert v["rule"] != "one_pair_pot_cap", street


def test_one_pair_pot_cap_blocks_huge_bets_and_raises_with_one_pair(exploits):
    exploits("one_pair_pot_cap")
    bet = _xdb([_phand(1, {0: ["5h", "6h"], 1: ["Kh", "7d"]}, [_pa(1, "flop", "bet", 120, 100, 0, -120, 80)])])
    v = _judge1(bet, 1)
    assert v["rule"] == "one_pair_pot_cap" and v["preferred"] == "check" and v["agree"] is False
    small = _xdb([_phand(1, {0: ["5h", "6h"], 1: ["Kh", "7d"]}, [_pa(1, "flop", "bet", 40, 100, 0, -40, 160)])])
    assert _judge1(small, 1)["rule"] != "one_pair_pot_cap"
    river = _xdb([_phand(1, {0: ["5h", "6h"], 1: ["Kh", "7d"]}, [_pa(1, "river", "bet", 40, 100, 0, -40, 160)])])
    assert _judge1(river, 1)["rule"] != "one_pair_pot_cap"
    big_set = _xdb([_phand(1, {0: ["5h", "6h"], 1: ["Kh", "Kd"]}, [_pa(1, "flop", "bet", 120, 100, 0, -120, 80)])])
    assert _judge1(big_set, 1)["rule"] != "one_pair_pot_cap"
    facing = [_pa(0, "flop", "bet", 20, 80, 0, -20, 180), _pa(1, "flop", "raise", 150, 100, 20, -150, 50)]
    con = _xdb([_phand(1, {0: ["5h", "6h"], 1: ["Kh", "7d"]}, facing)])
    v = _judge1(con, 1)
    assert v["rule"] == "one_pair_pot_cap" and v["preferred"] == "fold"


def test_monster_value_is_off_by_default_and_needs_a_real_monster():
    v = _judge1(_street_db("flop", ["Kh", "Kd"], to_call=80, pot=180), 1)
    assert v["rule"] != "monster_value"


def test_monster_value_calls_normal_bets_with_a_monster(exploits):
    exploits("monster_value")
    for cards in (["Kh", "Kd"],):
        v = _judge1(_street_db("flop", cards, to_call=80, pot=180), 1)  # facing_frac 0.8
        assert v["rule"] == "monster_value" and v["preferred"] == "call" and v["agree"] is False
    straight = _judge1(_street_db("river", ["5h", "6h"], to_call=80, pot=180), 1)
    assert straight["rule"] == "monster_value" and straight["preferred"] == "call"
    over = _judge1(_street_db("flop", ["Kh", "Kd"], to_call=150, pot=250), 1)  # facing_frac 1.5
    assert over["rule"] != "monster_value"
    called = _judge1(_street_db("flop", ["Kh", "Kd"], to_call=80, pot=180, action="call"), 1)
    assert called["rule"] != "monster_value"


def test_monster_value_leaves_weak_two_pair_board_hands_and_trips_with_one_hole_card(exploits):
    exploits("monster_value")
    two_pair = _judge1(_street_db("flop", ["Kh", "9h"], to_call=80, pot=180), 1)
    assert two_pair["rule"] != "monster_value"
    trips = _judge1(_street_db("flop", ["Kh", "7d"], to_call=80, pot=180, board=PAIRED_BOARD), 1)
    assert trips["rule"] != "monster_value"
    board_straight = ("5s", "6d", "7h", "8c", "9d")
    played = _judge1(_street_db("river", ["2h", "2d"], to_call=80, pot=180, board=board_straight), 1)
    assert played["rule"] != "monster_value"
    full = _judge1(_street_db("river", ["Kh", "9h"], to_call=80, pot=180, board=("Ks", "Kd", "9d", "2c", "3d")), 1)
    assert full["rule"] == "monster_value"  # full house on K K 9


def test_monster_value_bets_when_we_check_a_monster(exploits):
    exploits("monster_value")
    acts = [_pa(1, "flop", "check", None, 100, 0, 0, 200)]
    v = _judge1(_xdb([_phand(1, {0: ["5h", "6h"], 1: ["Kh", "Kd"]}, acts)]), 1)
    assert (v["rule"], v["preferred"], v["target_frac"], v["agree"]) == ("monster_value", "bet", 0.6, False)
    weak = _judge1(_xdb([_phand(1, {0: ["5h", "6h"], 1: ["Kh", "9h"]}, acts)]), 1)
    assert weak["rule"] != "monster_value"
    air = _judge1(_xdb([_phand(1, {0: ["5h", "6h"], 1: ["7h", "8d"]}, acts)]), 1)
    assert air["rule"] != "monster_value"


def test_big_price_fold_wins_over_oversize_for_a_junk_raise(exploits):
    exploits("oversize", "big_price_fold")
    players = [{"seat": s, "hole_cards": ["7c", "2d"] if s == 3 else ["2c", "3d"], "starting_stack": 200} for s in range(4)]
    acts = [_act(3, "raise", 6, 3, 2, -6, 194), _act(2, "raise", 40, 9, 4, -40, 160),
            _act(3, "raise", 150, 49, 34, -144, 50)]  # oversized junk 4-bet facing a 34-chip price, not all-in
    hand = {"hand_number": 1, "button_seat": 0, "blinds": {"1": 1, "2": 2}, "players": players, "board": [],
            "showdown": False, "shown": {}, "actions": acts}
    v = judge(_xdb([hand], nseats=4), "m_x", 1, seat=3, action_index=1)
    assert v["rule"] == "big_price_fold" and v["preferred"] == "fold"


# ---- pad_read: the raiser's profile as the teacher pad at that hand shows it; OFF unless named ----

def _pad_text(va, vb, raises, seat=3):
    return (f"H9/30 | me seat0 200 | rank 1/4\nseat{seat}: 200 | VPIP {va}/{vb} | raises {raises} | showed: - | "
            "talk-claims: 0 false\nplan: mid: tight value, fold marginal spots")


def _pad_db(cards, action, price=60, hand=9, extra_live=False):
    """Hand `hand`, 4 seats: seats 1 and 2 fold (unless extra_live), seat 3 raises to `price`, we (seat 0) act."""
    acts = [] if extra_live else [_act(1, "fold", None, 3, 2, 0, 200), _act(2, "fold", None, 3, 2, 0, 200)]
    acts.append(_raise_act(3, price))
    to_call = price
    change = 0 if action == "fold" else -to_call
    acts.append(_act(0, action, None, 3 + price, to_call, change, 200 + change))
    players = [{"seat": s, "hole_cards": (cards if s == 0 else ["2c", "3d"]), "starting_stack": 200} for s in range(4)]
    h = {"hand_number": hand, "button_seat": 0, "blinds": {"1": 1, "2": 2}, "players": players, "board": [],
         "showdown": False, "shown": {}, "actions": acts}
    return _xdb([h], nseats=4)


def _pad_judge(monkeypatch, pad, cards, action, price=60, hand=9, **kw):
    monkeypatch.setattr(_t, "PAD_INDEX", None if pad is None else {"_pads": {("m_x", 0, 1): {hand: pad}}})
    return judge(_pad_db(cards, action, price, hand, **kw), "m_x", hand, seat=0, action_index=0)


WIDE = _pad_text(7, 9, 4)     # VPIP 78%, 4 raises, 9 hands
TIGHT = _pad_text(1, 10, 1)   # VPIP 10%


def test_pad_read_is_off_by_default(monkeypatch):
    assert _pad_judge(monkeypatch, WIDE, ["7h", "7d"], "fold")["rule"] != "pad_read"


def test_pad_read_abstains_without_a_pad_index_or_pad(exploits, monkeypatch):
    exploits("pad_read")
    assert _pad_judge(monkeypatch, None, ["7h", "7d"], "fold")["rule"] != "pad_read"  # PAD_INDEX unset
    monkeypatch.setattr(_t, "PAD_INDEX", {"_pads": {("m_x", 0): {}}})                  # no pad for this hand
    assert judge(_pad_db(["7h", "7d"], "fold"), "m_x", 9, seat=0, action_index=0)["rule"] != "pad_read"
    assert _pad_judge(monkeypatch, "my own free text", ["7h", "7d"], "fold")["rule"] != "pad_read"  # not the schema
    assert _pad_judge(monkeypatch, "", ["7h", "7d"], "fold")["rule"] != "pad_read"                   # hand 1: empty pad
    assert _pad_judge(monkeypatch, _pad_text(7, 9, 4, seat=2), ["7h", "7d"], "fold")["rule"] != "pad_read"  # raiser not on pad


def test_pad_read_wide_aggressor_prefers_call_with_the_widened_set(exploits, monkeypatch):
    exploits("pad_read")
    for cards in (["7h", "7d"], ["Ah", "Th"], ["Ah", "Jd"], ["Kh", "Qh"]):
        v = _pad_judge(monkeypatch, WIDE, cards, "fold")
        assert v["rule"] == "pad_read" and v["preferred"] == "call" and v["agree"] is False, cards
        assert v["profile"] == {"hands": 9, "vpip": 7, "raises": 4}
    for cards in (["6h", "6d"], ["Ah", "Td"], ["Kh", "Qd"], ["9h", "8h"]):
        assert _pad_judge(monkeypatch, WIDE, cards, "fold")["rule"] != "pad_read", cards


def test_pad_read_wide_thresholds_come_from_the_pad_counts(exploits, monkeypatch):
    exploits("pad_read")

    def rule(va, vb, raises):
        return _pad_judge(monkeypatch, _pad_text(va, vb, raises), ["7h", "7d"], "fold")["rule"]

    assert rule(4, 6, 3) == "pad_read"      # 6 hands, 67%, 3 raises: all at the boundary
    assert rule(4, 5, 5) != "pad_read"      # only 5 hands observed
    assert rule(3, 6, 5) != "pad_read"      # VPIP 50% < 60%
    assert rule(4, 6, 2) != "pad_read"      # only 2 raises


def test_pad_read_wide_respects_the_45_percent_price_cap(exploits, monkeypatch):
    exploits("pad_read")
    assert _pad_judge(monkeypatch, WIDE, ["7h", "7d"], "fold", price=90)["rule"] == "pad_read"   # exactly 45%
    assert _pad_judge(monkeypatch, WIDE, ["7h", "7d"], "fold", price=100)["rule"] != "pad_read"  # 50%


def test_pad_read_needs_a_big_raise_or_shove(exploits, monkeypatch):
    exploits("pad_read")
    assert _pad_judge(monkeypatch, WIDE, ["7h", "7d"], "fold", price=20)["rule"] != "pad_read"  # not over 12 BB
    assert _pad_judge(monkeypatch, WIDE, ["7h", "7d"], "fold", price=200)["rule"] != "pad_read"  # shove: 100% of stack


def test_pad_read_does_not_force_folds_or_second_guess_calls_in_the_widened_set(exploits, monkeypatch):
    exploits("pad_read")
    assert _pad_judge(monkeypatch, WIDE, ["7h", "7d"], "call")["rule"] != "pad_read"
    assert _pad_judge(monkeypatch, WIDE, ["9h", "8h"], "call")["rule"] != "pad_read"


def test_pad_read_tight_raiser_narrows_the_range(exploits, monkeypatch):
    exploits("pad_read")
    for cards in (["Th", "Td"], ["Ah", "Qs"], ["Jh", "Jd"]):
        v = _pad_judge(monkeypatch, TIGHT, cards, "call")
        assert v["rule"] == "pad_read" and v["preferred"] == "fold" and v["agree"] is False, cards
    for cards in (["Qh", "Qd"], ["Ah", "Ad"], ["Ah", "Kh"], ["Ah", "Kd"]):  # QQ+ AKs AKo stay
        assert _pad_judge(monkeypatch, TIGHT, cards, "call")["rule"] != "pad_read", cards
    assert _pad_judge(monkeypatch, TIGHT, ["Th", "Td"], "fold")["rule"] != "pad_read"
    assert _pad_judge(monkeypatch, _pad_text(2, 10, 1), ["Th", "Td"], "call")["rule"] != "pad_read"  # 20% is not very tight
    # 1/8 = 12.5% > 12%: not very tight; 1/9 = 11% is
    assert _pad_judge(monkeypatch, _pad_text(1, 8, 1), ["Th", "Td"], "call")["rule"] != "pad_read"
    assert _pad_judge(monkeypatch, _pad_text(1, 9, 1), ["Th", "Td"], "call")["rule"] == "pad_read"
    assert _pad_judge(monkeypatch, _pad_text(0, 7, 0), ["Th", "Td"], "call")["rule"] != "pad_read"   # 7 hands only
    assert _pad_judge(monkeypatch, _pad_text(0, 8, 0), ["Th", "Td"], "call")["rule"] == "pad_read"


def test_pad_read_is_heads_up_only(exploits, monkeypatch):
    exploits("pad_read")
    v = _pad_judge(monkeypatch, WIDE, ["7h", "7d"], "fold", extra_live=True)  # seats 1 and 2 still live
    assert v["rule"] != "pad_read"
    assert _pad_judge(monkeypatch, TIGHT, ["Th", "Td"], "call", extra_live=True)["rule"] != "pad_read"


def test_pad_read_runs_before_big_price_fold(exploits, monkeypatch):
    exploits("pad_read", "big_price_fold")
    assert _pad_judge(monkeypatch, WIDE, ["7h", "7d"], "fold")["rule"] == "pad_read"
    v = _pad_judge(monkeypatch, TIGHT, ["Th", "Td"], "call")
    assert v["rule"] == "pad_read" and v["preferred"] == "fold"
    # no profile: big_price_fold still applies
    assert _pad_judge(monkeypatch, None, ["7h", "2d"], "call")["rule"] == "big_price_fold"


def test_pad_read_correction_text_and_args():
    from poker.sft.make_records import _correction_args, correction_reasoning

    call = {"rule": "pad_read", "preferred": "call", "agree": False}
    fold = {"rule": "pad_read", "preferred": "fold", "agree": False}
    assert "Call" in correction_reasoning(call, "77") and "Fold" in correction_reasoning(fold, "TT")
    for verdict in (call, fold):  # the text names the pad only when the record shows a rewritten teacher pad
        assert "scratchpad" in correction_reasoning(verdict, "77", pad_rewritten=True).lower()
        assert "scratchpad" not in correction_reasoning(verdict, "77").lower()
        assert "scratchpad" not in correction_reasoning(verdict, "77", pad_rewritten=False).lower()
    assert _correction_args(call, {"messages": []}) == {"action": "call"}


def test_raise_first_in_early_seats_raise_good_hands_instead_of_folding(exploits):
    exploits("raise_first_in")
    for seat, cards in ((3, ["7c", "7d"]), (3, ["Ad", "Td"]), (4, ["Ac", "Th"]), (4, ["Kh", "Jd"]), (4, ["Ah", "5h"])):
        v = _judge1(_pre_db(seat, cards, "fold", nseats=6), seat)
        assert (v["rule"], v["preferred"], v["target_amount"]) == ("raise_first_in", "raise", 5), (seat, cards)
    assert _judge1(_pre_db(3, ["Ah", "5h"], "fold", nseats=6), 3)["rule"] != "raise_first_in"  # A5s is not a UTG open
    assert _judge1(_pre_db(3, ["Kh", "9d"], "fold", nseats=6), 3)["rule"] != "raise_first_in"


def test_folding_a_premium_first_in_is_always_corrected(exploits):
    exploits("raise_first_in")
    for seat, cards in ((3, ["Ah", "Ad"]), (4, ["Ks", "Kh"]), (0, ["Ac", "Kd"]), (3, ["Ac", "Qd"])):
        v = _judge1(_pre_db(seat, cards, "fold", nseats=6), seat)
        assert v["rule"] == "raise_first_in" and v["preferred"] == "raise", (seat, cards)


# ---- teacher v7: rfi_gto, iso_raise, three_bet, bb_defend, vs_3bet, postflop_sizing; OFF unless named ----
# 6 seats: BTN 0, SB 1, BB 2, UTG 3, HJ 4, CO 5. Every v7 test spells out pot and to_call itself.

def _hero(seat, action, amount, pot, to_call, before=200):
    change = 0 if action in ("fold", "check") else -(amount if action == "raise" else to_call)
    return _act(seat, action, amount, pot, to_call, change, before + change)


def _limp(seat, pot=3):
    return _act(seat, "call", None, pot, 2, -2, 198)


def _open(seat, amount, pot=3, to_call=2):
    return _act(seat, "raise", amount, pot, to_call, -amount, 200 - amount)


def _v7(hero, cards, acts, nseats=6, stack=200, hand_no=1, stacks=None):
    """Judge the last action in `acts`, which must be hero's; earlier hero actions count in the action index."""
    players = [{"seat": s, "hole_cards": cards if s == hero else ["2c", "3d"], "starting_stack": (stacks or {}).get(s, stack)}
               for s in range(nseats)]
    hand = {"hand_number": hand_no, "button_seat": 0, "blinds": {"1": 1, "2": 2}, "players": players, "board": [],
            "showdown": False, "shown": {}, "actions": acts}
    mine = sum(1 for a in acts if a["seat"] == hero)
    return judge(_xdb([hand], nseats=nseats), "m_x", hand_no, seat=hero, action_index=mine - 1)


def _rfi(hero, cards, action, amount=None, to_call=2, stack=200):
    h = _hero(hero, action, amount, 3, 1 if hero == 1 else to_call, before=stack)
    return _v7(hero, cards, [h], stack=stack)


V7_RULES = ("rfi_gto", "iso_raise", "three_bet", "bb_defend", "vs_3bet", "postflop_sizing", "field_call")


def test_v7_rules_are_off_by_default():
    for v in (_rfi(3, ["Ah", "Qh"], "fold"), _rfi(1, ["7h", "2d"], "call")):
        assert v["rule"] not in V7_RULES


def test_rfi_gto_raises_in_range_hands_instead_of_folding_or_limping(exploits):
    exploits("rfi_gto")
    cases = ((3, ["Ah", "Qh"], 5), (3, ["4h", "4d"], 5), (4, ["6h", "6d"], 5), (4, ["Qh", "Jd"], 5),
             (5, ["8h", "7h"], 5), (5, ["Kh", "6h"], 5), (0, ["Kh", "8d"], 5), (0, ["5h", "3h"], 5),
             (1, ["Ah", "2d"], 6), (1, ["Qh", "5h"], 6))
    for seat, cards, size in cases:
        for action in ("fold", "call"):
            v = _rfi(seat, cards, action)
            assert (v["rule"], v["preferred"], v["agree"], v["target_amount"]) == ("rfi_gto", "raise", False, size), (seat, cards, action)


def test_rfi_gto_never_corrects_folds_or_raises_of_out_of_range_hands(exploits):
    exploits("rfi_gto")
    for seat, cards in ((3, ["Kh", "9h"]), (3, ["Ah", "5d"]), (4, ["8h", "7d"]), (5, ["7h", "2d"]), (0, ["9h", "4d"]), (1, ["Jh", "5d"])):
        assert _rfi(seat, cards, "fold")["rule"] != "rfi_gto", (seat, cards)
    assert _rfi(3, ["Kh", "9h"], "raise", 5)["rule"] != "rfi_gto"  # a loose open is not corrected
    assert _rfi(3, ["Ah", "Qh"], "raise", 5)["rule"] != "rfi_gto"  # in range and already raising


def test_rfi_gto_flags_open_limps_of_out_of_range_hands(exploits):
    exploits("rfi_gto")
    for seat, cards in ((3, ["9h", "8h"]), (5, ["7h", "2d"]), (1, ["7h", "2d"])):
        v = _rfi(seat, cards, "call")
        assert (v["rule"], v["preferred"], v["agree"]) == ("rfi_gto", "fold", False), (seat, cards)


def test_rfi_gto_needs_an_unopened_pot_a_seat_and_a_real_stack(exploits):
    exploits("rfi_gto")
    assert _rfi(2, ["Ah", "Qh"], "check", to_call=0)["rule"] != "rfi_gto"  # big blind
    limped = _v7(0, ["8h", "8d"], [_limp(3), _hero(0, "fold", None, 5, 2)])
    assert limped["rule"] != "rfi_gto"
    raised = _v7(0, ["8h", "8d"], [_open(3, 5), _hero(0, "fold", None, 8, 5)])
    assert raised["rule"] != "rfi_gto"
    assert _rfi(0, ["8h", "8d"], "fold", stack=40)["rule"] != "rfi_gto"  # 20bb: shove-or-fold territory


def test_iso_raise_isolates_limpers_with_good_hands(exploits):
    exploits("iso_raise")
    one = _v7(0, ["8h", "8d"], [_limp(3), _hero(0, "fold", None, 5, 2)])
    assert (one["rule"], one["preferred"], one["agree"], one["target_amount"]) == ("iso_raise", "raise", False, 6)
    two = _v7(0, ["Qh", "Jh"], [_limp(3), _limp(4, 5), _hero(0, "call", None, 7, 2)])
    assert (two["rule"], two["preferred"], two["target_amount"]) == ("iso_raise", "raise", 8)
    for cards in (["7h", "5h"], ["Ah", "3d"]):  # any suited or any ace on the button
        v = _v7(0, cards, [_limp(3), _hero(0, "fold", None, 5, 2)])
        assert v["rule"] == "iso_raise" and v["preferred"] == "raise", cards
    sb = _v7(1, ["Ah", "Jd"], [_limp(3), _hero(1, "call", None, 5, 1)])
    assert sb["rule"] == "iso_raise" and sb["target_amount"] == 6


def test_iso_raise_folds_overlimps_and_leaves_other_spots_alone(exploits):
    exploits("iso_raise")
    over = _v7(4, ["7h", "5h"], [_limp(3), _hero(4, "call", None, 5, 2)])  # suited junk from the hijack
    assert (over["rule"], over["preferred"], over["agree"]) == ("iso_raise", "fold", False)
    assert _v7(4, ["7h", "5h"], [_limp(3), _hero(4, "fold", None, 5, 2)])["rule"] != "iso_raise"
    assert _v7(0, ["8h", "8d"], [_limp(3), _hero(0, "raise", 6, 5, 2)])["rule"] != "iso_raise"
    assert _v7(0, ["8h", "8d"], [_open(3, 5), _hero(0, "fold", None, 8, 5)])["rule"] != "iso_raise"
    assert _v7(2, ["8h", "8d"], [_limp(3), _hero(2, "check", None, 5, 0)])["rule"] != "iso_raise"  # big blind


def test_three_bet_value_hands_raise_instead_of_calling_or_folding(exploits):
    exploits("three_bet")
    for action in ("call", "fold"):
        ip = _v7(5, ["Qh", "Qd"], [_open(3, 5), _hero(5, action, None, 8, 5)])  # cutoff vs UTG, in position
        assert (ip["rule"], ip["preferred"], ip["agree"], ip["target_amount"]) == ("three_bet", "raise", False, 15), action
    oop = _v7(2, ["Ah", "Kd"], [_open(0, 5), _hero(2, "call", None, 8, 3)])  # big blind vs button, out of position
    assert (oop["rule"], oop["preferred"], oop["target_amount"]) == ("three_bet", "raise", 20)
    vs_sb = _v7(2, ["9h", "9d"], [_open(1, 6, to_call=1), _hero(2, "call", None, 9, 4)])
    assert (vs_sb["rule"], vs_sb["target_amount"]) == ("three_bet", 18)  # the big blind acts last after the flop: 3x
    vs_btn = _v7(1, ["Ah", "Qh"], [_open(0, 5), _hero(1, "fold", None, 8, 4)])  # AQs vs a button open
    assert vs_btn["rule"] == "three_bet" and vs_btn["preferred"] == "raise"


def test_three_bet_never_forces_bluffs_or_corrects_what_it_cannot_judge(exploits):
    exploits("three_bet")
    assert _v7(5, ["Ah", "5h"], [_open(3, 5), _hero(5, "fold", None, 8, 5)])["rule"] != "three_bet"  # bluff, not forced
    assert _v7(5, ["Jh", "Jd"], [_open(3, 5), _hero(5, "call", None, 8, 5)])["rule"] != "three_bet"  # JJ vs UTG
    assert _v7(5, ["Qh", "Qd"], [_open(3, 5), _hero(5, "raise", 15, 8, 5)])["rule"] != "three_bet"
    assert _v7(5, ["Qh", "Qd"], [_open(3, 5), _limp(4, 8), _hero(5, "call", None, 10, 5)])["rule"] != "three_bet"  # caller
    assert _v7(5, ["Qh", "Qd"], [_act(3, "raise", 200, 3, 2, -200, 0), _hero(5, "fold", None, 203, 200)])["rule"] != "three_bet"


def _bb_defend(cards, action, opener=5, size=5):
    to_call = size - 2
    return _v7(2, cards, [_open(opener, size), _hero(2, action, None, 3 + size, to_call)])


def test_bb_defend_never_turns_a_fold_into_a_call(exploits):
    exploits("bb_defend")
    for cards in (["8h", "7h"], ["Ah", "Td"], ["Th", "9h"], ["Kh", "Jd"], ["9h", "4d"], ["7h", "2d"]):
        assert _bb_defend(cards, "fold")["rule"] != "bb_defend", cards
    assert _bb_defend(["Ah", "Td"], "fold", opener=3)["rule"] != "bb_defend"
    assert _bb_defend(["8h", "7h"], "fold", size=7)["rule"] != "bb_defend"


def test_bb_defend_tightens_against_early_openers_and_bigger_opens(exploits):
    exploits("bb_defend")
    assert _bb_defend(["Qh", "Td"], "call", opener=3)["preferred"] == "fold"  # QTo vs UTG is a fold
    assert _bb_defend(["Ah", "Td"], "call", opener=3)["rule"] != "bb_defend"  # ATo vs UTG stays
    assert _bb_defend(["8h", "7h"], "call", opener=3)["rule"] != "bb_defend"  # suited connectors are not flagged
    assert _bb_defend(["Kh", "9d"], "call", opener=3)["preferred"] == "fold"
    assert _bb_defend(["Ah", "8d"], "call", size=7)["preferred"] == "fold"  # A8o is dropped against a big open


def test_bb_defend_folds_offsuit_trash_calls_but_not_suited_ones(exploits):
    exploits("bb_defend")
    for cards in (["9h", "4d"], ["Kh", "6d"], ["Jh", "7d"], ["8h", "2d"]):
        v = _bb_defend(cards, "call")
        assert (v["rule"], v["preferred"], v["agree"]) == ("bb_defend", "fold", False), cards
    assert _bb_defend(["9h", "4h"], "call")["rule"] != "bb_defend"
    assert _bb_defend(["8h", "7d"], "call")["rule"] != "bb_defend"  # marginal connected offsuit hands are not flagged
    assert _bb_defend(["Kh", "9d"], "call")["rule"] != "bb_defend"  # K9o vs a late open at a good price is marginal
    assert _bb_defend(["Kh", "9d"], "call", size=7)["preferred"] == "fold"  # but not against a big open


def test_bb_defend_only_judges_a_big_blind_facing_one_plain_open(exploits):
    exploits("bb_defend")
    assert _v7(4, ["9h", "4d"], [_open(3, 5), _hero(4, "call", None, 8, 5)])["rule"] != "bb_defend"  # not the big blind
    assert _v7(2, ["9h", "4d"], [_open(5, 5), _limp(0, 8), _hero(2, "call", None, 10, 3)])["rule"] != "bb_defend"  # a caller
    assert _v7(2, ["9h", "4d"], [_open(5, 5), _act(0, "raise", 20, 8, 5, -20, 180), _hero(2, "call", None, 28, 18)])["rule"] != "bb_defend"
    assert _v7(2, ["9h", "4d"], [_act(5, "raise", 200, 3, 2, -200, 0), _hero(2, "call", None, 203, 198)])["rule"] != "bb_defend"


def test_bb_defend_leaves_value_hands_to_three_bet(exploits):
    exploits("bb_defend", "three_bet")
    v = _bb_defend(["Ah", "Kd"], "fold", opener=0)
    assert v["rule"] == "three_bet" and v["preferred"] == "raise"


def _3bet_seq(hero, cards, action, three_bettor, amount3=18, open_amount=5, hero_amount=None, allin=False):
    """Hero opens to `open_amount`, `three_bettor` re-raises to `amount3`, hero acts."""
    after_open = 200 - open_amount
    to_call = amount3 - open_amount
    three = _act(three_bettor, "raise", amount3, 3 + open_amount, open_amount, -amount3,
                 0 if allin else 200 - amount3)
    pot = 3 + open_amount + amount3
    change = 0 if action == "fold" else -(hero_amount - open_amount if action == "raise" else to_call)
    hero_act = _act(hero, action, hero_amount, pot, to_call, change, after_open + change)
    return _v7(hero, cards, [_open(hero, open_amount), three, hero_act])


def test_vs_3bet_continues_with_the_top_of_the_range_in_position_with_a_pair(exploits):
    exploits("vs_3bet")
    for cards in (["Qh", "Qd"], ["Ah", "Ks"], ["Ah", "Kd"]):
        v = _3bet_seq(0, cards, "fold", 1)
        assert (v["rule"], v["preferred"], v["agree"]) == ("vs_3bet", "call", False), cards
    for action, amount in (("fold", None), ("call", None)):
        v = _3bet_seq(0, ["Ah", "Ad"], action, 1, hero_amount=amount)
        assert (v["rule"], v["preferred"], v["target_amount"]) == ("vs_3bet", "raise", 41), action  # 2.3x the 3-bet
    big = _3bet_seq(0, ["Kh", "Kd"], "fold", 1, amount3=40)  # 35 to call: too big to 4-bet, still a continue
    assert big["rule"] == "vs_3bet" and big["preferred"] == "call"
    for cards in (["Jh", "Jd"], ["Th", "Td"], ["Ah", "Qh"]):  # in position (button vs a blind 3-bet): call
        v = _3bet_seq(0, cards, "fold", 1)
        assert (v["rule"], v["preferred"]) == ("vs_3bet", "call"), cards


def test_vs_3bet_leaves_the_rest_to_the_equity_rule_and_unclear_spots_alone(exploits):
    exploits("vs_3bet")
    for cards, action, amount in ((["Ah", "Jd"], "call", None), (["Kh", "Qs"], "call", None), (["7h", "7d"], "call", None),
                                  (["9h", "9d"], "call", None), (["Ah", "Qd"], "call", None), (["Ah", "5h"], "raise", 40)):
        assert _3bet_seq(0, cards, action, 1, hero_amount=amount)["rule"] != "vs_3bet", (cards, action)
    assert _3bet_seq(0, ["Ah", "Jd"], "fold", 1)["rule"] != "vs_3bet"  # folding is never corrected
    oop = _3bet_seq(5, ["Jh", "Jd"], "call", 0)  # cutoff open, button 3-bet: out of position, unclear
    assert oop["rule"] != "vs_3bet"
    assert _3bet_seq(0, ["Ah", "Jd"], "call", 1, allin=True, amount3=200)["rule"] != "vs_3bet"  # a jam
    assert _3bet_seq(0, ["Ah", "Jd"], "call", 1, amount3=60)["rule"] != "vs_3bet"  # 55 to call, over 20% of the stack
    cold = _v7(0, ["Ah", "Jd"], [_open(3, 5), _hero(0, "call", None, 8, 5)])  # we did not open
    assert cold["rule"] != "vs_3bet"


def test_vs_3bet_abstains_after_a_cold_call_between_the_3bet_and_us(exploits):
    exploits("vs_3bet")
    three = _act(1, "raise", 18, 8, 5, -18, 182)
    squeeze_call = _act(2, "call", None, 26, 18, -18, 182)
    hero = _act(0, "fold", None, 44, 13, 0, 195)
    v = _v7(0, ["Qh", "Qd"], [_open(0, 5), three, squeeze_call, hero])
    assert v["rule"] != "vs_3bet"


# ---- field_call: equity against the empirical field range; OFF unless named ----
def _synthetic_table(situation, counts, n=200):
    return {"meta": {}, situation: {"any": {"all": {"n": n, "counts": counts}}}}


@pytest.fixture
def field(monkeypatch, exploits):
    def setup(situation, counts):
        exploits("field_call")
        _t._FIELD_EQ.clear()
        monkeypatch.setattr(_t, "_field_table", lambda: _synthetic_table(situation, counts))
    yield setup
    _t._FIELD_EQ.clear()


def test_field_call_is_off_by_default(monkeypatch):
    monkeypatch.setattr(_t, "_field_table", lambda: _synthetic_table("3bet", {"AA": 200}))
    assert _3bet_seq(0, ["7h", "7d"], "call", 1)["rule"] != "field_call"


def test_field_call_folds_a_call_that_is_far_behind_the_range(field):
    field("3bet", {"AA": 200})
    v = _3bet_seq(0, ["7h", "7d"], "call", 1)  # 13 to call into 26: needs 33%, 77 has about 19% against aces
    assert (v["rule"], v["preferred"], v["agree"], v["situation"]) == ("field_call", "fold", False, "3bet")
    assert v["equity"] < v["required"] - 0.04
    v = _3bet_seq(0, ["9h", "9d"], "raise", 1, hero_amount=40)
    assert (v["rule"], v["preferred"]) == ("field_call", "fold")


def test_field_call_calls_when_the_range_is_weak_and_the_price_is_fine(field):
    field("3bet", {"72o": 200})
    v = _3bet_seq(0, ["Jh", "Jd"], "fold", 1)
    assert (v["rule"], v["preferred"], v["agree"]) == ("field_call", "call", False)
    assert v["equity"] > v["required"] + 0.06
    assert _3bet_seq(0, ["Jh", "Jd"], "call", 1)["rule"] != "field_call"  # a call that is fine stays alone
    assert _3bet_seq(0, ["9h", "9d"], "fold", 1)["rule"] != "field_call"  # outside the continue set: no fold -> call
    assert _3bet_seq(0, ["Th", "Td"], "fold", 1)["preferred"] == "call"  # TT joins in position
    assert _3bet_seq(0, ["Ah", "Qd"], "fold", 1)["preferred"] == "call"  # AQo joins in position
    assert _3bet_seq(5, ["Th", "Td"], "fold", 0, ) ["rule"] != "field_call"  # TT out of position stays out


def test_field_call_leaves_the_top_hands_to_vs_3bet(field):
    field("3bet", {"72o": 200})
    _t.EXPLOITS.add("vs_3bet")
    v = _3bet_seq(0, ["Ah", "Ad"], "fold", 1)
    assert v["rule"] == "vs_3bet" and v["preferred"] == "raise"


def test_field_call_big_price_fold_needs_more_equity(field, monkeypatch):
    field("jam", {"72o": 200})
    # facing a jam for 190 on a 200 stack: fold to call only when equity > required + 0.10
    raise_ = _act(3, "raise", 200, 3, 2, -200, 0)
    hero = _act(0, "fold", None, 203, 200, 0, 200)
    v = _v7(0, ["Ah", "Kd"], [raise_, hero])
    assert v["rule"] == "field_call" and v["preferred"] == "call"
    field("jam", {"AA": 100, "KK": 100})
    v = _v7(0, ["Ah", "5d"], [_act(3, "raise", 200, 3, 2, -200, 0), _act(0, "fold", None, 203, 200, 0, 200)])
    assert v["rule"] != "field_call"  # folding the worse hand is right


def test_field_call_abstains_multiway_and_when_the_table_is_missing(field, monkeypatch):
    field("3bet", {"AA": 200})
    three = _act(1, "raise", 18, 8, 5, -18, 182)
    cold = _act(2, "call", None, 26, 18, -18, 182)
    v = _v7(0, ["7h", "7d"], [_open(0, 5), three, cold, _act(0, "call", None, 44, 13, -13, 182)])
    assert v["rule"] != "field_call"
    monkeypatch.setattr(_t, "_field_table", lambda: None)
    assert _3bet_seq(0, ["7h", "7d"], "call", 1)["rule"] != "field_call"


def test_field_call_facing_a_single_open_uses_the_openers_position_and_size(field):
    field("open", {"AA": 200})
    # UTG opens to 5; we on the button call with 72o: needs 5/? ... equity far below the price
    v = _v7(0, ["7h", "2d"], [_open(3, 5), _hero(0, "call", None, 8, 5)])
    assert (v["rule"], v["preferred"], v["situation"]) == ("field_call", "fold", "open")


def test_v7_field_call_text_has_no_stat_words():
    from poker.sft.make_records import V7_EXPLOITS, V7_TEXTS
    assert "field_call" in V7_EXPLOITS and "field_call" in V7_TEXTS
    for pref in ("fold", "call"):
        text = V7_TEXTS["field_call"]({"preferred": pref, "equity": 0.31, "required": 0.4, "situation": "3bet"}, "77")
        assert text and not any(w in text.lower() for w in ("percent", "%", "data", "field", "sample", "team"))


# ---- postflop_sizing ----
DRY_BOARD = ("Ks", "9d", "4h", "2c", "3d")
WET_BOARD = ("9h", "8h", "6c", "2d", "3s")


def _sized(cards, amount, street="flop", board=DRY_BOARD, pot=100, action="bet"):
    acts = [_pa(1, street, action, amount, pot, 0, -amount, 200 - amount)]
    return _judge1(_xdb([_phand(1, {0: ["5h", "6h"], 1: cards}, acts, board=board)]), 1)


def test_postflop_sizing_is_off_by_default():
    assert _sized(["Kh", "7d"], 120)["rule"] != "postflop_sizing"


def test_postflop_sizing_trims_overbets_without_a_monster(exploits):
    exploits("postflop_sizing")
    v = _sized(["Kh", "7d"], 120)  # 1.2x pot with top pair on a dry board
    assert (v["rule"], v["preferred"], v["agree"], v["target_frac"], v["target_amount"]) == ("postflop_sizing", "bet", False, 0.33, 50)
    assert agrees(v, "bet", 33) and not agrees(v, "bet", 120) and not agrees(v, "check")
    wet = _sized(["Jh", "3h"], 130, board=WET_BOARD)
    assert (wet["rule"], wet["target_frac"], wet["target_amount"], wet["min_amount"]) == ("postflop_sizing", 0.7, 75, 50)
    assert agrees(wet, "bet", 70) and not agrees(wet, "bet", 30) and not agrees(wet, "bet", 130)
    river = _sized(["Kh", "7d"], 120, street="river")
    assert river["rule"] == "postflop_sizing" and river["target_frac"] == 0.6
    assert _sized(["Kh", "Kd"], 120)["rule"] != "postflop_sizing"  # a set may bet big
    assert _sized(["Kh", "7d"], 110)["rule"] == "postflop_sizing"  # 1.1x is the edge: the target still applies on a dry board


def test_postflop_sizing_fixes_clear_mis_sizes_by_texture(exploits):
    exploits("postflop_sizing")
    assert _sized(["Kh", "7d"], 75)["rule"] == "postflop_sizing"  # 75% pot on a dry board
    assert _sized(["Kh", "7d"], 50)["rule"] != "postflop_sizing"
    assert _sized(["Kh", "7d"], 33)["rule"] != "postflop_sizing"
    assert _sized(["Jh", "3h"], 25, board=WET_BOARD)["rule"] == "postflop_sizing"  # tiny bet on a wet board
    assert _sized(["Jh", "3h"], 66, board=WET_BOARD)["rule"] != "postflop_sizing"
    assert _sized(["Jh", "3h"], 150, board=WET_BOARD)["rule"] == "postflop_sizing"  # 1.5x pot, not all-in
    assert _sized(["Kh", "7d"], 200, pot=100)["rule"] != "postflop_sizing"  # all-in is left alone


def test_postflop_sizing_checks_weak_hands_that_bet(exploits):
    exploits("postflop_sizing")
    for street in ("flop", "turn"):
        v = _sized(["Jh", "7d"], 55, street=street)  # more than half the pot
        assert (v["rule"], v["preferred"], v["agree"]) == ("postflop_sizing", "check", False), street
    assert _sized(["Ah", "Qd"], 55)["rule"] != "postflop_sizing"  # ace-high keeps equity
    assert _sized(["Jh", "7h"], 55, board=("9h", "8h", "2c", "3s", "4d"))["rule"] != "postflop_sizing"  # flush draw
    assert _sized(["Jh", "7d"], 33, street="river")["rule"] != "postflop_sizing"  # river bluffs are not judged
    assert _sized(["Kh", "7d"], 33)["rule"] != "postflop_sizing"  # a pair is a normal bet


def test_postflop_sizing_lets_heads_up_small_stabs_alone(exploits):
    exploits("postflop_sizing")
    assert _sized(["Jh", "7d"], 33)["rule"] != "postflop_sizing"
    assert _sized(["Jh", "7d"], 50)["rule"] != "postflop_sizing"


def test_postflop_sizing_checks_weak_hands_in_multiway_pots(exploits):
    exploits("postflop_sizing")
    acts = [_pa(1, "flop", "bet", 33, 100, 0, -33, 167)]
    hand = _phand(1, {0: ["5h", "6h"], 1: ["Jh", "7d"], 2: ["2s", "2h"]}, acts, seats=(0, 1, 2))
    v = _judge1(_xdb([hand], nseats=3), 1)
    assert (v["rule"], v["preferred"]) == ("postflop_sizing", "check")


def test_postflop_sizing_exempts_gutshots(exploits):
    exploits("postflop_sizing")
    acts = [_pa(1, "flop", "bet", 55, 100, 0, -55, 145)]
    board = ("9h", "8c", "2d", "3s", "4d")
    hand = _phand(1, {0: ["5h", "6h"], 1: ["Jh", "7d"]}, acts, board=board)  # 7-8-9-J: a gutshot
    assert _judge1(_xdb([hand]), 1)["rule"] != "postflop_sizing"
    hand = _phand(1, {0: ["5h", "6h"], 1: ["Jh", "5d"]}, acts, board=("9h", "Kc", "2d", "3s", "4d"))  # no draw
    assert _judge1(_xdb([hand]), 1)["preferred"] == "check"


def test_board_is_wet_needs_three_ranks_inside_a_four_rank_window():
    assert not _t._board_is_wet(["9h", "7c", "5d"])  # 9-7-5 spans five ranks
    assert _t._board_is_wet(["9h", "8c", "6d"])
    assert _t._board_is_wet(["9h", "8c", "7d"])
    assert not _t._board_is_wet(["Ah", "9c", "4d"])
    assert _t._board_is_wet(["Ah", "3c", "2d"])  # ace plays low
    assert _t._board_is_wet(["9h", "8h", "2d"])  # two-tone


def test_postflop_sizing_skips_wet_boards_when_the_minimum_bet_exceeds_the_cap(exploits):
    exploits("postflop_sizing")
    acts = [_pa(1, "flop", "bet", 30, 100, 0, -30, 10)]  # only 40 behind: the 50 minimum on a wet board is out of reach
    hand = _phand(1, {0: ["5h", "6h"], 1: ["Jh", "3h"]}, acts, board=("9h", "8h", "6c", "2d", "3s"))
    assert _judge1(_xdb([hand]), 1)["rule"] != "postflop_sizing"


def test_v7_sizing_text_names_the_size_from_the_target():
    from poker.sft.make_records import V7_TEXTS
    for frac, word in (("0.33", "a third"), ("0.5", "half"), ("0.7", "two thirds"), ("0.75", "three quarters")):
        text = V7_TEXTS["postflop_sizing"]({"preferred": "bet", "target_frac": float(frac), "texture": "dry", "reason": "mis_size"}, "KQ")
        assert word in text and "%" not in text, (frac, text)


def test_postflop_sizing_ignores_raises(exploits):
    exploits("postflop_sizing")
    assert _sized(["Kh", "7d"], 120, action="raise")["rule"] != "postflop_sizing"


def test_field_call_does_not_defend_offsuit_junk(field):
    field("open", {"72o": 200})
    # BTN opens to 5, SB folds, BB folds J5o: raw equity beats the price, but offsuit junk plays badly out of position
    acts = [_open(0, 5), _act(1, "fold", None, 8, 4, 0, 199), _hero(2, "fold", None, 8, 3)]
    assert _v7(2, ["Jh", "5d"], acts)["rule"] != "field_call"
    assert _v7(2, ["8h", "7h"], acts)["rule"] != "field_call"  # a fold of an open is never turned into a call


def test_field_call_discounts_equity_out_of_position(field, monkeypatch):
    field("open", {"72o": 200})
    monkeypatch.setattr(_t, "_field_equity", lambda *a: 0.45)
    # SB facing a BTN open to 5: needs 4/12; 0.45 raw clears the margin, 0.45 realized out of position does not
    sb = [_open(0, 5), _hero(1, "fold", None, 8, 4)]
    assert _v7(1, ["9h", "8h"], sb)["rule"] != "field_call"
    monkeypatch.setattr(_t, "_field_equity", lambda *a: 0.55)
    # BTN facing a UTG open to 5: needs 5/13; in position 0.55 still clears it
    btn = [_open(3, 5), _hero(0, "fold", None, 8, 5)]
    assert _v7(0, ["9h", "8h"], btn)["rule"] != "field_call"  # no fold -> call against an open, whatever the equity


# ---- pot_odds_call: a tiny price for our last chips is a call with any two cards ----
# A short stack wins only the matched part of the pot, so the price is judged on the effective pot.

def _short_spot(cards, action, stack=15, jam=200, extra=()):
    """Seat 3 (UTG) jams for `jam`; hero (seat 0, button, nothing in) has `stack` chips left and acts last."""
    shove = _act(3, "raise", jam, 3, 2, -jam, 0)
    pot = 3 + jam + sum(-a["chip_change"] for a in extra)
    to_call = jam
    hero = _act(0, action, None, pot, to_call, 0 if action == "fold" else -stack, 0 if action != "fold" else stack)
    return _v7(0, cards, [shove, *extra, hero], stack=stack)


def _hu_short(cards, action):
    """Heads up: hero (small blind, seat 0) raised to 100 with 15 behind and the big blind jammed 300 over it."""
    raise_ = _act(0, "raise", 100, 3, 1, -99, 15)
    jam = _act(1, "raise", 300, 102, 98, -298, 0)
    hero = _act(0, action, None, 400, 200, 0 if action == "fold" else -15, 15 if action == "fold" else 0)
    return _v7(0, cards, [raise_, jam, hero], nseats=2, stack=115)


def test_effective_price_counts_only_the_chips_a_short_stack_can_win(exploits):
    # 15 chips left, nothing in, a jam for 995 in front: 15 to win 15 + 3 of blinds + our 15, about 45%
    row = {"street": "preflop", "to_call": 995, "pot_before": 998, "seat": 0, "position": "BTN", "facing_allin": 1}
    before = [{"street": "preflop", "seat": 3, "position": "UTG", "action": "raise", "chip_change": -995}]
    required, live = _t._effective_price(row, before, 15)
    assert required == pytest.approx(15 / 33) and live == 1
    assert _t._cheap_all_in(row, before, 15) is None
    exploits("pot_odds_call", "big_price_fold")
    v = _short_spot(["7h", "2d"], "fold", jam=995)
    assert v["rule"] != "pot_odds_call"
    called = _short_spot(["7h", "2d"], "call", jam=995)
    assert called["rule"] != "pot_odds_call"  # a 45% price is no pot-odds call


def test_pot_odds_call_calls_when_our_own_chips_make_the_price_cheap(exploits):
    exploits("pot_odds_call", "big_price_fold")
    v = _hu_short(["7h", "2d"], "fold")  # 100 already in, 15 behind: 15 to win 215
    assert (v["rule"], v["preferred"], v["agree"]) == ("pot_odds_call", "call", False)
    assert v["required"] == pytest.approx(15 / 230, abs=1e-4)
    called = _hu_short(["7h", "2d"], "call")
    assert called["rule"] != "big_price_fold" and called.get("preferred") != "fold"


def test_pot_odds_call_big_blind_with_one_chip_behind(exploits):
    exploits("pot_odds_call")
    opener = _act(3, "raise", 6, 3, 2, -6, 194)
    hero = _act(2, "fold", None, 9, 4, 0, 1)  # the big blind posted 2 and has 1 chip behind
    v = _v7(2, ["7h", "2d"], [opener, hero], stack=3)
    assert (v["rule"], v["preferred"], v["agree"]) == ("pot_odds_call", "call", False)
    assert v["required"] == pytest.approx(1 / 7, abs=1e-4)


def test_pot_odds_call_needs_an_all_in_spot_and_a_cheap_price(exploits):
    exploits("pot_odds_call")
    assert _short_spot(["7h", "2d"], "fold", stack=60)["rule"] != "pot_odds_call"  # 60 into 123: about 33%
    # a big stack facing a big raise that is not all-in is not an all-in spot
    raise_ = _act(3, "raise", 30, 3, 2, -30, 170)
    hero = _act(0, "fold", None, 33, 30, 0, 200)
    assert _v7(0, ["7h", "2d"], [raise_, hero])["rule"] != "pot_odds_call"


def test_pot_odds_call_is_off_by_default():
    assert _hu_short(["7h", "2d"], "fold")["rule"] != "pot_odds_call"


def test_big_price_fold_stays_off_when_the_effective_price_is_cheap(exploits):
    exploits("big_price_fold")
    # 30 to call (a big price) after we already put in 100 against a 3-bet to 130: 30 to win 260 is cheap
    open_ = _act(0, "raise", 100, 3, 1, -99, 101)
    three = _act(1, "raise", 130, 102, 98, -128, 70)
    hero = _act(0, "call", None, 230, 30, -30, 71)
    v = _v7(0, ["7h", "2d"], [open_, three, hero], nseats=2)
    assert v["rule"] != "big_price_fold"


def _price_spot(hand_no, stacks):
    open_ = _open(3, 30)
    hero = _act(5, "call", None, 33, 30, -30, 20)
    return _v7(5, ["7h", "2d"], [open_, hero], hand_no=hand_no, stacks=stacks)


def test_big_price_fold_skips_the_last_hand_when_we_are_already_last(exploits):
    exploits("big_price_fold")
    short = {s: 200 for s in range(6)} | {5: 50}
    assert _price_spot(29, short)["rule"] == "big_price_fold"  # earlier hands: still a fold
    assert _price_spot(30, short)["rule"] != "big_price_fold"  # last hand, smallest stack: no label
    mid = {s: 200 for s in range(6)} | {5: 50, 4: 40}
    assert _price_spot(30, mid)["rule"] == "big_price_fold"  # someone is below us: rank can still fall


def test_huge_price_continue_includes_tens(exploits):
    exploits("big_price_fold")
    assert "TT" in _t.HUGE_PRICE_CONTINUE
    assert _judge1(_pre_db(0, ["Th", "Td"], "call", before=[_raise_act(3, 200)]), 0)["rule"] != "big_price_fold"


# ---- facing_open_tight ----
def _fot(hero, cards, opener=3, size=5, action="call", extra=()):
    hero_act = _hero(hero, action, None, 3 + size, size - (2 if hero == 2 else 1 if hero == 1 else 0))
    return _v7(hero, cards, [_open(opener, size), *extra, hero_act])


def test_facing_open_tight_folds_calls_outside_the_call_set(exploits):
    exploits("facing_open_tight")
    for hero, cards in ((0, ["9h", "8h"]), (5, ["Ah", "9h"]), (0, ["Ah", "Td"]), (0, ["Qh", "Jd"]), (0, ["Kh", "Ts"]),
                        (5, ["Kh", "9h"]), (0, ["Th", "9d"])):
        v = _fot(hero, cards)
        assert (v["rule"], v["preferred"], v["agree"]) == ("facing_open_tight", "fold", False), (hero, cards)
        assert "oop" in v


def test_facing_open_tight_keeps_the_call_set(exploits):
    exploits("facing_open_tight")
    for hero, cards in ((0, ["Ah", "Th"]), (0, ["Kh", "Th"]), (5, ["Qh", "Th"]), (5, ["Jh", "Th"]), (0, ["Ah", "Qd"]),
                        (0, ["Kh", "Qd"]), (0, ["2h", "2d"]), (5, ["7h", "7d"])):
        assert _fot(hero, cards)["rule"] != "facing_open_tight", (hero, cards)


def test_facing_open_tight_pairs_need_position(exploits):
    exploits("facing_open_tight")
    v = _fot(1, ["7h", "7d"], opener=0)  # small blind vs a button open: out of position
    assert (v["rule"], v["preferred"], v["oop"]) == ("facing_open_tight", "fold", True)
    assert _fot(0, ["7h", "7d"], opener=3)["rule"] != "facing_open_tight"  # button vs UTG: in position


def test_facing_open_tight_never_turns_a_fold_into_a_call(exploits):
    exploits("facing_open_tight")
    for cards in (["Ah", "Ks"], ["7h", "7d"], ["Ah", "Ts"], ["9h", "8h"]):
        assert _fot(0, cards, action="fold")["rule"] != "facing_open_tight", cards


def test_facing_open_tight_leaves_value_hands_to_three_bet(exploits):
    exploits("facing_open_tight")
    assert _fot(5, ["Qh", "Qd"])["rule"] != "facing_open_tight"  # QQ vs UTG is a 3-bet hand
    exploits("three_bet", "facing_open_tight")
    v = _fot(5, ["Qh", "Qd"])
    assert v["rule"] == "three_bet" and v["preferred"] == "raise"


def test_facing_open_tight_spots(exploits):
    exploits("facing_open_tight")
    assert _fot(2, ["9h", "4d"])["rule"] != "facing_open_tight"  # big blind: bb_defend's job
    assert _fot(0, ["9h", "8h"], extra=[_limp(4, 8)])["rule"] != "facing_open_tight"  # a caller
    allin = _v7(0, ["9h", "8h"], [_act(3, "raise", 200, 3, 2, -200, 0), _hero(0, "call", None, 203, 200)])
    assert allin["rule"] != "facing_open_tight"
    three = _act(1, "raise", 18, 8, 5, -18, 182)
    assert _v7(0, ["9h", "8h"], [_open(3, 5), three, _hero(0, "call", None, 26, 18)])["rule"] != "facing_open_tight"


def test_facing_open_tight_is_off_by_default_and_listed():
    from poker.sft.make_records import V7_EXPLOITS, V7_TEXTS
    assert _fot(0, ["9h", "8h"])["rule"] != "facing_open_tight"
    assert "facing_open_tight" in V7_EXPLOITS and "facing_open_tight" in V7_TEXTS


# ---- rfi_gto: open sizes ----
def test_rfi_gto_cuts_back_oversized_first_in_raises(exploits):
    exploits("rfi_gto")
    for seat, cards, amount, size in ((5, ["Ah", "Qh"], 8, 5), (3, ["Ah", "Kh"], 7, 5), (1, ["Ah", "2d"], 8, 6),
                                      (0, ["Kh", "8d"], 12, 5), (5, ["Ah", "Qh"], 200, 5)):
        v = _rfi(seat, cards, "raise", amount)
        assert (v["rule"], v["preferred"], v["agree"], v["target_amount"]) == ("rfi_gto", "raise", False, size), (seat, amount)
        assert v["oversize"] and agrees(v, "raise", size) and not agrees(v, "raise", amount)


def test_rfi_gto_leaves_ordinary_open_sizes_alone(exploits):
    exploits("rfi_gto")
    for seat, cards, amount in ((5, ["Ah", "Qh"], 5), (5, ["Ah", "Qh"], 6), (3, ["Ah", "Kh"], 4), (1, ["Ah", "2d"], 6),
                                (1, ["Ah", "2d"], 7)):
        assert _rfi(seat, cards, "raise", amount)["rule"] != "rfi_gto", (seat, amount)
    assert _rfi(5, ["7h", "2d"], "raise", 12)["rule"] != "rfi_gto"  # outside the chart: size is not judged
    # a short all-in is not an oversize
    assert _t._rfi_gto({"street": "preflop", "hand_code": "AQs", "position": "CO", "to_call": 2, "facing_allin": 0,
                        "effective_stack": 28, "action": "raise", "amount": 28, "stack_after": 0},
                       [], [], 28, {}) is None


def test_rfi_gto_small_blind_complete(exploits):
    exploits("rfi_gto")
    out = _rfi(1, ["7h", "2d"], "call")  # completing with trash
    assert (out["rule"], out["preferred"]) == ("rfi_gto", "fold")
    inn = _rfi(1, ["Kh", "9h"], "call")  # completing with a chart hand: raise to 6
    assert (inn["rule"], inn["preferred"], inn["target_amount"]) == ("rfi_gto", "raise", 6)


def test_v7_texts_for_the_new_rules_use_plain_poker_words():
    from poker.sft.make_records import V7_TEXTS
    texts = [V7_TEXTS["facing_open_tight"]({"preferred": "fold", "oop": oop}, "98s") for oop in (True, False)]
    texts.append(V7_TEXTS["rfi_gto"]({"preferred": "raise", "target_amount": 5, "oversize": True}, "AQs"))
    for text in texts:
        assert text.strip().endswith(("Fold.", "Raise to 5."))
        assert not any(w in text.lower() for w in ("percent", "%", "data", "field", "sample", "team"))
