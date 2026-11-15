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
