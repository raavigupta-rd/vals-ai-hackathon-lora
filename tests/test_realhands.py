import json

import pytest

from poker import realhands as R
from poker.arena import db

M = "m_test"


def _con():
    return db.connect(":memory:")


def _act(con, hand, idx, seat, street, action, pot, to_call, scripted=0):
    con.execute("INSERT INTO actions (match_id, hand_number, idx, seat, street, action, pot_before, to_call, scripted) "
                "VALUES (?,?,?,?,?,?,?,?,?)", (M, hand, idx, seat, street, action, pot, to_call, scripted))


def _hole(con, hand, seat, c1, c2):
    con.execute("INSERT INTO hole_cards (match_id, hand_number, seat, card1, card2) VALUES (?,?,?,?,?)",
                (M, hand, seat, c1, c2))


def _hand(con, with_kk_folder=False):
    """Seat 2 raises (72o), hero seat 1 (AA) faces it, optional seat 3 (KK) folds later."""
    _hole(con, 1, 2, "7c", "2d")
    _hole(con, 1, 1, "As", "Ah")
    _act(con, 1, 0, 2, "preflop", "raise", 3, 2)
    _act(con, 1, 1, 1, "preflop", "call", 9, 6)  # hero
    if with_kk_folder:
        _hole(con, 1, 3, "Kc", "Kd")
        _act(con, 1, 2, 3, "preflop", "fold", 15, 6)


def test_fold_is_zero_and_call_is_positive_for_aces():
    con = _con()
    _hand(con)
    fold = R.spot_ev(con, M, 1, 1, 0, "fold")
    call = R.spot_ev(con, M, 1, 1, 0, "call")
    assert fold["ev"] == 0.0
    assert call["ev"] > 5.0  # about 0.87 * 15 - 6
    assert call["n_villains"] == 1 and call["to_call"] == 6 and call["pot"] == 9


def test_raise_is_priced_as_call():
    con = _con()
    _hand(con)
    assert R.spot_ev(con, M, 1, 1, 0, "raise")["ev"] == R.spot_ev(con, M, 1, 1, 0, "call")["ev"]


def test_later_folder_is_dropped_but_later_caller_is_kept():
    con = _con()
    _hand(con, with_kk_folder=True)
    dropped = R.spot_ev(con, M, 1, 1, 0, "call")
    assert dropped["n_villains"] == 1
    assert dropped["equity"] > 0.8  # vs 72o only, KK ignored
    con2 = _con()
    _hand(con2)
    _hole(con2, 1, 3, "Kc", "Kd")
    _act(con2, 1, 2, 3, "preflop", "call", 15, 6)
    kept = R.spot_ev(con2, M, 1, 1, 0, "call")
    assert kept["n_villains"] == 2
    assert kept["equity"] < dropped["equity"]


def test_earlier_folder_is_dropped():
    con = _con()
    _hole(con, 1, 3, "Kc", "Kd")
    _act(con, 1, 0, 3, "preflop", "fold", 3, 2)
    _hole(con, 1, 2, "7c", "2d")
    _act(con, 1, 1, 2, "preflop", "raise", 3, 2)
    _hole(con, 1, 1, "As", "Ah")
    _act(con, 1, 2, 1, "preflop", "call", 9, 6)
    assert R.spot_ev(con, M, 1, 1, 0, "call")["n_villains"] == 1


def test_action_index_counts_only_our_unscripted_actions():
    con = _con()
    _hole(con, 1, 2, "7c", "2d")
    _hole(con, 1, 1, "As", "Ah")
    _act(con, 1, 0, 1, "preflop", "fold", 3, 0, scripted=1)  # scripted, not counted
    _act(con, 1, 1, 2, "preflop", "raise", 3, 2)
    _act(con, 1, 2, 1, "preflop", "call", 9, 6)
    assert R.spot_ev(con, M, 1, 1, 0, "call")["to_call"] == 6


def test_unscorable_spots_return_reason():
    con = _con()
    _hole(con, 1, 1, "As", "Ah")
    _act(con, 1, 0, 1, "preflop", "raise", 3, 0)  # nothing to call
    _act(con, 1, 1, 1, "flop", "check", 9, 0)
    assert R.spot_ev(con, M, 1, 1, 0, "call")["ev"] is None
    assert R.spot_ev(con, M, 1, 1, 0, "call")["reason"] == "no_bet_faced"
    assert R.spot_ev(con, M, 1, 1, 1, "call")["reason"] == "not_preflop"
    assert R.spot_ev(con, M, 1, 1, 9, "call")["reason"] == "unmatched"


def test_fallback_to_aggressor_when_everyone_folds_to_hero_raise():
    con = _con()
    _hand(con)
    con.execute("UPDATE actions SET action='raise' WHERE idx=1")
    _act(con, 1, 2, 2, "preflop", "fold", 21, 12)  # raiser folds to our 3bet
    r = R.spot_ev(con, M, 1, 1, 0, "raise")
    assert r["ev"] is not None and r["fallback"] is True


def test_equity_is_deterministic():
    a = R.equity_vs_hands(("As", "Ah"), [("7c", "2d")], n=200, seed=3)
    assert a == R.equity_vs_hands(("As", "Ah"), [("7c", "2d")], n=200, seed=3)
    assert 0.8 < a < 0.95


def test_paired_bootstrap_detects_clear_gap_and_is_seeded():
    ref = {f"s{i}": -1.0 for i in range(30)}
    dep = {f"s{i}": 1.0 for i in range(30)}
    w = {k: 1.0 for k in ref}
    d = R.paired_bootstrap(ref, dep, w, seed=1)
    assert d["diff"] == pytest.approx(2.0) and d["lo"] > 1.9 and d["n"] == 30
    assert d == R.paired_bootstrap(ref, dep, w, seed=1)


def test_score_end_to_end(tmp_path, monkeypatch):
    con = _con()
    _hand(con)
    # second spot: hero 72o facing AA, orig call
    _hole(con, 2, 2, "As", "Ah")
    _hole(con, 2, 1, "7c", "2d")
    _act(con, 2, 0, 2, "preflop", "raise", 3, 2)
    _act(con, 2, 1, 1, "preflop", "call", 9, 6)
    joined = tmp_path / "j.jsonl"
    joined.write_text("\n".join(json.dumps({"span_id": s, "source": "arena", "kind": "action",
                                            "match": {"match_id": M, "hand": h, "seat": 1, "action_index": 0,
                                                      "verified": True}})
                                for s, h in (("a", 1), ("b", 2))) + "\n")
    rows = lambda acts: [{"span_id": s, "valid": v, "action": a, "group": "judged"}  # noqa: E731
                         for s, (a, v) in zip("ab", acts)]
    gate = {"spots": {"ref": rows([("fold", True), ("call", True)]),
                      "dep": rows([("call", True), ("call", False)])}}  # invalid counts as fold
    path = tmp_path / "g.json"
    path.write_text(json.dumps(gate))
    out = R.score(path, reference="ref", joined=joined, con=con, n=200, seed=0)
    dep, ref = out["deployments"]["dep"], out["deployments"]["ref"]
    assert dep["n_spots"] == 2 and dep["n_invalid_scored"] == 1 and dep["n_invalid_all"] == 1
    assert dep["mean_ev"] > ref["mean_ev"]  # dep calls with AA, ref folds it
    assert dep["orig_mean_ev"] == ref["orig_mean_ev"]
    assert out["deployments"]["dep"]["vs_ref"]["diff"] == pytest.approx(dep["wmean_ev"] - ref["wmean_ev"])
    assert "vs_ref" not in ref


def test_fold_is_unscorable_when_call_is_unscorable():
    con = _con()
    _hole(con, 1, 1, "As", "Ah")
    _act(con, 1, 0, 1, "preflop", "fold", 3, 2)  # nobody else ever put money in or has cards
    r = R.spot_ev(con, M, 1, 1, 0, "fold")
    assert r["ev"] is None and r["reason"] == "no_villain"


def test_three_way_tie_splits_equity():
    # same ranks, different suits: mostly three-way ties, equity near 1/3
    eq = R.equity_vs_hands(("As", "Ks"), [("Ah", "Kh"), ("Ad", "Kd")], n=600, seed=1)
    assert 0.28 < eq < 0.40


def test_blind_checking_the_option_stays_in_the_hand():
    con = _con()
    _hole(con, 1, 1, "As", "Ah")
    _hole(con, 1, 2, "7c", "2d")
    _hole(con, 1, 3, "Kc", "Kd")
    _act(con, 1, 0, 2, "preflop", "call", 3, 2)  # limper
    _act(con, 1, 1, 1, "preflop", "call", 5, 2)  # hero faces the limp
    _act(con, 1, 2, 3, "preflop", "check", 7, 0)  # BB checks the option
    r = R.spot_ev(con, M, 1, 1, 0, "call")
    assert r["n_villains"] == 2 and r["fallback"] is False


def test_non_fold_call_raise_actions_are_invalid_and_scored_as_fold():
    con = _con()
    _hand(con)
    for bad in ("check", "bet", "all_in"):
        r = R.spot_ev(con, M, 1, 1, 0, bad)
        assert r["ev"] == 0.0 and r["invalid"] is True
    assert R.spot_ev(con, M, 1, 1, 0, "call")["invalid"] is False
    assert R.spot_ev(con, M, 1, 1, 0, "fold")["invalid"] is False


def test_invalid_action_on_unscorable_spot_stays_unscorable():
    con = _con()
    _hole(con, 1, 1, "As", "Ah")
    _act(con, 1, 0, 1, "preflop", "fold", 3, 2)
    assert R.spot_ev(con, M, 1, 1, 0, "check")["ev"] is None


def test_call_is_capped_at_effective_stack():
    con = _con()
    _hand(con)
    con.execute("UPDATE hole_cards SET starting_stack=4 WHERE seat=1")
    con.execute("UPDATE hole_cards SET starting_stack=200 WHERE seat=2")
    r = R.spot_ev(con, M, 1, 1, 0, "call", n=300)
    assert r["call_amt"] == 4 and r["to_call"] == 6
    assert r["ev"] == pytest.approx(r["equity"] * (9 + 4) - 4)
    con.execute("UPDATE hole_cards SET starting_stack=3 WHERE seat=2")
    assert R.spot_ev(con, M, 1, 1, 0, "call", n=300)["call_amt"] == 3


def test_n_invalid_scored_vs_all_and_bootstrap_key_set(tmp_path):
    con = _con()
    _hand(con)  # span a: scorable
    _hole(con, 2, 1, "As", "Ah")  # span b: nobody else, unscorable
    _act(con, 2, 0, 1, "preflop", "fold", 3, 2)
    joined = tmp_path / "j.jsonl"
    joined.write_text("\n".join(json.dumps({"span_id": s, "source": "arena", "kind": "action",
                                            "match": {"match_id": M, "hand": h, "seat": 1, "action_index": 0,
                                                      "verified": True}})
                                for s, h in (("a", 1), ("b", 2))) + "\n")
    row = lambda s, a, v=True: {"span_id": s, "valid": v, "action": a, "group": "judged"}  # noqa: E731
    gate = {"spots": {"ref": [row("a", "call"), row("b", "fold")],
                      "dep": [row("a", "bet"), row("b", "call", False)]}}
    path = tmp_path / "g.json"
    path.write_text(json.dumps(gate))
    out = R.score(path, reference="ref", joined=joined, con=con, n=100, seed=0)
    dep = out["deployments"]["dep"]
    assert dep["n_spots"] == 1
    assert dep["n_invalid_scored"] == 1  # the bet on span a
    assert dep["n_invalid_all"] == 2
    assert out["deployments"]["ref"]["n_invalid_scored"] == 0
    assert dep["vs_ref"]["n"] == dep["n_spots"] == 1
    assert out["unscored_reasons"] == {"no_villain": 1}
    assert "invalid" in R.format_table(out).splitlines()[0]
