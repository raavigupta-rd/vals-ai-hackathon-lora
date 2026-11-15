import json
from pathlib import Path

import pytest

from poker.arena import db, stats

FIX = Path(__file__).parent / "arena_fixtures" / "match.json"


def mk_match(hands, mid="m_s", n=3, status="complete"):
    seats = [{"seat": s, "entry": {"id": f"e{s}", "kind": "team", "name": f"t{s}", "submission_name": "v1",
                                  "deployment_id": "d", "model": "m"}} for s in range(n)]
    return {"id": mid, "kind": "ranked", "status": status, "format": {"big_blind": 2}, "seats": seats,
            "results": [{"seat": s, "entry_id": f"e{s}", "net_bb": 10.0 * (s - 1), "hands": len(hands),
                         "place": s + 1, "eliminated_on_hand": 5 if s == 0 else None} for s in range(n)],
            "hands": hands}


def mk_hand(actions, board=(), showdown=False, hn=1, n=3, button=2):
    players = [{"seat": s, "hole_cards": ["As", "Ks"], "starting_stack": 200} for s in range(n)]
    return {"hand_number": hn, "button_seat": button, "blinds": {"0": 1, "1": 2}, "players": players,
            "actions": actions, "board": list(board), "showdown": showdown, "shown": {}}


def A(seat, street, action, pot=3, to_call=0, change=0, stack_after=100, amount=None, **kw):
    return dict(seat=seat, street=street, action=action, amount=amount, pot_before=pot, to_call=to_call,
                chip_change=change, stack_after=stack_after, **kw)


def run(matches, by="sub_key"):
    con = db.connect(":memory:")
    for m in matches:
        db.load_match(con, m)
    return stats.compute(con, by=by), con


def get(res, seat, stat):
    return res[f"e{seat}|v1|d" if "|" not in str(seat) else seat][stat]


def test_wilson_matches_known_value():
    r = stats.rate(5, 10)
    assert r["n"] == 10 and r["k"] == 5 and r["rate"] == 0.5
    assert r["ci95"][0] == pytest.approx(0.2366, abs=1e-3) and r["ci95"][1] == pytest.approx(0.7634, abs=1e-3)
    assert stats.rate(0, 0)["rate"] is None


def test_vpip_pfr_3bet_fold_to_3bet():
    # seat2 (btn) opens, seat0 3-bets, seat2 folds; seat1 folds (BB) facing 3-bet
    h = mk_hand([A(2, "preflop", "raise", 3, 2, -6, 194, 6),
                 A(0, "preflop", "raise", 9, 5, -18, 182, 18),
                 A(1, "preflop", "fold", 27, 16),
                 A(2, "preflop", "fold", 27, 12)])
    res, _ = run([mk_match([h])])
    assert get(res, 2, "vpip")["k"] == 1 and get(res, 2, "pfr")["k"] == 1
    assert get(res, 0, "3bet") ["k"] == 1 and get(res, 0, "3bet")["n"] == 1
    assert get(res, 2, "fold_to_3bet") ["k"] == 1 and get(res, 2, "fold_to_3bet")["n"] == 1
    assert get(res, 1, "vpip")["k"] == 0 and get(res, 1, "vpip")["n"] == 1
    assert get(res, 1, "3bet")["n"] == 0  # BB acts after the 3-bet
    assert get(res, 2, "3bet")["n"] == 0


def test_open_shove():
    h = mk_hand([A(2, "preflop", "raise", 3, 2, -200, 0, 200),
                 A(0, "preflop", "fold", 203, 199, 0, 199),
                 A(1, "preflop", "call", 203, 198, -198, 0)])
    res, _ = run([mk_match([h])])
    assert get(res, 2, "open_shove")["k"] == 1 and get(res, 2, "open_shove")["n"] == 1
    assert get(res, 0, "open_shove")["k"] == 0
    assert get(res, 0, "call_vs_allin")["n"] == 1 and get(res, 0, "call_vs_allin")["k"] == 0
    assert get(res, 1, "call_vs_allin")["k"] == 1


def test_cbet_and_fold_to_cbet_with_buckets():
    flop = ["2c", "7d", "9h"]
    h = mk_hand([A(2, "preflop", "raise", 3, 2, -6, 194, 6),
                 A(0, "preflop", "fold", 9, 5),
                 A(1, "preflop", "call", 9, 4, -4, 194),
                 A(1, "flop", "check", 12, 0),
                 A(2, "flop", "bet", 12, 0, -6, 188, 6),   # half pot cbet
                 A(1, "flop", "fold", 18, 6)],
                board=flop)
    res, _ = run([mk_match([h])])
    assert get(res, 2, "cbet")["k"] == 1 and get(res, 2, "cbet")["n"] == 1
    assert get(res, 1, "fold_to_cbet")["k"] == 1 and get(res, 1, "fold_to_cbet")["n"] == 1
    assert get(res, 1, "fold_to_bet_50_100")["k"] == 1  # 6 into 12 = 0.5 pot
    assert get(res, 1, "fold_to_bet_lt50")["n"] == 0
    assert get(res, 1, "wtsd")["n"] == 1 and get(res, 1, "wtsd")["k"] == 0


def test_fold_to_bet_allin_bucket_and_wtsd():
    flop = ["2c", "7d", "9h"]
    h = mk_hand([A(2, "preflop", "raise", 3, 2, -6, 194, 6),
                 A(1, "preflop", "call", 9, 4, -4, 194),
                 A(0, "preflop", "fold", 9, 5),
                 A(1, "flop", "bet", 12, 0, -194, 0, 194),
                 A(2, "flop", "call", 206, 188, -188, 6)],
                board=flop, showdown=True)
    res, _ = run([mk_match([h])])
    assert get(res, 2, "fold_to_bet_allin")["n"] == 1 and get(res, 2, "fold_to_bet_allin")["k"] == 0
    assert get(res, 2, "wtsd")["k"] == 1
    assert get(res, 2, "aggression_factor")["calls"] == 1


def test_flags_tokens_place_bust_and_netbb():
    h = mk_hand([A(2, "preflop", "fold", 3, 2, fallback=True, tokens={"in": 5, "out": 100, "reasoning": 0}),
                 A(0, "preflop", "fold", 3, 2, token_cap=True, tokens={"in": 5, "out": 300, "reasoning": 0})])
    res, _ = run([mk_match([h])])
    assert get(res, 2, "fallback_rate")["k"] == 1 and get(res, 2, "token_cap_rate")["k"] == 0
    assert get(res, 0, "token_cap_rate")["k"] == 1
    assert get(res, 2, "median_out_tokens") == 100
    assert get(res, 2, "mean_place") == 3 and get(res, 0, "bust_rate")["k"] == 1
    assert get(res, 2, "net_bb_per_100")["value"] == 1000.0  # +10bb over 1 hand


def test_team_grouping_merges_submissions():
    m1 = mk_match([mk_hand([A(2, "preflop", "raise", 3, 2, -6, 194, 6)])], mid="m_a")
    m2 = mk_match([mk_hand([A(2, "preflop", "fold", 3, 2)])], mid="m_b")
    m2["seats"][2]["entry"]["submission_name"] = "v2"
    by_sub, _ = run([m1, m2])
    by_team, _ = run([m1, m2], by="entry_id")
    assert len([k for k in by_sub if k.startswith("e2|")]) == 2
    assert by_team["e2"]["pfr"]["n"] == 2 and by_team["e2"]["pfr"]["k"] == 1


def test_probe_flags():
    s = {"n_actions": 400, "vpip": stats.rate(390, 400), "pfr": stats.rate(10, 400),
         "open_shove": stats.rate(0, 100), "fold_to_bet_all": stats.rate(5, 100)}
    assert "maniac_vpip" in stats.probe_flags(s)
    s2 = {"n_actions": 400, "vpip": stats.rate(5, 400), "pfr": stats.rate(2, 400),
          "open_shove": stats.rate(80, 100), "fold_to_bet_all": stats.rate(99, 100)}
    f = stats.probe_flags(s2)
    assert {"nit_vpip", "open_shover", "always_folds_to_bets"} <= set(f)
    assert stats.probe_flags({"n_actions": 5, "vpip": stats.rate(5, 5)}) == []


def test_hand_class():
    c = stats.hand_class({"card1": "Ah", "card2": "Kh"})
    assert c["code"] == "AKs" and c["kind"] == "suited" and "broadway" in c["tags"]
    c = stats.hand_class({"card1": "7s", "card2": "7d"})
    assert c["kind"] == "pair" and "small_pair" in c["tags"]
    c = stats.hand_class({"card1": "9h", "card2": "8h"})
    assert "suited_connector" in c["tags"]
    c = stats.hand_class({"card1": "As", "card2": "4d"})
    assert c["kind"] == "offsuit" and "ace_x" in c["tags"]
    assert stats.hand_class({"card1": None, "card2": None}) is None


def test_range_query():
    h = mk_hand([A(2, "preflop", "raise", 3, 2, -6, 194, 6)])
    _, con = run([mk_match([h])])
    out = stats.range_for(con, key="e2|v1|d", action="raise", street="preflop")
    assert out == {"AKs": 1}
    assert stats.range_for(con, key="e2|v1|d", action="raise", street="preflop", tag="pair") == {}
