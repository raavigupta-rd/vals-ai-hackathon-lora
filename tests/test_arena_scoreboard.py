import random

import pytest

from poker.arena import db, scoreboard
from tests.test_arena_stats import A, mk_hand


def mk(mid, places, started="2026-11-14T10:00:00Z", hands=None, names=None, kinds=None):
    n = len(places)
    names = names or [f"t{s}" for s in range(n)]
    kinds = kinds or ["team"] * n
    seats = [{"seat": s, "entry": {"id": f"e{s}", "kind": kinds[s], "name": names[s], "submission_name": "v1",
                                  "deployment_id": "dep12345678", "model": "m"}} for s in range(n)]
    res = [{"seat": s, "entry_id": f"e{s}", "net_bb": 5.0 * (n - places[s]), "ev_net_bb": 1.0, "hands": 10,
            "place": places[s], "eliminated_on_hand": 7 if places[s] == n else None} for s in range(n)]
    return {"id": mid, "kind": "ranked", "status": "complete", "started_at": started, "format": {"big_blind": 2},
            "seats": seats, "results": res, "hands": hands or []}


def load(matches):
    con = db.connect(":memory:")
    for m in matches:
        db.load_match(con, m)
    return con


def test_bootstrap_ci_is_seeded_and_brackets_mean():
    xs = [1, 2, 3, 4, 5, 6]
    a = scoreboard.bootstrap_mean_ci(xs)
    assert a == scoreboard.bootstrap_mean_ci(xs)
    assert a[0] <= 3.5 <= a[1]
    assert scoreboard.bootstrap_mean_ci([2, 2, 2]) == (2.0, 2.0)
    assert scoreboard.bootstrap_mean_ci([]) is None


def test_match_level_stats_and_min_matches():
    con = load([mk(f"m{i}", [1, 2, 3]) for i in range(3)] + [mk("m9", [3, 1, 2])])
    rows = scoreboard.build(con, min_matches=3)["submissions"]
    by = {r["name"]: r for r in rows}
    assert by["t0"]["matches"] == 4 and by["t0"]["mean_place"] == pytest.approx(1.5)
    assert by["t2"]["bust_rate"]["k"] == 3 and by["t2"]["bust_rate"]["n"] == 4
    assert by["t2"]["top3_rate"]["k"] == 4
    assert by["t0"]["mean_ev_bb"] == 1.0
    assert rows[0]["mean_place"] <= rows[-1]["mean_place"]
    assert scoreboard.build(con, min_matches=5)["submissions"] == []


def test_since_filter():
    con = load([mk("a", [1, 2, 3], started="2026-11-01T00:00:00Z"), mk("b", [1, 2, 3])])
    rows = scoreboard.build(con, min_matches=1, since="2026-11-10")["submissions"]
    assert rows[0]["matches"] == 1


def test_field_excludes_house_and_ours_flag():
    ms = [mk(f"m{i}", [1, 2, 3, 4], names=["LoRA", "Accord", "House nit", "Friends"],
             kinds=["team", "team", "house", "team"]) for i in range(3)]
    con = load(ms)
    out = scoreboard.build(con, min_matches=1)
    assert out["field"]["seat_matches"] == 9  # 3 team entries x 3 matches
    assert all(r["kind"] == "team" for r in out["submissions"])
    ours = scoreboard.build(con, min_matches=1, ours=True)
    assert [r["name"] for r in ours["submissions"]] == ["LoRA"]


def test_our_spot_diagnostics():
    pf = mk_hand([A(0, "preflop", "raise", 3, 2, -200, 0, 200),
                  A(1, "preflop", "call", 203, 200, -200, 0, fallback=1),
                  A(2, "preflop", "fold", 403, 200, tok_out=0)], n=3)
    pf["players"][1]["hole_cards"] = ["Ac", "Ad"]
    con = load([mk("m1", [1, 2, 3], hands=[pf])])
    r = {x["name"]: x for x in scoreboard.build(con, min_matches=1)["submissions"]}
    assert r["t0"]["preflop_shoves"] == 1
    assert r["t1"]["allin_call"]["k"] == 1 and r["t1"]["allin_call"]["n"] == 1
    assert r["t1"]["allin_call_hands"] == {"AA": 1}
    assert r["t1"]["fallback_rate"]["k"] == 1 and r["t1"]["fallback_rate"]["n"] == 1
    assert r["t2"]["allin_call"]["n"] == 1 and r["t2"]["allin_call"]["k"] == 0


def test_paired_comparison_between_ours():
    ms = []
    for i in range(6):
        m = mk(f"m{i}", [1, 2, 3], names=["LoRA", "LoRA", "Accord"])
        m["seats"][1]["entry"]["submission_name"] = "v2"
        ms.append(m)
    out = scoreboard.build(load(ms), min_matches=1, ours=True)
    p = out["paired"][0]
    assert p["diff_mean_place"] == pytest.approx(1.0)  # v2 place 2 vs v1 place 1
    assert p["ci95"][0] <= 1.0 <= p["ci95"][1]
    assert "independent" in p["method"]


def test_afk_actions_count_as_fallbacks_but_not_as_all_in_decisions():
    pf = mk_hand([A(0, "preflop", "raise", 3, 2, -200, 0, 200),
                  A(1, "preflop", "fold", 203, 200, 0, 200, afk=1)], n=2)
    con = load([mk("m1", [1, 2], hands=[pf])])
    r = {x["name"]: x for x in scoreboard.build(con, min_matches=1)["submissions"]}
    assert r["t1"]["fallback_rate"]["k"] == 1 and r["t1"]["fallback_rate"]["n"] == 1
    assert r["t1"]["allin_call"]["n"] == 0  # a forced fold is not a choice about the all-in
