import sqlite3

import pytest

from poker import profiles as P
from poker.ranges import ALL_CLASSES, combos_of

SCHEMA = """
create table matches (match_id text, started_at text, big_blind int);
create table entries (sub_key text, name text, kind text, submission_name text, version int);
create table seats (match_id text, seat int, sub_key text, place int);
create table actions (match_id text, hand_number int, idx int, sub_key text, street text, action text,
  amount real, to_call real, stack_after real, card1 text, card2 text, hand_code text, position text,
  raises_before int);
"""


def mk_db(tmp_path, acts, entries=None, matches=None, seats=None):
    p = tmp_path / "t.db"
    con = sqlite3.connect(p)
    con.executescript(SCHEMA)
    con.executemany("insert into entries values (?,?,?,?,?)", entries or [
        ("a|v1", "Alpha", "team", "v1", 1), ("a|v2", "Alpha", "team", "v2", 1),
        ("b|v1", "Beta", "team", "v1", 1), ("l|x", "LoRA", "team", "x", 1), ("h|x", "House nit", "house", None, None)])
    con.executemany("insert into matches values (?,?,2)", matches or [("m1", "2026-01-01"), ("m2", "2026-01-02")])
    con.executemany("insert into seats values (?,?,?,?)", seats or [
        ("m1", 0, "a|v1", 1), ("m1", 1, "b|v1", 3), ("m2", 0, "a|v2", 2), ("m2", 1, "b|v1", 1)])
    con.executemany("insert into actions values (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", acts)
    con.commit()
    con.close()
    return str(p)


def act(m, h, i, sk, action, code, pos, rb=0, street="preflop", amount=None, to_call=0, stack=100):
    return (m, h, i, sk, street, action, amount, to_call, stack, None, None, code, pos, rb)


def test_team_filter_and_latest(tmp_path):
    acts = [act("m1", 1, 0, "a|v1", "raise", "AKs", "BTN", amount=6),
            act("m1", 1, 1, "l|x", "raise", "AKs", "CO", rb=1, amount=18),
            act("m2", 1, 0, "a|v2", "fold", "72o", "BTN"),
            act("m2", 1, 1, "h|x", "raise", "AKs", "CO")]
    prof = P.build_profiles(mk_db(tmp_path, acts))
    assert set(prof["teams"]) == {"Alpha"}  # LoRA and house bots excluded
    a = prof["teams"]["Alpha"]
    assert a["latest_sub_key"] == "a|v2"
    assert a["all"]["positions"]["BTN"]["first_in_n"] == 2
    assert a["all"]["positions"]["BTN"]["open"] == 50.0
    assert a["latest"]["positions"]["BTN"]["first_in_n"] == 1
    assert a["latest"]["positions"]["BTN"]["fold"] == 100.0
    assert a["mean_place"] == 1.5


def test_first_in_excludes_after_limp_and_bb(tmp_path):
    acts = [act("m1", 1, 0, "b|v1", "call", "98s", "UTG"),
            act("m1", 1, 1, "a|v1", "raise", "AKs", "BTN", amount=8),
            act("m1", 1, 2, "a|v1", "fold", "AKs", "BB", rb=1)]
    prof = P.build_profiles(mk_db(tmp_path, acts))
    a = prof["teams"]["Alpha"]["all"]
    assert "BTN" not in a["positions"]  # a limper came first, so no first-in decision
    assert a["positions"]["BB"]["first_in_n"] == 0
    assert a["positions"]["BB"]["facing_n"] == 1
    assert a["positions"]["BB"]["facing_fold"] == 100.0
    assert prof["teams"]["Beta"]["all"]["positions"]["UTG"]["limp"] == 100.0


def test_vpip_pfr_open_size_and_ranges(tmp_path):
    acts = [act("m1", 1, 0, "a|v1", "raise", "AKs", "BTN", amount=6),
            act("m1", 2, 0, "a|v1", "fold", "72o", "BTN"),
            act("m1", 3, 0, "a|v1", "call", "98s", "CO", rb=1),
            act("m1", 4, 0, "a|v1", "raise", "QQ", "BTN", rb=1, amount=20),
            act("m1", 5, 0, "a|v1", "raise", "TT", "BTN", amount=200, stack=0)]
    a = P.build_profiles(mk_db(tmp_path, acts))["teams"]["Alpha"]["all"]
    assert a["hands"] == 5
    assert a["vpip"] == 80.0 and a["pfr"] == 60.0
    assert a["open_size_bb"] == 3.0 and a["open_size_n"] == 1  # all-in is a jam, not an open
    assert a["ranges"]["open"]["BTN"]["AKs"] == 1
    assert a["ranges"]["3bet"]["any"]["QQ"] == 1
    assert a["ranges"]["call_vs_open"]["CO"]["98s"] == 1
    assert a["ranges"]["jam"]["BTN"]["TT"] == 1
    assert a["first_in_by_class"]["BTN"]["72o"] == 1


def test_postflop(tmp_path):
    acts = [act("m1", 1, 0, "a|v1", "bet", "AKs", "BTN", street="flop"),
            act("m1", 1, 1, "a|v1", "check", "AKs", "BTN", street="turn"),
            act("m1", 1, 2, "a|v1", "call", "AKs", "BTN", street="river", to_call=5),
            act("m1", 1, 3, "a|v1", "fold", "AKs", "BTN", street="river", to_call=5)]
    a = P.build_profiles(mk_db(tmp_path, acts))["teams"]["Alpha"]["all"]
    assert a["postflop_aggr"] == 33.33 and a["postflop_n"] == 3
    assert a["fold_to_bet"] == 50.0 and a["fold_to_bet_n"] == 2


ORDER = ["AA", "KK", "AKs", "QQ", "72o"]


def toy_order():
    rest = [c for c in ALL_CLASSES if c not in ORDER]
    return ORDER[:4] + rest + ["72o"]


def test_adjusted_range_width_and_order():
    order = toy_order()
    base = {c: combos_of(c) / 1326 for c in ALL_CLASSES}
    tight = P.adjusted_range(base, 5, 100, prior_strength=10, order=order, base_freq=0.4, mode="strength")
    loose = P.adjusted_range(base, 90, 100, prior_strength=10, order=order, base_freq=0.4, mode="strength")
    assert sum(tight.values()) == pytest.approx(1.0)
    assert tight["AA"] > 0 and tight["72o"] < 1e-3
    # looser posterior includes more hands; mass on AA per combo shrinks
    assert loose["72o"] > tight["72o"]
    assert tight["AA"] / combos_of("AA") > loose["AA"] / combos_of("AA")


def test_adjusted_range_prior_shrinks_to_base():
    order = toy_order()
    base = {c: combos_of(c) / 1326 for c in ALL_CLASSES}
    a = P.adjusted_range(base, 0, 0, order=order, base_freq=0.25, mode="strength")
    b = P.adjusted_range(base, 25, 100, prior_strength=1e9, order=order, base_freq=0.25, mode="strength")
    assert P.posterior_width(0, 0, 0.25, 10) == pytest.approx(0.25)
    assert a["AA"] == pytest.approx(b["AA"], rel=1e-3)
    assert P.posterior_width(100, 100, 0.25, 10) == pytest.approx((2.5 + 100) / 110)


def test_overlap_and_loglik():
    p = {"AA": 0.5, "KK": 0.5}
    emp = {"AA": 1.0}
    assert P.overlap(emp, p) == pytest.approx(0.5)
    assert P.loglik({"AA": 4}, p, eps=0.0) == pytest.approx(__import__("math").log(0.5))


def test_tilt_matches_target_frequency_and_keeps_shape():
    order = toy_order()
    share = {c: combos_of(c) / 1326 for c in ALL_CLASSES}
    # field opens only pairs: rate 1 on pairs would be clipped, so use a field heavy on AA and thin elsewhere
    base = {c: (5.0 if c == "AA" else 1.0) * share[c] for c in ALL_CLASSES}
    t = sum(base.values())
    base = {c: v / t for c, v in base.items()}
    wide = P.adjusted_range(base, 80, 100, order=order, base_freq=0.3)
    narrow = P.adjusted_range(base, 5, 100, order=order, base_freq=0.3)
    assert sum(wide.values()) == pytest.approx(1.0)
    assert wide["72o"] > narrow["72o"]
    assert narrow["AA"] / share["AA"] > narrow["KK"] / share["KK"]  # field shape kept
