import sqlite3

import pytest

from poker import ranges as R


def test_hand_class_mapping():
    assert R.hand_class("As", "Ks") == "AKs"
    assert R.hand_class("Kd", "Ah") == "AKo"
    assert R.hand_class("7c", "7d") == "77"
    assert R.hand_class("2h", "Th") == "T2s"
    assert len(R.ALL_CLASSES) == 169


def test_combo_math():
    assert R.combos_of("AA") == 6 and R.combos_of("AKs") == 4 and R.combos_of("AKo") == 12
    assert sum(R.combos_of(c) for c in R.ALL_CLASSES) == 1326
    assert len(R.expand("AKs")) == 4 and len(R.expand("AKo")) == 12 and len(R.expand("QQ")) == 6


def _table(n_any, n_btn):
    def b(n, cls):
        return {"n": n, "counts": {cls: n}}
    return {"open": {"any": {"all": b(n_any, "AA")},
                     "BTN": {"all": b(n_btn, "KK"), "<=2.5x": b(5, "72o"), "2.5-4x": b(40, "QQ")}}}


def test_fallback_logic():
    t = _table(100, 10)
    assert R.range_probs(t, "open", "BTN")["AA"] > R.range_probs(t, "open", "BTN")["KK"]  # n<30 -> any
    t = _table(100, 50)
    p = R.range_probs(t, "open", "BTN")
    assert p["KK"] > p["AA"]
    assert R.range_probs(t, "open", "BTN", "2.5-4x")["QQ"] > 0.5
    p = R.range_probs(t, "open", "BTN", "<=2.5x")  # 5 samples -> position pool
    assert p["KK"] > p["72o"]
    assert abs(sum(R.range_probs(t, "open", "UTG").values()) - 1) < 1e-9


def test_equity_sanity_and_determinism():
    flat = {c: 1.0 for c in R.ALL_CLASSES}
    aa = R.equity_vs_range(("As", "Ah"), flat, n=1500, seed=1)
    assert 0.80 < aa < 0.90
    assert R.equity_vs_range(("7c", "2d"), {"AA": 1.0}, n=800, seed=1) < 0.20
    assert R.equity_vs_range(("As", "Ah"), flat, n=300, seed=5) == R.equity_vs_range(("As", "Ah"), flat, n=300, seed=5)


def test_build_filters_and_buckets(tmp_path):
    db = tmp_path / "a.db"
    c = sqlite3.connect(db)
    c.execute("create table entries(sub_key, name, kind)")
    c.execute("create table actions(match_id, hand_number, idx, seat, sub_key, street, action, amount, pot_before,"
              " to_call, stack_after, card1, card2, hand_code, position, raises_before)")
    c.executemany("insert into entries values(?,?,?)", [("t", "Spark", "team"), ("l", "LoRA", "team"), ("h", "Bot", "house")])
    rows = []
    for i, sk in enumerate(["t", "l", "h"]):
        rows.append((1, 1, i, i, sk, "preflop", "raise", 6.0, 3, 2, 194, "As", "Ah", "AA", "BTN", 0))
    rows.append((1, 2, 0, 0, "t", "preflop", "raise", 200.0, 3, 2, 0, "Ks", "Kh", "KK", "CO", 1))
    rows.append((1, 3, 0, 1, "t", "preflop", "fold", None, 3, 2, 100, "7s", "2h", "72o", "UTG", 0))
    c.executemany("insert into actions values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    c.commit()
    t = R.build(str(db))
    assert t["open"]["BTN"]["all"]["n"] == 1  # LoRA and house excluded
    assert t["open"]["BTN"]["2.5-4x"]["counts"]["AA"] == 1
    assert t["jam"]["any"]["all"]["counts"]["KK"] == 1
    assert t["first_in"]["BTN"]["open"] == 1 and t["first_in"]["UTG"]["fold"] == 1
