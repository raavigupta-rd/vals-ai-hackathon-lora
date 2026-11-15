import json
from pathlib import Path

from poker.arena import db

FIX = Path(__file__).parent / "arena_fixtures" / "match.json"


def raw():
    return json.loads(FIX.read_text())


def test_hand_code():
    assert db.hand_code("Ad", "Qd") == "AQs"
    assert db.hand_code("Tc", "Th") == "TT"
    assert db.hand_code("3d", "Jd") == "J3s"
    assert db.hand_code("9c", "Ts") == "T9o"
    assert db.hand_code(None, "As") is None


def test_positions():
    assert db.position_labels([0, 1, 2, 3, 4, 5], 5, {}) == {
        0: "SB", 1: "BB", 2: "UTG", 3: "HJ", 4: "CO", 5: "BTN"}
    assert db.position_labels([0, 1, 2], 1, {}) == {2: "SB", 0: "BB", 1: "BTN"}
    assert db.position_labels([2, 4], 4, {}) == {4: "HU_SB", 2: "HU_BB"}


def test_parse_fixture_shape():
    p = db.parse_match(raw())
    assert p["match"]["match_id"] == "m_20261114t231117_fd8e48"
    assert len(p["hands"]) == 4 and len(p["hole_cards"]) == 24
    assert len(p["actions"]) == 17 + 5 + 12 + 8
    assert len(p["seats"]) == 6 and len(p["entries"]) == 6
    keys = {e["sub_key"] for e in p["entries"]}
    assert "org_9197691edc5e|base|infer_66b2ff75a081" in keys
    assert "house:nit||" in keys


def test_parse_action_fields():
    p = db.parse_match(raw())
    first = p["actions"][0]  # hand 1, seat 2 folds Jh8c; button 5 => seat 2 is UTG
    assert (first["seat"], first["action"], first["hand_code"], first["position"]) == (2, "fold", "J8o", "UTG")
    assert first["players_remaining"] == 6 and first["raises_before"] == 0
    assert first["effective_stack"] == 200
    open_raise = p["actions"][1]  # seat 3 raises to 5 with QQ
    assert (open_raise["hand_code"], open_raise["position"]) == ("QQ", "HJ")
    assert open_raise["players_remaining"] == 5
    # raise to 5 on pot 3 facing 2: put in 5, call 2, raise 3 into pot 3+2
    assert open_raise["bet_frac"] == (5 - 2) / (3 + 2)
    nxt = p["actions"][2]
    assert nxt["raises_before"] == 1 and nxt["players_remaining"] == 5


def test_postflop_flags_and_pf_aggressor():
    p = db.parse_match(raw())
    post = [a for a in p["actions"] if a["street"] != "preflop"]
    assert post and all(a["raises_before"] >= 0 for a in post)
    assert any(a["preflop_aggressor"] for a in post)


def synthetic(actions):
    players = [{"seat": s, "hole_cards": ["As", "Ks"], "starting_stack": 100} for s in (0, 1)]
    return {"id": "m_x", "kind": "ranked", "status": "complete", "format": {},
            "seats": [{"seat": s, "entry": {"id": f"e{s}", "kind": "team", "name": f"t{s}",
                                            "submission_name": "v1", "deployment_id": "d"}} for s in (0, 1)],
            "results": [],
            "hands": [{"hand_number": 1, "button_seat": 0, "blinds": {"0": 1, "1": 2}, "players": players,
                       "board": [], "showdown": False, "shown": {}, "actions": actions}]}


def act(seat, street, action, amount, pot, to_call, change, stack_after):
    return {"seat": seat, "street": street, "action": action, "amount": amount, "pot_before": pot,
            "to_call": to_call, "chip_change": change, "stack_after": stack_after}


def test_allin_facing_flag():
    r = synthetic([act(0, "preflop", "raise", 100, 3, 1, -99, 0),
                   act(1, "preflop", "call", None, 102, 98, -98, 0)])
    p = db.parse_match(r)
    assert p["actions"][0]["facing_allin"] == 0
    assert p["actions"][1]["facing_allin"] == 1
    assert p["actions"][0]["position"] == "HU_SB"


def test_load_is_idempotent_and_skips_incomplete():
    con = db.connect(":memory:")
    assert db.load_match(con, raw()) and db.load_match(con, raw())
    assert con.execute("select count(*) from actions").fetchone()[0] == 42
    assert con.execute("select count(*) from matches").fetchone()[0] == 1
    live = raw() | {"id": "m_live", "status": "running"}
    assert db.load_match(con, live) is False
    assert con.execute("select count(*) from matches").fetchone()[0] == 1
