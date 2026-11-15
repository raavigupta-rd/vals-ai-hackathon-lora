import json

from poker.arena.join_traces import join

MATCH = {
    "id": "m_1", "started_at": "2026-11-15T00:00:00Z", "finished_at": "2026-11-15T00:30:00Z",
    "seats": [{"seat": 0, "entry": {"deployment_id": "infer_x", "name": "us"}},
              {"seat": 1, "entry": {"deployment_id": None, "name": "House"}}],
    "results": [{"seat": 0, "place": 3, "net_chips": -10, "final_chips": 190}],
    "hands": [{
        "hand_number": 6, "players": [{"seat": 0, "hole_cards": ["5c", "9s"], "starting_stack": 200}],
        "board": ["Jd", "4h", "3d"], "deltas": {"0": -2, "1": 2}, "stacks_after": {"0": 198},
        "actions": [
            {"seat": 0, "street": "preflop", "action": "check", "scripted": False},
            {"seat": 1, "street": "preflop", "action": "fold", "scripted": True},
            {"seat": 0, "street": "flop", "action": "bet", "amount": 4, "scripted": False},
        ],
    }],
}


def prompt(street, hole="5c 9s", board="-"):
    return [{"role": "user", "content": f"## Hand 6 of 30\nYour seat: 0 (position 1; 0 = small blind)\nStreet: {street}\nYour hole cards: {hole}\nBoard: {board}\nPot (including current bets): 5\nLegal actions: check, bet to 2-253\nCall submit_action with your decision."}]


def rec(span_id, ts, street, board="-", conv="vals-arena/m_1/hand-6/seat0", kind="action"):
    return {"span_id": span_id, "ts": ts, "deployment": "infer_x", "kind": kind, "messages": prompt(street, board=board),
            "arena": {"match_id": "m_1", "hand": 6, "seat": 0} if conv else None, "tool_calls": []}


def test_join_by_conversation_id_in_order():
    recs = [rec("b", "2026-11-15T00:10:02Z", "flop", "Jd 4h 3d"), rec("a", "2026-11-15T00:10:01Z", "preflop")]
    out = join(recs, [MATCH])
    by = {r["span_id"]: r for r in out}
    assert by["a"]["match"]["match_id"] == "m_1" and by["a"]["match"]["action_index"] == 0
    assert by["b"]["match"]["action_index"] == 1
    assert by["a"]["outcome"]["hand_delta"] == -2 and by["a"]["outcome"]["final_place"] == 3
    assert by["a"]["outcome"]["actual_action"] == {"action": "check", "amount": None}
    assert by["b"]["outcome"]["actual_action"]["action"] == "bet"
    assert "outcome" not in by["a"]["messages"][0]  # outcome kept outside the prompt
    assert by["a"]["match"]["verified"] is True


def test_join_fallback_without_conversation_id_uses_deployment_time_and_cards():
    r = rec("a", "2026-11-15T00:10:01Z", "preflop", conv=None)
    out = join([r], [MATCH])
    assert out[0]["match"]["match_id"] == "m_1" and out[0]["match"]["method"] == "window+cards"


def test_unmatched_and_cleanup():
    r = rec("z", "2026-11-16T00:10:01Z", "preflop", conv=None)
    c = rec("c", "2026-11-15T00:10:05Z", "preflop", kind="cleanup")
    out = {x["span_id"]: x for x in join([r, c], [MATCH])}
    assert out["z"]["match"] is None
    assert out["c"]["match"]["match_id"] == "m_1" and out["c"]["match"]["action_index"] is None


def test_fallback_needs_street_agreement_and_cleanup_unverified():
    r = rec("a", "2026-11-15T00:10:01Z", "river", board="Jd 4h 3d 2c 2d", conv=None)
    assert join([r], [MATCH])[0]["match"] is None
    c = rec("c", "2026-11-15T00:10:05Z", "preflop", kind="cleanup")
    assert join([c], [MATCH])[0]["match"]["verified"] is None
