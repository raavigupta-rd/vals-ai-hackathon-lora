import json

from poker.sft.make_records import correction_reasoning, select, target_from_reply


def joined(decision="d1", action="call", cutoff=False, reasoning="I think. Calling.", amount=None):
    args = {"action": action} | ({"amount": amount} if amount is not None else {})
    return {"span_id": decision, "source": "arena", "kind": "action", "cutoff": cutoff,
            "messages": [{"role": "system", "content": "rules"}, {"role": "user", "content": "state"}],
            "reasoning": reasoning, "content": "",
            "tool_calls": [{"name": "submit_action", "arguments": json.dumps(args)}],
            "match": {"match_id": "m", "hand": 1, "seat": 0, "action_index": 0, "verified": True}}


def test_target_from_reply_keeps_the_models_own_reasoning_and_call():
    target = target_from_reply(joined(action="raise", amount=6))
    assert target["reasoning"] == "I think. Calling."
    assert target["tool_call"] == {"name": "submit_action", "arguments": {"action": "raise", "amount": 6}}


def test_correction_reasoning_is_short_and_names_the_numbers():
    text = correction_reasoning({"rule": "allin_call", "preferred": "fold", "equity": 0.38, "required": 0.57,
                                 "to_call": 194, "pot": 213}, hand_code="96o")
    assert "38%" in text and "57%" in text and "96o" in text and len(text) < 600


def test_select_mixes_corrections_agreements_and_clean_abstains():
    rows = [(joined("bad", "call"), {"rule": "allin_call", "preferred": "fold", "agree": False, "equity": 0.3,
                                    "required": 0.55, "to_call": 190, "pot": 200}, "96o"),
            (joined("good", "fold"), {"rule": "allin_call", "preferred": "fold", "agree": True}, "72o"),
            (joined("cut", "check", cutoff=True), {"rule": None, "preferred": None, "agree": None}, "AKo")]
    rows += [(joined(f"a{i}", "check"), {"rule": None, "preferred": None, "agree": None}, "T9s") for i in range(10)]
    records, stats = select(rows, cycle=1, abstain_ratio=2.0, seed=0)
    by_id = {r["decision_id"]: r for r in records}
    assert by_id["bad"]["target"]["tool_call"]["arguments"] == {"action": "fold"}
    assert by_id["bad"]["source"] == "teacher_correction" and by_id["bad"]["weight"] > 1
    assert by_id["good"]["source"] == "teacher_agree"
    assert "cut" not in by_id  # a reply cut off at the cap is never a target
    assert stats["abstain_kept"] == 4  # 2 x (1 correction + 1 agreement)
    assert all(r["cycle"] == 1 for r in records)


def test_target_drops_an_amount_on_calls_checks_and_folds():
    target = target_from_reply(joined(action="call", amount=18))
    assert target["tool_call"]["arguments"] == {"action": "call"}
