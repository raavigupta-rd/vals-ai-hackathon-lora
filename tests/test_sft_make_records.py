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


def test_correction_reasoning_for_postflop_call_names_street_price_and_equity():
    verdict = {"rule": "postflop_call", "preferred": "fold", "equity": 0.21, "required": 0.35, "to_call": 80,
               "pot": 180, "street": "turn", "n": 12}
    text = correction_reasoning(verdict, hand_code="72o")
    assert "turn" in text and "21%" in text and "35%" in text and "72o" in text and "Fold" in text and len(text) < 600
    call = correction_reasoning(verdict | {"preferred": "call", "equity": 0.7}, hand_code="KK")
    assert "70%" in call and "call" in call.lower()


def test_select_mixes_corrections_agreements_and_clean_abstains():
    rows = [(joined("bad", "call"), {"rule": "allin_call", "preferred": "fold", "agree": False, "equity": 0.3,
                                    "required": 0.55, "to_call": 190, "pot": 200}, "96o"),
            (joined("good", "fold"), {"rule": "allin_call", "preferred": "fold", "agree": True}, "72o"),
            (joined("cut", "check", cutoff=True), {"rule": None, "preferred": None, "agree": None}, "AKo")]
    rows += [(joined(f"a{i}", "check"), {"rule": None, "preferred": None, "agree": None}, "T9s") for i in range(10)]
    records, stats = select(rows, cycle=1, abstain_ratio=2.0, seed=0)
    by_id = {r["decision_id"]: r for r in records}
    assert by_id["bad"]["target"]["tool_call"]["arguments"] == {"action": "fold"}
    assert by_id["bad"]["source"] == "teacher_correction" and by_id["bad"]["weight"] == 1.0  # template fallback
    assert by_id["good"]["source"] == "teacher_agree"
    assert "cut" not in by_id  # a reply cut off at the cap is never a target
    assert stats["abstain_kept"] == 4  # 2 x (1 correction + 1 agreement)
    assert all(r["cycle"] == 1 for r in records)


def test_target_drops_an_amount_on_calls_checks_and_folds():
    target = target_from_reply(joined(action="call", amount=18))
    assert target["tool_call"]["arguments"] == {"action": "call"}


def test_corrections_prefer_the_models_own_resampled_reply():
    verdict = {"rule": "allin_call", "preferred": "fold", "agree": False, "equity": 0.3, "required": 0.55,
               "to_call": 190, "pot": 200}
    rows = [(joined("own", "call"), verdict, "96o"), (joined("tmpl", "call"), dict(verdict), "J4o")]
    resampled = {"own": {"reasoning": "In my own words: fold.", "content": "", "arguments": {"action": "fold"}}}
    records, stats = select(rows, cycle=2, abstain_ratio=0, seed=0, resampled=resampled)
    by_id = {r["decision_id"]: r for r in records}
    assert by_id["own"]["source"] == "teacher_resampled" and by_id["own"]["weight"] == 3.0  # all-in calls are bust rules
    assert by_id["own"]["target"]["reasoning"] == "In my own words: fold."
    assert by_id["tmpl"]["source"] == "teacher_correction" and by_id["tmpl"]["weight"] == 1.0
    assert stats["resampled"] == 1 and stats["corrections"] == 2


def test_v3_rules_have_correction_text_and_bust_rules_weigh_more():
    assert "fold" in correction_reasoning({"rule": "facing_raise", "preferred": "fold"}, hand_code="K7o").lower()
    assert "fold" in correction_reasoning({"rule": "own_shove", "preferred": "fold"}, hand_code="J4o").lower()
    shove = {"rule": "allin_call", "preferred": "fold", "agree": False, "equity": 0.3, "required": 0.55,
             "to_call": 190, "pot": 200}
    raise_ = {"rule": "facing_raise", "preferred": "fold", "agree": False}
    own = {"s": {"reasoning": "own", "content": "", "arguments": {"action": "fold"}},
           "r": {"reasoning": "own", "content": "", "arguments": {"action": "fold"}}}
    records, _ = select([(joined("s", "call"), shove, "96o"), (joined("r", "call"), raise_, "K7o")],
                        cycle=3, abstain_ratio=0, seed=0, resampled=own)
    by_id = {r["decision_id"]: r for r in records}
    assert by_id["s"]["weight"] == 3.0 and by_id["r"]["weight"] == 2.0


def test_template_weight_zero_keeps_only_validated_corrections():
    verdict = {"rule": "facing_raise", "preferred": "fold", "agree": False}
    rows = [(joined("own", "call"), verdict, "K7o"), (joined("tmpl", "call"), dict(verdict), "J4o")]
    own = {"own": {"reasoning": "mine", "content": "", "arguments": {"action": "fold"}}}
    records, stats = select(rows, cycle=2, abstain_ratio=0, seed=0, resampled=own, template_weight=0)
    assert [r["decision_id"] for r in records] == ["own"] and stats["unvalidated_dropped"] == 1


def test_a_resampled_reply_is_ignored_when_the_teachers_preference_changed():
    verdict = {"rule": "allin_call", "preferred": "call", "agree": False, "equity": 0.7, "required": 0.5,
               "to_call": 190, "pot": 200}
    stale = {"x": {"reasoning": "fold it", "content": "", "arguments": {"action": "fold"}, "preferred": "fold"}}
    records, stats = select([(joined("x", "fold"), verdict, "KK")], cycle=3, abstain_ratio=0, seed=0,
                            resampled=stale, template_weight=0)
    assert records == [] and stats["unvalidated_dropped"] == 1


def test_weight_scale_multiplies_every_record_weight():
    rows = [(joined(f"a{i}", "check"), {"rule": None, "preferred": None, "agree": None}, "T9s") for i in range(3)]
    rows.append((joined("g", "fold"), {"rule": "allin_call", "preferred": "fold", "agree": True}, "72o"))
    records, _ = select(rows, cycle=3, abstain_ratio=5, seed=0, weight_scale=2.0)
    assert {r["weight"] for r in records} == {2.0}


def test_rules_filter_keeps_only_listed_corrections_and_treats_the_rest_as_anchors():
    shove = {"rule": "allin_call", "preferred": "fold", "agree": False, "equity": 0.3, "required": 0.55,
             "to_call": 190, "pot": 200}
    raise_ = {"rule": "facing_raise", "preferred": "fold", "agree": False}
    own = {k: {"reasoning": "mine", "content": "", "arguments": {"action": "fold"}, "preferred": "fold"} for k in "sr"}
    rows = [(joined("s", "call"), shove, "96o"), (joined("r", "call"), raise_, "K7o")]
    records, stats = select(rows, cycle=4, abstain_ratio=5, seed=0, resampled=own, template_weight=0,
                            rules=("allin_call", "own_shove"))
    by_id = {r["decision_id"]: r for r in records}
    assert by_id["s"]["source"] == "teacher_resampled"
    assert by_id["r"]["source"] == "self_clean"  # out-of-scope correction: base's own action kept as an anchor
    assert stats["out_of_scope"] == 1
