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


# ---- exploit corrections (v5) ----

from poker.sft.make_records import _correction_args


def _prompt(street, pot, legal, board="-", hole="7h 2s"):
    text = (f"Your seat: 1 (position 2; button)\nStreet: {street}\nYour hole cards: {hole}\nBoard: {board}\n"
            f"Pot (including current bets): {pot}\nLegal actions: {legal}\n"
            "  seat 0: x stack 100, bet 0\n  seat 1: y stack 100, bet 0")
    return {"messages": [{"role": "user", "content": text}]}


def test_exploit_rules_have_plain_correction_text_without_data_talk():
    for rule, word in (("river_fold", "Fold"), ("steal", "Raise"), ("stab", "Bet")):
        text = correction_reasoning({"rule": rule, "preferred": word.lower()}, hand_code="A5s")
        assert text.endswith(word + ".") and len(text) < 300
        assert not any(w in text.lower() for w in ("data", "field", "statistic", "team", "sample"))


def test_new_exploit_corrections_are_plain_poker():
    for rule, pref, word in (("monster_call", "call", "Call"), ("river_air_fold", "fold", "Fold")):
        text = correction_reasoning({"rule": rule, "preferred": pref}, hand_code="KK")
        assert text.endswith(word + ".") and len(text) < 300
        assert not any(w in text.lower() for w in ("data", "field", "statistic", "team", "sample", "percent"))
    assert _correction_args({"rule": "monster_call", "preferred": "call"}, {}) == {"action": "call"}
    assert _correction_args({"rule": "river_air_fold", "preferred": "fold"}, {}) == {"action": "fold"}


def test_exploit_correction_args_are_legal():
    assert _correction_args({"rule": "river_fold", "preferred": "fold"}, {}) == {"action": "fold"}
    steal = _correction_args({"rule": "steal", "preferred": "raise"},
                             _prompt("preflop", 3, "fold, call 2, raise to 4-200"))
    assert steal == {"action": "raise", "amount": 5}
    clamped = _correction_args({"rule": "steal", "preferred": "raise"},
                               _prompt("preflop", 3, "fold, call 2, raise to 6-200"))
    assert clamped == {"action": "raise", "amount": 6}
    capped = _correction_args({"rule": "steal", "preferred": "raise"}, _prompt("preflop", 3, "fold, call 2, raise to 3-4"))
    assert capped == {"action": "raise", "amount": 4}
    stab = _correction_args({"rule": "stab", "preferred": "bet"},
                            _prompt("flop", 20, "check, bet to 2-100", board="Ks 9d 4h"))
    assert stab == {"action": "bet", "amount": 10}
    small = _correction_args({"rule": "stab", "preferred": "bet"},
                             _prompt("flop", 3, "check, bet to 2-100", board="Ks 9d 4h"))
    assert small == {"action": "bet", "amount": 2}
    assert _correction_args({"rule": "stab", "preferred": "bet"}, _prompt("flop", 20, "check, call 5")) is None


def test_main_exploits_flag_sets_the_teacher_set(monkeypatch, tmp_path):
    import sys

    from poker import teacher
    from poker.sft import make_records

    monkeypatch.setattr(teacher, "EXPLOITS", set())
    joined_path = tmp_path / "j.jsonl"
    joined_path.write_text("")
    monkeypatch.setattr(sys, "argv", ["x", "--joined", str(joined_path), "--db", str(tmp_path / "a.db"),
                                      "--cycle", "1", "--out", str(tmp_path / "o.jsonl"),
                                      "--exploits", "river_fold,steal"])
    make_records.main()
    assert teacher.EXPLOITS == {"river_fold", "steal"}


# ---- oversize correction and silent targets ----

def test_oversize_correction_raises_to_the_cap_with_plain_text():
    verdict = {"rule": "oversize", "preferred": "raise", "target_amount": 18, "agree": False}
    args = _correction_args(verdict, _prompt("preflop", 9, "fold, call 6, raise to 10-200"))
    assert args == {"action": "raise", "amount": 18}
    text = correction_reasoning(verdict, hand_code="AKo")
    assert text == ("A smaller raise does the same job here and keeps the pot manageable; there is no need to build "
                    "a huge pot before the flop. Raise to 18.")
    assert not any(w in text.lower() for w in ("data", "field", "team", "player", "opponent"))


def _with_say(decision, action="raise", amount=6):
    record = joined(decision, action, amount=amount)
    record["tool_calls"] = [{"name": "submit_action",
                             "arguments": json.dumps({"action": action, "amount": amount, "say": "all-in! lets see that king!"})}]
    return record


def test_select_strips_say_from_every_target_by_default():
    bad = {"rule": "facing_raise", "preferred": "fold", "agree": False}
    rows = [(_with_say("bad", "call", None), bad, "K7o"), (_with_say("tmpl", "call", None), dict(bad), "J4o"),
            (_with_say("good"), {"rule": "unopened", "preferred": "raise", "agree": True}, "AA"),
            (_with_say("anc"), {"rule": None, "preferred": None, "agree": None}, "T9s")]
    own = {"bad": {"reasoning": "mine", "content": "", "arguments": {"action": "fold", "say": "ha"}, "preferred": "fold"}}
    records, _ = select(rows, cycle=7, abstain_ratio=5, seed=0, resampled=own, template_weight=1.0)
    assert {r["decision_id"] for r in records} == {"bad", "tmpl", "good", "anc"}
    assert all("say" not in r["target"]["tool_call"]["arguments"] for r in records)
    kept, _ = select(rows, cycle=7, abstain_ratio=5, seed=0, resampled=own, template_weight=1.0, strip_say=False)
    by_id = {r["decision_id"]: r for r in kept}
    assert by_id["bad"]["target"]["tool_call"]["arguments"]["say"] == "ha"
    assert "say" in by_id["good"]["target"]["tool_call"]["arguments"]
    assert "say" in by_id["anc"]["target"]["tool_call"]["arguments"]


def test_stats_count_targets_whose_say_was_removed():
    rows = [(_with_say("g"), {"rule": "unopened", "preferred": "raise", "agree": True}, "AA"),
            (_with_say("a"), {"rule": None, "preferred": None, "agree": None}, "T9s"),
            (joined("b", "check"), {"rule": None, "preferred": None, "agree": None}, "T9s")]
    _, stats = select(rows, cycle=7, abstain_ratio=5, seed=0)
    assert stats["say_stripped"] == 2  # counted over the kept records only


def test_keep_say_cli_flag(monkeypatch, tmp_path):
    import sys

    from poker.sft import make_records

    seen = {}
    monkeypatch.setattr(make_records, "select", lambda rows, **kw: seen.update(kw) or ([], {}))
    p = tmp_path / "j.jsonl"
    p.write_text("")
    base = ["x", "--joined", str(p), "--db", str(tmp_path / "a.db"), "--cycle", "1", "--out", str(tmp_path / "o.jsonl")]
    monkeypatch.setattr(sys, "argv", base)
    make_records.main()
    assert seen["strip_say"] is True
    monkeypatch.setattr(sys, "argv", base + ["--keep-say"])
    make_records.main()
    assert seen["strip_say"] is False


def test_cleanup_anchors_keep_valid_scratchpad_updates_only():
    from poker.sft.make_records import cleanup_anchors
    def cl(i, name="update_scratchpad", contents="notes", cutoff=False):
        return {"span_id": f"c{i}", "source": "arena", "kind": "cleanup", "cutoff": cutoff, "deployment": "d",
                "messages": [{"role": "system", "content": "r"}, {"role": "user", "content": "Now call update_scratchpad"}],
                "reasoning": "keep reads", "content": "",
                "tool_calls": [{"name": name, "arguments": json.dumps({"contents": contents} if name == "update_scratchpad" else {"action": "fold"})}]}
    pool = [cl(i) for i in range(6)] + [cl(9, name="submit_action"), cl(10, cutoff=True), cl(11, contents="x" * 6000)]
    out = cleanup_anchors(pool, n=10, cycle=4, seed=0)
    assert {r["decision_id"] for r in out} == {f"c{i}" for i in range(6)}
    assert all(r["target"]["tool_call"] == {"name": "update_scratchpad", "arguments": {"contents": "notes"}} for r in out)
    assert all(r["source"] == "cleanup_anchor" for r in out)
    assert len(cleanup_anchors(pool, n=3, cycle=4, seed=0)) == 3
