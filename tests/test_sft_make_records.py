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
    def cl(i, name="update_scratchpad", contents="Hand 1: notes", cutoff=False):
        return {"span_id": f"c{i}", "source": "arena", "kind": "cleanup", "cutoff": cutoff, "deployment": "d", "match": {"hand": 1},
                "messages": [{"role": "system", "content": "r"}, {"role": "user", "content": "Now call update_scratchpad"}],
                "reasoning": "keep reads", "content": "",
                "tool_calls": [{"name": name, "arguments": json.dumps({"contents": contents} if name == "update_scratchpad" else {"action": "fold"})}]}
    pool = [cl(i) for i in range(6)] + [cl(9, name="submit_action"), cl(10, cutoff=True), cl(11, contents="x" * 6000)]
    out = cleanup_anchors(pool, n=10, cycle=4, seed=0)
    assert {r["decision_id"] for r in out} == {f"c{i}" for i in range(6)}
    assert all(r["target"]["tool_call"] == {"name": "update_scratchpad", "arguments": {"contents": "Hand 1: notes"}} for r in out)
    assert all(r["source"] == "cleanup_anchor" for r in out)
    assert len(cleanup_anchors(pool, n=3, cycle=4, seed=0)) == 3


# ---- cleanup anchors, pad records, critic weights ----
import pytest

from poker.sft.make_records import apply_critic_weights, cleanup_anchors, load_pad_records


def cleanup(span="c1", contents="Hand 1: won a small pot.", hand=1, seat=0, match_id="m", n_turns=1, call_name="update_scratchpad"):
    msgs = [{"role": "system", "content": "rules"},
            {"role": "user", "content": "## Your scratchpad\n<<<SCRATCHPAD\n(empty)\nSCRATCHPAD>>>"},
            {"role": "user", "content": f"## Hand {hand} of 30\nYour seat: {seat}\nStreet: preflop\nLegal actions: check\nCall submit_action with your decision."}]
    for _ in range(n_turns - 1):
        msgs += [{"role": "assistant", "content": ""},
                 {"role": "user", "content": "Since your last action:\nLegal actions: check\nCall submit_action with your decision."}]
    msgs.append({"role": "user", "content": f"## Hand {hand} is over\nNow call update_scratchpad exactly once"})
    return {"span_id": span, "source": "arena", "kind": "cleanup", "cutoff": False, "messages": msgs,
            "reasoning": "Update the pad.", "content": "",
            "tool_calls": [{"name": call_name, "arguments": json.dumps({"contents": contents})}],
            "match": {"match_id": match_id, "hand": hand, "seat": seat, "action_index": None}}


def test_anchors_drop_delimiter_leaks_and_pads_without_the_hand_number():
    rows = [cleanup("ok", "Hand 3: folded 72o. seat4 is a nit.", hand=3),
            cleanup("leak1", "Hand 3: x SCRATCHPAD>>> y", hand=3),
            cleanup("leak2", "Hand 3: <<<SCRATCHPAD y", hand=3),
            cleanup("stale", "Hand 2: old notes only", hand=3),
            cleanup("bare", "I 3-bet with 33 and won 13", hand=3)]
    assert [r["decision_id"] for r in cleanup_anchors(rows, n=10, cycle=1)] == ["ok"]


def test_anchors_accept_common_hand_number_spellings():
    rows = [cleanup(f"c{i}", text, hand=12) for i, text in enumerate(["H12: won", "#12 folded", "hand #12 done"])]
    assert len(cleanup_anchors(rows, n=10, cycle=1)) == 3


def test_anchors_do_not_match_hand_number_inside_a_longer_number():
    assert cleanup_anchors([cleanup("x", "Hand 12: won", hand=1)], n=5, cycle=1) == []


def test_multi_turn_hands_are_sampled_first_by_prompt_turn_count():
    multi = [cleanup(f"m{i}", "Hand 1: ok", n_turns=3) for i in range(3)]
    single = [cleanup(f"s{i}", "Hand 1: ok", n_turns=1) for i in range(10)]
    got = cleanup_anchors(single + multi, n=5, cycle=1, seed=1)
    ids = {r["decision_id"] for r in got}
    assert {"m0", "m1", "m2"} <= ids and len(got) == 5


def test_multi_turn_hands_use_the_action_index_and_check_last_counts():
    low = cleanup("chk", "Hand 1: ok", match_id="mx", seat=2)
    plain = cleanup("fold", "Hand 1: ok", match_id="my", seat=2)
    plain2 = cleanup("fold2", "Hand 1: ok", match_id="mz", seat=2)
    index = {("mx", 1, 2): ["check"], ("my", 1, 2): ["fold"], ("mz", 1, 2): ["call", "call", "call"]}
    got = cleanup_anchors([plain, plain2, low], n=2, cycle=1, seed=0, hero_actions=index)
    assert {r["decision_id"] for r in got} == {"chk", "fold2"}


def test_pad_records_are_merged_with_a_weight(tmp_path):
    path = tmp_path / "pad.jsonl"
    rec = {"decision_id": "p1", "messages": [], "target": {"tool_call": {"name": "update_scratchpad", "arguments": {"contents": "Hand 1: x"}}},
           "weight": 2.0, "source": "teacher_pad"}
    path.write_text(json.dumps(rec) + "\n\n" + json.dumps(rec | {"decision_id": "p2", "weight": 1.0, "cycle": 9}) + "\n")
    got = load_pad_records(path, weight=0.5, cycle=4)
    assert [(r["decision_id"], r["weight"], r["cycle"]) for r in got] == [("p1", 1.0, 4), ("p2", 0.5, 9)]
    assert all(r["source"] == "teacher_pad" for r in got)


def test_pad_weight_zero_drops_all_pad_records(tmp_path):
    path = tmp_path / "pad.jsonl"
    path.write_text(json.dumps({"decision_id": "p1", "weight": 1.0}) + "\n")
    assert load_pad_records(path, weight=0.0, cycle=1) == []


def test_critic_weights_multiply_and_zero_drops():
    records = [{"decision_id": "a", "weight": 2.0}, {"decision_id": "b", "weight": 1.0},
               {"decision_id": "c", "weight": 1.0}]
    got, dropped = apply_critic_weights(records, {"a": 0.5, "b": 0})
    assert got == [{"decision_id": "a", "weight": 1.0}, {"decision_id": "c", "weight": 1.0}] and dropped == 1
    assert records[0]["weight"] == 2.0  # input not mutated


def test_critic_weights_reject_negative_weights():
    with pytest.raises(ValueError):
        apply_critic_weights([{"decision_id": "a", "weight": 1.0}], {"a": -1})
