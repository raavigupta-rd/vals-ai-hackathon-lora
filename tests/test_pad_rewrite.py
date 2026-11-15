import json

from poker.pad.rewrite import chain_index, rewrite_messages, rewrite_records, teacher_pad_at, truncated_chains

CLEANUP = ("## Hand {h} is over\nStacks now: seat0 {s0}, seat1 200, seat2 200, seat3 200, seat4 200, seat5 200.\n"
           "Now call update_scratchpad exactly once")


def cleanup(match_id="m", hand=1, seat=0, s0=190, source="arena", pad="(empty)"):
    msgs = [{"role": "system", "content": "Blinds are 1/2."},
            {"role": "user", "content": f"## Your scratchpad (private: only you can see it)\n<<<SCRATCHPAD\n{pad}\nSCRATCHPAD>>>\n0/5000 characters. You can rewrite it after this hand."},
            {"role": "user", "content": f"## Hand {hand} of 30\nYour seat: {seat} (position 0)\nStreet: preflop\nTable:\n"
             + "".join(f"  seat {i}: seat{i} stack 200, bet 0\n" for i in range(6))},
            {"role": "user", "content": CLEANUP.format(h=hand, s0=s0)}]
    return {"kind": "cleanup", "source": source, "span_id": f"c{match_id}{hand}{seat}", "messages": msgs,
            "match": {"match_id": match_id, "hand": hand, "seat": seat}}


def action_record(hand=2, seat=0, match_id="m", pad="my own notes"):
    block = f"## Your scratchpad (private: only you can see it)\n<<<SCRATCHPAD\n{pad}\nSCRATCHPAD>>>\n{len(pad)}/5000 characters. You can rewrite it after this hand."
    return {"decision_id": "d", "messages": [{"role": "system", "content": "rules"}, {"role": "user", "content": block},
                                             {"role": "user", "content": f"## Hand {hand} of 30\n..."}],
            "target": {"tool_call": {"name": "submit_action", "arguments": {"action": "fold"}}}}


def test_hand_one_is_the_empty_pad():
    index = chain_index([cleanup(hand=1)])
    assert teacher_pad_at(index, "m", 0, 1) == ""


def test_pad_at_hand_two_is_the_writer_output_for_hand_one():
    index = chain_index([cleanup(hand=1)])
    pad = teacher_pad_at(index, "m", 0, 2)
    assert pad.startswith("H1/30 | me seat0 190 | rank 6/6") and "seat1:" in pad


def test_chain_accumulates_counts_across_hands():
    index = chain_index([cleanup(hand=1), cleanup(hand=2)])
    assert teacher_pad_at(index, "m", 0, 3).startswith("H2/30")
    assert "VPIP 0/2" in teacher_pad_at(index, "m", 0, 3)


def test_gap_in_the_chain_returns_none():
    index = chain_index([cleanup(hand=1), cleanup(hand=3)])
    assert teacher_pad_at(index, "m", 0, 4) is None
    assert teacher_pad_at(index, "m", 0, 2) is not None


def test_other_seats_and_matches_do_not_mix():
    index = chain_index([cleanup(hand=1, seat=1), cleanup(hand=1, match_id="x")])
    assert teacher_pad_at(index, "m", 0, 2) is None


def test_non_arena_and_unparseable_rows_are_ignored():
    bad = cleanup(hand=1)
    bad["messages"] = bad["messages"][:2]
    assert teacher_pad_at(chain_index([cleanup(hand=1, source="other"), bad]), "m", 0, 2) is None


def test_rewrite_messages_uses_the_harness_wrapper():
    from poker_harness.prompts import scratchpad_block
    msgs = action_record()["messages"]
    out = rewrite_messages(msgs, "H1/30 | x")
    assert out[1]["content"] == scratchpad_block("H1/30 | x", 5000)
    assert msgs[1]["content"].count("my own notes") == 1  # input is not mutated
    assert out[0] == msgs[0] and out[2] == msgs[2]
    assert rewrite_messages(msgs, "")[1]["content"] == scratchpad_block("", 5000)


def test_rewrite_records_counts_and_keeps_unchainable_ones():
    index = chain_index([cleanup(hand=1)])
    ok, no_chain, cleanup_rec = action_record(hand=2), action_record(hand=9), action_record(hand=2) | {"decision_id": "c"}
    cleanup_rec["target"] = {"tool_call": {"name": "update_scratchpad", "arguments": {"contents": "x"}}}
    matches = {"d": ("m", 0, 2), "n": ("m", 0, 9), "c": ("m", 0, 2)}
    no_chain["decision_id"] = "n"
    out, stats = rewrite_records([ok, no_chain, cleanup_rec], matches, index)
    assert out[0]["messages"][1]["content"].count("H1/30") == 1 and "my own notes" not in out[0]["messages"][1]["content"]
    assert out[1] == no_chain and out[2] == cleanup_rec
    assert stats["action_records"] == 2 and stats["rewritten"] == 1 and stats["kept_no_chain"] == 1
    assert stats["skipped_non_action"] == 1


def test_strict_drops_unrewritten_pad_dependent_records_only():
    index = chain_index([cleanup(hand=1)])
    ok, no_chain, other = action_record(hand=2), action_record(hand=9), action_record(hand=9)
    no_chain["decision_id"], other["decision_id"] = "n", "o"
    matches = {"d": ("m", 0, 2), "n": ("m", 0, 9), "o": ("m", 0, 9)}
    out, stats = rewrite_records([ok, no_chain, other], matches, index, strict_ids={"d", "n"})
    assert [r["decision_id"] for r in out] == ["d", "o"]  # n: pad-dependent and unrewritable -> dropped
    assert stats["rewritten"] == 1 and stats["kept_no_chain"] == 1 and stats["dropped_no_chain"] == 1
    out, stats = rewrite_records([ok, no_chain, other], matches, index)  # strict off: nothing is dropped
    assert len(out) == 3 and stats["dropped_no_chain"] == 0


def test_incomplete_match_keys_are_skipped_and_counted():
    index = chain_index([cleanup(hand=1)])
    a, b = action_record(hand=2), action_record(hand=2)
    b["decision_id"] = "b"
    matches = {"d": ("m", 0, 2), "b": ("m", None, 2)}
    out, stats = rewrite_records([a, b, action_record() | {"decision_id": "zz"}], matches, index)
    assert len(out) == 3 and stats["rewritten"] == 1 and stats["kept_no_chain"] == 2
    assert stats["incomplete_match"] == 1  # "b" has a key but a missing part; "zz" has no key at all


def test_chain_index_skips_rows_with_incomplete_match_metadata():
    row = cleanup(hand=1)
    del row["match"]["match_id"]
    index = chain_index([row, cleanup(hand=1) | {"match": None}])
    assert teacher_pad_at(index, "m", 0, 2) is None
    assert index["_meta"]["incomplete"] == 2


def test_truncated_chains_are_reported_per_match():
    bad = cleanup(hand=3)
    bad["messages"] = bad["messages"][:2]  # unparseable cleanup in the middle
    index = chain_index([cleanup(hand=1), cleanup(hand=2), bad, cleanup(hand=4), cleanup(match_id="ok", hand=1)])
    assert truncated_chains(index) == {("m", 0): {"chain_end": 3, "last_hand": 4}}
    out, stats = rewrite_records([action_record(hand=5)], {"d": ("m", 0, 5)}, index)
    assert stats["truncated_chains"] == 1 and stats["unparseable_cleanups"] == 1
    assert stats["truncated_matches"] == ["m"]


def test_a_chain_with_a_gap_is_truncated_too():
    index = chain_index([cleanup(hand=1), cleanup(hand=3)])
    assert truncated_chains(index) == {("m", 0): {"chain_end": 2, "last_hand": 3}}


def test_rendered_pads_over_the_limit_are_refused():
    import pytest
    with pytest.raises(AssertionError):
        rewrite_messages(action_record()["messages"], "x" * 5001)
    rewrite_messages(action_record()["messages"], "x" * 5000)


def test_rewrite_v2_pads_use_the_v2_renderer():
    index = chain_index([cleanup(hand=1), cleanup(hand=2)])
    pad = teacher_pad_at(index, "m", 0, 3, version=2)
    assert " BB) | rank " in pad and "open " in pad
    assert teacher_pad_at(index, "m", 0, 3).startswith("H2/30 | me seat0 190 | rank")  # default stays v1
    assert " BB)" not in teacher_pad_at(index, "m", 0, 3)
    out, stats = rewrite_records([action_record(hand=3)], {"d": ("m", 0, 3)}, index, version=2)
    assert stats["rewritten"] == 1 and " BB) | rank " in out[0]["messages"][1]["content"]


def test_rewrite_v1_and_v2_chains_do_not_share_a_cache():
    index = chain_index([cleanup(hand=1)])
    a, b = teacher_pad_at(index, "m", 0, 2), teacher_pad_at(index, "m", 0, 2, version=2)
    assert a != b and teacher_pad_at(index, "m", 0, 2) == a
