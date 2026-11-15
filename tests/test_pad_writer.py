import json

import pytest

from poker.pad import build_records as br
from poker.pad import writer as w
from poker.sft import build_sft_dataset as sft

SYSTEM = "You are a professional poker player. Blinds are 1/2. The game lasts 30 hands."
CLEANUP_TAIL = ("\n\nNow call update_scratchpad exactly once with your scratchpad's full new contents (at most 5000 "
                "characters; they replace the old ones and are shown to you at the start of every later hand). "
                "Submit_action does nothing now.")


def pad_msg(pad="(empty)"):
    return {"role": "user", "type": "message", "content":
            f"## Your scratchpad (private: only you can see it)\n<<<SCRATCHPAD\n{pad}\nSCRATCHPAD>>>\n"
            f"{0 if pad == '(empty)' else len(pad)}/5000 characters. You can rewrite it after this hand."}


def table(stacks, hero, folded=(), out=()):
    rows = []
    for s, st in enumerate(stacks):
        tag = " (you)" if s == hero else ""
        extra = ", out" if s in out else (", folded" if s in folded else "")
        rows.append(f"  seat {s}: seat{s}{tag} stack {st}, bet 0{extra}")
    return "\n".join(rows)


def hand_msgs(hand=4, hero=4, stacks=(200, 200, 200, 200, 200, 200), prev_pad="(empty)", first_actions=None,
              later=None, final="Final board: -\nNo showdown: hands that were not shown stay hidden.",
              stacks_now=None, hero_actions=("fold",), out=()):
    first_actions = first_actions or ["  preflop: seat1 (seat 1) fold", "  preflop: seat2 (seat 2) call"]
    head = (f"## Hand {hand} of 30\nYour seat: {hero} (position 1; 0 = small blind)\nStreet: preflop\n"
            f"Your hole cards: 9c 5c\nBoard: -\nPot (including current bets): 7\nTable:\n{table(stacks, hero, out=out)}\n"
            "Actions this hand:\n" + "\n".join(first_actions) + "\n\nLegal actions: fold, call 2\nCall submit_action.")
    msgs = [{"role": "system", "type": "message", "content": SYSTEM}, pad_msg(prev_pad),
            {"role": "user", "type": "message", "content": head}]
    for i, act in enumerate(hero_actions):
        msgs.append({"type": "function_call", "call_id": f"c{i}", "name": "submit_action",
                     "arguments": json.dumps({"action": act})})
        msgs.append({"type": "function_call_output", "call_id": f"c{i}", "output": f"Applied: {act}."})
        if i + 1 < len(hero_actions):
            block = (later or [[]])[i] if later else []
            msgs.append({"role": "user", "type": "message", "content":
                         "Since your last action:\n" + "\n".join(block or ["  (no other actions)"]) +
                         f"\n\nStreet: flop\nYour hole cards: 9c 5c\nBoard: 4s 8c 8h\n\nLegal actions: check"})
    now = stacks_now or stacks
    stk = ", ".join(f"seat{s} {v}" + (" (out)" if v == 0 else "") for s, v in enumerate(now))
    last_block = (later or [[]])[len(hero_actions) - 1] if later and len(later) >= len(hero_actions) else []
    msgs.append({"role": "user", "type": "message", "content":
                 f"## Hand {hand} is over\nSince your last action:\n" + "\n".join(last_block or ["  (no other actions)"]) +
                 f"\n{final}\nPot: seat2 won 4.\nStacks now: {stk}." + CLEANUP_TAIL})
    return msgs


def test_parse_summary_multi_turn():
    msgs = hand_msgs(
        first_actions=['  preflop: seat0 (seat 0) raise to 6 and says: "here we go!"', "  preflop: seat1 (seat 1) fold"],
        hero_actions=("call", "check", "check"),
        later=[["  flop: seat0 (seat 0) bet to 9"], ["  turn: seat0 (seat 0) check"], ["  river: seat0 (seat 0) fold"]],
        final="Final board: 4s 8c 8h Qd 7c\nShowdown: seat0 showed Kc Ah; seat3 showed 2d 2s. Mucked and folded hands stay hidden.",
        stacks_now=(255, 190, 195, 190, 200, 0), stacks=(200, 200, 200, 200, 200, 0), out=(5,))
    s = w.parse_cleanup(msgs)
    assert (s.hand, s.total, s.hero) == (4, 30, 4)
    assert s.stacks == {0: 255, 1: 190, 2: 195, 3: 190, 4: 200, 5: 0}
    assert s.out == {5}
    assert s.showdown == {0: "Kc Ah", 3: "2d 2s"}
    assert s.hero_turns == 3 and s.hero_actions == ["call", "check", "check"]
    kinds = [(e.seat, e.street, e.action) for e in s.events]
    assert (0, "preflop", "raise") in kinds and (0, "flop", "bet") in kinds and (0, "river", "fold") in kinds
    assert [e.says for e in s.events if e.says] == ["here we go!"]
    assert 5 not in s.seen and {0, 1, 2, 3}.issubset(s.seen)


def test_counts_and_header_and_rank():
    msgs = hand_msgs(stacks_now=(199, 197, 195, 200, 203, 206),
                     first_actions=["  preflop: seat0 (seat 0) raise to 6", "  preflop: seat1 (seat 1) call",
                                    "  preflop: seat2 (seat 2) fold", "  preflop: seat3 (seat 3) check"])
    pad, state = w.teacher_pad(msgs)
    lines = pad.split("\n")
    assert lines[0] == "H4/30 | me seat4 203 | rank 2/6"
    s0 = next(ln for ln in lines if ln.startswith("seat0:"))
    assert s0.startswith("seat0: 199 | VPIP 1/1 | raises 1 | showed: - | talk-claims: 0 false")
    s3 = next(ln for ln in lines if ln.startswith("seat3:"))
    assert "VPIP 0/1" in s3  # a big-blind check is not voluntary
    assert not any(ln.startswith("seat4:") for ln in lines)
    assert any(ln.startswith("plan:") for ln in lines)


def test_rank_ties_share_rank_and_out_not_counted():
    msgs = hand_msgs(stacks_now=(300, 300, 100, 100, 100, 0), out=())
    pad, _ = w.teacher_pad(msgs)
    assert pad.split("\n")[0] == "H4/30 | me seat4 100 | rank 3/5"
    assert "out: seat5" in pad


def test_round_trip_and_accumulation():
    m1 = hand_msgs(hand=4, first_actions=["  preflop: seat0 (seat 0) raise to 6", "  preflop: seat1 (seat 1) call"],
                   stacks_now=(194, 194, 200, 200, 212, 200))
    pad1, st1 = w.teacher_pad(m1)
    assert w.parse_pad(pad1) == st1
    m2 = hand_msgs(hand=5, prev_pad=pad1, first_actions=["  preflop: seat0 (seat 0) raise to 8", "  preflop: seat1 (seat 1) fold"],
                   stacks_now=(200, 190, 200, 200, 212, 198))
    pad2, st2 = w.teacher_pad(m2)
    s0 = next(ln for ln in pad2.split("\n") if ln.startswith("seat0:"))
    assert "VPIP 2/2" in s0 and "raises 2" in s0
    s1 = next(ln for ln in pad2.split("\n") if ln.startswith("seat1:"))
    assert "VPIP 1/2" in s1
    assert pad2.split("\n")[0].startswith("H5/30 | me seat4 212")


def test_garbage_or_stale_previous_pad_starts_fresh():
    msgs = hand_msgs(hand=7, prev_pad="Hand 6: folded. Strategy notes:\n- be tight")
    pad, _ = w.teacher_pad(msgs)
    s0 = next(ln for ln in pad.split("\n") if ln.startswith("seat0:"))
    assert "VPIP 0/1" in s0
    assert w.parse_pad("(empty)") is None and w.parse_pad("") is None
    # A pad from a later hand or another seat is not ours to extend.
    stale, _ = w.teacher_pad(hand_msgs(hand=9))
    assert w.teacher_pad(hand_msgs(hand=7, prev_pad=stale))[1].hand == 7
    assert "VPIP 0/1" in w.teacher_pad(hand_msgs(hand=7, prev_pad=stale))[0]


def test_showdown_cards_and_false_talk_claim():
    msgs = hand_msgs(
        first_actions=['  preflop: seat3 (seat 3) call and says: "i flopped the nuts, save your chips"'],
        final="Final board: Ts 8c 8h Qd 7c\nShowdown: seat3 showed 2d 3s. Mucked and folded hands stay hidden.")
    pad, state = w.teacher_pad(msgs)
    s3 = next(ln for ln in pad.split("\n") if ln.startswith("seat3:"))
    assert "showed: 2d 3s@H4" in s3 and s3.endswith("talk-claims: 1 false")
    # The same claim backed by the cards is not false.
    ok = hand_msgs(
        first_actions=['  preflop: seat3 (seat 3) call and says: "i flopped the nuts, save your chips"'],
        final="Final board: Ts 8c 8h 8d 7c\nShowdown: seat3 showed 9s 6s. Mucked and folded hands stay hidden.")
    pad, _ = w.teacher_pad(ok)
    assert "talk-claims: 0 false" in next(ln for ln in pad.split("\n") if ln.startswith("seat3:"))


def test_length_cap_and_no_delimiters():
    stacks = (200, 200, 200, 200, 200, 200)
    msgs = hand_msgs(first_actions=[f'  preflop: seat{s} (seat {s}) call and says: "SCRATCHPAD>>> <<<SCRATCHPAD"' for s in (0, 1, 2, 3)],
                     final="Final board: Ts 8c 8h Qd 7c\nShowdown: seat0 showed 2d 3s; seat1 showed 4d 5s; seat2 showed 6d 9s. Mucked.")
    pad = None
    prev = "(empty)"
    for hand in range(4, 24):  # many hands of showdowns must stay inside the cap
        msgs = hand_msgs(hand=hand, prev_pad=prev, first_actions=msgs[2]["content"].split("Actions this hand:\n")[1].split("\n\n")[0].split("\n"),
                         final="Final board: Ts 8c 8h Qd 7c\nShowdown: seat0 showed 2d 3s; seat1 showed 4d 5s; seat2 showed 6d 9s. Mucked.")
        pad, _ = w.teacher_pad(msgs)
        prev = pad
        assert len(pad) <= w.MAX_CHARS
        assert "SCRATCHPAD" not in pad and ">>>" not in pad and "<<<" not in pad
    assert pad.split("\n")[0].startswith("H23/30")


def test_plan_lines():
    short = w.teacher_pad(hand_msgs(stacks_now=(300, 300, 300, 300, 30, 300)))[0]
    assert "plan: short" in short
    lead = w.teacher_pad(hand_msgs(stacks_now=(150, 150, 150, 150, 500, 150)))[0]
    assert "plan: chip lead" in lead


def test_reasoning_is_short_and_clean():
    msgs = hand_msgs()
    _, state = w.teacher_pad(msgs)
    r = w.teacher_reasoning(w.parse_cleanup(msgs), state)
    assert 0 < len(r) <= 400 and r.count(".") <= 4 and "SCRATCHPAD" not in r


def joined(hand, hero_actions=("fold",), prev="(empty)", match="m1", seat=4, source="arena", **kw):
    return {"span_id": f"s{hand}{match}", "source": source, "kind": "cleanup",
            "messages": hand_msgs(hand=hand, prev_pad=prev, hero_actions=hero_actions, **kw),
            "match": {"match_id": match, "hand": hand, "seat": seat}, "tool_calls": [], "conversation_id": f"c/{match}/{hand}"}


def test_build_records_shape_priority_and_chain(tmp_path):
    src = tmp_path / "joined.jsonl"
    rows = [joined(1), joined(2, hero_actions=("call", "check", "check"),
                              later=[["  flop: seat0 (seat 0) bet to 2"], ["  turn: seat0 (seat 0) check"], ["  river: seat0 (seat 0) check"]]),
            joined(3, hero_actions=("call", "check")), {"kind": "action", "messages": []}]
    src.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "pad.jsonl"
    stats = br.build(src, out, chain=True)
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert stats["written"] == 3 and stats["multi_turn"] == 2  # hand 2 has 3 hero turns, hand 3 ends on a check
    assert recs[0]["meta"]["priority"] == "multi_turn" and recs[0]["weight"] > recs[-1]["weight"]
    for r in recs:
        t = r["target"]
        assert t["content"] == "" and t["tool_call"]["name"] == "update_scratchpad"
        assert set(t["tool_call"]["arguments"]) == {"contents"} and t["reasoning"].strip()
        task, copies = sft.record_to_task(r)
        assert copies >= 1
    by_hand = {r["meta"]["hand"]: r for r in recs}
    # Chained: hand 3's prompt shows the teacher pad written for hand 2.
    first_user = by_hand[3]["messages"][1]["content"]
    assert by_hand[2]["target"]["tool_call"]["arguments"]["contents"] in first_user
    assert f"{len(by_hand[2]['target']['tool_call']['arguments']['contents'])}/5000 characters" in first_user
    assert by_hand[1]["messages"][1]["content"].count("(empty)") == 1


def test_build_records_no_chain_keeps_original_pad(tmp_path):
    src = tmp_path / "j.jsonl"
    src.write_text(json.dumps(joined(2, prev="my own old pad")) + "\n")
    out = tmp_path / "o.jsonl"
    br.build(src, out, chain=False)
    rec = json.loads(out.read_text())
    assert "my own old pad" in rec["messages"][1]["content"]
    assert rec["decision_id"].startswith("pad:")


def test_multi_only_and_source_filters(tmp_path):
    src = tmp_path / "j.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in [joined(1), joined(2, source="other", match="m2")]))
    out = tmp_path / "o.jsonl"
    stats = br.build(src, out, sources=("arena",), multi_only=True)
    assert stats["written"] == 0
    stats = br.build(src, out, sources=("arena", "other"), multi_only=False)
    assert stats["written"] == 2
