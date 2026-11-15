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


def table(stacks, hero, folded=(), out=(), button=None):
    rows = []
    for s, st in enumerate(stacks):
        tag = (" (you)" if s == hero else "") + (" [button]" if s == button else "")
        extra = ", out" if s in out else (", folded" if s in folded else "")
        rows.append(f"  seat {s}: seat{s}{tag} stack {st}, bet 0{extra}")
    return "\n".join(rows)


def hand_msgs(hand=4, hero=4, stacks=(200, 200, 200, 200, 200, 200), prev_pad="(empty)", first_actions=None,
              later=None, final="Final board: -\nNo showdown: hands that were not shown stay hidden.",
              stacks_now=None, hero_actions=("fold",), out=(), button=None):
    first_actions = first_actions or ["  preflop: seat1 (seat 1) fold", "  preflop: seat2 (seat 2) call"]
    head = (f"## Hand {hand} of 30\nYour seat: {hero} (position 1; 0 = small blind)\nStreet: preflop\n"
            f"Your hole cards: 9c 5c\nBoard: -\nPot (including current bets): 7\nTable:\n{table(stacks, hero, out=out, button=button)}\n"
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


# ---- pad v2 -------------------------------------------------------------------------------------------------------

def v2(msgs, prev=None):
    return w.teacher_pad(msgs, prev, version=2)


def v2_line(pad, prefix):
    return next(ln for ln in pad.split("\n") if ln.startswith(prefix))


def test_v2_header_line():
    pad, _ = v2(hand_msgs(stacks_now=(199, 197, 195, 200, 203, 206)))
    assert pad.split("\n")[0] == "H4/30 | me seat4 203 (101 BB) | rank 2/6 | gap up +3, down -3 | 26 left"


def test_v2_header_rank_one_and_out():
    top = v2(hand_msgs(stacks_now=(100, 100, 100, 100, 300, 100)))[0].split("\n")[0]
    assert "rank 1/6 | gap up +0, down -200" in top
    gone = v2(hand_msgs(stacks_now=(200, 200, 200, 200, 0, 200), hero_actions=("fold",)))[0].split("\n")[0]
    assert "rank out/5" in gone


def test_v2_opponent_fields_from_one_hand():
    msgs = hand_msgs(stacks_now=(199, 197, 195, 200, 203, 206),
                     first_actions=["  preflop: seat0 (seat 0) raise to 6", "  preflop: seat1 (seat 1) call",
                                    "  preflop: seat2 (seat 2) fold", "  preflop: seat3 (seat 3) check"])
    pad, _ = v2(msgs)
    assert v2_line(pad, "seat0:") == ("seat0: 199 | VPIP 1/1 | open 1/1 | 3bet 0/0 | jams 0 | folds-to-raise 0/0 | shown: -")
    assert "VPIP 1/1 | open 0/0 | 3bet 0/1 | jams 0 | folds-to-raise 0/1" in v2_line(pad, "seat1:")
    assert "VPIP 0/1 | open 0/0 | 3bet 0/1 | jams 0 | folds-to-raise 1/1" in v2_line(pad, "seat2:")
    assert not any(ln.startswith("seat4:") for ln in pad.split("\n"))


def test_v2_three_bet_and_jam_counts():
    msgs = hand_msgs(stacks_now=(50, 300, 200, 200, 200, 200),
                     first_actions=["  preflop: seat1 (seat 1) raise to 6", "  preflop: seat2 (seat 2) raise to 20",
                                    "  preflop: seat0 (seat 0) raise to 200", "  preflop: seat3 (seat 3) fold"])
    pad, _ = v2(msgs)
    assert "open 1/1 | 3bet 0/0" in v2_line(pad, "seat1:")
    assert "open 0/0 | 3bet 1/1 | jams 0" in v2_line(pad, "seat2:")
    assert "open 0/0 | 3bet 0/0 | jams 1" in v2_line(pad, "seat0:")  # a jam over a 3-bet is neither an open nor a 3-bet


def test_v2_hero_raise_counts_as_a_raise_faced():
    msgs = hand_msgs(hero_actions=("raise",),
                     first_actions=["  preflop: seat0 (seat 0) fold"],
                     later=[["  preflop: seat1 (seat 1) fold", "  preflop: seat2 (seat 2) call"]])
    msgs[3]["arguments"] = json.dumps({"action": "raise", "amount": 6})
    pad, _ = v2(msgs)
    assert "folds-to-raise 1/1" in v2_line(pad, "seat1:") and "folds-to-raise 0/1" in v2_line(pad, "seat2:")
    assert "folds-to-raise 0/0" in v2_line(pad, "seat0:")  # folded before the raise


def test_v2_blinds_fold_to_steals_line_and_state():
    # button seat5: SB seat0, BB seat1, CO seat4 opens, both blinds fold
    msgs = hand_msgs(hero=2, button=5, first_actions=["  preflop: seat3 (seat 3) fold", "  preflop: seat4 (seat 4) raise to 5",
                                                      "  preflop: seat5 (seat 5) fold", "  preflop: seat0 (seat 0) fold",
                                                      "  preflop: seat1 (seat 1) call"])
    pad, state = v2(msgs)
    assert "blinds: fold-to-steal 1/2" in pad
    assert w.parse_pad(pad).fts_f == 1 and w.parse_pad(pad).fts_g == 2


def test_v2_no_positions_predictions_or_generic_policy():
    pad, _ = v2(hand_msgs(stacks_now=(300, 300, 100, 100, 100, 100)))
    low = pad.lower()
    for bad in ("play tight", "marginal", "next hand", "next button", "plan:", "avoid"):
        assert bad not in low


def opp_state(**kw):
    return w.Opp(stack=kw.pop("stack", 200), va=kw.pop("va", 2), vb=kw.pop("vb", 10), **kw)


def mk_state(opps, stack=200, hand=20, version=2):
    return w.PadState(hand, 30, 4, stack, opps, {}, **({"version": version}))


def test_v2_exploit_lines_need_samples():
    opps = {3: opp_state(vb=9, va=4, jams=4), 5: opp_state(vb=3, va=3, jams=3), 1: opp_state(vb=12, va=4, jams=3)}
    pad = w.render(mk_state(opps), 2, version=2)
    assert "seat1 jams" not in pad  # 3 jams is under the threshold of 4
    assert "> seat3 jams wide (4/9): call jams with 77+ ATs+ AJo+ KQs" in pad
    assert "seat5 jams" not in pad  # 3 hands is not enough


def test_v2_steal_line_needs_samples():
    few = w.render(mk_state({1: opp_state()}) , 2, version=2)
    assert "blinds fold to steals" not in few
    st = mk_state({1: opp_state()})
    st.fts_f, st.fts_g = 5, 6
    assert "> blinds fold to steals (5/6): open BTN/CO wide" in w.render(st, 2, version=2)
    st.fts_f, st.fts_g = 2, 3
    assert "blinds fold to steals" not in w.render(st, 2, version=2)
    st.fts_f, st.fts_g = 1, 8  # they defend: no exploit
    assert "blinds fold to steals" not in w.render(st, 2, version=2)


def test_v2_calling_station_and_nit_lines():
    opps = {5: opp_state(vb=10, va=7, opens=0, open_opp=8, three_bet=0, jams=0),
            2: opp_state(vb=10, va=1, opens=1, open_opp=8)}
    pad = w.render(mk_state(opps), 2, version=2)
    assert "> seat5 calls a lot: value bet, do not bluff" in pad
    assert "> seat2 is tight (VPIP 1/10): respect its raises" in pad
    thin = w.render(mk_state({5: opp_state(vb=4, va=4, opens=0, open_opp=3)}), 2, version=2)
    assert "calls a lot" not in thin
    # VPIP carried from a v1 pad has few open chances: do not call it a station
    carried = w.render(mk_state({5: opp_state(vb=10, va=7, opens=0, open_opp=3)}), 2, version=2)
    assert "calls a lot" not in carried


def test_v2_fold_to_raise_line():
    pad = w.render(mk_state({2: opp_state(vb=10, va=3, ftr_f=6, ftr_g=7)}), 2, version=2)
    assert "> seat2 folds to raises (6/7): raise first-in and 3-bet light" in pad
    assert "folds to raises" not in w.render(mk_state({2: opp_state(vb=10, ftr_f=2, ftr_g=3)}), 2, version=2)


def test_v2_survival_line_only_when_short_or_leading_late():
    plain = w.render(mk_state({1: opp_state()}, stack=200, hand=10), 2, version=2)
    assert "survive" not in plain
    short = w.render(mk_state({1: opp_state()}, stack=20, hand=10), 2, version=2)
    assert "> survive: 10 BB or less, shove or fold" in short
    assert "survive" not in w.render(mk_state({1: opp_state()}, stack=24, hand=10), 2, version=2)  # 12 BB
    late = mk_state({1: opp_state()}, stack=300, hand=25)
    assert "> survive: leading late" in w.render(late, 2, version=2)
    early = mk_state({1: opp_state()}, stack=300, hand=10)
    assert "survive" not in w.render(early, 2, version=2)
    assert "survive" not in w.render(mk_state({1: opp_state(stack=400)}, stack=300, hand=25), 2, version=2)
    assert "survive" not in w.render(mk_state({1: opp_state(stack=280)}, stack=300, hand=25), 2, version=2)  # lead of 20 < 40
    assert "survive" not in w.render(mk_state({1: opp_state()}, stack=300, hand=20), 2, version=2)  # 10 hands left


def test_v2_survival_line_is_first_and_survives_trimming():
    opps = {i: opp_state(vb=10, va=8, opens=0, open_opp=9, jams=5, ftr_f=8, ftr_g=9) for i in range(5)}
    st = mk_state(opps, stack=18)
    assert w.exploit_lines(st, 2)[0].startswith("> survive")
    assert w.render_v2(st, 2, max_exploits=1).split("\n")[-1].startswith("> survive")
    assert "> survive" in w.fit_v2(st, 2)


def test_v2_afk_not_rendered_but_old_pads_parse():
    pad, _ = v2(hand_msgs())
    assert "afk" not in pad
    old = ("H4/30 | me seat4 203 (101 BB) | rank 2/6 | gap up +3, down -3 | 26 left\n"
           "seat0: 199 | VPIP 7/9 | open 1/5 | 3bet 0/3 | jams 4 | folds-to-raise 2/6 | afk 0 | shown: Kc Ah@H4")
    st = w.parse_pad(old)
    assert st.opps[0].jams == 4 and st.opps[0].ftr_g == 6 and st.opps[0].showed == [("Kc Ah", 4)]
    assert w.parse_pad(pad) is not None


def jam_hand(events, stacks=(200,) * 6, **kw):
    return v2(hand_msgs(stacks=stacks, first_actions=events, **kw))[0]


def test_v2_jam_counts_chips_across_streets():
    pad = jam_hand(["  preflop: seat0 (seat 0) raise to 50", "  preflop: seat1 (seat 1) call",
                    "  flop: seat0 (seat 0) bet to 150", "  flop: seat1 (seat 1) fold"])
    assert "jams 1" in v2_line(pad, "seat0:")  # 50 + 150 = the whole 200
    pad = jam_hand(["  preflop: seat0 (seat 0) raise to 6", "  preflop: seat1 (seat 1) call",
                    "  flop: seat0 (seat 0) bet to 20", "  flop: seat1 (seat 1) call",
                    "  turn: seat0 (seat 0) bet to 174"])
    assert "jams 1" in v2_line(pad, "seat0:")
    pad = jam_hand(["  preflop: seat0 (seat 0) raise to 6", "  preflop: seat1 (seat 1) call",
                    "  flop: seat0 (seat 0) bet to 100"])
    assert "jams 0" in v2_line(pad, "seat0:")


def test_v2_no_jam_for_forced_short_stack():
    pad = jam_hand(["  preflop: seat0 (seat 0) raise to 25"], stacks=(25, 200, 200, 200, 200, 200))
    assert "jams 0" in v2_line(pad, "seat0:")
    pad = jam_hand(["  preflop: seat0 (seat 0) all-in"], stacks=(25, 200, 200, 200, 200, 200))
    assert "jams 0" in v2_line(pad, "seat0:")


def test_v2_all_in_is_a_raise_in_sequencing():
    pad = jam_hand(["  preflop: seat0 (seat 0) all-in", "  preflop: seat1 (seat 1) call", "  preflop: seat2 (seat 2) fold"])
    assert "open 1/1 | 3bet 0/0 | jams 1" in v2_line(pad, "seat0:")
    assert "open 0/0 | 3bet 0/1 | jams 0 | folds-to-raise 0/1" in v2_line(pad, "seat1:")
    assert "folds-to-raise 1/1" in v2_line(pad, "seat2:")
    pad = jam_hand(["  preflop: seat0 (seat 0) raise to 6", "  preflop: seat1 (seat 1) allin"])
    assert "open 0/0 | 3bet 1/1 | jams 1" in v2_line(pad, "seat1:")


def test_v2_folds_to_raise_and_3bet_opps_count_once_per_hand():
    pad = jam_hand(["  preflop: seat0 (seat 0) raise to 6", "  preflop: seat1 (seat 1) raise to 20",
                    "  preflop: seat0 (seat 0) raise to 60", "  preflop: seat1 (seat 1) fold"])
    s1 = v2_line(pad, "seat1:")
    assert "3bet 1/1" in s1 and "folds-to-raise 0/1" in s1  # the 4-bet fold is its second decision facing a raise
    s0 = v2_line(pad, "seat0:")
    assert "folds-to-raise 0/1" in s0


def test_v2_isolation_raise_is_not_an_open():
    pad = jam_hand(["  preflop: seat3 (seat 3) call", "  preflop: seat4 (seat 4) raise to 8", "  preflop: seat5 (seat 5) fold"])
    assert "open 0/1" in v2_line(pad, "seat3:")
    assert "seat4:" not in pad  # hero is seat4: no line for us
    pad = jam_hand(["  preflop: seat3 (seat 3) call", "  preflop: seat5 (seat 5) raise to 8"])
    assert "open 0/0" in v2_line(pad, "seat5:")


def test_v2_three_handed_big_blind_is_not_a_stealer():
    # button seat0: SB seat1, BB seat2. The BB raising is not a steal, so the SB's fold is not a steal reply.
    pad = jam_hand(["  preflop: seat2 (seat 2) raise to 5", "  preflop: seat1 (seat 1) fold", "  preflop: seat0 (seat 0) fold"],
                   stacks=(200, 200, 200), hero=2, button=0)
    assert "blinds:" not in pad
    pad = jam_hand(["  preflop: seat0 (seat 0) raise to 5", "  preflop: seat1 (seat 1) fold"],
                   stacks=(200, 200, 200), hero=2, button=0)
    assert "blinds: fold-to-steal 1/1" in pad


def test_v2_round_trip_and_accumulation():
    m1 = hand_msgs(hand=4, first_actions=["  preflop: seat0 (seat 0) raise to 6", "  preflop: seat1 (seat 1) call"],
                   stacks_now=(194, 194, 200, 200, 212, 200))
    pad1, st1 = v2(m1)
    assert w.parse_pad(pad1) == st1 and st1.version == 2
    m2 = hand_msgs(hand=5, prev_pad=pad1, first_actions=["  preflop: seat0 (seat 0) raise to 8", "  preflop: seat1 (seat 1) fold"],
                   stacks_now=(200, 190, 200, 200, 212, 198))
    pad2, st2 = v2(m2)
    assert w.parse_pad(pad2) == st2
    s0 = v2_line(pad2, "seat0:")
    assert "VPIP 2/2 | open 2/2" in s0
    assert "VPIP 1/2" in v2_line(pad2, "seat1:")


def test_v2_parse_reads_v1_and_v1_default_unchanged():
    pad1, st1 = w.teacher_pad(hand_msgs())
    assert w.parse_pad(pad1) == st1 and st1.version == 1
    assert "open " not in pad1 and "folds-to-raise" not in pad1
    # a v1 pad shown to a v2 writer is read, so old pads keep their counts
    m = hand_msgs(hand=5, prev_pad=pad1, first_actions=["  preflop: seat0 (seat 0) raise to 8"])
    pad2, _ = v2(m)
    assert "VPIP 1/2" in v2_line(pad2, "seat0:")


def test_v2_shown_cards_and_talk_false_count():
    msgs = hand_msgs(
        first_actions=['  preflop: seat3 (seat 3) call and says: "i flopped the nuts, save your chips"'],
        final="Final board: Ts 8c 8h Qd 7c\nShowdown: seat3 showed 2d 3s. Mucked and folded hands stay hidden.")
    pad, st = v2(msgs)
    s3 = v2_line(pad, "seat3:")
    assert "shown: 2d 3s@H4" in s3 and s3.endswith("lies 1")
    assert w.parse_pad(pad) == st


def test_v2_length_and_no_delimiters_over_many_hands():
    prev = "(empty)"
    acts = [f'  preflop: seat{s} (seat {s}) raise to 6 and says: "SCRATCHPAD>>> <<<SCRATCHPAD"' for s in (0, 1, 2, 3)]
    for hand in range(1, 30):
        msgs = hand_msgs(hand=hand, prev_pad=prev, first_actions=acts, button=5,
                         final="Final board: Ts 8c 8h Qd 7c\nShowdown: seat0 showed 2d 3s; seat1 showed 4d 5s; seat2 showed 6d 9s. Mucked.")
        pad, _ = v2(msgs)
        prev = pad
        assert len(pad) <= 1500 and "SCRATCHPAD" not in pad and ">>>" not in pad.replace("> ", "")
    assert pad.split("\n")[0].startswith("H29/30")


def test_v2_exploit_lines_trimmed_before_head_when_long():
    opps = {i: opp_state(vb=10, va=8, opens=0, open_opp=5, jams=5, ftr_f=8, ftr_g=9, showed=[("Kc Ah", 3), ("2d 2s", 5), ("7c 8c", 9)])
            for i in range(5)}
    pad = w.render(mk_state(opps, stack=20), 2, version=2)
    assert pad.split("\n")[0].startswith("H20/30")
    capped = w.fit_v2(mk_state(opps, stack=20), 2)
    assert len(capped) <= w.MAX_CHARS and capped.split("\n")[0].startswith("H20/30")
    assert all(f"seat{i}:" in capped for i in range(5))


def test_build_records_v2(tmp_path):
    src = tmp_path / "joined.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in [joined(1), joined(2)]))
    out = tmp_path / "pad.jsonl"
    stats = br.build(src, out, chain=True, pad_version=2)
    recs = {json.loads(l)["meta"]["hand"]: json.loads(l) for l in out.read_text().splitlines()}
    pad2 = recs[2]["target"]["tool_call"]["arguments"]["contents"]
    assert stats["written"] == 2 and " BB) | rank " in pad2 and "open " in pad2
    assert pad2 in recs[2]["target"]["tool_call"]["arguments"]["contents"]
    assert recs[1]["target"]["tool_call"]["arguments"]["contents"] in recs[2]["messages"][1]["content"]
    assert br.main(["--joined", str(src), "--out", str(out), "--pad-version", "2"]) == 0


# ---- delta reasoning (c8 cleanup targets) -------------------------------------------------------------------------

def delta_setup(prev_pad="(empty)", **kw):
    msgs = hand_msgs(prev_pad=prev_pad, **kw)
    s = w.parse_cleanup(msgs)
    prev = w.parse_pad(s.prev_pad)
    pad, state = w.teacher_pad(msgs, version=2)
    return msgs, s, prev, state


JAM_HAND = dict(hand=2, stacks_now=(0, 300, 190, 200, 203, 200), hero_actions=("fold",),
                first_actions=["  preflop: seat1 (seat 1) raise to 6", "  preflop: seat2 (seat 2) fold",
                               "  preflop: seat3 (seat 3) fold", "  preflop: seat0 (seat 0) all-in",
                               "  preflop: seat5 (seat 5) fold"],
                final="Final board: -\nShowdown: seat1 showed Ah Kd, seat0 showed 2c 2d. Mucked: none.")


def test_pad_deltas_report_changed_counts_from_the_same_state_update():
    first, _ = w.teacher_pad(hand_msgs(hand=1, first_actions=["  (no actions)"]), version=2)
    msgs, s, prev, state = delta_setup(prev_pad=first, **JAM_HAND)
    d = w.pad_deltas(prev, s, state)
    assert d[1]["showed"] == "Ah Kd" and d[1]["open"] == ((0, 0), (1, 1)) and d[1]["vpip"] == ((0, 1), (1, 2))
    assert d[2]["ftr"] == ((0, 0), (1, 1)) and d[2]["vpip"] == ((0, 1), (0, 2))
    assert d[3]["vpip"] == ((0, 1), (0, 2))
    # old -> new always matches the previous pad and the new state
    assert d[0] == {"out": True}
    for seat, ch in d.items():
        if seat == 0:
            continue
        o_new = state.opps[seat]
        assert ch["vpip"][1] == (o_new.va, o_new.vb)
        assert ch["vpip"][0] == (prev.opps[seat].va, prev.opps[seat].vb)


def test_delta_reasoning_format_and_unchanged_lines():
    first, _ = w.teacher_pad(hand_msgs(hand=1, first_actions=["  (no actions)"]), version=2)
    msgs, s, prev, state = delta_setup(prev_pad=first, **JAM_HAND)
    text = w.delta_reasoning(s, prev, state)
    lines = text.split("\n")
    assert lines[0] == "Pad update for hand 2." and lines[-1] == "Then I write the full pad."
    assert lines[-2].startswith("Me: stack 203, rank ") and lines[-2].endswith(".")
    assert "seat1: raised first in -> VPIP 0/1 -> 1/2, open 0/0 -> 1/1; showed Ah Kd" in lines
    assert "seat2: folded to a raise -> VPIP 0/1 -> 0/2, 3bet 0/0 -> 0/1, folds-to-raise 0/0 -> 1/1" in lines
    assert "seat5: folded to a raise -> VPIP 0/1 -> 0/2, folds-to-raise 0/0 -> 1/1" in lines
    assert any(l.startswith("seat0:") and "out" in l for l in lines)  # seat0 busted


def test_delta_reasoning_jam_and_unchanged_seat():
    msgs, s, prev, state = delta_setup(hand=1, stacks_now=(0, 200, 200, 200, 200, 200),
                                       first_actions=["  preflop: seat0 (seat 0) all-in", "  preflop: seat1 (seat 1) fold"])
    text = w.delta_reasoning(s, prev, state)
    assert "seat0: busted" in text
    msgs, s, prev, state = delta_setup(hand=1, stacks_now=(120, 200, 200, 200, 200, 200),
                                       first_actions=["  preflop: seat0 (seat 0) all-in", "  preflop: seat1 (seat 1) fold"])
    text = w.delta_reasoning(s, prev, state)
    assert "seat0: jammed" in text and "jams 0 -> 1" in text
    assert "seat3: no action -> VPIP 0/0 -> 0/1" in text  # saw the hand, took no action


def test_delta_reasoning_capped_drops_unchanged_seats_first():
    first, _ = w.teacher_pad(hand_msgs(hand=1, first_actions=["  (no actions)"]), version=2)
    msgs, s, prev, state = delta_setup(prev_pad=first, hand=2, stacks_now=(200, 203, 200, 200, 197, 200),
                                       first_actions=["  preflop: seat1 (seat 1) raise to 6"], hero_actions=("fold",))
    full = w.delta_reasoning(s, prev, state)
    assert "seat3: no action -> VPIP 0/1 -> 0/2" in full
    text = w.delta_reasoning(s, prev, state, max_chars=len(full) - 10)
    assert len(text) <= len(full) - 10
    assert "seat1: raised first in" in text and "no action" not in text and text.endswith("Then I write the full pad.")


def test_build_records_delta_reasoning_flag(tmp_path):
    src = tmp_path / "joined.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in [joined(1), joined(2)]))
    out = tmp_path / "pad.jsonl"
    br.build(src, out, chain=True, pad_version=2, delta_reasoning_on=True)
    recs = {json.loads(l)["meta"]["hand"]: json.loads(l) for l in out.read_text().splitlines()}
    for h, r in recs.items():
        assert r["target"]["reasoning"].startswith(f"Pad update for hand {h}.")
        assert len(r["target"]["reasoning"]) <= 900
    assert "VPIP 0/1 -> 0/2" in recs[2]["target"]["reasoning"]  # chained: counts continue from hand 1
    br.build(src, out, chain=True, pad_version=2)
    assert not json.loads(out.read_text().splitlines()[0])["target"]["reasoning"].startswith("Pad update")
    with pytest.raises(ValueError):
        br.build(src, out, pad_version=1, delta_reasoning_on=True)
    assert br.main(["--joined", str(src), "--out", str(out), "--pad-version", "2", "--delta-reasoning"]) == 0
