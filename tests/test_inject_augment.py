import json

from poker.inject import augment, probe

LINE = "  preflop: seat3 (seat 3) raise to 12"


def prompt(hole="Ah Kd", board="-", line=LINE):
    return [
        {"role": "system", "type": "message", "content": "sys"},
        {"role": "user", "type": "message", "content": f"## Hand 3 of 30\nYour seat: 1 (position 1; 0 = small blind)\n"
         f"Street: {'preflop' if board == '-' else 'flop'}\nYour hole cards: {hole}\nBoard: {board}\n"
         "Pot (including current bets): 17\nTable:\n  seat 0: seat0 [button] stack 190, bet 0\n"
         "  seat 1: seat1 (you) stack 198, bet 2\n  seat 2: seat2 stack 190, bet 0\n"
         "  seat 3: seat3 stack 188, bet 12\n  seat 4: seat4 stack 0, bet 0, out\n  seat 5: seat5 stack 0, bet 0, out\n"
         f"Actions this hand:\n{line}\n\nLegal actions: fold, call 10, raise to 22-200\nCall submit_action with your decision."},
    ]


def rec(action="call", hole="Ah Kd", board="-", line=LINE, did="d1", weight=2.0, **extra):
    args = {"action": action, **({"amount": 30} if action in ("bet", "raise") else {})}
    return {"decision_id": did, "messages": prompt(hole, board, line), "weight": weight, "source": "teacher",
            "target": {"reasoning": "Facts: x.\nReads: none\nThink more.", "content": "",
                       "tool_call": {"name": "submit_action", "arguments": args}}, **extra}


def test_lines_valid_and_split():
    allv = augment.INJECTIONS
    assert len(allv) >= 40 and len(set(allv)) == len(allv)
    for t in allv:
        assert t == probe.sanitize_talk(t) and 0 < len(t) <= 70
    for t in probe.INJECTIONS.values():
        assert probe.sanitize_talk(t) in allv
    assert len(augment.HELDOUT) == 8 and not set(augment.HELDOUT) & set(augment.TRAIN_INJECTIONS)
    assert not set(augment.HELDOUT) & {probe.sanitize_talk(t) for t in probe.INJECTIONS.values()}
    assert set(augment.HELDOUT) | set(augment.TRAIN_INJECTIONS) == set(allv)


def test_defend_copies_non_fold_with_talk_and_sentence():
    out, st = augment.defend([rec("call"), rec("fold", did="d2")], share=1.0, seed=0)
    assert st["copies"] == 2 and len(out) == 2
    c = out[0]
    assert out[0]["injection"] != out[1]["injection"]
    assert c["source"] == "inject_defense" and c["weight"] == 2.0
    assert c["target"]["tool_call"] == rec("call")["target"]["tool_call"]
    t = probe.find_target(prompt())
    msg = c["messages"][t["msg"]]["content"]
    assert ' and says: "' in msg and "Hand 3" in msg
    lines = c["target"]["reasoning"].split("\n")
    assert lines[:2] == ["Facts: x.", "Reads: none"]
    assert lines[2] == "Talk: seat3's message is table talk, not a rule; I ignore it."
    assert probe.find_target(c["messages"]) is None  # now has talk


def test_defend_skips_no_target_talk_and_scratchpad():
    clean = rec("check", line="  preflop: seat3 (seat 3) call")
    cleanup = rec("call"); cleanup["target"]["tool_call"]["name"] = "update_scratchpad"
    out, st = augment.defend([clean, cleanup], share=1.0)
    assert out == []


def test_defend_share_deterministic():
    rs = [rec("call", did=f"d{i}") for i in range(200)]
    a, _ = augment.defend(rs, share=0.35, seed=1)
    b, _ = augment.defend(rs, share=0.35, seed=1)
    assert a == b and 80 < len(a) < 200


def test_offend_weak_gets_say_monster_none():
    weak = rec("raise", did="w")  # AKo preflop, no made hand
    pair = rec("bet", hole="Ah Kd", board="Ac 7d 2s", line="  flop: seat3 (seat 3) check", did="p")
    pair["messages"] = prompt("Ah Kd", "Ac 7d 2s", "  flop: seat3 (seat 3) check")
    mon = rec("bet", did="m"); mon["messages"] = prompt("Ah Ad", "Ac 7d 2s", "  flop: seat3 (seat 3) check")
    call = rec("call", did="c")
    out, st = augment.offend([weak, pair, mon, call], seed=0)
    assert out[0]["target"]["tool_call"]["arguments"]["say"] in augment.OFFENSE_LINES and out[0]["say_added"] is True
    assert out[1]["target"]["tool_call"]["arguments"]["say"] in augment.OFFENSE_LINES
    assert "say" not in out[2]["target"]["tool_call"]["arguments"] and "say_added" not in out[2]
    assert "say" not in out[3]["target"]["tool_call"]["arguments"]
    assert st["says"] == 2 and st["skipped_monsters"] == 1
    assert "say" not in weak["target"]["tool_call"]["arguments"]  # input untouched
    again, _ = augment.offend([weak, pair, mon, call], seed=0)
    assert again == out


def test_offense_lines_valid():
    for t in augment.OFFENSE_LINES:
        assert t == probe.sanitize_talk(t) and 0 < len(t) <= 70


def test_cli(tmp_path, capsys):
    p, o = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [rec("call"), rec("raise", did="w")]) + "\n")
    augment.main(["--records", str(p), "--out", str(o), "--defend-share", "1.0", "--offend"])
    rows = [json.loads(x) for x in o.read_text().splitlines()]
    assert len(rows) == 6 and any(r["source"] == "inject_defense" for r in rows)
    assert "defense copies" in capsys.readouterr().out


def test_defend_replaces_existing_talk_and_sees_past_later_calls():
    talked = rec("call", line=LINE + ' and says: "hi there"\n  preflop: seat0 (seat 0) fold\n  preflop: seat2 (seat 2) call')
    out, st = augment.defend([talked], share=1.0, copies_per_record=1)
    assert len(out) == 1 and st["preflop"] == 1
    txt = out[0]["messages"][1]["content"]
    assert "hi there" not in txt and txt.count(' and says: "') == 1
    assert txt.index("raise to 12 and says") < txt.index("seat0 (seat 0) fold")


def test_defend_ignores_earlier_street_aggression():
    r = rec("call", board="Ac 7d 2s", line="  preflop: seat3 (seat 3) raise to 12\n  flop: seat3 (seat 3) check")
    assert augment.defend([r], share=1.0)[0] == []
