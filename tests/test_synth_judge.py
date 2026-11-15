import json

from poker.synth.judge_spots import facing_allin, hand_number, judge_record, judge_spans, summarize
from poker.hints import parse_prompt

FIELD = ["AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo", "AQs"]


def prompt(hole="9s 9c", table=None, hand=2, legal="fold, call 197", street="preflop"):
    table = table or ["  seat 0: seat0 [button] stack 0, bet 199, all-in", "  seat 1: seat1 (you) stack 197, bet 1",
                      "  seat 2: seat2 stack 198, bet 2", "  seat 3: seat3 stack 200, bet 0, folded",
                      "  seat 4: seat4 stack 203, bet 0, folded", "  seat 5: seat5 stack 200, bet 0, folded"]
    text = "\n".join([f"## Hand {hand} of 30", "Your seat: 1 (position 0; 0 = small blind)", f"Street: {street}",
                      f"Your hole cards: {hole}", "Board: -", "Pot (including current bets): 202", "Table:", *table,
                      "Actions this hand:", "  preflop: seat0 (seat 0) raise to 199", "", f"Legal actions: {legal}",
                      "Call submit_action with your decision."])
    return [{"role": "system", "type": "message", "content": "sys"},
            {"role": "user", "type": "message", "content": "## Hand 1 is over\n## Your scratchpad\nnotes"},
            {"role": "user", "type": "message", "content": text}]


def span(messages, action="call", **kw):
    return {"span_id": kw.get("span_id", "s1"), "kind": "action", "source": "other", "deployment": "infer_x",
            "messages": messages, "reasoning": "r", "content": "", "cutoff": False,
            "tool_calls": [{"call_id": "c", "name": "submit_action", "arguments": json.dumps({"action": action})}]}


def test_hand_number_uses_the_current_decision_message():
    assert hand_number(prompt(hand=7)) == 7


def test_facing_allin_heads_up_only():
    state = parse_prompt(prompt())
    assert facing_allin(state) == {"shover": 0, "stack": 197}
    # a caller who matched the shove makes it multiway
    table = ["  seat 0: seat0 [button] stack 0, bet 199, all-in", "  seat 1: seat1 (you) stack 197, bet 1",
             "  seat 2: seat2 stack 1, bet 199", "  seat 3: seat3 stack 200, bet 0, folded",
             "  seat 4: seat4 stack 203, bet 0, folded", "  seat 5: seat5 stack 200, bet 0, folded"]
    assert facing_allin(parse_prompt(prompt(table=table))) is None
    # two shovers
    table = [table[0], table[1], "  seat 2: seat2 stack 0, bet 150, all-in", *table[3:]]
    assert facing_allin(parse_prompt(prompt(table=table))) is None


def test_no_price_or_postflop_is_not_a_spot():
    assert facing_allin(parse_prompt(prompt(legal="check, bet to 2-200"))) is None


def test_junk_call_is_judged_fold_and_marked_disagree():
    r = judge_record(span(prompt(hole="7s 2c"), "call"), FIELD)
    v = r["verdict"]
    assert r["source"] == "synth" and r["hand_number"] == 2 and r["hand_code"] == "72o"
    assert v["preferred"] == "fold" and v["actual"] == "call" and v["agree"] is False and v["rule"] == "allin_call"
    assert v["to_call"] == 197 and v["pot"] == 202 and "equity_lo" in v  # evidence rule on


def test_premium_call_is_agree_or_abstain_never_disagree():
    r = judge_record(span(prompt(hole="As Ac"), "call"), FIELD)
    assert r["verdict"]["agree"] is not False


def test_skips_arena_cleanup_and_non_spots():
    s = span(prompt(hole="7s 2c"))
    assert judge_record(s | {"source": "arena"}, FIELD) is None
    assert judge_record(s | {"kind": "cleanup"}, FIELD) is None
    assert judge_record(span(prompt(legal="check, bet to 2-200")), FIELD) is None
    assert judge_record(s | {"tool_calls": []}, FIELD) is None


def test_summary_counts_call_rate_when_teacher_folds():
    spans = [span(prompt(hole="7s 2c"), "call", span_id="a"), span(prompt(hole="8s 3c"), "fold", span_id="b")]
    records, stats = judge_spans(spans, FIELD)
    assert stats["spots"] == 2
    s = summarize(records)
    assert s["teacher_fold_n"] == 2 and s["call_rate_when_teacher_folds"] == 0.5
