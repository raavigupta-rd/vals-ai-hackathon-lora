import json

from poker.pad import check as ck
from poker.pad import writer as w
from tests.test_pad_writer import hand_msgs

HAND2 = dict(hand=2, stacks_now=(199, 197, 195, 200, 203, 206),
             first_actions=["  preflop: seat0 (seat 0) raise to 6", "  preflop: seat1 (seat 1) call",
                            "  preflop: seat2 (seat 2) fold", "  preflop: seat3 (seat 3) check"])


def setup():
    first, _ = w.teacher_pad(hand_msgs(hand=1), version=2)
    msgs = hand_msgs(prev_pad=first, **HAND2)
    teacher, _ = w.teacher_pad(msgs, version=2)
    return first, msgs, teacher


def test_perfect_pad_passes_everything():
    first, msgs, teacher = setup()
    r = ck.pad_check(first, msgs, teacher)
    assert r == {"parses": True, "header_ok": True, "seats_ok": True, "count_errors": 0, "exploit_lines_ok": True,
                 "extra_lines": 0}


def test_count_errors_count_each_wrong_field():
    first, msgs, teacher = setup()
    line = next(l for l in teacher.split("\n") if l.startswith("seat0:"))
    assert "VPIP 1/2 | open 1/1" in line
    bad = teacher.replace(line, line.replace("VPIP 1/2 | open 1/1", "VPIP 0/0 | open 0/0"))
    r = ck.pad_check(first, msgs, bad)
    assert r["parses"] and r["count_errors"] == 4


def test_missing_seat_and_wrong_header():
    first, msgs, teacher = setup()
    lines = teacher.split("\n")
    no_seat = "\n".join(l for l in lines if not l.startswith("seat2:"))
    r = ck.pad_check(first, msgs, no_seat)
    assert r["seats_ok"] is False and r["count_errors"] >= 9
    wrong = teacher.replace("H2/30", "H3/30")
    assert ck.pad_check(first, msgs, wrong)["header_ok"] is False


def test_garbage_and_junk_lines():
    first, msgs, teacher = setup()
    r = ck.pad_check(first, msgs, "seat0: VPIP 0(0%), open 0(0%) aggro 0")
    assert r["parses"] is False and r["count_errors"] is None and r["header_ok"] is False
    r = ck.pad_check(first, msgs, teacher + "\nhand 2 log: fold")
    assert r["parses"] and r["extra_lines"] == 1


def test_exploit_lines_compared_with_teacher():
    first, msgs, teacher = setup()
    assert ck.pad_check(first, msgs, teacher + "\n> seat0 jams wide (9/9): call jams")["exploit_lines_ok"] is False


def fake_response(pad):
    call = {"function": {"name": "update_scratchpad", "arguments": json.dumps({"contents": pad})}}
    return {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [call]}}]}


def row(hand, match="m", seat=4, prev="(empty)", **kw):
    return {"kind": "cleanup", "source": "arena", "span_id": f"s{match}{hand}",
            "messages": hand_msgs(hand=hand, prev_pad=prev, **kw), "match": {"match_id": match, "hand": hand, "seat": seat}}


class Echo:
    """Writes the teacher pad for the prompt it is given, except for every third call (invalid)."""

    def __init__(self):
        self.calls = 0

    def complete(self, model, messages, tools):
        self.calls += 1
        if self.calls % 3 == 0:
            return {"choices": [{"finish_reason": "stop", "message": {"content": "no tool"}}]}
        first = next(m["content"] for m in messages if m["role"] == "user")
        pad = first.split("<<<SCRATCHPAD\n")[1].split("\nSCRATCHPAD>>>")[0]
        return fake_response(pad if pad != "(empty)" else "H1/30 | me seat4 200 (100 BB) | rank 1/6 | gap up +0, down -0 | 29 left")


def test_build_cases_and_run(tmp_path):
    rows = [row(h) for h in range(1, 6)]
    cases = ck.build_cases(rows, n=3, n_first=1, seed=0)
    assert sum(1 for c in cases if c["hand"] == 1) == 1 and len(cases) == 4
    for c in cases:
        shown = c["messages"][1]["content"]
        assert c["prev_pad"] == ("" if c["hand"] == 1 else c["prev_pad"])
        assert (c["prev_pad"] or "(empty)") in shown
    res = ck.run(cases, Echo(), "dep", tools=[], concurrency=1)
    assert res["n"] == 4 and 0 < res["valid_rate"] < 1 and "mean_count_errors" in res
    assert len(res["failures"]) <= 3


def make_chain():
    m1 = hand_msgs(hand=1)
    t2, _ = w.teacher_pad(m1, version=2)
    m2 = hand_msgs(prev_pad=t2, **HAND2)
    t3, _ = w.teacher_pad(m2, version=2)
    return {"key": ("m", 0), "msgs": {1: m1, 2: m2}, "teacher_prev": {1: "", 2: t2}}, t2, t3


def test_self_chain_perfect_model_all_parse():
    chain, t2, t3 = make_chain()
    seen = []
    answers = iter([t2, t3])

    def fake(messages):
        seen.append(messages)
        return next(answers)

    steps = ck.run_chain(chain, fake)
    assert all(s["score"]["parses"] and s["score"]["count_errors"] == 0 for s in steps)
    assert any(t2 in m.get("content", "") for m in seen[1] if isinstance(m.get("content"), str))
    s = ck.summarize_chains([steps])
    assert s["parse_rate_hand1"] == 1 and s["games_all_parse"] == 1 and s["first_break_hand"] == [None]


def test_self_chain_junk_then_copy_never_parses():
    chain, t2, _ = make_chain()
    calls = []

    def fake(messages):
        calls.append(messages)
        return "junk"

    steps = ck.run_chain(chain, fake)
    assert not any(s["score"]["parses"] for s in steps)
    assert any("junk" in m.get("content", "") for m in calls[1] if isinstance(m.get("content"), str))
    s = ck.summarize_chains([steps])
    assert s["games_all_parse"] == 0 and s["first_break_hand"] == [1] and s["parse_rate_hand1"] == 0


def test_self_chain_failed_call_keeps_previous_pad():
    chain, t2, t3 = make_chain()
    calls = []

    def fake(messages):
        calls.append(messages)
        return None if len(calls) == 1 else t3

    steps = ck.run_chain(chain, fake)
    assert steps[0]["valid"] is False and steps[0]["score"] is None and steps[1]["valid"]
    assert t2 not in str(calls[1]) and "junk" not in str(calls[1])
    s = ck.summarize_chains([steps])
    assert s["valid_rate"] == 0.5 and s["first_break_hand"] == [1]
