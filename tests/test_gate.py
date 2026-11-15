import json

from poker import gate

PROMPT = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "## Hand 1 of 30\nYour seat: 1 (position 1; 0 = small blind)\nStreet: preflop\n"
     "Your hole cards: 6h Ac\nBoard: -\nPot (including current bets): 5\nTable:\n"
     "  seat 0: seat0 stack 199, bet 1, folded\n  seat 1: seat1 (you) stack 198, bet 2\n"
     "\nLegal actions: check, raise to 4-200\nCall submit_action with your decision."},
]


def response(calls, finish="tool_calls", tokens=100):
    msg = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}} for n, a in calls]}
    return {"choices": [{"message": msg, "finish_reason": finish}], "usage": {"completion_tokens": tokens}}


def test_parse_legal_check_and_raise():
    ok = gate.parse_reply(PROMPT, response([("submit_action", {"action": "check"})]))
    assert ok["valid"] and ok["action"] == "check" and not ok["cutoff"] and ok["tokens"] == 100
    assert gate.parse_reply(PROMPT, response([("submit_action", {"action": "raise", "amount": 6})]))["valid"]


def test_parse_illegal_and_malformed():
    assert not gate.parse_reply(PROMPT, response([("submit_action", {"action": "fold"})]))["valid"]
    assert not gate.parse_reply(PROMPT, response([("submit_action", {"action": "raise", "amount": 3})]))["valid"]
    two = response([("submit_action", {"action": "check"}), ("submit_action", {"action": "check"})])
    assert not gate.parse_reply(PROMPT, two)["valid"]
    assert not gate.parse_reply(PROMPT, response([("update_scratchpad", {"contents": "x"})]))["valid"]
    assert not gate.parse_reply(PROMPT, response([]))["valid"]


def test_cutoff_and_http_error():
    cut = gate.parse_reply(PROMPT, response([], finish="length", tokens=5000))
    assert cut["cutoff"] and not cut["valid"]
    err = gate.parse_reply(PROMPT, {"error": "HTTP 500"})
    assert not err["valid"] and err["tokens"] is None


def test_chat_tools_and_messages():
    tools = gate.chat_tools()
    assert [t["function"]["name"] for t in tools] == ["submit_action", "update_scratchpad"]
    assert all(t["type"] == "function" for t in tools)
    out = gate.chat_messages([{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}])
    assert out == [{"role": "user", "content": "a\nb"}]


def entry(i, preferred, rule="unopened"):
    record = {"span_id": f"s{i:03d}", "messages": PROMPT,
              "tool_calls": [{"name": "submit_action", "arguments": '{"action":"check"}'}]}
    return {"record": record, "verdict": {"rule": rule, "preferred": preferred}}


def test_original_action():
    assert gate.original_action(entry(1, None)["record"]) == "check"
    assert gate.original_action({"tool_calls": []}) is None


def test_select_deterministic_and_order_independent():
    entries = [entry(i, "fold" if i % 3 == 0 else None) for i in range(60)]
    a = gate.select_spots(entries, n_judged=None, n_other=10, seed=1)
    b = gate.select_spots(list(reversed(entries)), n_judged=None, n_other=10, seed=1)
    ids = lambda s: [(x["group"], x["record"]["span_id"]) for x in s]  # noqa: E731
    assert ids(a) == ids(b)
    assert sum(x["group"] == "judged" for x in a) == 20 and sum(x["group"] == "other" for x in a) == 10
    assert ids(a) != ids(gate.select_spots(entries, n_judged=None, n_other=10, seed=2))
    assert sum(x["group"] == "judged" for x in gate.select_spots(entries, n_judged=5, n_other=0, seed=1)) == 5


def test_aggregate():
    def r(group, action, preferred=None, rule=None, orig="check", valid=True, cutoff=False, tokens=100):
        return {"group": group, "action": action, "preferred": preferred, "rule": rule, "orig": orig,
                "valid": valid, "cutoff": cutoff, "tokens": tokens, "error": None}

    rows = [r("judged", "fold", "fold", "unopened"), r("judged", "call", "fold", "unopened"),
            r("judged", "check", "check", "free_fold"),
            r("other", "check"), r("other", "raise", valid=False, cutoff=True, tokens=5000)]
    s = gate.aggregate(rows)
    assert s["valid_rate"] == 0.8 and s["cutoff_rate"] == 0.2
    assert s["tokens_median"] == 100 and s["tokens_p90"] == 5000
    assert s["judged"]["agree"] == 2 and s["judged"]["by_rule"]["unopened"] == {"n": 2, "agree": 1, "agree_rate": 0.5}
    assert s["other"]["dist"]["check"] == 0.5 and s["other"]["drift"] == 0.5
    assert "valid" in gate.format_table({"base": s})


def test_run_gate_fake_client_and_retry():
    calls = []

    class Fake:
        def complete(self, model, messages, tools):
            calls.append(model)
            return response([("submit_action", {"action": "check"})])

    spots = gate.select_spots([entry(1, "check"), entry(2, None)], n_judged=None, n_other=5, seed=0)
    out = gate.run_gate(spots, ["a", "b"], Fake(), gate.chat_tools(), concurrency=2)
    assert sorted(calls) == ["a", "a", "b", "b"]
    assert out["summary"]["a"]["judged"]["agree_rate"] == 1.0

    class Resp:
        def __init__(self, code): self.status_code = code
        def json(self): return response([("submit_action", {"action": "check"})])

    class FakeHttpx:
        HTTPError = OSError
        def __init__(self): self.codes = [429, 503, 200]
        def post(self, *a, **k): return Resp(self.codes.pop(0))

    client = gate.HttpClient("k", sleep=lambda s: None)
    client.httpx = FakeHttpx()
    assert "choices" in client.complete("m", [], [])


def test_chat_messages_function_items():
    items = [{"type": "message", "role": "user", "content": "q"},
             {"type": "message", "role": "assistant", "content": "thinking"},
             {"type": "function_call", "call_id": "c1", "name": "submit_action", "arguments": '{"action":"check"}'},
             {"type": "function_call_output", "call_id": "c1", "output": "ok"}]
    out = gate.chat_messages(items)
    assert out[1]["tool_calls"][0]["function"]["name"] == "submit_action" and out[1]["content"] == "thinking"
    assert out[2] == {"role": "tool", "tool_call_id": "c1", "content": "ok"}


def test_repeats_sample_each_judged_spot_several_times():
    entries = [entry(i, "fold" if i % 3 == 0 else None) for i in range(60)]
    spots = gate.select_spots(entries, n_judged=None, n_other=10, seed=1, repeats=3)
    assert sum(x["group"] == "judged" for x in spots) == 60 and sum(x["group"] == "other" for x in spots) == 10


def _row(span, group, action, preferred=None, rule="allin_call"):
    return {"span_id": span, "group": group, "rule": rule if group == "judged" else None, "preferred": preferred,
            "orig": "call", "valid": True, "cutoff": False, "tokens": 500, "action": action}


def test_bootstrap_ci_brackets_the_mean_and_narrows_with_more_data():
    lo, hi = gate.bootstrap_ci([1, 0] * 10, seed=0)
    wide = hi - lo
    lo2, hi2 = gate.bootstrap_ci([1, 0] * 200, seed=0)
    assert lo < 0.5 < hi and lo2 < 0.5 < hi2 and hi2 - lo2 < wide


def test_paired_difference_uses_per_spot_means():
    base = [_row(f"s{i}", "judged", "call", "fold") for i in range(20) for _ in range(3)]
    cand = [_row(f"s{i}", "judged", "fold", "fold") for i in range(20) for _ in range(3)]
    d = gate.paired_difference(base, cand, seed=0)
    assert d["n_spots"] == 20 and d["diff"] == 1.0 and d["lo"] > 0.9


def test_mixing_counts_spots_where_replays_choose_different_actions():
    rows = [_row("a", "other", "check"), _row("a", "other", "bet"),
            _row("b", "other", "fold"), _row("b", "other", "fold")]
    m = gate.mixing(rows)
    assert m["spots"] == 2 and m["mixed_rate"] == 0.5 and m["entropy_bits"] > 0


def test_verdict_rejects_a_candidate_that_stops_mixing():
    ref = {"valid_rate": 1.0, "cutoff_rate": 0.0, "mixing": {"mixed_rate": 0.30}}
    good = {"valid_rate": 1.0, "cutoff_rate": 0.0, "mixing": {"mixed_rate": 0.25}}
    rigid = {"valid_rate": 1.0, "cutoff_rate": 0.0, "mixing": {"mixed_rate": 0.05}}
    better = {"diff": 0.2, "lo": 0.05, "hi": 0.3}
    assert gate.verdict(ref, good, better)["pass"] is True
    v = gate.verdict(ref, rigid, better)
    assert v["pass"] is False and "mixing" in " ".join(v["reasons"])
    worse = {"diff": -0.2, "lo": -0.35, "hi": -0.05}
    assert gate.verdict(ref, good, worse)["pass"] is False
