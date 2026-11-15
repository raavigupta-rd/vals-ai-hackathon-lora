import json

from poker.arena.pull_gateway import normalize_span, pull, parse_when

PROMPT = [
    {"role": "system", "type": "message", "content": "sys"},
    {"role": "user", "type": "message", "content": "## Hand 6 of 30\nYour seat: 0 (position 1; 0 = small blind)\nStreet: flop\nYour hole cards: 5c 9s\nBoard: Jd 4h 3d\nPot (including current bets): 5\nLegal actions: check, bet to 2-253\nCall submit_action with your decision."},
]
OUTPUT = [
    {"type": "reasoning", "content": "think"},
    {"type": "message", "role": "assistant", "content": "\n\n"},
    {"type": "function_call", "call_id": "c1", "name": "submit_action", "arguments": "{\"action\":\"check\"}"},
]


def row(span_id="s1", prompt=PROMPT, output=OUTPUT, conv="vals-arena/m_1/hand-6/seat0", finish="tool_calls"):
    attrs = {"finish_reason": finish, "conversation_id": conv, "inference_deployment_id": "infer_x",
             "sampling_params": "{\"max_completion_tokens\":5000}"}
    return {"span_id": span_id, "trace_id": "t1", "name": "inference.request", "start_time": "2026-11-15T00:21:04.868725",
            "duration_ms": 1468, "status": "completed", "tokens_in": 10, "tokens_out": 20, "cost": 0.0,
            "input": json.dumps(prompt), "output": json.dumps(output), "attributes": json.dumps(attrs)}


def test_normalize_action():
    r = normalize_span(row(), "infer_x")
    assert r["span_id"] == "s1" and r["deployment"] == "infer_x"
    assert r["ts"] == "2026-11-15T00:21:04.868725Z"
    assert r["kind"] == "action" and r["source"] == "arena"
    assert r["messages"] == PROMPT and r["tools"] is None
    assert r["reasoning"] == "think" and r["content"] == "\n\n"
    assert r["tool_calls"] == [{"call_id": "c1", "name": "submit_action", "arguments": "{\"action\":\"check\"}"}]
    assert r["usage"] == {"tokens_in": 10, "tokens_out": 20, "cost": 0.0}
    assert r["finish_reason"] == "tool_calls" and r["cutoff"] is False
    assert r["arena"] == {"match_id": "m_1", "hand": 6, "seat": 0}


def test_cleanup_cutoff_and_other_source():
    prompt = PROMPT + [{"role": "user", "type": "message", "content": "Now call update_scratchpad with the new contents."}]
    r = normalize_span(row(prompt=prompt, conv="eval-run-3", finish="length"), "infer_x")
    assert r["kind"] == "cleanup" and r["cutoff"] is True
    assert r["source"] == "other" and r["arena"] is None


def test_unreadable_payload_is_kept():
    bad = row()
    bad["input"] = "not json"
    r = normalize_span(bad, "infer_x")
    assert r["messages"] is None and r["kind"] == "action"


def test_parse_when():
    assert parse_when("2026-11-15T00:00:00Z") == "2026-11-15T00:00:00"
    assert len(parse_when("3h")) == 19


def test_pull_paginates_dedupes_and_is_idempotent(tmp_path):
    rows = [row(f"s{i}") for i in range(5)]
    calls = []

    def query(spec):
        calls.append(spec.offset)
        return type("R", (), {"rows": rows[spec.offset:spec.offset + spec.limit], "total": len(rows)})()

    out = tmp_path / "g.jsonl"
    first = pull("infer_x", "2026-11-15T00:00:00", None, out, query=query, page=2)
    assert first["written"] == 5 and calls == [0, 2, 4]
    second = pull("infer_x", "2026-11-15T00:00:00", None, out, query=query, page=2)
    assert second["written"] == 0 and second["skipped"] == 5
    ids = [json.loads(l)["span_id"] for l in out.read_text().splitlines()]
    assert ids == [f"s{i}" for i in range(5)]
