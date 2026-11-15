import json

from poker.synth.adapter import load_resampled, resample_spots, select, to_rows
from tests.test_synth_judge import FIELD, prompt, span
from poker.synth.judge_spots import judge_spans


class Client:
    """Replies with the given action every time, in chat-completions shape."""

    def __init__(self, actions):
        self.actions, self.n = list(actions), 0

    def complete(self, model, messages, tools):
        action = self.actions[min(self.n, len(self.actions) - 1)]
        self.n += 1
        call = {"id": "c", "type": "function", "function": {"name": "submit_action", "arguments": json.dumps({"action": action})}}
        return {"choices": [{"finish_reason": "tool_calls", "message": {"content": "", "reasoning_content": "price 49%, fold", "tool_calls": [call]}}]}


def spots():
    records, _ = judge_spans([span(prompt(hole="7s 2c"), "call", span_id="bad"),
                              span(prompt(hole="As Ac"), "call", span_id="good")], FIELD)
    return records


def test_select_uses_the_precomputed_verdict_with_no_db():
    records, stats = select(spots(), cycle=3, template_weight=0)
    assert stats["corrections"] == 0 and stats["unvalidated_dropped"] == 1  # template weight 0 drops unvalidated
    reply = {"reasoning": "fold it", "content": "", "arguments": {"action": "fold"}, "preferred": "fold"}
    records, stats = select(spots(), cycle=3, template_weight=0, resampled={"bad": reply})
    fixed = [r for r in records if r["source"] == "teacher_resampled"]
    assert len(fixed) == 1 and fixed[0]["decision_id"] == "bad" and fixed[0]["weight"] == 3.0  # allin_call is a bust rule
    assert fixed[0]["target"]["tool_call"]["arguments"] == {"action": "fold"}


def test_resample_plain_then_hinted(monkeypatch):
    import poker.sft.resample as rs

    monkeypatch.setattr(rs, "_valid_by_gate", lambda p, r: True)
    # plain sample succeeds
    out = resample_spots(spots(), Client(["fold"]), [], "dep", samples=2, hinted_samples=1, concurrency=1)
    assert [o["span_id"] for o in out] == ["bad"] and out[0]["reply"]["hinted"] is False
    # plain fails (always call), the hinted re-roll also calls: nothing validated
    out = resample_spots(spots(), Client(["call"]), [], "dep", samples=2, hinted_samples=1, concurrency=1)
    assert out[0]["reply"] is None and out[0]["tries"] == 3
    # plain fails once per sample, then a hinted fold succeeds
    out = resample_spots(spots(), Client(["call", "call", "fold"]), [], "dep", samples=2, hinted_samples=1, concurrency=1)
    assert out[0]["reply"]["hinted"] is True


def test_load_resampled_roundtrip(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps({"span_id": "a", "preferred": "fold", "reply": {"reasoning": "x", "arguments": {}}}) + "\n"
                    + json.dumps({"span_id": "b", "preferred": "fold", "reply": None}) + "\n")
    assert list(load_resampled(path)) == ["a"]
