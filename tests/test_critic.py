import json
import random

import pytest

from poker.critic import batches, merge

STATE = (
    "## Hand 3 of 30\nYour seat: 0 (position 0; 0 = small blind)\nStreet: preflop\nYour hole cards: 3s Jh\n"
    "Board: -\nPot (including current bets): 205\nTable:\n  seat 0: seat0 (you) stack 199, bet 1\n"
    "\nLegal actions: fold, call 199\nCall submit_action with your decision."
)


def record(span_id, action, *, source="arena", state=STATE, pad=True, kind="action"):
    msgs = [{"role": "system", "content": "sys", "type": "message"}]
    if pad:
        msgs.append({"role": "user", "content": "## Your scratchpad (private)\n<<<SCRATCHPAD\nsecret\nSCRATCHPAD>>>"})
    msgs += [{"role": "user", "content": state}, {"role": "assistant", "content": "\n\n"},
             {"type": "function_call", "name": "submit_action", "arguments": "{}", "call_id": "x"},
             {"role": "user", "content": "Since your last action:\n  preflop: seat5 fold"}]
    return {"span_id": span_id, "source": source, "deployment": "dep", "kind": kind, "messages": msgs,
            "tool_calls": [{"name": "submit_action", "arguments": json.dumps({"action": action})}],
            "match": {"verified": True}}


def verdict(rule="allin_call", preferred="fold", actual="call", **kw):
    agree = None if preferred is None else preferred == actual
    return {"rule": rule, "preferred": preferred, "actual": actual, "agree": agree} | kw


def test_render_state_drops_scratchpad_keeps_user_messages():
    text = batches.render_state(record("a", "call")["messages"])
    assert "secret" not in text and "SCRATCHPAD" not in text
    assert "Hand 3 of 30" in text and "Since your last action" in text and "sys" not in text


def test_facts_pot_odds_legal_actions_and_teacher_numbers():
    facts = batches.facts_for(STATE, verdict(equity=0.45, required=0.59, equity_lo=0.34, equity_hi=0.55, rule="allin_call"))
    assert facts["legal_actions"] == ["fold", "call 199"]
    assert facts["to_call"] == 199 and facts["pot"] == 205
    assert facts["pot_odds"] == pytest.approx(199 / 404, abs=1e-3)
    assert facts["equity"] == 0.45 and facts["required_equity"] == 0.59 and facts["equity_interval"] == [0.34, 0.55]
    assert "preferred" not in json.dumps(facts)
    assert "pot odds" in batches.facts_block(facts).lower()
    assert batches.facts_for(STATE.replace("call 199", "check"), None)["to_call"] == 0


def test_action_taken_parsed():
    assert batches.action_taken(record("a", "raise")) == "raise"
    assert batches.action_taken({"tool_calls": None}) is None


def test_select_stratifies_by_rule_and_is_deterministic():
    items = [{"span_id": f"a{i}", "verdict": verdict("r1")} for i in range(20)]
    items += [{"span_id": f"b{i}", "verdict": verdict("r2")} for i in range(3)]
    pick = batches.select(items, n=10, seed=1)
    rules = [i["verdict"]["rule"] for i in pick]
    assert len(pick) == 10 and rules.count("r2") == 3
    assert [i["span_id"] for i in pick] == [i["span_id"] for i in batches.select(items, n=10, seed=1)]


def test_write_batches_blinded_with_key(tmp_path):
    items = [{"span_id": f"s{i}", "deployment": "dep", "source": "arena", "messages": record("s", "call")["messages"],
              "action": "call", "verdict": verdict("allin_call", equity=0.4, required=0.5)} for i in range(7)]
    key = batches.write_batches(items, tmp_path, n_batches=3, seed=0)
    files = sorted(tmp_path.glob("batch_*.jsonl"))
    assert len(files) == 3
    rows = [json.loads(line) for f in files for line in f.read_text().splitlines()]
    assert len(rows) == 7 and len({r["id"] for r in rows}) == 7
    blob = json.dumps(rows)
    assert "s0" not in blob and "allin_call" not in blob and "secret" not in blob and "dep" not in blob
    assert set(rows[0]) >= {"id", "state", "action_taken", "facts"}
    saved = json.loads((tmp_path / "answer_key.json").read_text())
    assert saved == key and {k["span_id"] for k in key.values()} == {f"s{i}" for i in range(7)}
    assert key[rows[0]["id"]]["verdict"]["rule"] == "allin_call"
    assert (tmp_path / "INSTRUCTIONS.md").exists()


def test_load_decisions_arena_and_synth(tmp_path):
    arena = tmp_path / "joined.jsonl"
    arena.write_text("\n".join(json.dumps(r) for r in [
        record("a1", "call"), record("a2", "fold", kind="cleanup"), record("a3", "call", source="synth")]))
    synth = tmp_path / "spots.jsonl"
    synth.write_text(json.dumps(record("s1", "fold", source="synth") | {"verdict": verdict(preferred="fold", actual="fold")}))
    got = batches.load_decisions(arena, synth_paths=[synth], verdicts={"a1": verdict()})
    ids = {d["span_id"]: d for d in got}
    assert set(ids) == {"a1", "s1"}
    assert ids["a1"]["verdict"]["preferred"] == "fold" and ids["s1"]["verdict"]["agree"] is True
    assert ids["a1"]["action"] == "call"


# ---------------- merge ----------------

def rating(i, score, better=None, conf="high", reason="r"):
    return {"id": i, "score": score, "better_action": better, "confidence": conf, "reason": reason}


def make_key(specs):
    """specs: list of (id, rule, preferred, actual)."""
    return {i: {"span_id": f"sp_{i}", "rule": r, "actual": a, "verdict": verdict(r, p, a)} for i, r, p, a in specs}


def test_norm_confidence():
    assert merge.norm_confidence("high") == 0.9 and merge.norm_confidence(0.3) == 0.3
    assert merge.norm_confidence("garbage") == 0.0 and merge.norm_confidence(None) == 0.0


def test_critic_preference_uses_score_then_better_action():
    assert merge.critic_preference(rating("x", 9), "call") == "call"
    assert merge.critic_preference(rating("x", 2, "fold"), "call") == "fold"
    assert merge.critic_preference(rating("x", 5, "fold"), "call") is None  # ambiguous middle band
    assert merge.critic_preference(rating("x", 2, "bet"), "call") == "raise"


def test_calibration_per_rule():
    key = make_key([(f"k{i}", "r1", "fold", "call") for i in range(4)])
    key |= make_key([("z", "r1", None, "call")])  # teacher abstains: not counted
    rated = {f"k{i}": rating(f"k{i}", 2, "fold") for i in range(3)} | {"k3": rating("k3", 9)} | {"z": rating("z", 1, "fold")}
    cal = merge.calibration(key, rated)
    assert cal["r1"]["n"] == 4 and cal["r1"]["agree"] == 3 and cal["r1"]["calibration"] == 0.75


def test_weights_drop_upweight_and_corrections():
    spec = [(f"j{i}", "r1", "fold", "call") for i in range(8)]
    spec += [("hi", "r1", None, "call"), ("lo", "r1", None, "call"), ("bad", "r2", None, "call"),
             ("drop", "r1", None, "fold"), ("mid", "r1", None, "call"), ("up", "r1", None, "fold"),
             ("dropweak", "r1", None, "fold"), ("none", "none", None, "call")]
    key = make_key(spec)
    rated = {f"j{i}": rating(f"j{i}", 2, "fold") for i in range(7)} | {"j7": rating("j7", 9)}  # calibration 7/8
    rated |= {"hi": rating("hi", 2, "fold"), "lo": rating("lo", 2, "fold", conf="low"),
              "bad": rating("bad", 2, "fold"), "drop": rating("drop", 3, "call"), "mid": rating("mid", 6),
              "up": rating("up", 9), "dropweak": rating("dropweak", 2, "call", conf="low"),
              "none": rating("none", 1, "fold")}
    out = merge.build_weights(key, rated, min_n=5)
    w = out["weights"]
    assert w["sp_drop"] == 0.0 and w["sp_dropweak"] == 1.0   # drop needs high confidence
    assert w["sp_up"] == 1.5 and w["sp_mid"] == 1.0
    assert w["sp_hi"] == 0.0  # also scored <=3 with high confidence
    assert "sp_unrated" not in w
    corr = {c["span_id"]: c for c in out["corrections"]}
    assert set(corr) == {"sp_hi", "sp_drop"}    # r1 calibrated, teacher abstains, high-conf better action
    assert corr["sp_hi"]["better_action"] == "fold"
    # r2 has no judged spots -> no calibration -> no correction; "none" rule likewise
    assert out["calibration"]["r1"]["calibration"] == 0.875 and "r2" not in out["calibration"]


def test_corrections_need_calibration_and_min_n():
    key = make_key([(f"j{i}", "r1", "fold", "call") for i in range(4)] + [("c", "r1", None, "call")])
    rated = {f"j{i}": rating(f"j{i}", 9) for i in range(4)} | {"c": rating("c", 2, "fold")}  # critic always disagrees
    assert merge.build_weights(key, rated, min_n=2)["corrections"] == []
    key = make_key([(f"j{i}", "r1", "fold", "call") for i in range(2)] + [("c", "r1", None, "call")])
    rated = {f"j{i}": rating(f"j{i}", 2, "fold") for i in range(2)} | {"c": rating("c", 2, "fold")}
    assert merge.build_weights(key, rated, min_n=5)["corrections"] == []
    assert len(merge.build_weights(key, rated, min_n=2)["corrections"]) == 1


def test_load_ratings_skips_bad_and_dedupes(tmp_path):
    p = tmp_path / "rated_1.jsonl"
    p.write_text("\n".join([json.dumps(rating("a", 7)), "not json", json.dumps({"id": "b", "score": 99}),
                            json.dumps(rating("c", "x")), json.dumps(rating("a", 8))]))
    rated, bad = merge.load_ratings([p])
    assert set(rated) == {"a"} and rated["a"]["score"] == 8 and bad == 3


def test_cli_end_to_end(tmp_path):
    key = make_key([(f"j{i}", "r1", "fold", "call") for i in range(5)] + [("c", "r1", None, "call")])
    (tmp_path / "answer_key.json").write_text(json.dumps(key))
    rows = [rating(f"j{i}", 2, "fold") for i in range(5)] + [rating("c", 2, "fold"), rating("ghost", 5)]
    (tmp_path / "rated_a.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "w.json"
    merge.main(["--key", str(tmp_path / "answer_key.json"), "--ratings", str(tmp_path / "rated_a.jsonl"),
                "--out", str(out), "--min-n", "3"])
    assert json.loads(out.read_text())["sp_c"] == 0.0
    assert json.loads((tmp_path / "w.corrections.json").read_text())[0]["span_id"] == "sp_c"
    assert json.loads((tmp_path / "w.calibration.json").read_text())["r1"]["calibration"] == 1.0


def test_teacher_agreement_vetoes_a_critic_drop():
    key = make_key([("tv", "r1", "call", "call"), ("td", "r1", "fold", "call")])
    rated = {"tv": rating("tv", 2, "fold"), "td": rating("td", 2, "fold")}
    w = merge.build_weights(key, rated, min_n=1)["weights"]
    assert w["sp_tv"] == 1.0  # the teacher's math agrees with the action taken: keep it as an anchor
    assert w["sp_td"] == 0.0
