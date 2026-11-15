import json

import pytest
from ac2.runtime import FunctionCall, Message, Reasoning, Task

from poker.sft import build_sft_dataset as b

PROMPT = """## Hand 3 of 30
Your seat: 0 (position 1; 0 = small blind)
Street: flop
Your hole cards: Ah Kd
Board: Jd 4h 3d
Pot (including current bets): 10
  seat 0: you, stack 90, bet 0
  seat 1: opp, stack 90, bet 0
Legal actions: check, bet to 2-90
Call submit_action with your decision."""
FACING = PROMPT.replace("check, bet to 2-90", "fold, call 4, raise to 8-90")


def rec(prompt=PROMPT, args=None, **updates):
    out = {
        "messages": [{"role": "system", "content": "You play poker."}, {"role": "user", "content": prompt}],
        "target": {"reasoning": "Top pair, bet for value.", "content": "",
                   "tool_call": {"name": "submit_action", "arguments": args or {"action": "bet", "amount": 6}}},
        "source": "unit", "decision_id": "d1",
    }
    return out | updates


def test_task_layout_and_tools():
    task, copies = b.record_to_task(rec())
    assert copies == 1 and isinstance(task, Task)
    kinds = [(i.type, getattr(i, "role", None)) for i in task.input]
    assert kinds == [("message", "system"), ("message", "user"), ("reasoning", None), ("function_call", None)]
    assert isinstance(task.input[2], Reasoning) and task.input[2].content.startswith("Top pair")
    call = task.input[3]
    assert isinstance(call, FunctionCall) and call.name == "submit_action"
    assert json.loads(call.arguments) == {"action": "bet", "amount": 6}
    assert [t["name"] for t in task.env_params["tools"]] == ["submit_action", "update_scratchpad"]
    assert task.env_params["decision_id"] == "d1"


def test_content_becomes_assistant_message_between_reasoning_and_call():
    task, _ = b.record_to_task(rec(target={**rec()["target"], "content": "Betting."}))
    assert [i.type for i in task.input[2:]] == ["reasoning", "message", "function_call"]
    assert task.input[3].role == "assistant"


@pytest.mark.parametrize("args,reason", [
    ({"action": "fold"}, "illegal"),            # cannot fold when check is free
    ({"action": "bet", "amount": 1}, "illegal"),  # below the minimum
    ({"action": "bet", "amount": 91}, "illegal"),
    ({"action": "bet"}, "arguments"),
    ({"action": "raise", "amount": 6}, "illegal"),  # wrong aggressive kind
    ({"action": "check", "amount": 3}, "arguments"),
    ({"action": "bet", "amount": 6, "bogus": 1}, "arguments"),
    ({"action": "jam"}, "arguments"),
])
def test_illegal_or_malformed_arguments_rejected(args, reason):
    with pytest.raises(b.Rejected) as err:
        b.record_to_task(rec(args=args))
    assert err.value.reason == reason


def test_legal_facing_bet_actions():
    for args in ({"action": "fold"}, {"action": "call"}, {"action": "raise", "amount": 20}):
        b.record_to_task(rec(prompt=FACING, args=args))
    with pytest.raises(b.Rejected):
        b.record_to_task(rec(prompt=FACING, args={"action": "check"}))


def test_say_is_allowed_and_must_be_a_string():
    b.record_to_task(rec(args={"action": "check", "say": "gg"}))
    with pytest.raises(b.Rejected):
        b.record_to_task(rec(args={"action": "check", "say": 5}))


def test_unparseable_prompt_kept_unless_strict():
    odd = rec(prompt="Something else entirely")
    task, _ = b.record_to_task(odd)
    assert task is not None
    with pytest.raises(b.Rejected) as err:
        b.record_to_task(odd, strict=True)
    assert err.value.reason == "parse"


def test_reasoning_budget_and_empty_reasoning():
    long = rec(target={**rec()["target"], "reasoning": "x" * 40_000})
    with pytest.raises(b.Rejected) as err:
        b.record_to_task(long)
    assert err.value.reason == "too_long"
    b.record_to_task(long, max_reasoning_tokens=20_000)
    empty = rec(target={**rec()["target"], "reasoning": "  "})
    with pytest.raises(b.Rejected) as err:
        b.record_to_task(empty)
    assert err.value.reason == "no_reasoning"


def test_exactly_one_submit_action():
    bad = rec(target={**rec()["target"], "tool_call": {"name": "update_scratchpad", "arguments": {"contents": "x"}}})
    with pytest.raises(b.Rejected) as err:
        b.record_to_task(bad)
    assert err.value.reason == "format"
    with pytest.raises(b.Rejected):
        b.record_to_task({**rec(), "target": {"reasoning": "r", "content": ""}})


def test_prompt_must_end_before_the_assistant_turn():
    messages = rec()["messages"] + [{"role": "assistant", "content": "oops"}]
    with pytest.raises(b.Rejected) as err:
        b.record_to_task(rec(messages=messages))
    assert err.value.reason == "format"


def test_weights_by_duplication_are_deterministic():
    assert b.copies_for(None, "a") == 1
    assert b.copies_for(1.0, "a") == 1
    assert b.copies_for(3.0, "a") == 3
    assert b.copies_for(0, "a") == 0
    assert b.copies_for(2.5, "a") == b.copies_for(2.5, "a") in (2, 3)
    assert b.copies_for(100, "a") == b.MAX_COPIES
    hits = sum(b.copies_for(0.5, f"id{i}") for i in range(2000))
    assert 850 < hits < 1150  # fractional part acts as a probability
    with pytest.raises(b.Rejected):
        b.copies_for(-1, "a")


def test_build_tasks_counts_duplicates_and_weights():
    records = [rec(decision_id="a", weight=2), rec(decision_id="b"), rec(decision_id="b"),
               rec(decision_id="c", args={"action": "fold"}), rec(decision_id="z", weight=0)]
    tasks, stats = b.build_tasks(records)
    assert len(tasks) == 3 and stats["kept"] == 2 and stats["duplicate"] == 1
    assert stats["illegal"] == 1 and stats["weight_zero"] == 1 and stats["copies"] == 3


def test_load_records_reports_bad_json(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps(rec()) + "\nnot json\n\n")
    records = list(b.load_records(path))
    assert len(records) == 2 and records[1] is None


def test_token_estimate_is_conservative():
    assert b.estimate_tokens("a" * 350) == 100


def test_cleanup_records_train_an_update_scratchpad_call():
    from poker.sft.build_sft_dataset import record_to_task
    from ac2.runtime import FunctionCall
    record = {"decision_id": "c1", "source": "cleanup_anchor",
              "messages": [{"role": "system", "content": "rules"},
                           {"role": "user", "content": "Hand 3 is over.\nNow call update_scratchpad with the full new contents."}],
              "target": {"reasoning": "Note the reads.", "content": "",
                         "tool_call": {"name": "update_scratchpad", "arguments": {"contents": "S2 tight. S4 shoves wide."}}}}
    task, copies = record_to_task(record)
    call = task.input[-1]
    assert isinstance(call, FunctionCall) and call.name == "update_scratchpad" and copies == 1
    assert json.loads(call.arguments) == {"contents": "S2 tight. S4 shoves wide."}


def test_cleanup_targets_reject_overlong_or_malformed_contents():
    import pytest
    from poker.sft.build_sft_dataset import Rejected, record_to_task
    base = {"decision_id": "c2", "messages": [{"role": "system", "content": "r"}, {"role": "user", "content": "Hand over.\nNow call update_scratchpad now."}],
            "target": {"reasoning": "x", "content": "", "tool_call": {"name": "update_scratchpad", "arguments": {}}}}
    with pytest.raises(Rejected):
        record_to_task(base)
    base["target"]["tool_call"]["arguments"] = {"contents": "y" * 5001}
    with pytest.raises(Rejected):
        record_to_task(base)
