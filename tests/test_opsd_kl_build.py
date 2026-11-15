import json

import pytest
from ac2.runtime import FunctionCall, HintedTask, Message, Reasoning

from poker.opsd_kl import build as b

PROMPT = """## Hand 3 of 30
Your seat: 0 (position 1; 0 = small blind)
Street: preflop
Your hole cards: 9s Td
Board: -
Pot (including current bets): 9
Legal actions: fold, call 5, raise to 10-110
Call submit_action with your decision."""


def rec(span="s1", action="call", deployment=b.BASE_DEPLOYMENT, **updates):
    out = {"span_id": span, "deployment": deployment, "cutoff": False, "usage": {"tokens_out": 300},
           "messages": [{"role": "system", "content": "You play poker."}, {"role": "user", "content": PROMPT}],
           "reasoning": "I think about it.", "content": "\n\n",
           "tool_calls": [{"call_id": "c1", "name": "submit_action", "arguments": json.dumps({"action": action})}]}
    return out | updates


def entry(record, agree=None, rule=None):
    verdict = {"agree": agree, "rule": rule, "preferred": "fold", "equity": 0.3, "required": 0.5,
               "to_call": 5, "pot": 9, "street": "preflop"}
    return {"record": record, "verdict": verdict}


def code_for(_):
    return "T9s"


def test_correction_has_hint_and_anchor_has_neutral_hint():
    entries = [entry(rec("bad"), False, "facing_raise"), entry(rec("ok"), True)]
    tasks, stats = b.build(entries, {"bad"}, code_for, anchor_ratio=1.0, correction_copies=1)
    kinds = {t.env_params["kind"]: t for t in tasks}
    assert set(kinds) == {"correction", "anchor"} and stats["tasks"] == 2
    assert "Fold." in kinds["correction"].opsd_hint_selections[0].hint
    assert kinds["anchor"].opsd_hint_selections[0].hint.endswith(b.NEUTRAL)
    assert "Fold." not in kinds["anchor"].opsd_hint_selections[0].hint


def test_input_is_prompt_plus_recorded_reply_and_hint_sits_at_the_system_boundary():
    args = json.dumps({"action": "raise", "amount": 12})
    task = b.hinted_task(rec(tool_calls=[{"call_id": "c1", "name": "submit_action", "arguments": args}]), "x", "anchor")
    sel = task.opsd_hint_selections[0]
    assert sel.rollout_prefix_len == 2
    assert sel.mutation.item_boundary == 1 and sel.mutation.action == "append_to_previous_item"
    assert isinstance(task.input[2], Reasoning) and task.input[2].content == "I think about it."
    call = task.input[-1]
    assert isinstance(call, FunctionCall) and json.loads(call.arguments) == {"action": "raise", "amount": 12}
    assert [t["name"] for t in task.env_params["tools"]] == ["submit_action", "update_scratchpad"]
    assert HintedTask.model_validate(task.model_dump())


def test_copies_bust_rules_and_anchor_ratio():
    entries = [entry(rec("a"), False, "facing_raise"), entry(rec("b"), False, "allin_call")]
    entries += [entry(rec(f"n{i}"), None) for i in range(20)]
    tasks, stats = b.build(entries, {"a", "b"}, code_for, anchor_ratio=3.0, correction_copies=2)
    assert stats["correction_tasks"] == 2 + 3 and stats["anchors"] == 15
    assert sum(t.env_params["kind"] == "anchor" for t in tasks) == 15


def test_anchor_ratio_is_capped_by_the_pool_and_unvalidated_disagreements_are_excluded():
    entries = [entry(rec("a"), False, "facing_raise"), entry(rec("unval"), False, "facing_raise"), entry(rec("n"), None)]
    tasks, stats = b.build(entries, {"a"}, code_for, anchor_ratio=10, correction_copies=1)
    assert stats["anchors"] == 1 and stats["disagree_excluded"] == 1


@pytest.mark.parametrize("change", [{"cutoff": True}, {"tool_calls": []}, {"reasoning": "  "},
                                    {"reasoning": "x" * 20000}, {"usage": {"tokens_out": 5000}}])
def test_unclean_replies_are_never_tasks(change):
    assert b.hinted_task(rec(**change), "x", "anchor") is None
    tasks, stats = b.build([entry(rec(**change))], set(), code_for)
    assert tasks == [] and stats["unclean"] == 1


def test_other_deployments_are_ignored():
    tasks, _ = b.build([entry(rec(deployment="infer_other"))], set(), code_for)
    assert tasks == []


def test_illegal_reply_is_dropped():
    assert b.hinted_task(rec(action="check"), "x", "anchor") is None  # check is not legal facing a bet


def test_anchor_hint_is_a_bare_neutral_line_and_blank_content_is_kept():
    task = b.hinted_task(rec(), b.NEUTRAL, "anchor")
    hint = task.opsd_hint_selections[0].hint
    assert hint == "\n\n" + b.NEUTRAL and "coaching" not in hint.lower()
    assert isinstance(task.input[3], Message) and task.input[3].content == "\n\n"  # replay the exact tokens
    corr = b.hinted_task(rec(), "Fold.", "correction")
    assert "coaching" in corr.opsd_hint_selections[0].hint.lower()


def _say_rec(**kw):
    args = json.dumps({"action": "raise", "amount": 12, "say": "all-in! lets see that king!"})
    return rec(tool_calls=[{"call_id": "c1", "name": "submit_action", "arguments": args}], **kw)


def test_replayed_reply_drops_say_by_default_and_keeps_it_on_request():
    task = b.hinted_task(_say_rec(), "x", "anchor")
    assert json.loads(task.input[-1].arguments) == {"action": "raise", "amount": 12}
    kept = b.hinted_task(_say_rec(), "x", "anchor", strip_say=False)
    assert json.loads(kept.input[-1].arguments)["say"].startswith("all-in")


def test_build_passes_strip_say_to_every_task():
    entries = [entry(_say_rec(span_id="bad"), False, "facing_raise"), entry(_say_rec(span_id="n"), None)]
    tasks, _ = b.build(entries, {"bad"}, code_for, anchor_ratio=1.0, correction_copies=1)
    assert all("say" not in json.loads(t.input[-1].arguments) for t in tasks)
    tasks, _ = b.build(entries, {"bad"}, code_for, anchor_ratio=1.0, correction_copies=1, strip_say=False)
    assert all("say" in json.loads(t.input[-1].arguments) for t in tasks)
