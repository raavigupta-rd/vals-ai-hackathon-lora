import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from opsd.build_dataset import check_provenance, decision_tasks, verify_source
from poker.trace_quality import audit
from vals_tools.eval import eval_config
from vals_tools.score import summarize

ROOT = Path(__file__).resolve().parents[1]
PROMPT = """Your seat: 0 (position 0; 0 = small blind)
Street: preflop
Your hole cards: As Kd
Board: -
Pot (including current bets): 3
Legal actions: fold, call 1, raise to 4-200"""


def span(**changes):
    return {"name": "get_completion", "status": "OK", "start_time": 0, "end_time": 1,
            "span_id": "s", "input": json.dumps([{"type": "message", "role": "user", "content": PROMPT}]),
            "output": json.dumps([{"type": "function_call", "call_id": "c", "name": "submit_action", "arguments": '{"action":"fold"}'}]),
            "tokens_in": 100, "tokens_out": 100, "job_id": "eval_train", "dataset": "pokerbots-train", "datapoint_id": "tsk_a"} | changes


def test_explicit_game_datasets_and_grading_timeouts_are_serialized():
    spec = importlib.util.spec_from_file_location("grpo", ROOT / "rl/train.py")
    grpo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grpo)
    args = grpo.parser().parse_args(["--train-dataset", "pokerbots-train-game", "--eval-dataset", "pokerbots-select-game", "--grading-timeout", "3600"])
    config = grpo.run_config(args, "game").to_json()
    assert config["experiment_cfg"]["tasks"]["datasets"][0]["ac2_dataset_name"] == "pokerbots-train-game"
    assert config["experiment_cfg"]["eval"]["datasets"][0]["ac2_dataset_name"] == "pokerbots-select-game"
    assert config["experiment_cfg"]["tasks"]["custom_harness"]["grading_timeout_seconds"] == 3600
    assert config["tags"]["dataset"] == "pokerbots-train-game"
    assert eval_config("games", deployment="p/base", grading_timeout=7200).custom_harness.grading_timeout_seconds == 7200


def test_game_metrics_use_hands_and_pair_final_place_separately():
    rows = {"a": dict(ev_bb=60, hands=30, realized_bb=30, place=1, policy_decisions=50, policy_format_errors=1, busted=False),
            "b": dict(ev_bb=-10, hands=10, realized_bb=-20, place=6, policy_decisions=10, policy_format_errors=0, busted=True)}
    base = {"a": rows["a"] | dict(place=3, ev_bb=0), "b": rows["b"] | dict(place=5, ev_bb=-20)}
    result = summarize(rows, base)
    assert result["games"] == 2 and result["hands"] == 40
    assert result["ev_bb_per_hand"] == 1.25
    assert result["mean_final_place"] == 3.5 and result["bust_rate"] == .5
    assert result["failed_game_rate"] == .5 and result["decision_failure_rate"] == round(1 / 60, 4)
    assert result["vs_base"]["mean_final_place_diff"] == -.5
    assert "format_error_hands" not in result
    failed = summarize({"failed": {"failure": True}})
    assert failed["games"] == 1 and failed["hands"] == 0 and failed["failed_game_rate"] == 1
    assert failed["ev_bb_per_hand"] is None


@pytest.mark.parametrize("changes,reason", [
    ({"tokens_out": None}, "missing_tokens"), ({"tokens_out": 0}, "missing_tokens"),
    ({"attributes": '{"truncated":true}'}, "truncated"), ({"end_time": ""}, "status"),
    ({"output": json.dumps([{"type":"function_call", "call_id":"c", "name":"submit_action", "arguments":'{"action":"raise","amount":201}'}])}, "illegal"),
    ({"output": json.dumps([{"type":"function_call", "call_id":"c", "name":"submit_action", "arguments":'{"action":"raise","amount":4.5}'}])}, "arguments"),
])
def test_opsd_rejects_unfinished_truncated_unknown_length_and_illegal_actions(changes, reason):
    tasks, stats = decision_tasks({"trace_id": "t", "spans": [span(**changes)]}, "short hint", 2500)
    assert not tasks and stats[reason] == 1


def test_trace_provenance_cannot_be_relabelled_as_training_data():
    trace = {"spans": [span()]}
    check_provenance(trace, "eval_train", "pokerbots-train", {"tsk_a"})
    wrong = copy.deepcopy(trace)
    wrong["spans"][0]["dataset"] = "pokerbots-eval"
    with pytest.raises(ValueError):
        check_provenance(wrong, "eval_train", "pokerbots-train", {"tsk_a"})
    for dataset in ("pokerbots-eval", "pokerbots-select", "pokerbots-test-train"):
        with pytest.raises(ValueError, match="training dataset"):
            verify_source(None, "id", dataset)


def test_quality_report_counts_cutoffs_without_conflating_whole_task_tokens():
    report = audit([{"trace_id": "t", "spans": [span(), span(tokens_out=5000, attributes='{"truncated":true}')] }])
    assert report["counts"]["decision_cutoffs"] == 1
    assert report["decision_cutoff_rate"] == .5
    assert report["tokens"]["task_output"]["sum"] == 5100
    assert report["tokens"]["task_all_request_tokens_upper_bound"]["sum"] == 5300


def test_reward_variance_ignores_retries_and_incomplete_groups():
    from poker.reward_diagnostics import summarize_rewards
    def row(task, sample, value, time):
        return dict(job_id="train_a", generated_step=1, datapoint_id=task, agent_trace_id=sample, score=value, created_at=time, parent_score_id="")
    rows = [row("a", "1", 0, 0), row("a", "2", 0, 0), row("b", "1", 0, 0), row("b", "2", 4, 0), row("c", "1", 0, 0), row("b", "2", 5, 1)]
    result = summarize_rewards(rows, 2)
    assert result["complete_groups"] == 2 and result["incomplete_groups"] == 1
    assert result["no_variance_fraction"] == .5 and result["all_zero_groups"] == 1
    assert result["scored_rollouts"] == 5


def test_bust_status_requires_a_finished_task_and_final_cleanup_evidence():
    from poker.trace_quality import bust_status_by_task
    prompt = [{"type": "message", "role": "user", "content": PROMPT},
              {"type": "message", "role": "user", "content": "Stacks now: seat0 0, seat1 400.\nNow call update_scratchpad exactly once."}]
    trace = {"spans": [span(input=json.dumps(prompt))], "scores": []}
    assert bust_status_by_task([trace]) == {}
    trace["scores"] = [{"score": 1}]
    assert bust_status_by_task([trace]) == {"tsk_a": True}
