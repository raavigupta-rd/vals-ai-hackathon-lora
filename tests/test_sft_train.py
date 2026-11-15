import json
import subprocess
import sys
from pathlib import Path

import pytest

from poker.sft import train

ROOT = Path(__file__).resolve().parents[1]


def args_for(*argv):
    return train.parser().parse_args(["--dataset", "d", *argv])


def test_steps_for_epochs():
    assert train.steps_for(2, 100, 32) == 7
    assert train.steps_for(0.01, 10, 32) == 1


def test_epochs_need_examples_and_exclude_steps():
    with pytest.raises(SystemExit):
        train.resolve_steps(args_for("--epochs", "2"))
    with pytest.raises(SystemExit):
        train.resolve_steps(args_for("--epochs", "2", "--examples", "10", "--steps", "3"))
    assert train.resolve_steps(args_for("--epochs", "2", "--examples", "100")) == 7


def test_config_is_sft_with_one_sample_and_no_eval_fields():
    a = args_for("--lr", "2e-5")
    cfg = train.run_config(a, "n", train.resolve_steps(a)).to_json()
    exp = cfg["experiment_cfg"]
    assert exp["method"]["type"] == "sft"
    assert exp["step"]["samples_per_problem"] == 1 and exp["step"]["problem_batch_size"] == 32
    assert exp["trainer"]["learning_rate"] == 2e-5
    assert "apply_chat_template_kwargs" not in exp["tasks"]
    assert cfg["model"] == "Qwen/Qwen3.6-35B-A3B"


def test_base_checkpoint_replaces_model():
    a = args_for("--base-checkpoint", "ckpt_abc")
    cfg = train.run_config(a, "n", None).to_json()
    assert "ckpt_abc" in json.dumps(cfg["model"])


def test_dry_run_prints_config_and_submits_nothing():
    out = subprocess.run([sys.executable, str(ROOT / "poker/sft/train.py"), "--dataset", "d", "--dry-run"],
                         capture_output=True, text=True, cwd=ROOT, check=True).stdout
    assert "This launch sets" in out and "GPUs while running" in out
