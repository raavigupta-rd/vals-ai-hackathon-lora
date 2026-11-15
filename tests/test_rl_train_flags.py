"""rl/train.py flags: warm start, group filter, mixture, sampling controls."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rl_train_flags", ROOT / "rl" / "train.py")
grpo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grpo)


def cfg(*argv: str) -> dict:
    return grpo.run_config(grpo.parser().parse_args(list(argv)), "n").to_json()


def exp(*argv: str) -> dict:
    return cfg(*argv)["experiment_cfg"]


def test_defaults_do_not_send_new_fields() -> None:
    c = cfg()
    text = json.dumps(c, default=str)
    for key in ("group_filter", '"checkpoint"', "temperature", "sample_timeout", "routing_replay", "filter_truncated", "mixture", "ratio"):
        assert key not in text
    assert c["model"] == "Qwen/Qwen3.6-35B-A3B"
    assert [d["ac2_dataset_name"] for d in c["experiment_cfg"]["tasks"]["datasets"]] == ["pokerbots-train"]


def test_base_checkpoint_sets_model_checkpoint() -> None:
    c = cfg("--base-checkpoint", "ckpt_abc")
    assert c["model"]["checkpoint"] == "ckpt_abc"
    assert "Qwen3.6" not in json.dumps(c["model"])


def test_group_filter_std_and_probe_samples() -> None:
    gf = exp("--group-filter-std", "0.05", "--probe-samples", "4")["method"]["group_filter"]
    assert gf["by_reward_std"] is True
    assert gf["min_reward_std"] == 0.05
    assert gf["probe_samples"] == 4


def test_group_filter_std_alone() -> None:
    gf = exp("--group-filter-std", "0.1")["method"]["group_filter"]
    assert gf["by_reward_std"] is True and gf["min_reward_std"] == 0.1
    assert "probe_samples" not in gf or gf["probe_samples"] is None


def test_probe_samples_needs_filter() -> None:
    with pytest.raises(SystemExit):
        cfg("--probe-samples", "4")


def test_probe_samples_not_above_group() -> None:
    with pytest.raises(SystemExit):
        cfg("--group-filter-std", "0.1", "--probe-samples", "17")


def test_filter_truncated() -> None:
    assert exp("--filter-truncated")["method"]["filter_truncated_samples"] is True


def test_temperature_and_sample_timeout() -> None:
    r = exp("--temperature", "0.9", "--sample-timeout", "1200")["rollout"]
    assert r["temperature"] == 0.9
    assert r["sample_timeout"] == 1200


def test_routing_replay() -> None:
    assert exp("--routing-replay")["trainer"]["use_rollout_expert_routing_replay"] is True


def test_mix_builds_configured_mixture() -> None:
    e = exp("--mix", "spots:3", "--mix", "cleanup:1.5")
    ds = e["tasks"]["datasets"]
    assert [(d["ac2_dataset_name"], d["ratio"]) for d in ds] == [("spots", 3.0), ("cleanup", 1.5)]
    assert e["tasks"]["mixture"] == "configured"


def test_mix_with_train_dataset_keeps_it_at_ratio_one() -> None:
    ds = exp("--train-dataset", "main", "--mix", "extra:2")["tasks"]["datasets"]
    assert [(d["ac2_dataset_name"], d["ratio"]) for d in ds] == [("main", 1.0), ("extra", 2.0)]


def test_train_dataset_alone_still_works() -> None:
    ds = exp("--train-dataset", "main")["tasks"]["datasets"]
    assert [d["ac2_dataset_name"] for d in ds] == ["main"]


@pytest.mark.parametrize("bad", ["nocolon", "a:x", "a:0", "a:-1", ":2"])
def test_mix_rejects_bad_values(bad: str) -> None:
    with pytest.raises(SystemExit):
        grpo.parser().parse_args(["--mix", bad])
