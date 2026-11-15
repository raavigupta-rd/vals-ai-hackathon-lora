"""SFT of Qwen3.6-35B-A3B on poker decisions (AC2 `SFTConfig`), optionally continuing from a prior checkpoint.

SFT trains directly on the conversations stored in the tasks (build them with `build_sft_dataset.py`): no agent,
environment, grader, rollouts or eval run, loss covers the assistant turns only. Every setting this script sends is
a flag below; a flag left unset is not sent, so AC2 uses its own default (learning rate 1e-5 with a cosine decay to
1e-6 and 3% warmup, weight decay 0.1). SFT needs `samples_per_problem=1`, one training dataset and forces eval off.
`apply_chat_template_kwargs` is not accepted for SFT, so the thinking setting cannot be sent here.

Iterate by chaining runs: base -> sft1 -> sft2 with `--base-checkpoint ckpt_...` (a registered checkpoint ID of the
previous run). Training has no poker eval: serve a checkpoint and play it through the harness.

    uv run python poker/sft/train.py --dataset pokerbots-sft1 --dry-run
    uv run python poker/sft/train.py --dataset pokerbots-sft2 --base-checkpoint ckpt_abc --epochs 2 --examples 3000 --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from typing import Any

from ac2.sdk import Client
from ac2.sdk.model import ModelConfig
from ac2.sdk.run_config import (
    ExperimentConfig,
    InternalConfig,
    InternalLaunchConfig,
    RolloutConfig,
    RunConfig,
    SFTConfig,
    StepConfig,
    TasksConfig,
    TrainDatasetConfig,
    TrainerConfig,
)


def _set(**values: Any) -> dict[str, Any]:
    """Only the values that were given: anything None is left to AC2's default."""
    return {key: value for key, value in values.items() if value is not None}


def steps_for(epochs: float, examples: int, batch: int) -> int:
    """Training steps that pass over `examples` tasks `epochs` times at `batch` tasks per step."""
    return max(1, math.ceil(epochs * examples / batch))


def resolve_steps(args: argparse.Namespace) -> int | None:
    if args.steps is not None and args.epochs is not None:
        raise SystemExit("pass --steps or --epochs, not both")
    if args.epochs is None:
        return args.steps
    if args.examples is None:
        raise SystemExit("--epochs needs --examples N (the number of tasks in the dataset, duplicates included)")
    return steps_for(args.epochs, args.examples, args.batch)


def run_config(args: argparse.Namespace, name: str, steps: int | None) -> RunConfig:
    launch = _set(tensorboard_enabled=args.tensorboard or None)  # AC2 places the run; teams can't pick a cluster
    # A registered checkpoint ID alone names the model; ModelConfig rejects model and checkpoint together.
    model = ModelConfig(checkpoint=args.base_checkpoint) if args.base_checkpoint else args.model
    return RunConfig(
        model=model,
        name=name,
        description=f"SFT on poker decisions, {args.dataset}" + (f", from {args.base_checkpoint}." if args.base_checkpoint else "."),
        tags={"benchmark": "pokerbots", "method": "sft", "dataset": args.dataset},
        experiment_cfg=ExperimentConfig(
            method=SFTConfig(),
            step=StepConfig(
                samples_per_problem=1,  # required by SFT
                problem_batch_size=args.batch,
                **_set(
                    num_train_steps=steps,
                    checkpoint_interval=args.checkpoint_every,
                    keep_last_checkpoints=args.keep_checkpoints,
                    seed=args.seed,
                ),
            ),
            tasks=TasksConfig(datasets=(TrainDatasetConfig(ac2_dataset_name=args.dataset),)),
            trainer=TrainerConfig(**_set(learning_rate=args.lr)),
            rollout=RolloutConfig(**_set(max_response_len=args.max_response_len, max_total_len=args.max_total_len)),
        ),
        **_set(internal_cfg=InternalConfig(launch=InternalLaunchConfig(**launch)) if launch else None),
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default="vals-hackathon")
    p.add_argument("--dataset", required=True, help="SFT dataset from build_sft_dataset.py (one dataset only)")
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B", help="base model (arena entries are this model or fine-tunes of it)")
    p.add_argument("--base-checkpoint", help="registered checkpoint ID (ckpt_...) of a prior run to continue from")
    p.add_argument("--dry-run", action="store_true", help="print the settings and configuration, submit nothing")
    g = p.add_argument_group("set by default; change them with these flags")
    g.add_argument("--batch", type=int, default=32, help="tasks per step (default 32)")
    g.add_argument("--no-tensorboard", dest="tensorboard", action="store_false", help="don't record TensorBoard logs")
    g.add_argument("--checkpoint-every", type=int, default=10, help="steps between checkpoints (default 10)")
    a = p.add_argument_group("length of the run (default: AC2's 20 steps)")
    a.add_argument("--steps", type=int, help="training steps")
    a.add_argument("--epochs", type=float, help="passes over the data; needs --examples")
    a.add_argument("--examples", type=int, help="tasks in the dataset, for --epochs")
    o = p.add_argument_group("left to AC2's defaults unless you set them")
    o.add_argument("--lr", type=float, help="peak learning rate (AC2's SFT default 1e-5, cosine to 1e-6)")
    o.add_argument("--max-response-len", type=int, help="token budget after the first prompt (AC2 default 8192)")
    o.add_argument("--max-total-len", type=int, help="prompt plus reply token limit")
    o.add_argument("--keep-checkpoints", type=int, help="checkpoints to keep (the docs say SFT always keeps 20)")
    o.add_argument("--seed", type=int, help="random seed")
    return p


def main() -> None:
    p = parser()
    args = p.parse_args()
    if args.base_checkpoint and not args.base_checkpoint.startswith("ckpt_"):
        p.error("--base-checkpoint must be a registered checkpoint ID such as ckpt_abc")
    steps = resolve_steps(args)
    name = f"pokerbots-sft-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    config = run_config(args, name, steps)
    if args.dry_run:
        print("This launch sets (everything else is AC2's default):")
        skip = {"project", "dry_run", "help", "examples", "epochs"}
        for action in p._actions:  # noqa: SLF001
            value = getattr(args, action.dest, None)
            if action.dest not in skip and value not in (None, False):
                print(f"  {action.dest} = {value}")
        if steps is not None:
            print(f"  num_train_steps = {steps}" + (f" ({args.epochs} epochs of {args.examples} tasks)" if args.epochs else ""))
        print("  samples_per_problem = 1, eval = off (required by SFT)")
        print("GPUs while running: about 4 for Qwen3.6-35B-A3B (training only; SFT has no inference or eval replicas)")
        print(json.dumps(config.to_json(), indent=1, default=str))
        return
    run = Client(project=args.project).train.run(config)
    print(json.dumps({"train_id": run.train_id, "name": name}))


if __name__ == "__main__":
    main()
