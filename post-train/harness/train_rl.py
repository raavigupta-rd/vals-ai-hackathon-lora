"""GRPO-style RL on harness-format problems with Tinker (tinker_cookbook.rl.train).

    uv run python -m harness.train_rl problems_path=data/synth_ioi/train.jsonl \
        eval_problems_path=data/synth_ioi/heldout.jsonl max_steps=5

Defaults are sized for a ~$100 budget: 8 problems x 4 rollouts = 32 rollouts/step.
Rough cost: ~1.5M sampled + ~1.5M trained tokens/step  ->  ~$3-4 per step on Qwen3.5-9B.
Groups where every rollout gets the same reward carry no signal and are dropped before training.
Checkpoints (incl. sampler weights for submission) are listed in <log_path>/checkpoints.jsonl.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

import chz

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from config import BASE_MODEL, PROJECT_ID  # noqa: E402

os.environ.setdefault("TINKER_PROJECT_ID", PROJECT_ID)

from tinker_cookbook import checkpoint_utils, cli_utils  # noqa: E402
from tinker_cookbook.rl.rollout_strategy import RetryOnFailure  # noqa: E402
from tinker_cookbook.rl.train import Config, main  # noqa: E402

from harness.rl_env import CPDatasetBuilder  # noqa: E402


@chz.chz
class CLIConfig:
    problems_path: str = str(ROOT / "data" / "synth_ioi" / "problems.jsonl")
    eval_problems_path: str | None = None
    exclude_ids: str = ""
    solutions_path: str | None = str(ROOT / "data" / "synth_ioi" / "solutions.jsonl")  # for hinted eval
    eval_group_size: int = 4

    model_name: str = BASE_MODEL
    load_checkpoint_path: str | None = None  # e.g. an SFT checkpoint (tinker://.../weights/...)
    renderer_name: str | None = "qwen3_5"  # thinking ON
    lora_rank: int = 32

    group_size: int = 4
    groups_per_batch: int = 8
    epochs: int = 10
    learning_rate: float = 2e-5
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0
    max_steps: int | None = None

    max_tokens: int = 32_768  # per turn (runner caps each turn at the remaining context)
    max_trajectory_tokens: int = 64_000  # Qwen3.5-9B on Tinker has a 64K context
    think_short_p0: float = 1.0
    think_short_anneal_steps: int = 6
    max_turns: int = 100
    score_mode: str = "subtask"
    turn_bonus: float = 0.04
    turn_bonus_horizon: int = 10
    remove_constant_reward_groups: bool = True

    save_every: int = 5
    eval_every: int = 2
    log_path: str | None = None
    wandb_project: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"
    seed: int = 0


async def cli_main(c: CLIConfig) -> None:
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=c.model_name, explicit_renderer_name=c.renderer_name, load_checkpoint_path=c.load_checkpoint_path,
    )
    run_name = (f"cprl-{Path(c.problems_path).stem}-g{c.group_size}x{c.groups_per_batch}-lr{c.learning_rate}-"
                f"{datetime.now().strftime('%m%d-%H%M')}")
    log_path = c.log_path or str(ROOT / "runs" / "rl" / run_name)
    dataset_builder = CPDatasetBuilder(
        problems_path=c.problems_path, eval_problems_path=c.eval_problems_path, exclude_ids=c.exclude_ids,
        batch_size=c.groups_per_batch, group_size=c.group_size, model_name=c.model_name,
        renderer_name=renderer_name, epochs=c.epochs, seed=c.seed, max_turns=c.max_turns,
        max_trajectory_tokens=c.max_trajectory_tokens, max_generation_tokens=c.max_tokens,
        score_mode=c.score_mode, turn_bonus=c.turn_bonus, turn_bonus_horizon=c.turn_bonus_horizon,
        solutions_path=c.solutions_path, eval_group_size=c.eval_group_size,
        think_short_p0=c.think_short_p0, think_short_anneal_steps=c.think_short_anneal_steps,
    )
    config = Config(
        learning_rate=c.learning_rate, dataset_builder=dataset_builder, model_name=c.model_name,
        recipe_name="cp_rl", renderer_name=renderer_name, lora_rank=c.lora_rank, max_tokens=c.max_tokens,
        temperature=c.temperature, kl_penalty_coef=c.kl_penalty_coef, log_path=log_path,
        load_checkpoint_path=c.load_checkpoint_path, save_every=c.save_every, eval_every=c.eval_every,
        max_steps=c.max_steps, remove_constant_reward_groups=c.remove_constant_reward_groups,
        wandb_project=c.wandb_project, wandb_name=run_name, session_name=run_name,
        rollout_error_tolerance=RetryOnFailure(max_retries=1),
    )
    cli_utils.check_log_dir(log_path, behavior_if_exists=c.behavior_if_log_dir_exists)
    print(f"log_path={log_path}")
    await main(config)


if __name__ == "__main__":
    asyncio.run(cli_main(chz.entrypoint(CLIConfig)))
