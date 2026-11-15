"""RL environment over any harness-format problems.jsonl (e.g. data/synth_ioi/problems.jsonl).

Each EnvGroupBuilder runs `group_size` rollouts of one problem through the Vals-like env
(run_cpp + submit tools, EXIT/turn-limit/context-overflow/parse-error endings).

Reward is computed once per rollout in `compute_group_rewards`, NOT inside the env, so every
ending is graded the same way the real grader does: the last submission that compiled is
judged on hidden tests (small 40 + large 60, all-or-nothing per half by default).

    reward = score / 100
           + turn_bonus * max(0, 1 - turns / turn_bonus_horizon)   if score == 100   (tie-breaker: fewer turns)

Never train on the problems you evaluate on: keep a held-out problems file for harness.eval.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

import chz

from tinker_cookbook import model_info, renderers
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, Metrics, RLDataset, RLDatasetBuilder, Trajectory
from tinker_cookbook.tokenizer_utils import get_tokenizer

from harness import sandbox
from harness.env import THINK_SHORT, build_env, grade, hint_text
from harness.problems import load


async def _no_reward(_history) -> tuple[float, dict[str, float]]:
    return 0.0, {}


@dataclass
class CPEnvGroupBuilder(EnvGroupBuilder):
    problem: dict
    model_name: str
    renderer_name: str
    group_size: int
    max_turns: int = 100
    max_trajectory_tokens: int = 48_000
    max_generation_tokens: int = 16_384
    score_mode: str = "subtask"
    turn_bonus: float = 0.04
    turn_bonus_horizon: int = 10
    hint: str = ""  # eval-only truncation hint appended to the statement
    tag: str = "train"
    # "think short" system instruction given to each rollout with this probability (annealed to 0 over
    # training, so the habit is learned without the cue; held-out eval never gets it)
    instruct_prob: float = 0.0
    seed: int = 0
    _msg_envs: list = field(default_factory=list, repr=False)

    async def make_envs(self) -> Sequence[Env]:
        renderer = renderers.get_renderer(self.renderer_name, get_tokenizer(self.model_name))
        envs = []
        self._msg_envs = []
        self._instructed = []
        rng = random.Random(self.seed)
        for _ in range(self.group_size):
            instructed = rng.random() < self.instruct_prob
            self._instructed.append(instructed)
            env, msg_env = build_env(
                self.problem, renderer, max_turns=self.max_turns,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens, hint=self.hint,
                system_extra=THINK_SHORT if instructed else "",
            )
            msg_env.reward_fn = _no_reward  # graded in compute_group_rewards instead
            envs.append(env)
            self._msg_envs.append(msg_env)
        return envs

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        out = []
        for i, env in enumerate(env_group):
            msg_env = env.message_env
            tools = msg_env.cp_tools
            g = await grade(self.problem, tools.last_compiled_code, self.score_mode)
            score = g["score"]
            turns = msg_env._turn_count
            bonus = 0.0
            if score == 100.0 and self.turn_bonus:
                bonus = self.turn_bonus * max(0.0, 1.0 - turns / self.turn_bonus_horizon)
            out.append((score / 100.0 + bonus, {
                "score": score,
                "solved": float(score == 100.0),
                "small_only": float(score == 40.0),
                "submitted": float(g["submitted"]),
                "compiled": float(g["compiled"]),
                "turns": float(turns),
                "n_submissions": float(tools.n_submissions),
                "n_runs": float(tools.n_runs),
                "turn_bonus": bonus,
                "instructed": float(self._instructed[i]) if i < len(getattr(self, "_instructed", [])) else 0.0,
                "instruct_prob": self.instruct_prob,
            }))
        return out

    def logging_tags(self) -> list[str]:
        return [self.tag]


class CPDataset(RLDataset):
    def __init__(self, builders: list[CPEnvGroupBuilder], batch_size: int):
        self.builders = builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return self.builders[index * self.batch_size:(index + 1) * self.batch_size]

    def __len__(self) -> int:
        return len(self.builders) // self.batch_size  # drop the ragged last batch


@chz.chz
class CPDatasetBuilder(RLDatasetBuilder):
    problems_path: str
    batch_size: int  # groups (problems) per step
    group_size: int  # rollouts per problem
    model_name: str = "Qwen/Qwen3.5-9B"
    renderer_name: str | None = None
    eval_problems_path: str | None = None
    epochs: int = 1
    seed: int = 0
    max_turns: int = 100
    max_trajectory_tokens: int = 48_000
    max_generation_tokens: int = 16_384
    score_mode: str = "subtask"
    turn_bonus: float = 0.04
    turn_bonus_horizon: int = 10
    exclude_ids: str = ""  # comma-separated problem ids to skip (e.g. eval overlap)
    eval_group_size: int = 4  # unhinted rollouts per held-out problem (tag "heldout")
    solutions_path: str | None = None  # if set, add 1 hinted rollout per held-out problem (tag "hinted")
    hint_min: float = 0.2
    hint_max: float = 0.8
    think_short_p0: float = 1.0  # initial probability of the think-short instruction per training rollout
    think_short_anneal_steps: int = 6  # linearly decays to 0 by this step (0 disables the instruction)

    def _builders(self, problems: list[dict], group_size: int, renderer_name: str, tag: str = "train",
                  hints: list[str] | None = None) -> list[CPEnvGroupBuilder]:
        return [CPEnvGroupBuilder(tag=tag, hint=(hints[i] if hints else ""),
            problem=p, model_name=self.model_name, renderer_name=renderer_name, group_size=group_size,
            max_turns=self.max_turns, max_trajectory_tokens=self.max_trajectory_tokens,
            max_generation_tokens=self.max_generation_tokens, score_mode=self.score_mode,
            turn_bonus=self.turn_bonus, turn_bonus_horizon=self.turn_bonus_horizon,
        ) for i, p in enumerate(problems)]

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        sandbox.ensure_container()
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(self.model_name)
        skip = {s for s in self.exclude_ids.split(",") if s}
        problems = [p for p in load(self.problems_path) if p["id"] not in skip]
        if self.eval_problems_path:
            eval_ids = {p["id"] for p in load(self.eval_problems_path)}
            overlap = eval_ids & {p["id"] for p in problems}
            if overlap:
                raise ValueError(f"train/eval overlap on {len(overlap)} problems, e.g. {sorted(overlap)[:5]}")
        rng = random.Random(self.seed)
        order = []
        for _ in range(self.epochs):
            ep = problems[:]
            rng.shuffle(ep)
            order += ep
        builders = self._builders(order, self.group_size, renderer_name)
        for i, b in enumerate(builders):
            step = i // self.batch_size
            b.seed = self.seed * 100_003 + i
            if self.think_short_anneal_steps > 0:
                b.instruct_prob = max(0.0, self.think_short_p0 * (1 - step / self.think_short_anneal_steps))
        train = CPDataset(builders, self.batch_size)
        test = None
        if self.eval_problems_path:
            ev = load(self.eval_problems_path)
            builders = self._builders(ev, self.eval_group_size, renderer_name, tag="heldout")
            if self.solutions_path:
                sols = {s["id"]: s for s in map(json.loads, open(self.solutions_path))}
                hv = [p for p in ev if p["id"] in sols]
                n = len(hv)
                levels = [self.hint_min + (self.hint_max - self.hint_min) * i / max(1, n - 1) for i in range(n)]
                hints = [hint_text(sols[p["id"]]["fast_cpp"], h) for p, h in zip(hv, levels)]
                builders += self._builders(hv, 1, renderer_name, tag="hinted", hints=hints)
            test = CPDataset(builders, len(builders))
        print(json.dumps({"train_problems": len(problems), "epochs": self.epochs, "steps": len(train),
                          "rollouts_per_step": self.batch_size * self.group_size}))
        return train, test
