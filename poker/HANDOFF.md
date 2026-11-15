# Finish the poker entry: combined A/B handoff

**Consolidation update:** the user has authorized committing/pushing all completed work and merging A/B onto
the shared `poker` branch. Use `plan.md`, `poker/STATUS.md` and the timestamped `poker/run_ledger.json` for current
instructions and state. Do not reapply the transfer patch in this workspace. `poker-a`/`poker-b` references below
describe the original historical transfer, not the active branch workflow. Read `README (2).md` for the added
strategy priorities, subject to the confirmed event rules and platform capabilities.

This is an execution handoff, not a claim that the project is complete. Person A's commits through `815c9bf`
were pulled with `git pull --ff-only origin poker-a` into local `poker-b`; all uncommitted B work was preserved.
No training job or arena submission was launched by the review.

## Instructions to paste into the other chat

Continue the poker hackathon from the combined A/B state. Read `CLAUDE.md`, the updated `plan.md`,
`poker/STATUS.md`, `poker/CAPABILITIES.md` and this handoff first. You are finishing integration and the training /
evaluation loop, not restarting setup. Preserve local edits and inspect live AC2 state before creating jobs.
Only one training job may run at once. Keep the live deployment running. Do not restart the shared harness
during a run. Arena submission/deployment deletion still needs the explicit user request required by CLAUDE.md.

### 0. Obtain the actual combined code

Person A's latest reviewed remote commit is `815c9bf` on `origin/poker-a`. The B work is still local and
uncommitted in `C:\Projects\ValsHackathon\vals-ai-hackathon-lora` on `poker-b`; it has NOT been pushed.
Do not assume `git pull origin poker-b` will retrieve it.

If this chat shares that workspace, inspect the existing diff and continue there without reapplying the archive.
On another machine, obtain `poker-ab-handoff.zip` from the user. Extract to a staging directory, not over a dirty
checkout. Bring a clean integration checkout to A's commit (or a reviewed descendant), then run
`git apply --check <staging>/code.patch` before applying it. If it fails, inspect the overlap and merge manually;
do not reset or overwrite local work. Copy the staged `data_bundle/out/` contents into the integration checkout's
`out/` directory after checking for existing datasets. The archive excludes credentials, the virtual environment,
raw PokerBench downloads and raw replays. Authenticate on the target machine using its own AC2 login.

Review and commit the combined implementation and small reports before sharing through a reviewed PR. Generated
JSONL stays ignored. Do not publish a branch or PR merely because this document mentions one; follow the user's
authorization for external writes.

### 1. Recheck live state and finish the baseline

Latest observation during this review (recheck; jobs can advance):

- Project: `vals-hackathon`, ID `prj_200dcc41ef41`; pushed project version exists.
- `vals-hackathon/base`: ready, deployment `infer_d3d2b6b92171`.
- `vals-hackathon/ac-vals-model`: configured external house model.
- One harness connector connected. This does not prove which grader revision is loaded in its Python process.
- Base smoke: `eval_992f2c5ead5b`, running on `pokerbots-smoke`.
- Earlier house-model smoke evals succeeded: `eval_15d0c3b30e5b`, `eval_50d9b110f7e1`.
- No training jobs and no base train/eval runs were listed at inspection.
- Starter datasets exist. Game datasets already contain 400 train, 100 select and 200 eval tasks.

Finish or diagnose the existing base smoke; do not create duplicate evals blindly. Verify the always-on harness
has the current placement grader code loaded. A local Git pull or pushed empty AC2 wheel does not update a
running harness process. Apply any harness restart only between jobs, coordinated with its owner.

After smoke succeeds, collect these two distinct base evals using bounded concurrency appropriate to the harness:

```bash
uv run python -m vals_tools.eval --dataset pokerbots-eval --deployment vals-hackathon/base
uv run python -m vals_tools.eval --dataset pokerbots-train --deployment vals-hackathon/base
```

Record IDs, model/deployment revision, dataset identity and grader revision. Only the **train** eval feeds OPSD.
The held-out eval is for comparison. Export traces and report per-decision failures/cutoffs, response-token
median/p99, cleanup failures, and BB per hand. `vals_tools.score` alone does not calculate all those metrics.
The plan's 17/68 cutoff finding concerns the house model; measure base rather than assuming the same failure rate.
Do not infer an exploitable 20–30% failure rate across the field from that house-model sample.

### 2. Close the integration gaps before longer-game training

These are findings from the combined code review, not failing unit tests:

1. **Training dataset selection:** `rl/train.py` always trains on `<prefix>-train`. A's uploaded dataset is
   `pokerbots-train-game`, which cannot be selected through the current prefix convention. Add an explicit
   `--train-dataset` override while retaining the existing default. Use `--eval-dataset pokerbots-select-game`
   for game checkpoint selection. Test the exact serialized launch configuration.
2. **Scoring game objectives:** the grader now emits `place`, but `vals_tools.score` ignores it. Extend reporting
   with mean final place, bust rate from logs where available, failed-game rate, decision-level failure rate and
   paired game comparisons. Its current `hands` count is actually score rows/tasks; on a game dataset each row is
   a whole game, and its EV total is not divided by actual hands. Keep games and hands explicit.
3. **Whole-game token budget:** 24,576 is the original single-hand budget. Measure a small multi-hand pilot's
   complete transcript, including observations and cleanup, before choosing response/context limits and grading
   timeouts. Confirm supported model context and connector payload size (<1 MiB). Do not blindly reuse the
   single-hand settings or multiply beyond supported limits.
4. **Sparse game rewards:** any policy format error currently zeros the entire game reward. Measure the fraction
   of zero-reward games and groups with no reward variance. A low decision failure rate can still spoil many
   30-hand games. Stabilize single hands/short games before moving to longer games; do not quietly weaken the
   failure rule to make a chart improve.
5. **Dataset upload duplication:** `make_game_datasets.py` appends to an existing dataset. Its existing counts are
   correct; do not rerun it blindly. Use versioned names or make upload behavior safely idempotent if changing it.

### 3. Train a small decide-fast candidate first

Follow phase 3 in `plan.md`: retain event thinking and tool rules, teach concise valid decisions, and evaluate
poker quality as well as format success. Begin with a small GRPO pilot/checkpoint on single-hand train tasks;
measure throughput before committing to the full 60-step recipe. Inspect a `--dry-run` configuration first.
Keep the 5,000-token per-decision and 10,000-token cleanup event caps unchanged. Respect the shared 10-GPU quota
and the team's rule of one training job at a time. The existing script estimates 8 GPUs with in-run eval or 6
without, plus live deployments.

Prepare the offline OPSD alternative while the GRPO slot is occupied, but do not launch it concurrently:

- Use only base **train** traces, with verified provenance. Enforce that source in code before an upload; the
  current builder trusts the supplied eval ID/path.
- For decide-fast, select completed decisions under roughly 2,500 tokens (`--max-tokens 2500` already filters
  strictly below that threshold). Reject missing token counts, truncated completions, malformed arguments and
  actions illegal for their prompt. The current one-call/count/cap checks do not establish all those properties.
- Add a dedicated short hint option (`--hint-file` or equivalent) for “strength, position, price, then act.”
  The existing `--generic-hint` is the original tight-aggressive hint, not that decide-fast experiment.
- Keep hints teacher-only and preserve base's actual reply. Keep `env_params.tools` aligned with the trace tools.
- Dry-run and inspect 20 real decisions, including incremental updates. Report parsing failures, missing label
  matches and how much data was filtered out before uploading a versioned hinted dataset.
- Then coordinate the 4-GPU offline OPSD slot, train, and evaluate without the hint.

The existing richer hint builder can be evaluated as a separate experiment. It uses heuristic sampled equity,
pot odds and exact-card/board/seat label matching with nearby prices/stacks. It has tests, not demonstrated
playing-strength benefits. Avoid treating approximate equity as solver truth or promising broad label coverage.

### 4. Evaluate, then advance to placement games

Deploy a candidate checkpoint under a new name, verify it actually answers with `submit_action`, and smoke-test
before large evals. Compare it with base on the same held-out deals. Advance only when failure rates improve
without unacceptable loss in poker results. Use the select sets to choose checkpoints and reserve held-out sets
for final comparisons.

After the integration fixes and measured short-game pilot, train on placement games. Evaluate on 30-hand games
with the placement metric, carried stacks and scratchpad. Compare paired games against base, and retain the
original single-hand held-out comparison as a general check. If a model improves only by exploiting flaky house
opponents, require broader scripted/real-model testing before declaring it better.

### 5. Use B's data and scouting with their actual limitations

- PokerBench: 574,200 schema-valid converted records; 11,000 source-test records separate. Cleaned labels:
  13,565 train and 1,609 validation. All 15,174 passed validation/legality checks. No shared train/validation hand
  families or exact states. Preflop folds are capped at 25% of each split.
- Source hand IDs are absent, so conservative test-family exclusion removes most rows. The retained set is
  heavily preflop; validation has **no postflop examples**. Do not call this a balanced complete poker dataset.
  Do not spend the first training window rebuilding it; decide-fast OPSD can begin with clean base traces.
- Nine public replays yielded early profiles for three teams. Refresh with
  `python -m poker.arena_replays --fetch --limit 20`. Track sample sizes and submission versions; review probe
  flags and avoid confidently calibrating bots from one hand or one game. Give Sunday evidence priority later.
- Original `sim.py`, its helpers and the existing Pluribus conversion were not found locally or in the fetched
  A branch. Ask for an exact location once. If absent, explicitly defer this work; do not fabricate a port or
  delay the main training/evaluation loop waiting for 200k teacher examples. Solver/probe extras are optional.

### 6. Platform constraints

Confirmed plan: managed harness GRPO and data-only offline OPSD. Public platform SFT exists, but our team's SFT
eligibility is unconfirmed. No public arbitrary differentiable-loss/gradient API was found. Requests with
`logprobs: true` to our ready base deployment returned HTTP 400; a matching normal request returned HTTP 200.
Do not build this entry around a custom-loss/raw-training-logprob API. General one_step OPSD/token masking support
does not confirm permission on this event account; obtain a concrete supported interface/access answer first.

### 7. Completion criteria and final handoff

Keep a run ledger with eval/train/checkpoint/deployment IDs, dataset versions, harness commit, settings and
results. Update `plan.md` and `poker/STATUS.md` with observed outcomes. Do not mark training complete after only
preparing a dataset or launching a job.

Finish with a selected, evaluated model; a 30-hand official-rule check with zero failed decisions; a documented
comparison against base in chip results and final rank; and a concrete final deployment ready for A's explicit
submission action. If no trained candidate beats the validated baseline, say so and retain the best validated
entry rather than claiming improvement. Follow the plan's final selection/submission cutoff and confirm its
timezone with the organizers. Keep the submitted deployment alive through the tournament.

## Verification at this integration point

After pulling A: 23 root tests passed; 106 harness tests passed, one skipped. On Windows the harness subprocess
test needs `PYTHONPATH=harness/src`. `poker/make_game_datasets.py --dry-run` generated the expected task counts,
carried stacks and `grader_params.reward=placement`. These checks validate local code, not the remote process's
loaded grader revision or model quality.
