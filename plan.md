# Poker data plan

Vals Hackathon, poker track. Written Sat Nov 14, 1:45 PM, revised against the starter README the same day.
Submissions close Sun Nov 15, 2 PM.

This plan covers data only. Training, RL, and deployment get their own plans, but this one has to feed the two
training paths the starter actually supports (GRPO and offline OPSD), so it names them where it matters.

## Team and setup

Two people. Each step below names its owner as A or B.

| | A: infra and RL (name: ___) | B: data and scouting (name: ___) |
|---|---|---|
| Machine | The always-on cloud box: harness, connector, deployments | Laptop: local repo, PokerBench download, local harness tests |
| Steps | 1, 6, 7, 9a, 9d, 9f, all GRPO runs | 2, 3, 4, 5, 8, 9b, 9c, 9e, 10 |
| Owns | `harness/`, `grading.py`, `rl/`, the arena entry | `poker/`, `opsd/`, opponent profiles, the probe log |

Rules between us
- Each person works on their own branch off `poker` (`poker-a`, `poker-b`) and merges back through a reviewed PR.
- Only A submits to the arena, so the live entry is never a surprise. B asks A for probe windows.
- Only one training job runs at a time. Agree in chat whose job is next before launching.
- A restarts the harness only between runs, since a running run grades with whatever harness is up.

### First hour

1. Both: `ac2 login`, `uv sync`, `(cd harness && uv sync)`, then `ac2 mcp install --agent claude` and restart Claude Code. The AC2 MCP lets Claude read runs, evals and traces directly.
2. A: get the always-on box and run README steps 1 to 6 on it.
3. A: deploy `base` and submit it to the arena (step 9d), so it collects games while we build.
4. B: start step 8 (replay scouting) and download PokerBench (step 1).
5. A: start 9a, the placement reward, tests first.

### GPU budget (10 GPUs, shared by both of us)

| Job | GPUs | Notes |
|---|---|---|
| Arena entry (base, probe or trained) | 1 | Always running. Never delete it. |
| RL run with `--no-eval` | 6 | A's lane. Evaluate checkpoints with a separate deployment. |
| Eval deployment or probe build | 1 to 2 | Delete when idle |
| OPSD run | 4 | B's lane. Only when no RL run is going. |

### Claude setup

Both of us run Claude Code on Opus 5.5 as the main session. The main session plans, makes design calls, and reviews.
It hands bulk work to subagents and checks their output before using it.

Which agent does what

| Task | Agent | Model |
|---|---|---|
| Reward design, plan changes, picking the final model, reviewing reward or harness diffs | Main session | Opus 5.5 |
| PokerBench converter, replay parser, task builders, repeated edits, running pytest | `sonnet-worker` | Sonnet 5.5 |
| Finding code or conventions across the repo | `Explore` | Sonnet 5.5 |
| Reading web pages and docs (AC2 docs, pokerbots.ac pages) | `web-fetch` | Sonnet 5.5 |
| Labeling replay decisions in bulk, after code filters cut the data down | `sonnet-worker` with `model: haiku` | Haiku 5.5 |
| Review before merging into `poker` | `code-reviewer` | Sonnet 5.5 |

Rules for agents
- For data work, write deterministic filters first (regex, dedupe, legality checks). Send only what is left to a model.
- Run independent subagents in parallel, in one message.
- Use `/fast` for quick fix-and-retry loops during setup.

Skills by task

| Task | Skill |
|---|---|
| 9a placement reward and any grader change | `superpowers:test-driven-development` (the pytest suite already exists) |
| Harness, connector or rollout failures | `superpowers:systematic-debugging` |
| Building a plan step with workers | `superpowers:subagent-driven-development` |
| Reading pokerbots.ac replays if there is no API | `claude-in-chrome` |
| Opponent profiles and their confidence | `data:explore-data`, `data:statistical-analysis` |
| One shared opponent-profile page for both of us | `data:build-dashboard` |
| Watching a training run | `/loop`, for example `ac2 train get <train id>` every 20 minutes |
| Before merging reward or harness changes | `code-review` |
| Commits | `commit` (never pushes) |

## What the README settles

The first draft had six open questions. The README answers most of them.

| Question | Answer from the README | What it changes |
|---|---|---|
| Heads-up or 6-max? | 6-max, seats `seat0` to `seat5` | Drop every heads-up source (ACPC) and the heads-up teacher runs |
| Stack depth and blinds? | Blinds 1/2, 200 chips (100 BB) to start. A game is 30 hands, chips carry over, no rebuys, a busted seat is out | Stacks drift during a game, so we need data at many depths, not just 100 BB |
| What is the goal? | Rank at the end of the game (OpenSkill), plus bb/100. Seats still in rank above busted seats | Busting is the worst outcome. Survival matters more than chip EV in big all-in spots |
| State and action format? | The harness renders the prompt. The model answers with one `submit_action(action, amount, say)` tool call. Amounts are "to" totals | We do not write our own renderer. We take prompts from the harness (`harness/src/poker_harness/prompts.py`) or from traces |
| Reasoning and time limit? | Qwen thinking is on. No time limit, but 5,000 output tokens per decision and 10,000 for the cleanup turn. A cut-off reply is check or fold | Long reasoning is a real risk. Data and hints must push toward short, decisive thinking |
| Can we see opponent hands? | Arena games and replays are on pokerbots.ac. In a hand, a seat sees only the current hand plus its own scratchpad | Opponent study comes from arena replays. Cross-hand memory must go through the 5,000-character scratchpad |

Still open, to ask the organizers:
- Is plain SFT available on team orgs? The README documents only GRPO and data-only offline OPSD ("BYOH-only"). If SFT is not available, a JSONL of (state, action) labels cannot be trained on directly. See "Where the data goes" below.
- Are outside datasets (PokerBench, Pluribus hands) allowed?
- Can arena replays be downloaded in bulk, or only viewed?
- How many submissions are allowed per hour? The README says only that the latest submission counts.

## Where the data goes

Without SFT, labeled decisions have three uses. This is the main change from the first draft.

| Use | How the data enters | Starter code |
|---|---|---|
| A. Offline OPSD hints | Each decision in a base-model trace gets its own hint built from our labels (equity, the PokerBench or teacher action, pot odds). The teacher is base plus that hint, so the student learns to act as if it knew the label. | `opsd/build_dataset.py` already takes a `hint` per decision. Today it passes one global `HINT`. We change it to compute the hint per decision. |
| B. RL task design | Pick deck seeds, seats, stack depths and hand counts so GRPO trains on the spots that matter (short stacks, 3-bet pots, multi-hand games with the scratchpad). | Tasks are `env_params` in the format of `vals_tools/tasks.py`. `harness/src/poker_harness/spots.py` can find seeds by hole cards. |
| C. Offline evals | Agreement with PokerBench test labels and the teacher, plus format checks, before we spend GPU time on live evals. | `vals_tools.score` for the live part |

If SFT turns out to be available, the labels can also be used directly, rendered through the harness prompt and
answered as a `submit_action` tool call with a short thinking block.

Offline OPSD only reweights tokens base already wrote. A hint that asks for something base never does needs
`rollout_mode="one_step"`. Keep that in mind when hints disagree with base often.

## Sources

| # | Source | Size | Role | Status |
|---|---|---|---|---|
| 1 | Teacher bot (our own `sim.py`) | Unlimited | Labels for hints (use A), spot finder (use B) | Not in this repo. Move it into `poker/` and switch it to 6-max, 200 chips |
| 2 | Pluribus hands (PHH dataset, MIT license) | 15,169 decisions from 10,000 hands, 6-max, 100 BB | Strong-play labels for hints | Converted elsewhere (`out/pluribus.jsonl`), not in this repo |
| 3 | PokerBench (solver labels) | 60k preflop and 500k postflop train spots, 6-max | Solver-quality labels for hints | Not downloaded yet |
| 4 | PokerBench test sets | 1k preflop and 10k postflop | Offline eval only, never in hints or tasks | Not downloaded yet |
| 5 | Base model traces on `pokerbots-train` | One eval of 2,000 hands | The replies OPSD trains on | Needs the `base` deployment (README step 7 and 10) |
| 6 | Arena replays (pokerbots.ac) | Grows all event | Opponent study and scratchpad notes | Not started |
| 7 | Solver spots we make (TexasSolver or postflop-solver) | Minutes per spot | Optional, for short-stack spots | Not started |

Dropped: ACPC bot hands (heads-up only) and the PokerStars 2009 human hands. The opponents are other teams' Qwen
models, not humans, so arena replays are a better guide to them.

Notes
- PokerBench must be downloaded on a laptop. Hugging Face is blocked from the Claude cloud workspace.
- TexasSolver is AGPL-3.0. We only use its output, and we do not copy its code into this repo.

## Record schema

Every label source converts to the same record. Put the schema in `poker/schemas/decision_record.schema.json`.

```json
{
  "id": "tch-7-123",
  "source": "teacher | solver | pokerbench | phh | trace",
  "format": {"players": 6, "sb": 1, "bb": 2, "starting_stack": 200},
  "state": {
    "hand_index": 4, "street": "flop", "hole": ["As", "Kd"], "board": ["Ah", "7c", "2d"],
    "seat": 3, "position": "BTN", "pot": 12, "to_call": 0,
    "stacks": [180, 240, 200, 190, 150, 240], "effective_stack": 180,
    "legal": {"fold": true, "check": true, "call": 0, "raise_min": 2, "raise_max": 190},
    "history": [{"street": "preflop", "seat": 3, "action": "raise", "to": 5}]
  },
  "action": {"type": "raise", "to": 8},
  "reason": "Top pair top kicker, value bet 0.66 pot.",
  "meta": {"equity": 0.84},
  "solver_freqs": null
}
```

Rules
- Amounts are "to" totals for the street, in chips, as in the harness.
- Stacks are per seat, so records can carry depths other than 200 chips.
- `reason` stays one short line. It feeds hints, and the model must not learn to think past 5,000 tokens.
- We do not store prompt text. The harness renders prompts, so we take them from traces or `prompts.py`.

## Pipeline

All new code goes in `poker/`. `src/pokerbots/` is the AC2 project package and stays empty on purpose.

### Step 1. Download and baseline (owner: A, 30 min)

```
pip install huggingface_hub
huggingface-cli download RZ412/PokerBench --repo-type dataset --local-dir data/pokerbench
```

In parallel, the deployment lane deploys `base` and runs the base eval on `pokerbots-eval` and on `pokerbots-train`
(README steps 7 and 10). The train eval gives us source 5.

Done when PokerBench is in `data/pokerbench/` and we have both base eval ids.

### Step 2. Convert PokerBench (owner: B, 1 hour)

- Write `poker/pokerbench_to_records.py`. Use the CSV files, because they hold structured fields (position, board, pot, action history, hole cards, correct decision).
- Nobody has inspected the files, so budget time for surprises.
- PokerBench may not give full stacks. Fill what it has and leave the rest null.
- Keep train and test apart. Test goes to `out/eval/`.

Done when 100% of records pass the schema and a spot check of 20 records matches the source prompts.

### Step 3. Port and run the teacher (owner: B, 45 min)

- Copy `sim.py` and its helpers into `poker/`. Switch it to 6 seats, blinds 1/2, and 200-chip stacks.
- Add stack variety: run it at 20, 50, 100 and 200 BB effective, since stacks drift over 30 hands.
- Check its legality against the official engine (`harness/src/poker_harness/engine/`), not only its own rules.
- Do not filter records by whether the hand won. One hand result is mostly luck.

Done when we have at least 200k 6-max records with zero illegal actions under the official engine.

### Step 4. Clean and split (owner: B, 45 min)

Write `poker/build_records.py`.
1. Validate every record against the schema. Drop and count failures.
2. Check that each action is legal for its own `legal` field.
3. Remove exact duplicates.
4. Split by hand id, so one hand never lands in two splits.
5. Cap preflop folds at about 25% of records.
6. Write `out/train.jsonl`, `out/val.jsonl`, and `stats.json` with counts by source, street, action and stack depth.

### Step 5. Per-decision OPSD hints (owner: B, 1.5 hours)

- Change `opsd/build_dataset.py` so the hint is computed per decision instead of the global `HINT`.
- For each decision in the base train trace, parse the state from the prompt, then build a short hint: equity against a sensible range (`harness/src/poker_harness/engine/equity.py`), pot odds, and the label of the nearest PokerBench or teacher spot.
- Add fixed rules that every hint carries: keep thinking short, always end with one `submit_action` call, and avoid marginal all-ins that risk busting.
- Keep the README checks: one `submit_action` call, reply under the token cap, `env_params.tools` from `opsd/tools.json`.

Done when the builder uploads `pokerbots-opsd-train` and a dry run shows sensible hints for 20 random decisions.

### Step 6. RL task sets (owner: A, 1 hour)

Build extra datasets in the `vals_tools/tasks.py` format. Keep the starter splits as they are, so evals stay
comparable with base.

| Dataset | What it adds |
|---|---|
| `pokerbots-train-depth` | Single hands at 20 to 200 BB (`starting_stack`) |
| `pokerbots-train-game` | Multi-hand games (`total_hands` 6 to 10, `reset_stacks: false`) so the scratchpad is read and busting is possible |
| `pokerbots-train-spots` | Seeds chosen with `spots.py` for hands the model plays badly in traces |

Check every multi-hand task stays under the 1 MiB connector limit and that the GRPO token budget
(`--max-response-len`) covers the longer games.

### Step 7. Eval sets (owner: A, 30 min)

| Set | Use |
|---|---|
| PokerBench test (11k spots) | Fast offline accuracy check of checkpoint traces |
| `pokerbots-eval` vs the base eval | The real verdict: `vals_tools.score --base`, paired per deal |
| `play_local.py --hands 30` | Format check under the event rules before any submission: 0 failed decisions |
| `play_local.py --opponents bots` | Stress test against scripted `maniac`, `nit` and `calling_station` players |

### Step 8. Arena scouting (owner: B, ongoing from Saturday evening)

- Read replays on pokerbots.ac. Note how other entries react to large bets and shoves, how often they fold to aggression, and how often their decisions fail (cut off, illegal).
- Turn what we see into (a) short reads the model can write to its scratchpad and (b) hint text for step 5.
- Replays cover every game, not only ours, so this needs no special entry of our own.

### Step 9. Exploit the field (owner: A for 9a, 9d, 9f; B for 9b, 9c, 9e, from Saturday evening)

The goal is the best final placement against the entries actually in the arena, not GTO play or chip profit.

**What the judges said (Sat Nov 14).**
1. Submissions are unlimited.
2. Entries that play a fixed style to study other entries are allowed.
3. Each new submission starts with a fresh rating, so games a probe loses cost nothing.
4. The leaderboard before 2 PM is only for development. It does not count.
5. Final standings come from one Official Tournament after 2 PM, played by every team's last submission before the cutoff.
6. Test the baseline model in the arena too, to see where it is weak and improve it.

What this means for us:
- Probes are free. Run as many as GPUs allow, at any time before the final entry.
- The ladder field is not the tournament field. The ladder mixes real models with other teams' probes and test builds. The tournament has only each team's final model, and those come in late. Profiles must filter out probe-like entries and weight Sunday games most.
- Auto-pause does not apply in Official Tournaments, but a failed decision is still a forced check or fold.

The tournament format is not given. What we know: at most 15 teams, so at most 15 models, one live entry per team.
- Plan for the unknown: assume several 30-hand games with 6 of the 15 models drawn per table. The model must do well against any mix, not one table.
- With at most 14 opponents, we can profile each team, not just broad styles. Track profiles by team (team pages on the arena site), since a team's final model often plays like its earlier ones.
- One live entry per team means a probe replaces our real model while it plays. Alternate them: probe windows and real-model windows, logged in the 9e probe log.

**9a. Placement reward.** Fork `grading.py` so a 30-hand game (`total_hands: 30`, `reset_stacks: false`) scores the
policy seat's final rank, with busting scored lowest and later busts above earlier ones. Keep failures at 0 and real
results above them. Test with `cd harness && uv run pytest`. This does not depend on the judges' answers and is
likely the biggest single gain.

**9b. Opponent profiles from replays.** From the step 8 scouting, measure per team (at most 14): VPIP, PFR, fold to
a shove, fold to a bet of pot or more, how often they bust, and how often their decisions fail. Keep one profile per
team, and also group them into a few styles for training.

**9c. A simulated field.** Tune the scripted players (`harness/src/poker_harness/engine/scripted.py`) in our
training fork so each profile matches its measured numbers. Train GRPO on 30-hand games against a mix of these
profiles with the placement reward. Scripted seats need no opponent calls and no GPUs. Mix in some
`ac-vals-model` seats so the model does not overfit to bots.

**9d. Baseline in the arena.** Submit the base model first and leave it in for a block of games. This is the
control: every probe and every trained model is compared with it.
- Pull its arena decisions from the deployment's traces (`client.analytics.query` or the console's Requests tab, see video 7) and the replays.
- List its leaks: spots where it folds too much, calls too much, thinks past the token cap, or busts.
- Each leak becomes RL tasks (step 6) or hint text (step 5) for the next model.

**9e. Probe entries.** Allowed and unlimited, so run them on a schedule.
- A probe is a Qwen deployment trained to play one fixed style, for example a short OPSD run with a "maniac" or "nit" hint. It costs a training run (4 GPUs) plus 1 GPU while it plays, so build two or three styles, not more.
- Each probe answers one question that replays cannot, such as how each entry reacts to an early-position open shove, or whether entries steal blinds against a nit.
- Keep a log per probe: style, time window, games played, and what each opponent did. That log feeds 9b.
- Probes and the baseline need about 15 to 25 minutes before their first rating, so give each a window of at least an hour.
- When reading the ladder, tag entries that look like other teams' probes (extreme, one-note styles). Keep them out of the profiles for the final field.

**9f. The final entry.** Only the last submission before 2 PM plays the tournament.
- Refresh the profiles on Sunday morning first, from the latest non-probe entries.
- Choose the final model by Sunday 12:30 PM. Judge it on mean final rank in local 30-hand games against the simulated field, and on its ladder games. Treat the ladder rating as a noisy signal, since that field holds probes.
- Run `play_local.py --hands 30` on it (0 failed decisions), then submit it by 1:15 PM. Submit nothing after it.
- Keep its deployment running until the tournament ends. Never delete it.

Done when the placement-reward model beats the chip-reward model on mean final rank in local 30-hand games against
the simulated field, and is not worse than base on `pokerbots-eval`.

### Step 10. Solver spots (optional, overnight)

- Run TexasSolver for short-stack spots (20 to 50 BB), where push or fold decisions matter most.
- Store the mixed strategy in `solver_freqs` and use it in hints.

## Quality gates

A dataset version ships only if all of these hold.
- 100% of records pass the schema.
- 0 illegal actions under the official engine.
- No hand id in two splits, and no PokerBench test record in hints or tasks.
- OPSD tasks pass the README checks (one tool call, under the token cap, tools listed).
- `stats.json` is committed next to the data.

## Timeline for today

| Time | Task |
|---|---|
| 2:00 to 2:30 | Step 1. Submit the base model to the arena so we are on the leaderboard. |
| 2:30 to 4:00 | Steps 2, 3 and 4 in parallel. Base train eval running. |
| 4:00 to 5:30 | Step 5, per-decision hints. First OPSD run launched. |
| 5:30 to 6:30 | Steps 6 and 7. RL task sets handed to the RL lane. |
| After 6:30 | Step 8 scouting, steps 9a to 9c. Baseline in the arena (9d). |
| Overnight | Probe entries (9e), one style per window. Step 10 if it still looks useful. |
| Sun 8 AM to 12:30 PM | Refresh opponent profiles. Train against the updated field. Probes still allowed. Pick the final model. |
| Sun by 1:15 PM | Check and submit the final entry (9f). No submissions after it. |
| After 2 PM | Official Tournament. Keep the entry deployment up. |

## Repo layout

```
plan.md
README.md
poker/          teacher, converters, hint builder, task builders
poker/schemas/  JSON schemas
data/           raw downloads (gitignored)
out/            generated datasets (gitignored except stats.json and small samples)
```

Add this to `.gitignore`. Also add `poker/`, `data/` and `out/` to `.ac2ignore`, so they stay out of the AC2 upload.

```
data/
out/*.jsonl
```

## Risks

| Risk | What we do |
|---|---|
| SFT is not available | Labels feed OPSD hints and RL tasks instead (see "Where the data goes") |
| Offline OPSD cannot teach what base never does | Try `rollout_mode="one_step"`, or lean on GRPO |
| The model thinks past 5,000 tokens and gets folded | Hints push short thinking. Track the cut-off rate in every eval. |
| The model busts early and ranks last | Multi-hand RL tasks and a survival rule in every hint |
| The teacher is weak | Use it for spot finding and depth coverage. Lean on PokerBench and RL for strength. |
| Outside data is not allowed | Fall back to teacher labels, equity hints and RL only |
| Too many folds in the data | Cap preflop folds in step 4 and check `stats.json` |
| The model overfits to scripted opponents | Keep `ac-vals-model` seats in the mix and judge on `pokerbots-eval` too |
| Profiles go stale as teams resubmit | Refresh them Sunday morning and weight recent games more |
| Profiles built from other teams' probes, not their final models | Tag and drop probe-like entries. Weight Sunday games most. |
| We overfit to the ladder field, and the tournament field differs | Keep `ac-vals-model` seats and `pokerbots-eval` as a general check |
| A probe or test model is still the entry at 2 PM | Final entry by 1:15 PM, then no more submissions |
