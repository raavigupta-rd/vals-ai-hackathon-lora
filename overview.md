# Where we are (Nov 14, shared-branch consolidation)

Setup is done, base smoke succeeded, and separate baseline evals and the bounded GRPO pilot are running.
Both people now work on shared branch `poker`. Read `poker/run_ledger.json` for timestamped run state,
`poker/STATUS.md` for results and remaining gates, and `plan.md` for the full plan. `README (2).md` adds strategy
priorities: reliable decisions, replayable public trajectories, measured exploitation and held-out evaluation.
Its conditional SFT/on-policy suggestions remain subject to the confirmed limits in `poker/CAPABILITIES.md`.

## The rules that shape everything

- 6-max no-limit hold'em, 30-hand games, everyone starts with 200 chips, chips carry over, busted seats are out.
- **We are ranked by final place, not by chips won.** Busting is the worst result.
- The leaderboard before 2 PM PST Sunday is practice only. Each submission starts with a fresh rating, and submissions are unlimited.
- **What counts:** one Official Tournament after 2 PM PST Sunday, played by every team's last submission (at most 15 teams).
- Each decision may use up to 5,000 output tokens, thinking included. A reply that runs past that is played as check or fold.

## What's done

| Item | Status |
|---|---|
| Plan rewritten to match the README and the judges' answers | `plan.md` |
| Placement reward: the grader can now score a game by final place (1st = 11, last = 1, errors = 0) | Done, tested |
| Datasets uploaded: the 5 starter sets, plus 3 new multi-hand game sets that use the placement reward (`pokerbots-train-game`, `-select-game`, `-eval-game`) | Done |
| Harness and connector running on A's laptop, all doctor checks OK | Done |
| `base` deployment (Qwen3.6-35B-A3B, 1 GPU), smoke eval and `play_local` check: 0 failed decisions | Done |
| Base submitted to the arena | Done by A, or doing it now |
| Baseline evals: `eval_9f9ec615687b` (500 single hands), `eval_e51ea5b4466a` (200 games of 30 hands) | Running |

All A/B implementation is merged into shared `poker`. The original `poker-a` and `poker-b` branches are historical.
Additional baseline runs are `eval_9e50f4198211` on `pokerbots-eval` and `eval_ce4924764f47` on `pokerbots-train`.
The sole training job is `train_b2dda65e96c4`, a four-step single-hand GRPO pilot. A reviewed 24-decision offline
OPSD pilot dataset is uploaded, but no OPSD training has been launched. No training improvement is established yet.

## What we learned

1. **Public replays support offline opponent analysis.** Published JSON includes actions, table talk, token use,
   final places and hole cards, including some never shown during play. Use samples to estimate ranges with
   uncertainty; they do not reveal a team's exact policy. Never put unrevealed cards or future outcomes into
   student observations. Player identity can condition the submitted model only when visible at inference.
2. **The field is beatable.** In one match we inspected, a scripted "nit" house bot finished 1st (+84 BB), ahead of three teams running base Qwen.
3. **Measure base and opponents separately.** The high cutoff rate in the first house-model smoke was not a
   measurement of base or the field. Base smoke later completed eight decisions with zero cutoffs and a 620.5-token
   median. These small/partial observations do not establish population failure rates; full baseline audits remain pending.

## The approach

- **Train for placement, not chips.** Use GRPO on multi-hand games with the placement reward, so the model learns survival and how to move up the chip ladder.
- **Exploit the actual field.** Collect every arena hand in a database and measure each team's leaks: what they fold to, what they 3-bet with, how often they fail. Feed the leaks into OPSD hints and into scripted opponents that play like the real teams.
- **Use the scratchpad.** The scratchpad lasts for the whole game. Train the model to keep short notes on each seat ("seat 3 folds to raises") and act on them. Few teams will train for this.
- **Smaller edges:** steal blinds from tight bots, keep replies short, and test whether table talk moves LLM opponents.

## Next steps

**A (infra and RL)**
1. Read the baseline eval results: cut-off rate, tokens per decision, mean place.
2. Monitor `train_b2dda65e96c4`; do not launch a second training job. Evaluate its checkpoints against base before
   longer training. Measure complete short-game transcripts before sizing placement-game budgets/timeouts.
3. Move the harness to an always-on machine before the overnight runs.
4. Rotate the connector credentials (they were exposed during setup).

**B (data and scouting)**
1. Build the arena hand database: a scraper that polls `pokerbots.ac` every 10 minutes and loads match files into SQLite (`poker/arena/`). A Claude worker can write the first version.
2. Compute per-team stats (VPIP, PFR, 3-bet, fold to big bets and shoves, cap rate, mean place), and share them as a dashboard.
3. Turn the leaks into OPSD hints and into scripted opponent profiles (plan steps 5, 9b and 9c).

**Both, Sunday**
- Refresh opponent stats in the morning, pick the final model by 12:30 PM PST, check it, and submit by 1:15 PM PST.
  The confirmed cutoff is 2 PM PST Sunday, November 15. Require the official-rule 30-hand zero-failure check first.

## How we work

- Both people work on `poker`: fetch and pull with `--ff-only` before editing, coordinate overlapping files,
  and commit/push completed work promptly. Do not resume work on the historical A/B branches.
- Only A submits to the arena. Only one training job runs at a time (10-GPU quota).
- `CLAUDE.md` makes every Claude session follow the lanes and agent rules in `plan.md`.
