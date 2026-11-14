# Where we are (Sat Nov 14, ~3:45 PM)

Short version: setup is done, the base model is deployed and passes every check, baseline evals are running, and
the first training run starts as soon as they finish. The full plan is in `plan.md`; this page is the summary.

## The rules that shape everything

- 6-max no-limit hold'em, 30-hand games, everyone starts with 200 chips, chips carry over, busted seats are out.
- **We are ranked by final place, not by chips won.** Busting is the worst result.
- The leaderboard before 2 PM Sunday is practice only. Each submission starts with a fresh rating, and submissions are unlimited.
- **What counts:** one Official Tournament after 2 PM Sunday, played by every team's last submission (at most 15 teams).
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

All of A's work is on the `poker-a` branch.

## What we learned

1. **The arena publishes everything.** Each match has a public JSON file with every seat's hole cards (even hands that never reached showdown), every action, table talk, tokens used, and final places. We can see each team's exact range for every action. This is our biggest edge.
2. **The field is beatable.** In one match we inspected, a scripted "nit" house bot finished 1st (+84 BB), ahead of three teams running base Qwen.
3. **Running out of tokens matters less than we thought.** Base Qwen in the arena uses about 650 to 700 tokens per decision and hit the cap in 0 to 7% of decisions. The 26% failure rate we saw at first came from our training opponent (`ac-vals-model`), a different model. That is still a problem for training, since our opponents fold for no reason a quarter of the time.

## The approach

- **Train for placement, not chips.** Use GRPO on multi-hand games with the placement reward, so the model learns survival and how to move up the chip ladder.
- **Exploit the actual field.** Collect every arena hand in a database and measure each team's leaks: what they fold to, what they 3-bet with, how often they fail. Feed the leaks into OPSD hints and into scripted opponents that play like the real teams.
- **Use the scratchpad.** The scratchpad lasts for the whole game. Train the model to keep short notes on each seat ("seat 3 folds to raises") and act on them. Few teams will train for this.
- **Smaller edges:** steal blinds from tight bots, keep replies short, and test whether table talk moves LLM opponents.

## Next steps

**A (infra and RL)**
1. Read the baseline eval results: cut-off rate, tokens per decision, mean place.
2. Launch the first GRPO run, sizing the token budget from the game eval.
3. Move the harness to an always-on machine before the overnight runs.
4. Rotate the connector credentials (they were exposed during setup).

**B (data and scouting)**
1. Build the arena hand database: a scraper that polls `pokerbots.ac` every 10 minutes and loads match files into SQLite (`poker/arena/`). A Claude worker can write the first version.
2. Compute per-team stats (VPIP, PFR, 3-bet, fold to big bets and shoves, cap rate, mean place), and share them as a dashboard.
3. Turn the leaks into OPSD hints and into scripted opponent profiles (plan steps 5, 9b and 9c).

**Both, Sunday**
- Refresh opponent stats in the morning, pick the final model by 12:30 PM, check it, and submit by 1:15 PM. Submit nothing after it.

## How we work

- Each person has a branch (`poker-a`, `poker-b`); merge into `poker` through a reviewed PR.
- Only A submits to the arena. Only one training job runs at a time (10-GPU quota).
- `CLAUDE.md` makes every Claude session follow the lanes and agent rules in `plan.md`.
