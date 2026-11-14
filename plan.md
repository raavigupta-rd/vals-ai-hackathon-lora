# Poker data plan

Vals Hackathon, poker track. Written Sat Nov 14, 1:45 PM. Submissions close Sun Nov 15, 2 PM.

This plan covers data only. Training, RL, and deployment get their own plans.

## Goal

Build one clean JSONL dataset of poker decisions that we can use for SFT on Applied Compute, plus a held-out eval set, by 7 PM tonight.

## Open questions that block parts of this plan

We need these answers from the organizers. Each one changes the data.

| Question | What it changes |
|---|---|
| Heads-up or 6-max? | Which sources we use and the teacher format |
| Stack depth and blinds? | Teacher settings and solver spots |
| Exact state and action format? | The prompt renderer (step 5) |
| Is reasoning text allowed, and what is the time limit per action? | Whether records keep the `reason` line |
| Can we see opponent hand histories? | Whether we build opponent stats |
| Any rules on outside data? | Whether PokerBench and human hands are allowed |

Until we have answers, we build for both heads-up and 6-max at 100bb.

## Sources

| # | Source | Size | Role | Status |
|---|---|---|---|---|
| 1 | Teacher bot self-play (our own) | Unlimited | Main SFT source, exact match format | Generator works |
| 2 | Pluribus hands (PHH dataset, MIT license) | 15,169 decisions from 10,000 hands, 6-max, 100bb | Strong-play labels | Converted, `out/pluribus.jsonl` |
| 3 | PokerBench (solver labels) | 60k preflop and 500k postflop train spots, 6-max | Solver-quality labels | Not downloaded yet |
| 4 | PokerBench test sets | 1k preflop and 10k postflop | Offline eval only, never train | Not downloaded yet |
| 5 | PokerStars 2009 human hands (PHH dataset) | 1.1 GB, 200NL and 1000NL | Opponent modeling only, never labels | Downloaded, not converted |
| 6 | ACPC bot hands (Zenodo, 20 GB) | About 280M heads-up hands | Optional, only if the format is heads-up | Not downloaded |
| 7 | Solver spots we make (TexasSolver or postflop-solver) | Minutes per spot | Optional, for the real stack depth | Not started |

Notes
- Sources 3, 4, and 6 must be downloaded on a laptop. Hugging Face and Zenodo are blocked from the Claude cloud workspace.
- The usage terms for the human hands (source 5) are not verified. We use them for stats only and do not ship them.
- TexasSolver is AGPL-3.0. We only use its output, and we do not copy its code into this repo.

## Record schema

Every source converts to the same record. The full schema is in `schemas/decision_record.schema.json`.

```json
{
  "id": "hu-7-123",
  "source": "teacher | solver | pokerbench | phh | selfplay",
  "format": {"players": 2, "bb": 2, "stack_bb": 100},
  "config": "balanced",
  "state": {
    "street": "flop", "hole": ["As", "Kd"], "board": ["Ah", "7c", "2d"],
    "position": "BTN", "in_position": true, "pot": 12, "to_call": 0,
    "stack": 190, "effective_stack": 190,
    "legal": {"fold": true, "check": true, "call": 0, "raise_min": 2, "raise_max": 190},
    "history": [{"street": "preflop", "seat": "hero", "action": "raise", "to": 5, "pot_frac": 0.67}]
  },
  "action": {"type": "raise", "to": 8, "allin": false},
  "reason": "Equity 0.84 vs top 40% range, value bets 0.66 pot.",
  "meta": {"equity": 0.84, "branch": "value_bet"},
  "hand_result_bb": 6.0,
  "solver_freqs": null
}
```

Rules
- Raise amounts are "raise to" totals for the street.
- The prompt text is not stored. We render it from `state` in step 5, so a format change does not force us to rebuild the data.

## Pipeline

### Step 1. Download what is missing (owner: ___, 15 min)

```
pip install huggingface_hub
huggingface-cli download RZ412/PokerBench --repo-type dataset --local-dir data/pokerbench
```

Done when the train and test JSON and CSV files are in `data/pokerbench/`.

### Step 2. Convert PokerBench (owner: ___, 1 hour)

- Write `src/pokerbench_to_records.py`. Use the CSV files, because they hold structured fields (position, board, pot, action history, hole cards, correct decision).
- The converter does not exist yet. Nobody has inspected the files, so budget time for surprises.
- PokerBench gives the pot and the options but may not give full stacks. Fill what the files have and leave the rest null.
- Keep train and test apart. Test goes to `out/eval/`.

Done when 100% of records pass the schema and a spot check of 20 records matches the source prompts.

### Step 3. Generate teacher data (owner: ___, 30 min, then rerun as needed)

```
cd src && export PYTHONPATH=.
python3 sim.py gen --fmt hu --hands 100000 --out ../out/hu_teacher.jsonl "balanced:lag,nit,station,maniac,tag,random"
```

- Measured speed on one core is about 1,600 heads-up records per second and about 900 6-max records per second. 200k records takes a few minutes.
- Use several hero configs for variety (balanced, tag, and balanced with `randomness=0.5`).
- Vary the stack depth once we know the format (for example 20, 50, 100, 200bb).
- Do not filter records by whether the hand won. One hand result is mostly luck, and that filter teaches the model to copy lucky plays.

Done when we have at least 200k records in the real format with zero illegal actions.

### Step 4. Clean and split (owner: ___, 45 min)

Write `src/build_dataset.py`. It does these things.
1. Validate every record against the schema. Drop and count failures.
2. Check that each action is legal for its own `legal` field.
3. Remove exact duplicates (same state and same action).
4. Split by hand id, so decisions from one hand never land in both train and val.
5. Balance. Preflop folds will dominate, so cap them (target about 25% of records).
6. Write `out/train.jsonl`, `out/val.jsonl`, and a `stats.json` with counts by source, street, and action.

Starting mix for SFT. Change it after the first eval.

| Source | Share |
|---|---|
| Teacher, real format | 40% |
| PokerBench train | 40% |
| Pluribus | 10% (only if the format is 6-max) |
| Solver spots at the real stack depth | 10% (if we get to it) |

If the format is heads-up, drop Pluribus and move its share to the teacher. PokerBench is 6-max, so in a heads-up format use only its blind-versus-blind and button-versus-blind spots at full weight.

### Step 5. Render prompts (owner: ___, 30 min after we get the interface)

- Write `src/render.py` with two functions. `state_to_prompt(state)` produces the exact text the hackathon harness sends. `action_to_text(action)` produces the exact text it expects back.
- Test it with a round trip. Render the action, parse it with the harness parser, and get the same action back for 100% of records.

Done when `out/sft_train.jsonl` holds prompt and completion pairs in the Applied Compute format.

### Step 6. Eval sets (owner: ___, 30 min)

| Set | Use |
|---|---|
| PokerBench test (11k spots) | Fast accuracy check for each checkpoint |
| `out/val.jsonl` | Teacher agreement and legal-action rate |
| Live matches against the preset pool | The real metric, in bb/100 with a standard error |

The paper behind PokerBench reports that a higher score there went with a higher win rate in play. Treat that as a guide and confirm with matches.

### Step 7. Opponent stats from human hands (optional, after step 6)

- Parse the PokerStars PHH hands and compute VPIP, PFR, fold to bet, and aggression by player type.
- Use the numbers to tune the preset opponents, so the RL pool looks like real weak players.

### Step 8. Solver spots (optional, overnight)

- Run TexasSolver on Modal for spots at the real stack depth and bet sizes.
- Store the mixed strategy in `solver_freqs` and sample the label from it.

## Quality gates

A dataset version ships only if all of these hold.
- 100% of records pass the schema.
- 0 illegal actions.
- No hand id appears in both train and val.
- No PokerBench test record appears in train.
- `stats.json` is committed next to the data.

## Timeline for today

| Time | Task |
|---|---|
| 2:00 to 2:30 | Get the interface. Download PokerBench. |
| 2:30 to 4:00 | Steps 2, 3, and 4 in parallel |
| 4:00 to 5:00 | Step 5, render to the real format |
| 5:00 to 6:00 | Step 6, eval sets. First SFT file handed to training. |
| After 7:00 | Steps 7 and 8 if they still look useful |

## Repo layout

```
plan.md
README.md
src/            teacher bot, simulator, converters
schemas/        JSON schemas
data/           raw downloads (gitignored)
out/            generated datasets (gitignored except stats.json and small samples)
```

Add this to `.gitignore`.

```
data/
out/*.jsonl
__pycache__/
```

The raw data and the big JSONL files stay out of git. Anyone can rebuild them with the commands in this plan.

## Risks

| Risk | What we do |
|---|---|
| The real format differs from our schema | The prompt is rendered late (step 5), so only the renderer changes |
| The teacher is weak (it only clearly beats a random bot) | Use it for format and legality. Lean on PokerBench and RL for strength. |
| PokerBench is 6-max and the event is heads-up | Use the matching spots only and raise the teacher share |
| Outside data is not allowed | Fall back to teacher and self-play data only |
| The dataset teaches too many folds | Cap preflop folds in step 4 and check `stats.json` |
