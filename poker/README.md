# Person B: data and scouting

Run these commands from the repository root on shared branch `poker`. This package is local tooling and stays out of AC2
project uploads. Person A owns the live harness, GRPO jobs and arena submissions.

## Local setup (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install jsonschema pokerkit pytest
```

The full project also needs the real `ac2>=0.3.97` SDK from the authenticated Applied Compute package index.
The public PyPI package named `ac2` is a placeholder and does not provide the CLI or SDK.
This checkout now has AC2 0.3.97 installed and authenticated in `.venv`. Use `.\.venv\Scripts\ac2.exe` directly
or activate the virtual environment. On Windows, set `$env:PYTHONIOENCODING = 'utf-8'` for CLI tables.

## PokerBench conversion and cleaning

The CSVs come from [RZ412/PokerBench](https://huggingface.co/datasets/RZ412/PokerBench), Apache-2.0.
Put the four `*_game_scenario_information.csv` files in `data/pokerbench/`. Keep the original source files.
Download the two test `*_prompt_and_label.json` files for the reference audit.

```powershell
.\.venv\Scripts\python.exe -m poker.pokerbench_to_records --limit 1000 --out out/pilot
.\.venv\Scripts\python.exe -m poker.pokerbench_to_records
.\.venv\Scripts\python.exe -m poker.audit_pokerbench
.\.venv\Scripts\python.exe -m poker.build_records out/raw/pokerbench.jsonl --held-out out/eval/pokerbench.jsonl
.\.venv\Scripts\python.exe -m pytest tests/test_poker_data.py -q
```

- `out/raw/pokerbench.jsonl`: all schema-valid source training records, including explicitly unverified rows.
- `out/eval/pokerbench.jsonl`: source test data, preserved separately. Never pass this as a hint-label source.
- `out/train.jsonl`, `out/val.jsonl`: engine-verified, integer-chip records after cleaning.
- `out/pokerbench_stats.json`, `out/stats.json`: conversion and cleaning counts, including exclusions.
- `out/pokerbench_audit.json`: 20 content-based checks against original test prompts.

The converter scales source SB=.5/BB=1/stack=100 to event SB=1/BB=2/stack=200. It maps postflop OOP/IP to
absolute positions from the preflop participants and strips board cards beyond the evaluation street.
CSV and JSON rows have different orders, so the audit matches content, never row indices.

Betting histories are replayed through the official engine. The source explicitly assumes unmentioned seats
folded, so only missing preflop folds are inferred. Pot mismatches, missing postflop actions and illegal labels
fail verification. These rows remain in the raw conversion for inspection; the cleaner excludes them.
Some solver sizes become fractional event chips; these are also excluded rather than rounded.

PokerBench has no original hand IDs. The split unit is a conservative scenario family (preflop line and flop),
shared across holdings, later streets and labels. A family present in test is excluded from training entirely.
This sacrifices data volume to prevent leakage. Exact decision states are also deduplicated. Splits are stable
SHA-256 partitions by family; preflop folds are capped at 25% separately in train and validation.

## Per-decision OPSD

After A supplies the **base eval on `pokerbots-train`**, with the real SDK installed and authenticated:

```powershell
.\.venv\Scripts\python.exe opsd/build_dataset.py --eval <TRAIN_EVAL_ID> --dataset pokerbots-opsd-train --records out/train.jsonl --dry-run --hint-report out/hint_review.json
```

Review the 20 hints, then repeat without `--dry-run` to upload. The builder retains the starter tool-call and
token-cap checks. `--generic-hint` restores the original static hint for comparison. `--equity-samples 0` disables
Monte Carlo equity while preserving pot odds, matched labels and the format/survival rules.

Hints parse only prompt-side public state, including the latest incremental update. Equity uses a documented
tight-range prior and the official engine's PokerKit hand evaluator, with deterministic runout sampling.
It is approximate pot share, not solver truth. Labels must match hole cards, board, seat, nearby pot/call size,
stack depth and legal action bounds. If no safe match exists, the hint says so. Test and validation labels are
rejected. Hints never see the sampled reply or private opponent cards.

Offline OPSD reweights base replies; it cannot introduce an action base never samples. The event's BYOH-only
restriction excludes online OPSD. Coordinate the single training slot with A before launching offline OPSD.

For the decide-fast experiment use `--hint-file poker/decide_fast_hint.txt --max-tokens 2500` and the train eval
`eval_ce4924764f47`. The builder now checks the live eval's base-model/dataset provenance and every trace's
task membership, token counts, completion status and action legality. A running source eval may supply a
versioned snapshot of its completed tasks; unfinished tasks are always excluded. Inspect the 20
state/action/hint entries in `--hint-report`, then pass its printed `--reviewed-report-sha256` for an upload to a
new versioned name. Existing hinted datasets are refused, never appended blindly.

`python -m poker.trace_quality <trace-export.jsonl> --out <report.json>` reports per-decision/cleanup cutoff and
failure rates and token median/p99. It labels sums of repeated request tokens as upper bounds; these are not
exact training transcript lengths. Long-game training remains gated on measured short-game token use.

## Scouting and probes

```powershell
.\.venv\Scripts\python.exe -m poker.scout_arena
```

The first snapshot showed no completed matches or team entries. There is no replay evidence to support opponent
profiles in that initial snapshot. A later refresh found games, and `arena_replays.py` now downloads the public
Raw JSON links and adapts the observed `vals-arena.match/1` schema. `out/scouting/arena_status.json` records the latest
observation and source. Refresh periodically. Derive rates only from observed opportunities, retain sample counts,
separate submission versions, flag probe-like entries and weight newer tournament-relevant games more heavily.

`scouting/probe_log.csv` is the handoff log for A's submission windows. Each window must last at least an hour;
leave unknown observations blank. B proposes style hints and analyzes responses; A submits the deployment.

`python -m poker.profile_logs <exported-game.json> --identities <players-to-teams.json>` computes VPIP, PFR,
fold-to-shove, fold-to-pot-bet, bust and failure rates from **official-format complete game logs**. Identity entries
must provide `team`, `submission` and `probe`; unknown players and probes are excluded. Each rate includes its
opportunity denominator and a Wilson interval. Forced folds on failed decisions are not strategic folds. This
adapter consumes harness-style logs; the arena adapter below handles the distinct public replay format.

```powershell
.\.venv\Scripts\python.exe -m poker.arena_replays --fetch --limit 20
```

This fetches publicly linked raw JSON into `data/replays/`, normalizes the observed arena schema and writes
`out/scouting/arena_profiles.json`. House entries are excluded from team profiles. `--exclude-probes <entry IDs>`
marks known probes; probe detection needs human review. These early ladder samples should not be treated as
stable final-tournament profiles. The initial run downloaded nine replays and observed three team entries.

## Dependencies still needed

- Original `sim.py` and helpers: they are absent from this checkout, so step 3 cannot be honestly called a port.
- Any existing Pluribus conversion: also absent here.
- Base train-eval ID or exported traces for a real OPSD dry run/upload. AC2 credentials now work.
- More replay evidence before profiles justify a calibrated simulated field; the initial profiles are provisional.
- Coordination with A before any 4-GPU OPSD/probe training job. Only A submits to the arena.

Optional TexasSolver work remains behind the core data/trace path. No solver output is claimed or fabricated.
