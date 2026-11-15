# Person B execution status

## Shared working branch and strategy

The user requested consolidation onto shared branch `poker`; `poker-a` and `poker-b` are historical references.
All completed implementation, reports and reviewed OPSD work are being committed and merged with A's latest
`overview.md`. The strategy reference `README (2).md` is now in the repo. `plan.md` records its applicable
priorities and the confirmed platform limits. The Sunday cutoff and earlier selection/submission targets are PST.
This repository consolidation does not restart the harness, alter deployments or launch another training job.

Current completed checks: 35 root tests passed; 106 harness tests passed, one skipped. The sole training run
remains `train_b2dda65e96c4`; see the timestamped run ledger for observed status and results. Its first four
graded rollouts had zero failures but identical rewards, so no learning benefit is established yet.

## Combined integration execution — Nov 14

- Combined A/B implementation published as `593a185` to `origin/poker-b`; archive extracted to ignored staging,
  not reapplied to the shared workspace. Existing changes were preserved.
- Base smoke `eval_992f2c5ead5b` succeeded: four hands, eight policy decisions, zero decision/cleanup failures.
  Output-token median 620.5, p99 864.79, max 865; cleanup max 955. `out/base_smoke_quality.json` records the audit.
  The observed single-hand all-request token upper bound was 15,668; it deliberately overcounts repeated
  prompts and is not the trainer's exact transcript length. Long-game limits remain unmeasured.
- Separate baseline jobs: held-out `eval_9e50f4198211` and train-only `eval_ce4924764f47`, both concurrency 8 and
  grading timeout 900s. Existing A select/game evals were preserved. No existing dataset uploads were rerun.
- The sole GRPO pilot is `train_b2dda65e96c4`: four single-hand steps, 4 tasks x 4 attempts, checkpoints every 2,
  concurrency 4, six estimated GPUs, no in-run eval, 24,576/32,768 response/context budgets and 1,200s grading timeout.
  These single-hand limits are not approved for placement-game training. See `poker/run_ledger.json` before any
  further launch. The base deployment is preserved; no arena submission or deployment deletion was performed.
- Reviewed gaps fixed locally: explicit train dataset; configurable eval grading timeout; game/hand denominators,
  mean final place, failure rates and paired game/place reporting; existing game uploads now refuse an occupied
  name. OPSD checks train-eval provenance and task membership, event tools, completed spans, token counts,
  truncation, JSON arguments and prompt legality. A dedicated decide-fast hint and review-hash upload gate exist.
- Added loaded grader fingerprint in health/scores and bust-status artifacts. **The live remote process has not
  been restarted**, so those additions are not yet active. Old smoke scores prove placement-aware grading is loaded,
  but not the exact source hash. No local harness/connector process was found. Ask the harness owner for its host
  and perform a coordinated idle-only update before placement-game training.
  A's newly merged `overview.md` identifies the original harness as running on A's laptop; its exact host/process
  revision still needs confirmation from A. That overview also notes moving it to an always-on host before overnight work.
- Integration checks: 33 root tests passed; 26 grading/server tests passed after telemetry-only edits. Earlier
  full harness run passed 106 with one skip. A full base-trace 20-hint review/upload and checkpoint comparison
  remain pending. Teacher/Pluribus source and optional solver/probe work are explicitly deferred.

### Train-only OPSD snapshot review

Reviewed 20 real decisions from completed base-train tasks, including seven incremental updates and all four
streets. All were legal and under 2,500 tokens. The versioned data-path pilot
`pokerbots-opsd-fast-20261114-pilot-v1` now contains 24 tasks. Eight unfinished task traces and 18 cleanup turns
were excluded; no retained decision had missing counts, truncation, parsing, argument or legality failures.
See `out/opsd_fast_hint_review.json` and `out/opsd_fast_review_notes.md` for the exact reviewed hash and source
manifest. No held-out/select traces were used. This static decide-fast experiment does not use PokerBench labels
or heuristic equity. No OPSD training was launched alongside the running GRPO pilot.

Live-export correction: the SDK's basic download can reuse a cached snapshot. `poker/export_traces.py` requests
`refresh=True` and saves a local SHA-256. Partial game trace prefixes are explicitly excluded from completed-game
token/failure summaries. The first 30-hand game snapshot had 330 observed decisions and one cutoff, but zero
finished games; it cannot establish a whole-game budget or failed-game rate.

The sections below retain the initial B milestone for context.

Work is on `poker-b`, branched from `poker`, with no arena submission or training launch.

Integration update: pulled Person A's work through `815c9bf` into this branch with a fast-forward. Local B changes
remain uncommitted and unpublished. The combined suite passed 23 root tests and 106 harness tests, with one skip.
`poker/HANDOFF.md` lists the remaining integration and training steps. The latest live check found base smoke
`eval_992f2c5ead5b` running, one connector, all eight starter/game datasets and no training jobs. Recheck state;
the older observations below are historical snapshots, not live status.

| Plan step | State | Evidence / next dependency |
|---|---|---|
| 2: PokerBench converter | Implemented and run | 574,200 schema-valid records, 11,000 source-test records kept separate; 20/20 original-prompt checks matched |
| 3: Port teacher | Waiting for source | No `sim.py` or helpers exist in this checkout; do not claim a substitute as a port |
| 4: Clean and split | Implemented and run for PokerBench | 13,565 train, 1,609 validation; `out/stats.json` records all exclusions |
| 5: Per-decision hints | Implemented; real trace dry run pending | Pot odds, sampled tight-range equity, matched clean training labels, format/survival rules; requires base train eval/traces |
| 8 / 9b: Scouting and profiles | Started with actual public replays | Nine exported games; three team entries in `out/scouting/arena_profiles.json` |
| 9c: Simulated field | Pending stronger profiles and A handoff | Current ladder samples are too small for reliable calibration; no shared engine edits or GRPO launch |
| 9e: Probes | Log prepared | `poker/scouting/probe_log.csv`; coordinate training slot and submission window with A |
| 10: Solver spots | Optional, not started | Core trace path takes priority |

Data limitations: source hand IDs are absent, so all scenario families seen in test are excluded conservatively.
This removes 527,827 source-training records before later checks. Another 2,004 lack engine verification,
8,586 have fractional event-chip states, 19 are duplicates, and 9,590 exceed the preflop-fold cap.
Most retained labels are preflop; the validation split contains no postflop records. Do not use it as a complete
postflop evaluation. Original PokerBench test data remains separate for that purpose.

Initial observations (early ladder, one submission version per team):

- Martingale: 135 hands over six games, VPIP 43/135, PFR 21/135; folded 32/55 times facing pot-size-or-larger
  pressure and 5/14 times facing a shove. Four failed decisions out of 196. These numbers have wide uncertainty
  and correlated observations; they do not establish a dependable exploit.
- Friends: 30 hands in one game. Insufficient evidence to calibrate a team-specific opponent.
- Spark: one dealt hand before busting in the downloaded sample. No dependable style estimate.

Rates and Wilson intervals, team IDs and submission identities are in the profile JSON. Replays are linked from
[the arena match list](https://www.pokerbots.ac/matches). Refresh profiles as teams submit new models and review
which entries are probes before using them to represent the final tournament field.

AC2 0.3.97 is installed and authenticated in `.venv`. The doctor passed every project/dataset check for
`prj_200dcc41ef41`. The `base` deployment is ready (`infer_d3d2b6b92171`). At inspection, the only evals were
successful table-smoke (`eval_50d9b110f7e1`) and smoke (`eval_15d0c3b30e5b`); neither is train data and neither
was used for OPSD. No training runs existed. A's base eval on `pokerbots-train` is still needed.

Verification so far: 12 data/scouting tests passed. The harness suite had 101 passes and one skip, plus a
subprocess import failure that passed when rerun with `PYTHONPATH` set to `harness/src`. Root SDK-dependent
tests now pass as well. The dynamic-hint integration test checks that hints receive only the prompt, preserve
base's reply exactly, and are added only to the teacher selection. A real base-trace OPSD dry run remains pending.
