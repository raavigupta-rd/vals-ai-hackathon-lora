# Vals Poker Bot — Post-Training Strategy

## Objective

Post-train the **allowed Qwen 35B checkpoint** using Applied Compute to finish **top 3 on the hackathon leaderboard**. The leaderboard aggregates performance across many games, so the goal is **repeatable advantage against the actual opponent pool**, not maximum variance in any one game.

**Working assumptions to verify in the harness:** the allowed base checkpoint and exact model name, submission artifact format, training methods exposed by Applied Compute, tournament scoring formula, number of hands and players per game, legal actions/bet-size discretization, `say` delivery, inference token limit (~5k reasoning + output per action), opponent identity visibility, and whether local games against competing teams are permitted. Do not assume the published public benchmark's settings are identical to this hackathon.

## What to optimize (in order)

| Priority | Target | What to measure | Optimization idea |
|---|---|---|---|
| P0 | **Valid, reliable decisions** | Invalid-action rate, timeouts, malformed outputs, latency/tokens per action | Force legal action schema; compress reasoning/output; handle edge cases |
| P0 | **Actual leaderboard performance** | Aggregate score/rating, mean chip change, elimination rate, top-3 rate **per game** (diagnostic), across held-out seeds/seats | Select checkpoints by repeated tournaments, not training loss |
| P1 | **Opponent exploitation** | Fold-to-raise by street/size, call vs large bets, 3-bet behavior, showdown frequency | Train situation-specific counters against observed opponents |
| P1 | **Preflop competence** | EV/leak by position, stack depth, effective stack, preflop action | Reliable opening/3-bet/call/fold ranges; avoid marginal stack-offs |
| P1 | **Bet sizing and fold equity** | Profit by size and street, opponent fold rate conditional on size, pot odds | Calibrate raises to exploit overfolding without overbetting indiscriminately |
| P1 | **Risk-adjusted tournament decisions** | Mean score **and** downside risk over many games | Prefer positive-EV aggression; do not blindly maximize volatility or survival |
| P2 | **Postflop hand/range reasoning** | Value/bluff EV, equity realization, expensive errors on turn/river | Equity/range estimates; pot-odds and board-texture-aware decisions |
| P2 | **Adaptation to observed behavior** | Gains against known vs unseen bots; performance drift | Condition on opponent features; use smoothing for small samples |
| P3 | **`say` / table talk** | Action-response change and net EV relative to no-message controls | Test short, allowed messages; keep only if measured gain |

**Primary target:** maximize the organizer's *actual aggregate leaderboard metric*. Track all other metrics as diagnostics. If the leaderboard uses TrueSkill or another formula, optimize that explicitly once confirmed. Since multiple games are aggregated, high per-game variance is not a substitute for higher expected score; also compare uncertainty across repeated runs.

## Main thesis

A small field (currently ~6 teams, subject to change) makes targeted exploitation potentially more valuable than training a perfectly general poker agent. But training directly on winning hands is a trap: poker outcomes are noisy. **Estimate the quality of actions against the actual opponents, then train Qwen on those actions.** Retain a competent general policy in case the field changes or opponents adapt.

### Exploit hypotheses to test (not assume)

1. **Overfolding under elimination pressure:** opponents may fold too frequently to large bets when effective stacks are threatened.
2. **Overly passive play:** a conservative bot may check/call or fold where aggression is profitable.
3. **Predictable bet sizing:** opponent raise size may reveal strength or correlate with bluff frequency.
4. **Poor multiway play:** heads-up heuristics may fail in 10-player pots.
5. **Poor adaptation:** opponents may maintain fixed behavior despite your aggression.
6. **Susceptibility to `say`:** table talk is visible and marked as possibly false; test whether it *actually* changes decisions. Respect hackathon rules and keep messages within permitted interaction.

Do **not** equate aggression with loose calling. Attack excessive folds, but avoid marginal calls or random all-ins. In a simple heads-up river bluff with zero equity when called, bluffing B into pot P needs fold probability > B/(P+B) to break even; this is only an illustrative special case, not a general multiway/tournament rule.

## Data collection

Record every observable decision from local games or approved matches against competing bots, **not only winning hands**. Include:

- Game, hand, seed, seat, street, position, blinds/antes, remaining players.
- Hero hole cards; community cards; pot, effective stacks, to-call amount, min/max legal raise, legal action set.
- Public betting history, each opponent's visible actions/sizing, revealed showdowns where available.
- Opponent identity **only if actually available to the submitted model**; otherwise derive behavioral features from public history.
- Our chosen action, bet amount, `say` text, model checkpoint, latency, token usage, validity.
- Terminal stack/chip changes, eliminations, game result, and leaderboard metric if available.

Never use unrevealed opponents' hole cards or future outcomes as **student inputs**. Such information can be used offline to analyze logs or evaluate a teacher, but student training examples must match inference-time observability. If opponent full state is exposed only by a simulator, avoid learning to depend on it.

### Baseline-first trajectory collection (start immediately)

**Treat the running base Qwen as an instrumented data generator from minute one.** Start logging now, even if the model is weak. The first purpose is to learn how the *opponents* play; the second is to identify which states our own model mishandles. Do not wait for SFT or OPSD before collecting data.

For **every decision**, persist a replayable record, preferably JSONL or Parquet, containing:

- `run_id`, `game_id`, `hand_id`, `decision_id`, `timestamp`, `seed`, `seat`, `street`, `model_version`, `opponent_versions` if known.
- **Exact student observation/prompt** as received, including chat/say messages, visible betting history, current legal actions and bet-size constraints. Save the tokenizer/prompt-format version.
- Our **sampled action, size, say, output validity, token count, latency**, and **action probabilities/logprobs if exposed** (including sampling temperature). Do not assume the API provides probabilities.
- The **public action sequence of every opponent**, with player/seat identifiers, bet sizes, stack and pot changes, including folds that reveal no hand.
- Revealed showdown cards and **terminal outcomes**; if full hidden state exists in the simulator, store it in a separate restricted teacher/evaluation dataset, never as a student input.
- **State snapshot before and after each decision**, so action legality, pot odds, and effective stack can be reconstructed without guessing.

At the **trajectory level**, link each decision to its full hand and game, including final chips, eliminations, standings, and any available aggregate score. Track whether a hand ends without showdown: this is essential to avoid treating unobserved cards as known.

**Minimum viable logger:** append one structured record per action; maintain stable player IDs; capture complete public state and outcome; make sessions replayable. A small reliable logger is higher priority than sophisticated training.

### From collected games to OPSD data

The log has **three distinct uses**, which should not be conflated:

1. **Opponent modeling (works immediately):** aggregate observed opponent actions conditional on street, sizing, effective stack, position, and game context. Use shrinkage / uncertainty intervals for sparse opponents. This does not require correct labels for our actions.
2. **SFT / value labeling (works with old logs):** re-evaluate logged student observations with a stronger action-value estimator or other teacher, and create `(observable_state, teacher_preferred_legal_action)` examples. Don't imitate actions merely because the hand was won; outcome is a noisy training label.
3. **Actual on-policy self-distillation (requires fresh rollouts):** after each checkpoint update, sample actions with the **current** student in games against the opponent pool. At each sampled decision, query the *same checkpoint as a privileged teacher* (or the approved OPSD teacher setup) with extra **training-only** equity/range/action-value information, obtain teacher/student token distributions if the trainer supports them, and train with the specified distillation loss. Maintain the current rollout policy version and avoid mixing stale trajectories without an explicitly justified off-policy correction. Teacher observation must not leak into student inference inputs.

**Caution on privileged labels:** revealing actual enemy hole cards can produce an omniscient teacher whose recommendations cannot be implemented from public information. Prefer calibrated counterfactual action values based on plausible ranges, or marginalize over hidden cards. On-policy distillation also requires access to the necessary logits/logprobs and loss; if Applied Compute only offers SFT, turn teacher assessments into supervised labels instead.

**Suggested data loop:**

```text
baseline Qwen -> games vs opponent pool -> append full public trajectories
                                         -> opponent behavior report
                                         -> teacher action-value labels -> SFT checkpoint
new checkpoint -> fresh on-policy games -> privileged teacher evaluation
               -> OPSD (if supported) -> held-out match evaluation
               -> keep winner, repeat
```

**Validation:** hold out complete matches and seeds, rotate seats, and evaluate against the actual current opponents plus at least one unknown/adversarial policy. Do not train on all games and then claim performance on those same games is an unbiased estimate.

### Opponent profiling

Estimate features conditionally instead of using one global aggression number:

- Fold to bet/raise by **street**, sizing bucket, pot odds, position, stack-to-pot ratio, and tournament pressure.
- Voluntarily put money in pot (VPIP), preflop raise (PFR), 3-bet frequency, continuation-bet and check-raise rates.
- Showdown tendencies, response to all-in pressure, bluff/value evidence **when cards are revealed**.
- Sample count and uncertainty for every estimate. Use prior smoothing and back off to global population behavior with sparse data.

Do not infer an opponent is exploitable from a handful of folds. Rotate seats and vary seeds so estimates do not confound cards, positions, or luck.

## Training plan

### Stage 0 — Baseline and harness (must do)

1. Run the supplied reference Qwen model end-to-end; save baseline tournament scores.
2. Read action parser, prompt construction, legal action rules, `say` visibility, scoring code, and submission validation.
3. Build a robust serializer: concise state in, **valid action + legal bet size + optional `say`** out.
4. Establish local evaluation with fixed seeds/opponent pool and seat rotations. Confirm reproducibility and expected runtime.

### Stage 1 — Strong supervised warm start (preferred first)

1. Generate realistic states from the actual opponent pool plus varied synthetic/general opponents.
2. Label with **good actions**, not hindsight winners: known poker heuristics, equity simulation, pot-odds/range analysis, and—where feasible—estimated action EV vs opponent tendencies.
3. Use a small, structured output target to conserve the ~5k-token per-action budget, and ensure action-size legality.
4. SFT the permitted Qwen checkpoint on Applied Compute. Check training/validation loss **and** real game performance.
5. Mine costly mistakes (invalid bet, bad stack-off, failed value bet, missed profitable bluff) for targeted re-training.

### Stage 2 — Opponent-targeted refinement

- Collect games against each real competitor if allowed; maintain a versioned opponent dataset.
- Generate counter-strategy examples for *observed* weaknesses, especially fold thresholds and sizing.
- Mix in general/novel opponents to avoid catastrophic overfitting and protect performance if teams update their policies.
- Retain opponent IDs only when available at inference. Otherwise teach adaptation from visible histories.
- Prefer **on-policy rollouts** of your newest checkpoint, since exploit data collected by an old policy may not represent the states your new policy reaches.

### Stage 3 — OPSD / RL only if supported and time remains

OPSD (on-policy self-distillation) is a candidate, **not the starting requirement**. It needs a meaningful privileged teacher signal and access to the training/logprob mechanics Applied Compute actually permits. Prefer teacher inputs like estimated legal-action values and simulated range/equity features over simply exposing actual hidden hole cards: a teacher that always knows the enemy's cards can teach an unrealizable policy.

If custom KL/logprob objectives are unavailable, fall back to **offline action-value labeling + SFT**. GRPO/RL from sparse game outcomes is higher-variance and should be attempted only after a competitive SFT baseline and a working evaluation loop.

## Experiment matrix

Run each candidate on **held-out games/seeds with seat rotations**. Compare aggregate score and confidence intervals / uncertainty, not isolated lucky runs.

| Experiment | Change | Success criterion |
|---|---|---|
| E0 | Reference Qwen | Reproducible baseline; near-zero invalid actions |
| E1 | General poker SFT | Better aggregate result vs E0 |
| E2 | Opponent-targeted SFT | Better vs known competitors **without collapsing** on held-out opponents |
| E3 | More selective aggressive sizing | Improved EV/aggregate score against observed overfolders |
| E4 | Opponent features in prompt | Meaningful lift vs identical model without profile |
| E5 | `say`: empty vs ordinary bluff/pressure | Controlled improvement in opponent actions **and** game outcomes |
| E6 | On-policy refinement / OPSD if feasible | Demonstrable gain vs strongest previous checkpoint |

Important ablations: no opponent information; no `say`; general-only vs opponent-specific labels; conservative vs aggressive sizing; different training mixes and prompt lengths. Use **paired seeds and seat rotations** wherever feasible, and keep a holdout set of full sessions, not randomly split actions from the same hands.

## Avoidable failures

- **Outcome imitation:** do not label a bad shove “good” because it got lucky.
- **Hidden-information leakage:** train student only on what is observable at decision time.
- **Aggression without fold equity:** large bluffs fail against calling stations and multiway pots.
- **Overfitting leaderboard noise:** a few wins do not prove improvement; use repeated evaluation.
- **Unnecessary long chain-of-thought:** 5k-token action budget means latency and truncation matter; use compact outputs and measure whether more reasoning helps.
- **Changing opponents:** re-evaluate after team submissions update; retain robust fallback play.
- **Unverified assumptions:** respect exact scoring/submission rules; verify what is allowed for opponent testing, table talk, and generated training data.

## Implementation checklist

- [ ] Confirm checkpoint, Applied Compute training API, scoring method, and submission requirements.
- [ ] Run reference bot; verify legal outputs, bet sizing, timeouts, and tokens/action.
- [ ] **Start baseline trajectory logging now**: every observation, action, opponent play, `say`, legal bet, and outcome.
- [ ] Save replayable per-action JSONL and stable identifiers; store privileged simulator state separately.
- [ ] Write per-opponent profile report and generate teacher labels from logged states.
- [ ] Run baseline games with fixed seeds/seat rotations.
- [ ] Test whether opponents actually overfold to large bets / elimination pressure.
- [ ] Generate high-quality labeled state→action examples.
- [ ] SFT permitted Qwen; evaluate against reference and real opponent pool.
- [ ] Train targeted counter-strategies; validate on new seeds and unfamiliar bots.
- [ ] Ablate `say`; keep only if legal and demonstrably beneficial.
- [ ] Attempt OPSD/on-policy refinement only if supported and clearly worth remaining time.
- [ ] Submit **best evaluated checkpoint**, not automatically the latest.

## Decision rule

**Optimize robust aggregate performance against the real field.** If evidence shows systematic overfolding, increase *profitable, situational aggression*. If not, prioritize fundamentals and opponent adaptation. The strongest approach is whichever checkpoint wins across repeated held-out tournaments—not whichever sounds most sophisticated.
