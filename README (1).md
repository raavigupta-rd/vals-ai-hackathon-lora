# Vals Poker Bot — Six-Max Multi-Wave OPSD Battle Plan

> **Goal:** Finish top 3 on an **aggregate, multi-game** poker leaderboard by consistently outperforming the actual competing bots. The objective is better *expected aggregate evaluation performance*, not high single-game variance, raw survival, or merely winning the occasional huge pot.
>
> **Table configuration:** **6 players per table (six-max)**, as confirmed by the team. This is distinct from the number of teams on the leaderboard. Do not use 10-player hand charts or 10-player positional assumptions.
>
> **Operating premise (verify, don't assume):** 10 GPUs available; each OPSD job consumes 4 GPUs; two simultaneous jobs are therefore *potentially* feasible, leaving 2 GPUs. The permitted submission is a post-trained Qwen ~35B model. Opponents' public actions and `say` are visible. Current student can play against competing bots. Exact GPU topology, resource sharing, Qwen checkpoint, trainer API, scoring, allowed automation, and opponent-access rules must be confirmed in the actual environment.

## TL;DR: the highest-leverage strategy

1. **Start logging immediately:** every decision and complete public trajectory from the running **current student** against real competitors, including failed actions, bet sizes, `say`, public opponent behavior and game outcomes.
2. **Build a robust evaluation harness first:** fixed, held-out evaluation matches; seat rotations; versioned opponent snapshots; actual aggregate-score estimator. Never select from one good tournament.
3. **Optimize hints cheaply before training:** take identical student-prefix states, try several teacher hint variants, reroll the hinted teacher, and grade legal-action quality using range-aware, opponent-aware action-value estimates. Do not trust plausible-sounding explanations.
4. **Launch two *controlled*, parallel 4-GPU OPSD jobs:** A = poker fundamentals / equity-and-action-value hints; B = same information **plus opponent-specific behavioral estimates**. Same starting checkpoint, data budget, and training config. Two jobs = 8 GPUs, *if* resource isolation and memory actually permit it.
5. **Evaluate A vs B vs baseline on held-out matches against the actual field**, plus a sanity-check unfamiliar opponent. Select by the leaderboard's real aggregate metric and uncertainty, not training loss or maximum variance.
6. **Use the next training round to test the winning hypothesis**, not an arbitrary new architecture: improve the best hint family, data selection, or mix of opponents. Run RL only as a measured challenger after OPSD, if the platform exposes online environment rewards and enough time remains.
7. **Submission:** deploy only the best validated, legally formatted, inference-ready checkpoint. Keep a fallback.

## 0. Unblockers and hard constraints

Inspect and document these *before* spending both 4-GPU allocations:

- [ ] Exact base model/checkpoint and whether training adapters, merged weights, or only provider-hosted checkpoints can be submitted.
- [ ] Is a 4-GPU OPSD job truly supported for this model? Is this **4 devices on the same node / same memory topology**, or 4 pooled GPUs? Can two jobs run concurrently without resource contention?
- [ ] Whether teacher scoring, student sampling and rollout serving are already included in each job's 4-GPU allocation. **Remaining 2 GPUs might not be enough to host Qwen 35B inference**, depending on quantization and architecture; use platform-provided inference or CPU-side analysis if needed.
- [ ] Trainer details: on-policy refresh frequency, privileged-hint interface, supported KL/objective, checkpoint saving, resume, grad batch, loss masks, action output format, teacher/student logprobs.
- [ ] Exact scoring: chip rank, elimination order, TrueSkill, tournament placement, averaging, or a custom metric. Use the actual scoring code if supplied.
- [ ] Exact observation/output contract, legal raise sizes, `say` length and routing, 5k-token cap, timeouts and invalid-output penalties.
- [ ] Can bots be targeted or only sampled through a matchmaking API? Can opponent versions change mid-test? Are identical seeds/seat rotations possible? Confirm scraping/observation and table-talk compliance with event rules.

**Hard rule:** Do not let infrastructure assumptions delay a single functional baseline and data logger. If two 4-GPU jobs are not genuinely possible, run A then B sequentially with the same evaluation protocol.

## 1. Optimize these, in this order

| Priority | Knob | Testable hypothesis | Measurement |
|---|---|---|---|
| P0 | **Action validity / speed** | Schema/latency mistakes are costly | Invalid %, timeout %, tokens/action, p95 latency |
| P0 | **Real aggregate score** | Model improves reliably across many games | Organizer score or exact local proxy, CI, opponent breakdown |
| P1 | **Teacher hint information** | Better grounded hints give better OPSD | Teacher reroll action quality, final hint-free student score |
| P1 | **Opponent-exploiting policy** | Real bots overfold, underbluff, overcall or mis-size situationally | Conditional actions and chip EV vs each opponent |
| P1 | **Training distribution** | High-impact decisions carry greater learning value | Held-out gain per rollout/training token |
| P1 | **Bet size and aggression** | Selective pressure beats generic risky play | Profit by pot, street, size, opponent, stack-to-pot ratio |
| P2 | **Opponent conditioning** | Public behavior statistics improve play | Same checkpoint vs ablated opponent summary |
| P2 | **Hint confidence / gating** | Low-quality hints hurt the student | Filtered vs unfiltered paired experiment |
| P2 | **On-policy data freshness** | Current-policy states matter more than stale baseline states | Fresh vs mixed replay training gain |
| P3 | **Joint `(action, say)`** | Some messages alter profitable opponent behavior | Matched same-action/same-state `say` A/B, eventual chip EV |
| P3 | **Post-OPSD RL** | Outcome optimization beats the best teacher alone | OPSD vs OPSD→RL, same held-out games |

**Not a priority:** random learning-rate sweeps, maximizing bluff frequency, optimizing single-game variance, long explanations, or chasing leaderboard noise.

## 2. Collect the actual competitor distribution — continuously

The *current* student should roll out against the opposing teams from minute one. Every update to the student triggers **fresh on-policy rollouts**. Old logs still serve opponent profiling and offline diagnostics but are not silently treated as current-policy trajectories.

### Data schema (one row per decision)

Store append-only JSONL/Parquet and reference separate game/hand tables:

```json
{
  "run_id": "r001", "game_id": "g042", "hand_id": "h11", "decision_id": "d07",
  "seed": 1234, "hero_seat": 3, "street": "turn",
  "student_checkpoint": "qwen_base", "rollout_policy_version": "v0",
  "opponent_version_ids": ["opp_a_v2", "opp_b_v1"],
  "student_observation": "EXACT PROMPT/OBSERVATION AS SENT TO MODEL",
  "public_state": {
    "hole_cards": ["Ah", "Qh"], "board": ["Kd", "9h", "3s", "2h"],
    "pot_bb": 22, "hero_stack_bb": 64, "to_call_bb": 8,
    "legal_actions": ["fold", "call", "raise"],
    "min_raise_bb": 16, "max_raise_bb": 64,
    "public_history": [], "visible_say_messages": []
  },
  "model_output_raw": "...", "action": "call", "bet_size_bb": 8,
  "say": "", "valid": true, "latency_ms": 800, "tokens_used": 240,
  "action_logprobs": null, "sampling_temperature": 0.7,
  "opponent_actions": [], "showdown_reveals": null,
  "terminal_chip_delta_bb": null, "match_final_score": null
}
```

*Schema example only: don't fabricate unavailable fields; fill outcomes later when hands/matches complete. The hero's cards are observable; other players' unrevealed cards are not.*

**Also retain**: full game outcome, final stacks, eliminations, all legal actions, bet-size conventions, opponent bet/say timestamps, prompts and tokenization versions, and whether hand ended by fold or showdown. Store any simulator-only hidden state **separately**, protected from student inference data.

### Opponent model: what matters

Estimate by opponent **and context**, with numerator/denominator and uncertainty:

- VPIP, preflop raise and 3-bet; fold-to-open/3-bet.
- Fold-to-bet/raise by street, **bet-to-pot size**, effective stack and position; distinguish facing a river bet from preflop play.
- Call/raise/fold on all-in pressure; action-frequency changes as stacks shrink.
- Continuation bets, check-raises, bet-size patterns, showdown frequency, hand-strength evidence *only at revealed showdowns*.
- Influence of chat: same betting action with `say` vs silence; control for selection bias and opponent mix.

Use Bayesian smoothing / pooled priors and backoff from sparse buckets to wider opponent groups. Do not infer a bot's policy from 3 folds. Check whether opponent IDs actually appear to the submitted student; otherwise condition on public action history, not hidden team labels.

### Holdouts and contamination safeguards

- Split by **entire match/session and seed**, never random individual actions from the same hand.
- Version opponents: `opponent_id + bot_version + time window`. They may submit updates.
- Save a locked evaluation seed/seat set not used to select teacher hints, train, or tune bet sizing.
- When exact replay against a changing remote bot is impossible, rotate randomized seats, use multiple sessions and report wider uncertainty.
- Log data-collection policies; samples chosen because a model already expected a fold are not unbiased evidence of message efficacy.

## 3. Cheap teacher-hint search before spending training GPUs

Applied Compute's **Oct. 7, 2026 hint auto-optimization work** suggests testing proposed hints using *fresh teacher continuations from the same prefix*, then training only on successful or sufficiently credible hints. This is directionally relevant, **not** a validated poker recipe.

Research: https://www.appliedcompute.com/research/scaling-on-policy-self-distillation

### Candidate hint families

**A — Fundamental poker / action-value teacher (control):**

> Estimate hero's equity against plausible ranges using only public betting signals. Compute the pot odds and legal action sizes. Compare fold, call and several legal raise sizes by expected value; consider position, multiway participation, effective stacks, and uncertainty. Prefer well-supported, profitable decisions; do not assume unseen cards or future events are known. Respond in the exact required action schema.

**B — Opponent-exploit teacher (treatment):**

> Use all the fundamental estimates **plus** provided opponent behavior summaries, with their sample sizes and uncertainty. Evaluate how observed fold/call/raise frequencies change expected value for each legal action and bet size. Exploit clear deviations from strong play without assuming a small sample is decisive. If evidence is weak, fall back to sound baseline strategy. Do not use opponent identity/features unavailable in the actual student observation at inference, unless they can be inferred from public histories. Respond in the exact required action schema.

**Optional later candidate families** (don't full-train all simultaneously):

- **C — Calibration / uncertainty:** B but abstain from specific exploitation when data are sparse; shrink estimates toward general population.
- **D — Counterfactual best-action:** B plus simulation estimates `Q(s,a)` across **legal** action-size choices; distinguish Monte Carlo noise from real improvement.
- **E — Error-specialist:** B on selected high-regret errors only: bad stack-offs, missed value, overbet bluffs, illegal sizes, wrong multiway odds.
- **F — Tournament metric:** B plus actual known aggregation objective if it meaningfully changes decisions; avoid speculative 'survival weighting'.

**Critical leakage rule:** Hints may contain privileged *analysis* at training time, including estimates based on aggregate past opponent observations, but must **not** present actual unrevealed opponent hole cards or future results as if the student could know them. For Monte Carlo action estimates, marginalize plausible hidden states rather than choosing the realized cards in the logged hand.

### Fast hint-selection protocol

1. Sample a stratified set of difficult states: preflop 3-bet, multiway pots, river bluff catchers, stack pressure, value-bet sizes, significant chip-EV mistakes.
2. Recreate **identical** student prefixes, same checkpoint and legal-action schema for A, B, and any screened candidate hints.
3. Sample teacher *continuations* using multiple seeds; do not grade only a single edited answer or the persuasiveness of its prose.
4. Grade each legal decision with a cheap, consistent action-quality oracle: equity/range simulations, expected action value estimates, rules and opponent-response models. Compare the teacher's selected action to estimated best legal action, with uncertainty. Keep the grader independent of the particular hint language.
5. Screen candidates by improvement over unhinted teacher, not by KL shift. Retain only hints with repeatable improvement and no catastrophic illegal-action or value errors.
6. Choose A vs B as the first expensive parallel OPSD comparison. Avoid selecting a hint solely because it 'sounds advanced'.

**Do not require perfect action values.** No-limit multiway poker EV estimation is itself approximate. Use it to reject obvious bad labels and prioritize promising hints, then let actual matches decide.

## 4. GPU allocation and parallel OPSD experiments

### Default allocation (subject to hardware validation)

| Resource | GPUs | Job | Controlled variable |
|---|---:|---|---|
| Group A | 4 | **OPSD-A: fundamental teacher** | Equity/range/pot odds/action values only |
| Group B | 4 | **OPSD-B: opponent-aware teacher** | A **plus** measured opponent tendencies/uncertainty |
| Unallocated capacity | 2 | Game runner, light inference, teacher-hint screening, evaluation and logs *if memory/serving permits* | None |

**Do not interpret 2 spare GPUs as guaranteed capacity for Qwen 35B.** If serving shares the training pool or needs >2 GPUs, prioritize trainer requirements and use available hosted inference/CPU simulations. Two active jobs may compete for CPU, I/O or rollout servers even if GPU arithmetic fits.

### First comparison must be fair

- **Identical** starting checkpoint, parser, output tokens, training hyperparameters, effective batch, optimizer, checkpoint schedule, train-step/token budget and sampling regime.
- **Same initial on-policy student rollouts** from the common starting checkpoint, if the OPSD API permits replay. If rollout freshness requires each job to sample independently, use matched seed/seat/opponent distributions and record that this is a noisier experiment.
- Only systematic intervention = teacher hint content / privileged opponent summary for B. Data and evaluation budgets otherwise matched.
- For both runs, teacher and student are versions of the **same run-specific model**, per the OPSD implementation; when models diverge, each run must collect **its own fresh on-policy student rollouts**. Do **not** swap B's new rollouts into A and call them on-policy.
- Checkpoints: baseline → early checkpoint → middle checkpoint → final checkpoint. Evaluate intermediate models; later is not automatically better.
- Save runs/configs/hint versions to reproducible manifest. Reserve time to evaluate and submit rather than training until deadline.

### Training-cycle sketch

```text
               Real competing bots (versioned)
                            |
                  Current Qwen student v0
                            |
                 Full public trajectory log
                            |
              Teacher-hint proxy screening
                            |
                 Common v0 starting state
                     /             \
             4 GPUs /               \ 4 GPUs
              OPSD-A                 OPSD-B
       equity / action EV      same + opponent stats
                  |                  |
             checkpoint A1      checkpoint B1
                  |                  |
      fresh A-policy rollouts  fresh B-policy rollouts
                  |                  |
             checkpoint A2      checkpoint B2
                     \             /
              held-out multi-game eval
                            |
             choose strongest checkpoint
                            |
               next targeted iteration
```

## 5. Evaluation and checkpoint selection: the real optimization objective

**Primary metric:** organizer's actual aggregate ranking/scoring function, computed over many games. The announced 'top 3 teams get prizes' is a **prize cutoff**, not proof that the scoring function is per-game top-3 finishing probability.

Track:

- Actual aggregate score or the closest exact local reconstruction.
- Mean chip profit per hand/game, variance and worst-quantile outcomes (diagnostics).
- Elimination frequency, game placement, rank shifts (diagnostics only unless scoring uses them).
- Paired A–B score differences on the same opponent versions/seeds/seat rotations; uncertainty intervals.
- Results *per opponent*: don't let strong performance vs one easy bot mask losses vs the rest.
- Invalid action rate, latency and tokens per action.

**Statistical note:** hundreds of actions in one game are correlated; bootstrap at **match/session level**, not at individual-action level. With few matches, report uncertainty candidly. Prefer confidence intervals and repeatable wins over leaderboard movement caused by one seed.

### Sequential decision gates

1. **Validity gate:** any candidate with material invalid-action/timeout regression is rejected until fixed.
2. **Teacher gate:** candidate hinted teacher improves validated legal-action quality vs unhinted teacher on a stratified proxy set.
3. **OPSD gate:** candidate is at least comparable to baseline and shows positive held-out aggregate performance at intermediate checkpoint.
4. **Promotion gate:** challenger improves the aggregate metric against a stable competitor snapshot by a meaningful margin and no catastrophic opponent-specific regression; if uncertain, gather more held-out matches.
5. **Submission gate:** checkpoint passes exact serving and file-format validation; choose highest-confidence strong model, not newest.

## 6. After the A/B results: allocate the *second* round intelligently

| Observation | Next experiment | Why |
|---|---|---|
| B beats A robustly | **B vs B+calibrated exploit**: improve shrinkage, opponent features, bet-size-specific fold estimates | Tests whether more reliable profiling adds value |
| A beats B | **A vs A+selective opponent hints** only in high-support states | Opponent summaries may be noisy/misleading |
| Both beat baseline, nearly tied | **Best vs best+high-regret example selection** | Improve data efficiency rather than adding noisy complexity |
| Neither beats baseline | Debug teacher grader, legal-action interface, hint leakage, train loss and rollout freshness | More compute will not fix broken supervision |
| A or B loses to a changed competitor version | Refresh opponent models and rerun held-out evaluation | Avoid overfitting to stale opponent policies |
| Strong OPSD plateau + robust eval | **Optional RL challenger** from strongest checkpoint, if online RL supported | Direct reward may discover exploit missed by teacher |

### Targeted data-selection experiment

Compare uniform current-policy trajectories with a mixture that *upweights*:

- Decisions with large estimated action-value gaps or high chip impact.
- Nontrivial bet-size choices; stack-pressure spots; multiway decisions; river marginal calls.
- States where actual opponents have a **well-supported** exploitable tendency.
- Failure states where teacher reroll changes an obviously suboptimal action into a legal high-value action.

Keep representative ordinary hands; training only on dramatic river situations damages basic play. Don't label actions as good because a lucky showdown won. Match total training tokens across arms.

### Optional RL branch — only if it can be evaluated fairly

- **Start from the strongest OPSD checkpoint**, not a weak baseline, unless doing a small RL-only control.
- Train against the *actual permitted opponent pool* or a versioned proxy if available, using on-policy environment feedback.
- Start with **full-hand net chip change normalized by BB**, then compare an actual-tournament-score reward if available. Never assume hand win rate is the right target.
- Use many hands, seat rotations, reward normalization and conservative KL controls to guard against a lucky sparse reward signal.
- Run matched held-out games against strongest OPSD checkpoint. Keep RL only if it improves *aggregate metric*; otherwise submit OPSD.
- Do not spend the last available eval/submission window building an unvalidated RL trainer.

## 7. `say` is a separate, low-cost factorial ablation

The bot's public `say` is visible to other LLM agents, who are warned that table talk could be false. Treat it as an **uncertain** game mechanic, not a guaranteed jailbreak.

Test a *small* set of permitted messages (empty/control; straightforward strength claim; equity doubt; tournament-pressure language; misleading weakness) under **identical game states and identical betting action**. Log resulting fold/call/raise rates and downstream chip outcomes. Do not assume a higher fold rate helps when value betting; action and message should align. Avoid rule-prohibited impersonation or instructions that attempt to access private data.

**If** results are repeatable and profitable, teach joint `(legal betting action, say)` choices. Otherwise leave `say` empty and spend effort on bet sizing and teacher quality. Do not train on the subset of individual messages that happened to induce a fold; evaluate average causal effects by opponent and context.

## 8. Execution order / operational checklist

### Immediate (before training)

- [ ] Verify Qwen checkpoint, 4-GPU/job claim, two-job concurrency, rollout serving, submission format and real scoring.
- [ ] Instrument **current student** to log exact public observations/actions and all visible opponent plays; save match outcomes.
- [ ] Run current baseline vs competitor pool; establish eval snapshots, seed/seat scheduling and fixed holdout.
- [ ] Generate opponent behavior report with counts and uncertainty.
- [ ] Build a cheap, legal-action / EV proxy grader for teacher continuations; screen A/B hints on common states.

### First full training round

- [ ] Start **OPSD-A** and **OPSD-B** from the same baseline, matched budgets; verify isolation and checkpointing.
- [ ] Keep each run's post-update trajectories on-policy with its own student checkpoint.
- [ ] Train without leaking actual unseen cards/outcomes to the student.
- [ ] Evaluate intermediate A/B checkpoints and baseline on held-out competitor snapshots and seats.
- [ ] Save one-page result matrix: `model / hint / train tokens / aggregate score / CI / per-bot score / validity / latency`.

### Exploitation/iteration

- [ ] Promote best validated model.
- [ ] Run **one** next experiment from the decision table (teacher calibration, data selection, or targeted opponent hints).
- [ ] Optionally test small `say` ablation and RL challenger **only if** core pipeline is healthy.
- [ ] Refresh evaluation if opponents update; stop training in time for submission and final validation.

## 9. Failure modes that will lose the hackathon

- Optimizing **variance** rather than repeatable aggregate performance.
- Training on realized winning hands (hindsight luck) instead of action quality.
- Giving the teacher actual enemy cards and then distilling impossible perfect-information responses.
- Mistaking opponent overfolding in a few hands for an exploitable pattern.
- Comparing two OPSD runs with different starting states, rollout distributions or train budgets and attributing the difference to hints.
- Comparing new checkpoints only to stale weak opponents or using training-game outcomes as held-out evidence.
- Spending all 10 GPUs on training without enough capacity/time to evaluate.
- Treating 2 unused GPUs as automatically sufficient to serve a full Qwen checkpoint.
- Wasting the 5k-token action budget on verbose explanations instead of reliable legal actions.
- Optimizing `say` or RL before confirming that either produces real aggregate lift.

## 10. Source & what is genuinely known

- Applied Compute, **"Scaling On-Policy Self-Distillation with Hint Auto-Optimization"** (Oct. 7, 2026): https://www.appliedcompute.com/research/scaling-on-policy-self-distillation — supports fresh teacher rerolls as a hint proxy and reports benefits from filtering weak hints on its studied tasks. **Poker transfer is unproven.**
- Applied Compute, **"Enriching Domain-Specific Post-Training with Multi-Teacher On-Policy Distillation"**: https://www.appliedcompute.com/research/multi-teacher-on-policy-distillation — discusses serving/training GPU resource tradeoffs; *not* proof of a particular GPU arrangement for this event.
- Hardware allocation (10 total, 4 per job), actual competitor access, allowed base model, `say` behavior, and aggregate leaderboard design are **team-reported hackathon assumptions** pending harness verification, not specifications proven by these papers.

---

**Decision rule:** The model with the strongest repeatable **held-out aggregate performance against current opponents** wins. The *first* experiment should isolate **teacher hint quality and opponent information**, not compare two uncalibrated piles of training changes.


## 11. Six-max strategy specification (corrected table size)

- Seats are **6-handed**, not 10-handed: use UTG, HJ, CO, BTN, SB, BB when all six seats are occupied; adapt position and number of live players after eliminations.
- Compared with a full-ring table, opening ranges generally widen, blind pressure recurs more often, and blind-versus-blind / button-versus-blind situations matter more. Never copy full-ring preflop charts into training without adjustment.
- Track effective stack (not just hero stack), open size, 3-bet/4-bet frequency, squeeze opportunities, position, ante/blind schedule, number of players still active, and ICM-like tournament effects *only if* the scoring actually warrants them.
- For GTO references, begin with **six-max preflop charts** appropriate to the actual rake/ante/stacks and no-limit format, then use approximate postflop solver-inspired heuristics. A true six-player no-limit GTO solution is computationally intractable here; do not call approximations exact GTO.
- Reserve some evaluation against a conservative **GTO-inspired anchor** to detect when field-specific exploitation becomes dangerously unbalanced.

## 12. Multi-wave training roadmap (two 4-GPU slots)

**Experimental discipline:** For each wave, launch two jobs with identical starting weights, budget, rollout access, holdout and evaluation protocol. Change *one major causal factor* between A/B. Fresh student rollouts are required after updates; only use identical starting trajectories for the initial controlled comparison. Keep 2 GPUs for services only if viable, otherwise use inference endpoints. Promote on held-out multi-game aggregate performance, not training loss or a lucky session. All plans are contingent on available wall time/compute.

| Wave | Slot A (4 GPUs) | Slot B (4 GPUs) | Decision / hypothesis |
|---|---|---|---|
| **0: Instrumentation** | No training: log baseline games, opponent profiles, errors, and seat rotations | No training: build evaluator and teacher-hint screening | Establish robust baseline; measure legal action/timeout rate |
| **1: Teacher context** | OPSD with six-max fundamentals: pot odds, range equity, SPR, legal action EV | Same hints + smoothed opponent fold/call/raise tendencies and action EV | Does opponent-aware information improve policy over grounded poker alone? |
| **2: GTO foundation** | Champion of wave 1 + six-max preflop / balanced-range guidance | Same initialization + the same hints but more explicit solver-inspired frequencies, defenses, and value/bluff balance | Does a stronger GTO anchor improve aggregate outcomes? |
| **3: Exploit strength** | GTO-informed policy + conservative exploitation, shrinking estimates toward prior | Same policy + stronger exploitation of opponent-specific deviations with calibrated uncertainty | Where is the best robustness–exploitation tradeoff? |
| **4: Data selection** | Uniform on-policy samples across positions, stacks, streets | Prioritized high-regret, large-pot, marginal and known opponent-weakness spots, with a coverage floor | Does selective sampling improve reward per train token? |
| **5: Teacher quality** | All valid hints (baseline filtering) | Reject or downweight low-confidence/noisy teacher hints, use agreement/EV checks | Does confidence gating avoid bad pseudo-labels? |
| **6: Position & stack specialists** | Position-conditioned six-max preflop and blind play | Effective-stack, SPR and short-stack-specific hints | Which decisions are still leaking the most EV? |
| **7: `say` communication** | Same policy, empty or neutral `say` controls | Policy allowed to choose tested poker-relevant `say` types jointly with bets | Does communication change *actual aggregate returns*? |
| **8: Authority / doubt stress tests** | Table talk focused on poker-range uncertainty and overthinking cues | Claimed system/evaluator notices and competence-pressure text as **adversarial test categories** | Does source-authority confusion measurably change actions despite untrusted-chat warning? |
| **9: Opponent freshness** | Specialize on currently observed field, conditioned on stable opponent identities if exposed | Mixed training against current field + earlier snapshots + diverse fallback opponents | Is specialization robust to competitor updates? |
| **10: RL challenger (conditional)** | Best OPSD checkpoint remains frozen as control | OPSD→online RL on six-max tournament outcomes if supported | Does direct outcome reward improve held-out aggregate score? |
| **11: Finalist comparison** | Best aggregate checkpoint | Best robust/checkpoint ensemble substitute (single legal model only) | Re-run fixed holdout plus fresh opponent snapshots; select one allowed checkpoint |

**Run ordering:** Waves 1–3 are highest priority after logging. Waves 4–6 are conditional on diagnostics; waves 7–8 are lower priority and must not crowd out fundamental strategy. Waves 9–11 are conditional on time, competition drift and platform features. Do not automatically execute all waves.

### GTO-inspired training labels and checks

1. Build a **six-max preflop** library indexed by position, stack, previous raises, blind/ante context and action sizing; prefer robust open/defend/3-bet guidance instead of one rigid action per hand.
2. For postflop states, estimate opponent range equity and legal-action EV via simulation where feasible. Use board texture, nut advantage, blockers, SPR and position; keep uncertainty with estimates.
3. Train mixed strategies where appropriate, not deterministic always-bet/always-fold rules; track divergence from the GTO-inspired anchor only where exploitation is evidence-backed.
4. Audit for overfolding BB defense, overbluffing multiway, excessive preflop all-ins, missed thin value, and nonstandard invalid sizings.
5. Test against exploit-prone field bots **and** an unfamiliar robust anchor; a strategy that only beats one version of one bot is fragile.

### `say` experiments: precise scope and measurement

Treat system-style notices, evaluator impersonation, reasoning-pressure messages, and doubt/gaslighting language as **candidate adversarial interventions**, not trusted instructions or known-effective attacks. Confirm they are allowed by tournament rules. The first test measures vulnerability; deployment requires demonstrable held-out aggregate improvement. Avoid making claims about real organizer rule changes as if they were true.

**Factors to sweep (one at a time):** category (silence, ordinary bluff, pot-odds doubt, competence/overthinking cue, purported authority), message length, frequency, street, bet sizing, whether hero is bluffing/value-betting, target opponent archetype, and repetition/adaptation. Use identical hero action/state and paired controlled repetitions for causal tests; then test full trajectories because table talk changes future opponent behavior.

**Success criterion:** better estimated *net chip EV / actual multi-game leaderboard metric*, not merely an increased fold rate. For value bets, making opponents fold can be bad. Retain no messages that reliably cause timeouts, invalid actions, or opponent safeguards in prohibited ways.

### Teacher-hint candidates (screen before using 4 GPUs)

- **H0:** Exact observable state only — negative control.
- **H1:** Six-max preflop action/range charts and postflop pot odds.
- **H2:** H1 + Monte Carlo range equity and confidence intervals.
- **H3:** H2 + approximate action EV for each legal bet size.
- **H4:** H3 + publicly inferred opponent-profile posterior (with shrinkage and observation counts).
- **H5:** H4 + six-max solver-inspired mixing and an exploit penalty for poorly supported deviations.
- **H6:** H5 + calibrated `say` effect estimates (only after blinded A/B demonstrates an effect).

Quick-screen hints on the **same starting decision set** by legal-action validity, teacher-vs-reference action regret, and reliability across positions/stacks. The best hint is the one that yields a stronger *hint-free student* after OPSD, not necessarily the teacher that sounds most confident. Do not leak opponent hole cards that are unavailable during deployment as if they were direct recommendations; if used for diagnostic oracle estimates, integrate out hidden information across ranges for student-compatible supervision.

### Operational scorecard and stop conditions

- Primary: actual held-out aggregate tournament score / ranking-proxy **with uncertainty** across repeated games, seeds, opponent snapshots and seat rotations.
- Secondary: mean chip delta, per-opponent delta, illegal-action %, timeout %, latency, calibration of fold predictions, catastrophic bust frequency, preflop/river EV leaks.
- Only promote if improvement exceeds expected noise, remains positive on a fresh holdout, and doesn't cause major regressions on key opponent types; compare A/B with matching inference budgets.
- Drop a candidate early if training breaks output validity, hints show no teacher improvement, or table talk gives no reproducible causal lift.
- Every wave records: checkpoint ID, parent, seed set, GPU-hours, rollouts per opponent, hint version, sample weights, trainer config, `say` variant, evaluator version, and aggregate score/CIs.

### If there is only time for three training pairs

1. **Pair 1:** six-max fundamentals vs six-max fundamentals + opponent hints (wave 1).
2. **Pair 2:** winning initialization with and without stronger GTO-inspired balancing (wave 2).
3. **Pair 3:** conservative versus aggressive-but-calibrated exploitation (wave 3).

Reserve the final evaluation window and submit whichever checkpoint has the strongest repeated **multi-game** performance. The prompt-injection tests can proceed as small offline/online A/B experiments without consuming a full OPSD training slot until evidence justifies it.
