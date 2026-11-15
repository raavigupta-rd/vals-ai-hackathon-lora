# Harness and training capability map (Sat Nov 14, ~10 PM)

Distilled from a 12-reader review of `harness/`, `rl/`, `opsd/`, the installed ac2 SDK and the Applied Compute docs, plus a completeness pass. Full notes: `scratchpad/harness_map/digest.txt` (local only).

## The scratchpad, exactly

- It is the first user message of every hand: `## Your scratchpad (private: only you can see it)` and the contents between `<<<SCRATCHPAD` and `SCRATCHPAD>>>`. It is the only memory that crosses hands; earlier hands are never shown.
- After every hand the seat gets a cleanup turn with its own 10,000-token budget. The prompt lists the actions since our last move, the board, any showdown cards, the pot and everyone's stacks, then says `Now call update_scratchpad exactly once...`.
- Only `too_long` and `malformed` updates get one retry. A reply with no `update_scratchpad` call, a reply cut at the cap, or an error leaves the pad unchanged with no retry. A too-long pad is cut from the tail, so the important lines must come first.
- Calling `update_scratchpad` during a hand is an invalid turn (check or fold).
- In the arena, c2b-s12 answered 34% of cleanup turns with `submit_action` (60% after hands with three or more of its own turns), so a third of its hand-starts showed a stale pad. 27.5% of its updates copied the `SCRATCHPAD>>>` wrapper into the contents. Base fails 1.5%.

## What tasks can control (env_params)

- `seed` (required; deals are deterministic per seed and hand number), `seats`, `total_hands`, `reset_stacks`, `starting_stack`, `policy_seats`, `opponent_model`, `submit_fields` (adds `p_win`, `read`, `say`), `ev_labels`/`ev_runouts`, timeouts.
- Gotchas: without `preset: "event"` a task silently plays 10 seats, 3 hands, reset stacks, no scratchpad. Without `opponent_model`, non-policy seats are also played by the policy. `reset_stacks: true` changes the system prompt.
- Scripted bots (`random`, `maniac`, `nit`, `calling_station`, `shover`) exist in the engine but the server could not use them; `env_params.opponent_bots` (training-only fork) adds that.

## Rewards (harness `grading.py`)

- One scalar per rollout; AC2 has no per-step reward. Failures score exactly 0.
- The default EV reward is clipped to +-5 BB over the whole game, and 91% of 30-hand games exceed it, so it is nearly binary on long games. One format error anywhere zeroes the whole game.
- Unused deterministic signals: `fold_regret_bb` labels, cleanup outcomes, `p_win` calibration against true equity, EV labels on single hands against known opponent cards.

## GRPO

- The pilot was starved by the opponent LLM: 44% of opponent actions timed out and were played as check or fold, and its one complete group scored [7.5, 7.5, 7.5, 7.5] (no variance, no gradient).
- Unused SDK knobs: warm start from an SFT checkpoint, zero-variance group filter (`by_reward_std`, probe samples), `filter_truncated_samples`, temperature, routing replay for the MoE, multi-dataset mixtures, sample timeout.
- Plan: GRPO on verifiable settings (single-hand spots and shove-heavy tables against scripted bots with EV-based rewards; cleanup drills with the deterministic cleanup reward), warm-started from the best SFT checkpoint.

## OPSD

- Allowed for team orgs: offline OPSD (data only). Online OPSD is ruled out. `one_step` (the student samples its own reply) runs without the harness and is unconfirmed for our org; ask Applied Compute.
