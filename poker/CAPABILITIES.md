# AC2 capability verification — November 14, 2026

Checked against official public docs, installed AC2 SDK 0.3.97, the event README and the authenticated account.

## Training interfaces

The [training configuration](https://docs.appliedcompute.com/platform/training/configuration) exposes managed
GRPO/RL, SFT, OPSD and decision-head methods. The SDK's `ExperimentConfig.method` accepts `RLConfig`, `SFTConfig`,
`OPSDConfig` and `DecisionConfig`. No public custom differentiable-loss callback, arbitrary forward/backward
training API or client-supplied gradient update was found. This does not rule out a separate private/staff API.

Custom poker rewards are possible through the harness grader. Rewards are scores supplied to managed RL;
they are not an arbitrary replacement for the trainer's loss. RL exposes advantage estimators, importance
weighting, entropy and KL controls. Decision heads offer predefined cross-entropy or Brier losses and require
separate access. They are not a generic custom-loss hook for the arena's generative model.

[OPSD](https://docs.appliedcompute.com/platform/training/opsd) internally uses teacher/student token probabilities
and exposes token selection/masking through `TokenJudgeConfig`. That does not establish user access to raw
training token-logprob tensors for an arbitrary objective. An internal `log_probs_chunk_size` field is a staff
configuration knob, not a public logprob-return API.

The [SFT guide](https://docs.appliedcompute.com/platform/training/sft) confirms platform support. Our event README
explicitly guarantees harness GRPO/evals and data-only offline OPSD. It does not confirm SFT eligibility, and
rules out online OPSD, custom images/launch arguments and SSH into training jobs. Confirm team access before
relying on SFT, one_step OPSD, token-judge variants or private training interfaces.

**Update, Sat Nov 14 ~4:15 PM:** the organizers confirmed to the user that our team may use `SFTConfig`. One_step
OPSD, token-judge variants and private interfaces remain unconfirmed.

## Live inference logprob check

Tested `https://inference.appliedcompute.com/v1/chat/completions` against the ready `vals-hackathon/base` deployment.
Each request used a harmless short prompt, eight output tokens and no stream.

| Request | Result |
|---|---|
| Normal request | HTTP 200; response logprobs null |
| Same request plus `logprobs: true` | HTTP 400, generic Invalid request |
| Same request plus `logprobs: true, top_logprobs: 3` | HTTP 400, generic Invalid request |

Raw generation logprobs are therefore not working through the tested route/deployment. The generic error does
not identify whether the restriction is gateway-, backend-, model- or account-specific. It does not establish
that every AC2 endpoint rejects logprobs. No public arbitrary-transcript scoring/logprob API was established in
the reviewed docs/SDK. The [serving guide](https://docs.appliedcompute.com/platform/serving) documents OpenAI-style
requests but does not promise these optional fields.

Planning conclusion: use confirmed managed GRPO/offline OPSD. Treat a custom-loss plus raw-token-logprob training
design as unavailable until Applied Compute supplies a supported interface and confirms team access.

## Decisions requiring caution

- Original PokerBench hand IDs are absent. Conservative family exclusions protect test data but discard most
  source rows. The retained validation split has no postflop records.
- Hint parsing, arithmetic and task construction have tests; hints have not been shown to improve poker results.
- Equity uses few samples and a heuristic opponent range. It is not solver equity.
- Label lookup requires exact visible cards/board/seat and nearby pot/call/stack sizes. This prioritizes precision
  and will leave many real decisions without a matched label.
- Ladder profiles have small, correlated samples and unreviewed probe flags; they cannot yet justify confident
  opponent-specific strategies or final-field calibration.
- No training job, arena submission or change to Person A's shared harness was made by this work.
