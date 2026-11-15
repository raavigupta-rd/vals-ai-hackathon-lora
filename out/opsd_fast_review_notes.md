# Real decide-fast hint review

Reviewed 20 real decisions from completed tasks in base train eval `eval_ce4924764f47`, dataset
`pokerbots-train`. Source snapshot: `traces/eval_ce4924764f47.jsonl`, SHA-256
`af5dca8dc33689822e34c0c58b2192cb83b8e69578a6ff81c1d42eb0293c6022`.

Reviewed report SHA-256: `a4e9762e845f0e2e3c1f9c8427ae15cedca48ce2f7778348b5e3674821696746`.
The snapshot retained 24 decisions and skipped eight unfinished task traces; 18 cleanup turns were excluded.
No retained decisions had missing counts, truncation, malformed arguments, parse failures or illegal actions.

The 20 inspected state/action pairs cover preflop, flop, turn and river; seven include incremental updates.
Tokens range from 205 to 2,005, below the strict 2,500 threshold. Checks included free checks, folds facing bets,
calls, raises within street-total bounds, and flop/turn value bets. The state parser used the latest visible
board and price. Optional amounts on call/check are ignored by the event parser and are not mislabeled as raises.

The hint is the short `poker/decide_fast_hint.txt`: strength, position, price, then one legal tool call. It asks
for concise reasoning and avoids speculative placement arithmetic. It contains no hidden cards, future board,
test labels or sampled-reply content. The sampled base reply is preserved, and only the teacher receives the hint.
This experiment deliberately does not use PokerBench labels or heuristic equity; label-match coverage is not
applicable. It is a small data-path pilot, not evidence of improved strategy or representative base cutoff rates.

Upload target: new versioned dataset `pokerbots-opsd-fast-20261114-pilot-v1`, never an append to an existing dataset.
The full baseline eval continues independently. No OPSD training may run while the GRPO pilot owns the slot.
