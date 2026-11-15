# SFT data (Track 2: post-train Qwen3.5-9B on Tinker)

Each file is the exact training set of one Tinker LoRA SFT run, gzip-compressed JSONL. One line = one
conversation: `{"id", "rating", "n_tokens", "messages", ...}`. `tool_calls` are stored as dicts
(`ToolCall.model_dump()`); load them with `ToolCall.model_validate` (see `post-train/sft/train_sft.py`).

| File | Run | Thinking | Content | Checkpoint |
|---|---|---|---|---|
| `sft_b_v1.train.jsonl.gz` | sft_b_v1 | Off (`qwen3_5_disable_thinking`) | 3,000 R1 traces, conversation <= 12k tokens; reasoning as visible text | `tinker://8cdc1923-63fd-5c09-ad50-e95cd5c29372:train:0/sampler_weights/final` |
| `sft_a_v1.train.jsonl.gz` | sft_a_v1 | On (`qwen3_5`) | Same 3,000 traces; reasoning inside `<think>` | `tinker://f542873c-b70b-5a1c-b9ba-0e4b28975b58:train:0/sampler_weights/final` |
| `sft_a_v2_stop.train.jsonl.gz` | sft_a_v2_stop (data built, NOT trained yet) | On (`qwen3_5`) | 3,000 traces, conversation <= 8k tokens; 4 tool schemas; submit -> EXIT | — |

`sft_b_v1.problem_ids.txt`: the 1,773 problem ids used by sft_b_v1 and sft_a_v1 (to exclude from eval sets).

## Source and filtering

- Traces: DeepSeek-R1 (open-source) from `open-r1/codeforces-cots`, config `solutions_decontaminated`.
- Problems: only the `open-r1/codeforces` **verifiable train split**. No test-split (2024-25) problems.
- Kept only traces that finished, closed `</think>`, contain a code block, and fit the token cap.
- Every solution was compiled (`g++ -std=c++20 -O2 -include bits/stdc++.h`) and accepted on the problem's
  official tests (with the dataset checker for multi-answer problems). Generated tests were not used.
- Curriculum: rows are ordered by rating, easy -> hard (the trainer does not reshuffle).
  Mix: sft_b_v1 / sft_a_v1 = 1,410 easy, 950 medium, 640 hard; sft_a_v2_stop = 2,020 / 900 / 80.
- No text written by Anthropic or OpenAI models. All reasoning and code is R1's; tool results are real
  compile/run outputs or fixed JSON. sft_a_v2_stop's final turn is `<think></think>` + `EXIT`.

## Conversation shape

```
system  tool declarations (+ system prompt for some schemas)
user    instructions + problem statement
assistant  [reasoning] + explanation, then a submit tool call with the C++ code
tool    sample-test result (all passed)
assistant  EXIT
```

Tool schemas: sft_b_v1 / sft_a_v1 use `cpp_executor(cpp_code)` + `submission(cpp_code)`. sft_a_v2_stop
varies them per example: ours (same names), harness `run_cpp(code, input)` + `submit(code)`, ioi-agent
prompt, and generic `compile_and_run(source, stdin)` + `submit_solution(source)`.

Thinking-on runs train one example per assistant turn (`build_supervised_examples`), because the
`qwen3_5` template drops earlier turns' `<think>` text from history.

## Results so far (shared 25-problem eval, open-r1 test split, Modal, 40/60 scoring)

base thinking-off 16.7 | sft_b_v1 9.6 | sft_a_v1 8.0 | base thinking-on 5.8. Both R1-SFT runs score below
the untrained base model; context overflow (still reasoning at the 16k per-turn cap) is the main failure.

## Rebuild

Scripts are in `post-train/sft/` and `post-train/baseline/`:
`build_sft_data.py` (judge R1 traces) -> `convert_thinkon.py` (reasoning into `<think>`) ->
`build_stop_data.py` (tool schemas, submit -> EXIT) -> `train_sft.py` (curriculum LoRA SFT on Tinker).
