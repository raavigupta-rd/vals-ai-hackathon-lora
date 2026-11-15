# Local Track-2 harness

A local stand-in for the Vals grader, used to score Tinker checkpoints before spending a real submission.

How it mirrors the real grader (from the track page and `vals-ai/ioi-agent`):
- One hard competitive programming problem per task. The model writes C++20, compiled with `g++ -std=c++20 -O2 -include bits/stdc++.h`.
- Two tools: `run_cpp(code, input)` runs code on any stdin; `submit(code)` judges on the **sample tests only**.
- Up to 50 submissions and 100 turns. The episode ends on `EXIT`, the turn limit, or repeated malformed tool calls.
- The **last submission that compiles** is graded on hidden tests: 40 points for the small half, 60 for the large half (split by input size), all-or-nothing per half. `score_fraction` gives partial credit instead.

Unconfirmed: the real tool names and schemas, the prompt wording, and whether points are partial within a half.

## Setup

```bash
open -a Docker          # the sandbox runs in a networkless gcc:14 container
uv run tinker auth login
```

## Data

- Eval set: `open-r1/codeforces` test split (2024–25 contests), rated 1400–2600, not interactive, with official and generated tests, including large inputs. Python checkers are kept for problems with several valid answers. **Never train on these problems.**
- `python -m harness.problems` builds `data/eval/cf_test.jsonl`. It needs `data/open_r1_cf/{test.parquet,generated_tests/,selected_contests.txt}`.

## Run

```bash
uv run python -m harness.eval --tag base                     # base Qwen3.5-9B
uv run python -m harness.eval --model-path tinker://... --tag sft1
uv run python -m harness.eval --n 20 --tag quick             # 20 problems spread over ratings
```

Writes `runs/<tag>/results.jsonl` (score, turns, tokens and full transcript per problem) and `summary.json`.

## Files

- `sandbox.py`: Docker compile/run with a read-only `/work` and a writable `/rw`, 1 GiB memory, CPU limits, runs as nobody.
- `runner_in_container.py`: runs and grades tests inside the container, with token compare or checker.
- `problems.py`: builds the dataset.
- `env.py`: tools, Vals-style termination, grading. A cookbook `Env`, so it can be reused for RL.
- `eval.py`: runs a batch of problems and summarizes them.
