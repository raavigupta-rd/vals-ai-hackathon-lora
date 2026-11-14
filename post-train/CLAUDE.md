# Post-Train with Tinker — hackathon track

This folder holds the work for the "Post-Train with Tinker" track. Read these rules before you write
training code, build datasets, or prepare a submission. A rule break disqualifies the team.

## Goal

- Post-train a small model on Tinker to solve hard competitive programming problems in C++ (C++20).
- The team with the highest score on the hidden test set wins.
- If two teams have the same score, the team whose model used fewer total turns wins.
- Allowed methods: SFT, RL, expert iteration, or distillation from a larger open-source model.

## Rules (hard constraints)

- Do not try to read, extract, store, or share the scoring problems. Do not train the model to leak them.
- Do not train on the scoring problems or on solutions to them.
- Public datasets are allowed. Distillation from larger **open-source** models is allowed.
- **Do not distill from Anthropic or OpenAI models.** This includes Claude. Claude must not write C++
  solutions, reasoning traces, or any other text that goes into a training dataset. Claude may write the
  pipeline code (data loaders, training loops, eval harness), but it must not write the training data.
- Submit only checkpoints that the team trained from a base model.
- The submitted model must have **at most 9B active parameters**. A mixture-of-experts (MoE) model
  qualifies when its active parameter count is 9B or less, for example a 35B-A3B model.
- Staff review scoring traces, token counts, and turn counts to find cheating. If a method is not
  clearly allowed, ask the staff first.

## Submission

1. Save the checkpoint with `save_weights_for_sampler`. This gives a `tinker://` path.
2. Use "Submit a model" and enter the `tinker://` path, the Tinker API key, and the sampling settings.
3. The system first runs basic tests to make sure the checkpoint works. If they pass, grading starts.

Limits:
- 4 submission attempts per hour per team. Each new submission or edit uses one attempt.
- Only one submission can be processing at a time.
- **Deadline: Sunday at 2pm.**
- The team is judged on its best submission. The test set is private; you see only your score and rank.

## Scoring environment (what the model must handle)

- Each task is one hard competitive programming problem. The model writes a C++20 solution, which is
  graded on hidden tests.
- The model has two tools:
  1. **Compile and run** C++ on any input.
  2. **Submit** a solution. The tool reports the result on the sample tests.
- The model can submit up to **50 times**. The grader scores the last submission that compiles.
- The task stops when the model answers `EXIT` or runs out of turns.
- **Repeated malformed tool calls end the task.** Train the model to produce correct tool calls.
- Output length and thinking settings can be changed in "Advanced Settings" at submission time.
- Ties are broken by fewest overall turns, so do not waste turns. Stop with `EXIT` after an accepted
  solution.
- Small hidden tests are worth 40 points. Large hidden tests are worth 60 points. Efficient algorithms
  matter, not only correct ones.

## Problem format

Each problem is one text file. It contains the statement, input/output format, constraints, and sample
tests. The model also gets the sample tests as files, plus the time and memory limits (the example uses
2 seconds and 1 GiB per test).

Example (not in the test set):

```
DISTINCT STALLS

A market street has N stalls in a row, numbered 1 to N. Stall i sells
fruit of kind A_i. You will be asked Q questions. Each question gives
two stalls L and R, and asks how many different kinds of fruit are sold
by the stalls L, L+1, ..., R.

Input
The first line contains N and Q.
The second line contains A_1 ... A_N.
Each of the next Q lines contains L and R.

Output
For each question print the answer on its own line.

Constraints
1 <= N, Q <= 200000
1 <= A_i <= 10^9
1 <= L <= R <= N

Sample input
5 3
1 2 1 3 2
1 3
2 5
3 3

Sample output
2
3
1
```

## Resources

- Tinker console: sign in, then claim the credits from the "Your credits" box.
- Tinker docs: the SDK, training loops, and sampling.
- Tinker Cookbook recipes: ready-made SFT, RL, and distillation recipes.
- "Save a checkpoint" page: how to use `save_weights_for_sampler` to get the `tinker://` path.
- "Models" page: which base models Tinker serves, and their context sizes. Pick the base model from this
  list.
