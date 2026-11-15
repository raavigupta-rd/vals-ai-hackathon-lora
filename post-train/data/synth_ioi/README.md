# Synthetic Vals Track-2 / IOI-Style Benchmark & Training Suite (`synth_ioi`)

40 original, cross-verified competitive programming problems (`synth_01` – `synth_40`) authored in the exact **Vals Track-2 (`DISTINCT STALLS`)** single-file format (`2 seconds`, `1024 MiB`, `40-point small` + `60-point large` hidden test split).

## Verification Guarantees
Every problem in this suite was built and verified in 8 batches of 5 (`batch_01.py` .. `batch_08.py`):
1. **Correctness (`fast.cpp == brute.cpp`)**: Both the optimal C++20 solution (`fast.cpp`) and an independent $O(N^2)$ / subtask brute-force solution (`brute.cpp`) compile under `g++ -std=c++20 -O2 -include bits/stdc++.h` and produce **identical token outputs** across all sample and `hidden_small` tests.
2. **40 / 100 Subtask Calibration**:
   - `brute.cpp` passes 100% of `hidden_small` (`40.0` points) in `< 0.15s` and reliably `TLE`s (`> 2.0s`) on `hidden_large`.
   - `fast.cpp` passes 100% of both `hidden_small` and `hidden_large` (`100.0` points).

## Files
- `problems.jsonl`: 40 problems in the exact `harness.problems` schema (hidden test files live under `.cache/work/ro/tests/synth_XX/`).
- `solutions.jsonl`: Paired `fast_cpp` (100-pt), `brute_cpp` (40-pt + stress-test oracle), `tags`, `rating`, and `notes` for SFT trajectory generation / hint-guided distillation / RL.
- `problems/synth_XX/{statement.txt, fast.cpp, brute.cpp}`: Individual per-problem bundles.

## Usage with `harness.eval`
```bash
uv run python -m harness.eval --problems data/synth_ioi/problems.jsonl --tag synth40
```

## Rebuild All 8 Batches
```bash
for b in 01 02 03 04 05 06 07 08; do
    .venv/bin/python data/synth_ioi/batch_${b}.py
done
```
