# `curated_r1` — Verified Public Codeforces Dataset for `lenz-sft-2`

Cloud-curated and `gcc:14`-verified dataset built by [`harness/build_curated_r1.py`](../../harness/build_curated_r1.py) from public unauthenticated HuggingFace datasets:
- [`open-r1/codeforces`](https://huggingface.co/datasets/open-r1/codeforces) (`train`, `test`, and `generated_tests/test_cases_XXXX.parquet`)
- [`open-r1/codeforces-submissions`](https://huggingface.co/datasets/open-r1/codeforces-submissions) (`selected_accepted`, `selected_incorrect`)

## Splits & Verified Counts (`stats.json`)

| File | Source Split | Problems | Rating Range | Single-Instance | Median Max Input | P90 Max Input |
|---|---|---:|---|---:|---:|---:|
| `eval.jsonl` + `eval_solutions.jsonl` | 2024–2025 `test` | **56** | `1400–2600` | `8.9%` (`5/56`) | `2.08 MB` | `5.67 MB` |
| `train.jsonl` + `train_solutions.jsonl` | `train` (pre-cutoff) | **600** | `1500–2600` | `95.8%` (`575/600`) | `1.30 MB` | `4.18 MB` |

All hidden test pairs (`Small` 40 pts + `Large` 60 pts) are staged on the Modal Volume `cp-tests` under `/work/tests/r1_<contest_id>_<index>/` and verified end-to-end against real accepted C++20 submissions using the exact chroot runner (`g++ -std=c++20 -O2 -pipe -static`, strict wall-clock limit `<= 0.80 * time_limit`).

## Filtering & Anti-Contamination Guarantees
- **Deterministic stdio matching only:** excludes interactive tasks, custom checkers (`generated_checker` not null), floating-point tolerance tasks, and multi-answer prompts (`"print any"`, etc.).
- **Zero train/eval overlap:** `train.jsonl` and `eval.jsonl` have strictly disjoint `id`, `or1_id`, `contest_id`, Div1/Div2 `aliases`, and normalized titles, and avoid overlap with legacy `cf_test.jsonl` (82 IDs) and `sft_b_v1.problem_ids.txt` (1,773 IDs).
- **Large-$N$ scaling enforcement:** drops trivially small test sets unless empirically verified TLE against a human TLE C++ submission.

## Train Rating Distribution (`600` problems)
`1500`: 52 | `1600`: 46 | `1700`: 79 | `1800`: 68 | `1900`: 77 | `2000`: 76 | `2100`: 61 | `2200`: 63 | `2300`: 18 | `2400`: 19 | `2500`: 26 | `2600`: 15
