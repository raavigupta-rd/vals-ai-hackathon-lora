"""Merge critic ratings (rated*.jsonl) with the answer key: calibration vs the teacher, and a weights file.

Rating line: {id, score 1-10, better_action, confidence, reason}. `id` is the blinded id from the answer key.

Calibration (per teacher rule, on judged spots only, i.e. where the teacher has a preferred action): the critic's
implied preference is the action taken when score >= GOOD, its better_action when score <= BAD (ambiguous in
between, which counts as a disagreement), and calibration = share equal to the teacher's preferred action.

Weights ({span_id: weight}): drop (0) anchors scored <= 3 by a high-confidence critic, upweight (1.5) those scored
8-10, everything else 1. "Critic-only corrections" are high-confidence better_action on spots where the teacher
abstains, within a rule whose calibration is >= 0.75 on >= min_n judged spots; they are written to a separate file
for re-roll validation (never used directly as labels).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

GOOD = 6
BAD = 4
DROP_SCORE = 3
UP_SCORE = 8
CONF_WORDS = {"high": 0.9, "medium": 0.6, "low": 0.3}
ACTIONS = {"fold", "check", "call", "raise", "bet"}


def _act(action):
    if not isinstance(action, str):
        return None
    action = action.strip().lower().split()[0] if action.strip() else None
    if action == "bet":
        return "raise"
    return action if action in ACTIONS else None


def norm_confidence(value) -> float:
    if isinstance(value, str):
        return CONF_WORDS.get(value.strip().lower(), 0.0)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def critic_preference(rating: dict, actual) -> str | None:
    score = rating["score"]
    if score >= GOOD:
        return _act(actual)
    if score <= BAD:
        return _act(rating.get("better_action"))
    return None


def load_ratings(paths) -> tuple[dict, int]:
    """Valid ratings by id (later lines win) and the count of rejected lines."""
    rated, bad = {}, 0
    for path in paths:
        for line in Path(path).open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                score = float(row["score"])
                if not (1 <= score <= 10) or not isinstance(row.get("id"), str):
                    raise ValueError
            except (ValueError, KeyError, TypeError):
                bad += 1
                continue
            rated[row["id"]] = row | {"score": score}
    return rated, bad


def _teacher_pref(entry):
    verdict = entry.get("verdict") or {}
    return _act(verdict.get("preferred"))


def calibration(key: dict, rated: dict) -> dict:
    stats = defaultdict(lambda: {"n": 0, "agree": 0, "agree_high_conf": 0, "n_high_conf": 0})
    for rid, rating in rated.items():
        entry = key.get(rid)
        if not entry or _teacher_pref(entry) is None:
            continue
        s = stats[entry.get("rule") or "none"]
        ok = critic_preference(rating, entry.get("actual")) == _teacher_pref(entry)
        s["n"] += 1
        s["agree"] += ok
        if norm_confidence(rating.get("confidence")) >= 0.7:
            s["n_high_conf"] += 1
            s["agree_high_conf"] += ok
    for s in stats.values():
        s["calibration"] = s["agree"] / s["n"]
    return dict(stats)


def build_weights(key: dict, rated: dict, *, min_n: int = 20, min_calibration: float = 0.75, high_conf: float = 0.7,
                  up_conf: float = 0.5, up_weight: float = 1.5) -> dict:
    cal = calibration(key, rated)
    weights, corrections = {}, []
    for rid, rating in rated.items():
        entry = key.get(rid)
        if not entry:
            continue
        conf = norm_confidence(rating.get("confidence"))
        score = rating["score"]
        span = entry["span_id"]
        teacher_agrees = _teacher_pref(entry) is not None and _teacher_pref(entry) == _act(entry.get("actual"))
        if score <= DROP_SCORE and conf >= high_conf and not teacher_agrees:  # the teacher's math has a veto
            weights[span] = 0.0
        elif score >= UP_SCORE and conf >= up_conf:
            weights[span] = up_weight
        else:
            weights[span] = 1.0
        better, actual = _act(rating.get("better_action")), _act(entry.get("actual"))
        rule = entry.get("rule") or "none"
        c = cal.get(rule)
        if (_teacher_pref(entry) is None and better and better != actual and score <= BAD and conf >= high_conf
                and c and c["n"] >= min_n and c["calibration"] >= min_calibration):
            corrections.append({"span_id": span, "id": rid, "rule": rule, "actual": actual, "better_action": better,
                                "score": score, "confidence": conf, "reason": rating.get("reason"),
                                "rule_calibration": round(c["calibration"], 3)})
    return {"weights": weights, "corrections": corrections, "calibration": cal}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--key", type=Path, default=Path("data/critic/batches/answer_key.json"))
    parser.add_argument("--ratings", type=Path, nargs="+", required=True, help="rated*.jsonl files")
    parser.add_argument("--out", type=Path, default=Path("data/critic/weights.json"))
    parser.add_argument("--min-n", type=int, default=20, help="judged spots a rule needs before its calibration counts")
    parser.add_argument("--min-calibration", type=float, default=0.75)
    parser.add_argument("--high-conf", type=float, default=0.7)
    parser.add_argument("--up-weight", type=float, default=1.5)
    args = parser.parse_args(argv)
    key = json.loads(args.key.read_text())
    rated, bad = load_ratings(args.ratings)
    unknown = [r for r in rated if r not in key]
    result = build_weights(key, rated, min_n=args.min_n, min_calibration=args.min_calibration,
                           high_conf=args.high_conf, up_weight=args.up_weight)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result["weights"], indent=1))
    args.out.with_suffix(".corrections.json").write_text(json.dumps(result["corrections"], indent=1))
    args.out.with_suffix(".calibration.json").write_text(json.dumps(result["calibration"], indent=1))
    w = result["weights"].values()
    print(json.dumps({"rated": len(rated), "rejected_lines": bad, "unknown_ids": len(unknown),
                      "dropped": sum(v == 0 for v in w), "upweighted": sum(v > 1 for v in w),
                      "corrections": len(result["corrections"]),
                      "calibration": {r: round(c["calibration"], 3) for r, c in result["calibration"].items()}}))


if __name__ == "__main__":
    main()
