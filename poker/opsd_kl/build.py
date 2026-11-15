"""Offline-OPSD dataset that approximates  L = L_OPSD(corrections) + lambda * KL(student || base)  on anchors.

Every task is one recorded arena decision by the BASE deployment. Input = the exact prompt plus the reply base wrote
(offline OPSD replays the policy's own tokens, so we start from base). The teacher is the frozen base model plus a
teacher-only hint appended to the system message (HintMutation append_to_previous_item at boundary 1, as in
opsd/build_dataset.py):

- correction tasks: decisions the teacher (poker.teacher.judge) disagreed with and make_records validated
  (`source == "teacher_resampled"`). The hint is the coaching text the validated resample used (resample.HINT_HEADER +
  make_records.correction_reasoning). Duplicated --correction-copies times (bust rules allin_call/own_shove: 3).
- anchor tasks: other clean base decisions. The hint is neutral, so the teacher is (nearly) base and the OPSD loss is
  (nearly) KL(student || base). lambda is set by --anchor-ratio = anchors per correction task (copies included).

The SDK refuses an empty hint list and a blank hint (HintedTask min_length=1, HintSelection.validate_hint), so
anchors cannot carry "no hint". They carry the same header with "No special notes for this decision." instead.

    uv run python -m poker.opsd_kl.build --records R.jsonl            # dry run (default)
    uv run python -m poker.opsd_kl.build --records R.jsonl --upload --dataset NAME
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path

from ac2.runtime import FunctionCall, HintedTask, HintMutation, HintSelection, Message, Reasoning

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from poker.sft.build_sft_dataset import TOOLS, Rejected, _prompt_items, estimate_tokens  # noqa: E402
from poker.sft.make_records import BUST_RULES, correction_reasoning  # noqa: E402
from poker.sft.resample import HINT_HEADER  # noqa: E402
from poker.trace_quality import legal_reply  # noqa: E402

BASE_DEPLOYMENT = "infer_d3d2b6b92171"
NEUTRAL = "No special notes for this decision."
MAX_REASONING_TOKENS = 2_500
MAX_TOKENS = 5_000  # the event's per-reply cap
BUST_COPIES = 3
SUBMIT = "submit_action"


def clean_reply(record: dict):
    """(prompt items, reply items) for a clean base reply, or None: one submit_action, not cut off, legal, short."""
    calls = [c for c in record.get("tool_calls") or [] if c.get("name") == SUBMIT]
    reasoning = record.get("reasoning") or ""
    tokens = (record.get("usage") or {}).get("tokens_out")
    if record.get("cutoff") or len(calls) != 1 or len(record.get("tool_calls") or []) != 1 or not reasoning.strip():
        return None
    if estimate_tokens(reasoning) > MAX_REASONING_TOKENS or (type(tokens) is int and tokens >= MAX_TOKENS):
        return None
    try:
        prompt = _prompt_items(record["messages"])
    except Rejected:
        return None
    args = calls[0].get("arguments")
    args = args if isinstance(args, str) else json.dumps(args, separators=(",", ":"))
    call = FunctionCall(call_id=calls[0].get("call_id") or f"call_{record['span_id']}", name=SUBMIT, arguments=args)
    prompt_json = [i.model_dump(mode="json") for i in prompt]
    if legal_reply(prompt_json, [call.model_dump(mode="json")]):
        return None
    reply = [Reasoning(content=reasoning)]
    if record.get("content"):  # keep even blank content: offline OPSD replays the exact tokens the model wrote
        reply.append(Message(role="assistant", content=record["content"]))
    return prompt, reply + [call]


def hinted_task(record: dict, hint: str, kind: str, rule: str | None = None) -> HintedTask | None:
    built = clean_reply(record)
    if built is None:
        return None
    prompt, reply = built
    env = {"tools": TOOLS, "kind": kind, "source_span": record["span_id"], "deployment": record.get("deployment")}
    if rule:
        env["rule"] = rule
    return HintedTask(
        input=prompt + reply,
        env_params=env,
        opsd_hint_selections=[HintSelection(
            # Anchors get a bare neutral line (the SDK forbids an empty hint), so the teacher stays as close to
            # base as possible and the loss there approximates KL to base. Corrections get the coaching header.
            hint=("\n\n" + hint) if kind == "anchor" else HINT_HEADER + hint,
            rollout_prefix_len=len(prompt),
            mutation=HintMutation(item_boundary=1, action="append_to_previous_item", separator=""),
        )],
    )


def build(entries, correction_ids, code_for, *, anchor_ratio=3.0, correction_copies=2, seed=0,
          deployment=BASE_DEPLOYMENT):
    """entries: [{"record", "verdict"}] from poker.gate.load_entries. Returns (tasks, stats)."""
    stats = {"corrections": 0, "correction_tasks": 0, "bust_corrections": 0, "correction_dropped": 0,
             "anchor_pool": 0, "anchors": 0, "disagree_excluded": 0, "unclean": 0}
    corrections, pool = [], []
    for entry in entries:
        record, verdict = entry["record"], entry["verdict"]
        if record.get("deployment") != deployment:
            continue
        if record["span_id"] in correction_ids:
            try:
                task = hinted_task(record, correction_reasoning(verdict, code_for(record)), "correction", verdict.get("rule"))
            except (ValueError, KeyError):
                task = None
            if task is None:
                stats["correction_dropped"] += 1
                continue
            bust = verdict.get("rule") in BUST_RULES
            copies = BUST_COPIES if bust else correction_copies
            stats["corrections"] += 1
            stats["bust_corrections"] += bust
            corrections += [task.model_copy(deep=True) for _ in range(copies)]
        elif verdict.get("agree") is False:
            stats["disagree_excluded"] += 1  # a spot we would change but have no validated correction for
        else:
            task = hinted_task(record, NEUTRAL, "anchor")
            if task is None:
                stats["unclean"] += 1
            else:
                pool.append(task)
    stats["correction_tasks"] = len(corrections)
    stats["anchor_pool"] = len(pool)
    rng = random.Random(seed)
    rng.shuffle(pool)
    anchors = pool[: round(anchor_ratio * len(corrections))]
    stats["anchors"] = len(anchors)
    tasks = corrections + anchors
    rng.shuffle(tasks)
    stats["tasks"] = len(tasks)
    return tasks, stats


def render(task: HintedTask) -> dict:
    shown = task.model_dump(mode="json")
    shown["env_params"] = {**shown["env_params"], "tools": [t["name"] for t in shown["env_params"]["tools"]]}
    return shown


def main() -> None:
    from poker.gate import load_entries
    from poker.teacher import hand_code

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default="vals-hackathon")
    p.add_argument("--records", required=True, type=Path, help="poker.sft.make_records output (teacher_resampled rows)")
    p.add_argument("--joined", type=Path, default=Path("data/arena/joined.jsonl"))
    p.add_argument("--db", type=Path, default=Path("data/arena/arena.db"))
    p.add_argument("--deployment", default=BASE_DEPLOYMENT, help="the base deployment; offline OPSD replays its tokens")
    p.add_argument("--anchor-ratio", type=float, default=3.0, help="anchor tasks per correction task (lambda)")
    p.add_argument("--correction-copies", type=int, default=2, help="copies per correction (bust rules: 3)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="default: nothing is uploaded")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--dataset", help="required with --upload")
    args = p.parse_args()
    if args.upload and (args.dry_run or not args.dataset):
        p.error("--upload needs --dataset NAME and excludes --dry-run")
    if args.anchor_ratio < 0 or args.correction_copies < 1:
        p.error("--anchor-ratio must be >= 0 and --correction-copies >= 1")
    ids = set()
    for line in args.records.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("source") == "teacher_resampled":
            ids.add(row["decision_id"])
    con = sqlite3.connect(args.db)

    def code_for(record):
        m = record["match"]
        row = con.execute("SELECT card1, card2 FROM hole_cards WHERE match_id=? AND hand_number=? AND seat=?",
                          (m["match_id"], m["hand"], m["seat"])).fetchone()
        return hand_code([row[0], row[1]]) if row and row[0] else "this hand"

    entries = load_entries(args.joined, args.db, args.deployment)
    tasks, stats = build(entries, ids, code_for, anchor_ratio=args.anchor_ratio,
                         correction_copies=args.correction_copies, seed=args.seed, deployment=args.deployment)
    stats["records_corrections"] = len(ids)
    print(json.dumps(stats | {"mode": "upload" if args.upload else "dry-run"}))
    for kind in ("correction", "anchor"):
        task = next((t for t in tasks if t.env_params["kind"] == kind), None)
        if task:
            print(f"One rendered {kind} task (tools elided):")
            print(json.dumps(render(task), indent=1))
    if not args.upload:
        print("Dry run: nothing uploaded. Pass --upload --dataset NAME to upload.")
        return
    if not tasks:
        p.error("no tasks to upload")
    from ac2.sdk import Client
    client = Client(project=args.project)
    client.datasets.create(args.dataset, required_types=["hinted"], exist_ok=False)
    for start in range(0, len(tasks), 500):
        client.datasets.add_tasks(args.dataset, tasks=tasks[start:start + 500])
    print(f"uploaded {len(tasks)} tasks to {args.dataset}")


if __name__ == "__main__":
    main()
