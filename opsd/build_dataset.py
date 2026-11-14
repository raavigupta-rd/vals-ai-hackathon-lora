"""Build an offline-OPSD dataset from the base model's own hands: one `HintedTask` per decision.

1. Run base on the train deals through the harness (`python -m vals_tools.eval --dataset pokerbots-train
   --deployment <base deployment>`).
2. Export that eval's traces and turn each policy decision into a task:
   - `input`: the decision's prompt (system message, user messages, earlier replies and tool results) followed by the
     reply base actually wrote (reasoning, text and its `submit_action` call);
   - `opsd_hint_selections`: one selection at the end of the prompt (`rollout_prefix_len`) that appends the hint to
     the system message (`append_to_previous_item` at item boundary 1). Only the teacher sees it;
   - `env_params.tools`: the tool schemas the model saw (`submit_action` and `update_scratchpad`, from tools.json).
     Training refuses a transcript with tool calls unless the tools are listed here.
3. Upload as a `hinted` dataset.

Keep only clean decisions: exactly one `submit_action` call and a reply under the token cap. The after-hand
`update_scratchpad` turns are skipped. Use train deals only.

    uv run python opsd/build_dataset.py --eval eval_xxx --dataset pokerbots-opsd-train --dry-run
"""

from __future__ import annotations

import argparse
import json
import hashlib
import random
import sys
from collections.abc import Callable
from pathlib import Path

from ac2.runtime import FunctionCall, HintedTask, HintMutation, HintSelection, Item, Tool
from ac2.sdk import Client
from pydantic import TypeAdapter
from pydantic import ValidationError
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hint import HINT, SEPARATOR  # noqa: E402
from poker.hints import HintBuilder, parse_prompt  # noqa: E402
from poker.trace_quality import cutoff, is_cleanup, legal_reply  # noqa: E402

ITEMS = TypeAdapter(list[Item])
SUBMIT = "submit_action"
SCRATCHPAD = "update_scratchpad"
# The tools every seat sees at an event table: submit_action (with `say`, since talk is on) and update_scratchpad.
# They must match what the model saw, field for field; tools.json is generated from the harness and a harness test
# checks it.
TOOLS = [Tool.model_validate(tool).model_dump() for tool in json.loads((Path(__file__).parent / "tools.json").read_text())]


def decision_tasks(trace: dict, hint: str | Callable, max_tokens: int) -> tuple[list[HintedTask], dict[str, int]]:
    """One task per clean policy decision in a trace, with counts of what was kept and why the rest was skipped."""
    spans = sorted(
        (s for s in trace["spans"] if s["name"] == "get_completion"),
        key=lambda s: s["start_time"],
    )
    stats = {key: 0 for key in ("kept", "cleanup", "format", "truncated", "missing_tokens", "status", "parse", "arguments", "illegal")}
    stats["decisions"] = len(spans)
    tasks = []
    for span in spans:
        if span["status"] != "OK" or not span.get("end_time"):
            stats["status"] += 1
            continue
        try:
            prompt = ITEMS.validate_json(span["input"])
            reply = ITEMS.validate_json(span["output"])
        except (ValueError, ValidationError):
            stats["format"] += 1
            continue
        calls = [item for item in reply if isinstance(item, FunctionCall)]
        prompt_json = [i.model_dump(mode="json") for i in prompt]
        if is_cleanup(prompt_json):
            stats["cleanup"] += 1
            continue
        if len(calls) != 1 or calls[0].name != SUBMIT:
            stats["format"] += 1
            continue
        tokens = span.get("tokens_out")
        if type(tokens) is not int or tokens <= 0:
            stats["missing_tokens"] += 1
            continue
        if cutoff(span) or tokens >= max_tokens:
            stats["truncated"] += 1
            continue
        error = legal_reply(prompt_json, [i.model_dump(mode="json") for i in reply])
        if error:
            stats[error] += 1
            continue
        decision_hint = hint(prompt) if callable(hint) else hint
        tasks.append(
            HintedTask(
                input=prompt + reply,
                env_params={"tools": TOOLS, "source_trace": trace["trace_id"], "source_span": span.get("span_id"), "source_tokens_out": tokens},
                opsd_hint_selections=[
                    HintSelection(
                        hint=decision_hint,
                        rollout_prefix_len=len(prompt),
                        mutation=HintMutation(item_boundary=1, action="append_to_previous_item", separator=SEPARATOR),
                    )
                ],
            )
        )
        stats["kept"] += 1
    return tasks, stats


def check_provenance(trace: dict, eval_id: str, dataset: str, task_ids: set[str]) -> None:
    spans = [s for s in trace["spans"] if s["name"] == "get_completion"]
    if not spans:
        return
    if any(s.get("job_id") != eval_id or s.get("dataset") != dataset or s.get("datapoint_id") not in task_ids for s in spans):
        raise ValueError("trace source does not match the verified base train eval and dataset membership")


def verify_source(client, eval_id, dataset, *, allow_running=False):
    import re
    if not re.search(r"(^|-)train($|-)", dataset) or re.search(r"(^|-)(eval|select|test|val)($|-)", dataset):
        raise ValueError("OPSD source must be an explicit training dataset, never eval/select/test/validation")
    run = client.eval.get(eval_id)
    if run.status not in (("succeeded", "running") if allow_running else ("succeeded",)):
        raise ValueError("source eval must have succeeded (or be running for an explicitly partial dry run)")
    if {t.dataset for t in run.tasks} != {dataset}:
        raise ValueError("source eval dataset does not match --source-dataset")
    source_model = run.launch_params.get("custom_harness", {}).get("model", {}).get("deployment")
    if run.tags.get("seat_model") != "vals-hackathon/base" or source_model != "vals-hackathon/base":
        raise ValueError("source must be the recorded base deployment, not a checkpoint or house opponent")
    refs = client.datasets.active_tasks(dataset)
    task_ids = {t.id for t in refs}
    # Event preset tools are generated/tested against the official harness. Refuse non-event source tasks.
    for task in client.datasets.load_tasks(dataset):
        params = task.env_params if hasattr(task, "env_params") else task.get("env_params", {})
        if params.get("preset") != "event" or params.get("submit_fields", ["action", "amount"]) != ["action", "amount"]:
            raise ValueError("source task tools do not match the audited event schema")
    return run, task_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="vals-hackathon")
    parser.add_argument("--eval", required=True, help="base model's eval on the train deals")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max-tokens", type=int, default=5_000, help="replies this long count as truncated (the event's cap)")
    parser.add_argument("--traces", type=Path, help="reuse a downloaded trace export")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--records", type=Path, help="cleaned train.jsonl labels; never test or validation records")
    parser.add_argument("--equity-samples", type=int, default=64, help="sampled tight-range equity per decision; 0 disables")
    parser.add_argument("--generic-hint", action="store_true", help="use the starter global hint instead of per-decision hints")
    parser.add_argument("--hint-report", type=Path, help="write hints for 20 deterministic random decisions for review")
    parser.add_argument("--hint-file", type=Path, help="dedicated short teacher hint; e.g. poker/decide_fast_hint.txt")
    parser.add_argument("--source-dataset", default="pokerbots-train")
    parser.add_argument("--allow-running-dry-run", action="store_true", help="inspect completed decisions from a running train eval; never uploads")
    parser.add_argument("--reviewed-report-sha256", help="SHA-256 of the 20-decision report reviewed before upload")
    args = parser.parse_args()
    if args.equity_samples < 0:
        parser.error("--equity-samples must be nonnegative")
    if args.allow_running_dry_run and not args.dry_run:
        parser.error("--allow-running-dry-run requires --dry-run")
    hint_builder = args.hint_file.read_text(encoding="utf-8") if args.hint_file else HINT if args.generic_hint else HintBuilder(args.records, equity_samples=args.equity_samples)
    client = Client(project=args.project)
    source, task_ids = verify_source(client, args.eval, args.source_dataset, allow_running=args.allow_running_dry_run)

    path = args.traces or Path(f"traces/{args.eval}.jsonl")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        Client(project=args.project).traces.download(args.eval, path)
    tasks, totals = [], {}
    with path.open() as lines:
        for line in lines:
            trace = json.loads(line)
            check_provenance(trace, args.eval, args.source_dataset, task_ids)
            built, stats = decision_tasks(trace, hint_builder, args.max_tokens)
            tasks.extend(built)
            totals = {k: totals.get(k, 0) + stats[k] for k in stats}
    random.Random(0).shuffle(tasks)
    print(json.dumps(totals | {"tasks": len(tasks)}))
    if args.hint_report:
        args.hint_report.parent.mkdir(parents=True, exist_ok=True)
        sample = random.Random(0).sample(tasks, min(20, len(tasks)))
        report = {"source_eval": args.eval, "source_dataset": args.source_dataset,
                  "manifest": source.tasks[0].task_manifest_uri, "counts": totals, "decisions": [
                    {"source_trace": task.env_params["source_trace"], "source_span": task.env_params["source_span"],
                     "tokens_out": task.env_params["source_tokens_out"], "state": parse_prompt(task.input[:task.opsd_hint_selections[0].rollout_prefix_len]),
                     "action": next(i.arguments for i in task.input[task.opsd_hint_selections[0].rollout_prefix_len:] if isinstance(i, FunctionCall)),
                     "hint": task.opsd_hint_selections[0].hint} for task in sample]}
        report_text = json.dumps(report, indent=2)
        report_hash = hashlib.sha256(report_text.encode()).hexdigest()
        if not args.dry_run and args.reviewed_report_sha256 != report_hash:
            parser.error("review report differs or has not been reviewed; run --dry-run, inspect 20 decisions, then pass --reviewed-report-sha256")
        args.hint_report.write_text(report_text, encoding="utf-8")
        print(json.dumps({"hint_report_sha256": report_hash}))
    if args.dry_run or not tasks:
        return
    if not args.hint_report or len(tasks) < 20:
        parser.error("upload requires --hint-report and at least 20 retained decisions to review first")
    client.datasets.create(args.dataset, required_types=["hinted"], exist_ok=False)
    for start in range(0, len(tasks), 500):
        client.datasets.add_tasks(args.dataset, tasks=tasks[start : start + 500])
    print(f"uploaded {len(tasks)} tasks to {args.dataset}")


if __name__ == "__main__":
    main()
