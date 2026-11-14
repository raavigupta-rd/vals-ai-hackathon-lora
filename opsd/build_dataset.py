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
import random
import sys
from collections.abc import Callable
from pathlib import Path

from ac2.runtime import FunctionCall, HintedTask, HintMutation, HintSelection, Item, Tool
from ac2.sdk import Client
from pydantic import TypeAdapter
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hint import HINT, SEPARATOR  # noqa: E402
from poker.hints import HintBuilder  # noqa: E402

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
        (s for s in trace["spans"] if s["name"] == "get_completion" and s["status"] == "OK"),
        key=lambda s: s["start_time"],
    )
    stats = {"decisions": len(spans), "kept": 0, "cleanup": 0, "format": 0, "truncated": 0}
    tasks = []
    for span in spans:
        prompt = ITEMS.validate_json(span["input"])
        reply = ITEMS.validate_json(span["output"])
        calls = [item for item in reply if isinstance(item, FunctionCall)]
        if calls and calls[0].name == SCRATCHPAD:  # the after-hand cleanup turn, not a poker decision
            stats["cleanup"] += 1
            continue
        if len(calls) != 1 or calls[0].name != SUBMIT:
            stats["format"] += 1
            continue
        if int(span.get("tokens_out") or 0) >= max_tokens:
            stats["truncated"] += 1
            continue
        decision_hint = hint(prompt) if callable(hint) else hint
        tasks.append(
            HintedTask(
                input=prompt + reply,
                env_params={"tools": TOOLS, "source_trace": trace["trace_id"]},
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
    args = parser.parse_args()
    if args.equity_samples < 0:
        parser.error("--equity-samples must be nonnegative")
    hint_builder = HINT if args.generic_hint else HintBuilder(args.records, equity_samples=args.equity_samples)

    path = args.traces or Path(f"traces/{args.eval}.jsonl")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        Client(project=args.project).traces.download(args.eval, path)
    tasks, totals = [], {"decisions": 0, "kept": 0, "cleanup": 0, "format": 0, "truncated": 0}
    with path.open() as lines:
        for line in lines:
            built, stats = decision_tasks(json.loads(line), hint_builder, args.max_tokens)
            tasks.extend(built)
            totals = {k: totals[k] + stats[k] for k in totals}
    random.Random(0).shuffle(tasks)
    print(json.dumps(totals | {"tasks": len(tasks)}))
    if args.hint_report:
        args.hint_report.parent.mkdir(parents=True, exist_ok=True)
        sample = random.Random(0).sample(tasks, min(20, len(tasks)))
        args.hint_report.write_text(json.dumps([
            {"source_trace": task.env_params["source_trace"], "hint": task.opsd_hint_selections[0].hint}
            for task in sample
        ], indent=2), encoding="utf-8")
    if args.dry_run or not tasks:
        return
    client = Client(project=args.project)
    client.datasets.create(args.dataset, required_types=["hinted"], exist_ok=True)
    for start in range(0, len(tasks), 500):
        client.datasets.add_tasks(args.dataset, tasks=tasks[start : start + 500])
    print(f"uploaded {len(tasks)} tasks to {args.dataset}")


if __name__ == "__main__":
    main()
