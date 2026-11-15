"""Turn reviewed poker decisions into AC2 SFT tasks, in exactly the event's chat format.

Input: a JSONL file, one decision per line:

    {"messages": [<system>, <user>, ...],        # exactly what the model saw (chat messages or SDK items)
     "target": {"reasoning": str,                # Qwen thinking text
                "content": str,                  # visible assistant text, may be ""
                "tool_call": {"name": "submit_action", "arguments": {...}}},
     "weight": 1.0,                              # optional
     "source": str, "decision_id": str}

Output: one `Task` (dataset type `traces`) per decision. `input` is the prompt items followed by the reply items
in the order the event traces store them: `Reasoning`, an optional assistant `Message`, then the `FunctionCall`.
SFT trains every assistant message and function call and masks user, system and tool-output items. The tool
schemas (`opsd/tools.json`) go in `env_params["tools"]`.

Weights: the SDK has no per-task weight, so a weight becomes copies of the task: floor(weight) plus one more with
probability equal to the fractional part (decided by a hash of `decision_id`, so it is repeatable). Weight 0 drops
the record. Copies are capped at MAX_COPIES.

    uv run python poker/sft/build_sft_dataset.py --records decisions.jsonl                      # dry run (default)
    uv run python poker/sft/build_sft_dataset.py --records decisions.jsonl --upload --dataset NAME
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

from ac2.runtime import FunctionCall, Item, Message, Reasoning, Task, Tool
from pydantic import TypeAdapter, ValidationError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from poker.hints import parse_prompt  # noqa: E402
from poker.trace_quality import legal_reply  # noqa: E402

ITEMS = TypeAdapter(list[Item])
SUBMIT = "submit_action"
ARG_KEYS = {"action", "amount", "say"}
MAX_COPIES = 10
CHARS_PER_TOKEN = 3.5  # conservative for English plus poker notation
DEFAULT_MAX_REASONING_TOKENS = 2_500
TOOLS = [Tool.model_validate(t).model_dump() for t in json.loads((ROOT / "opsd" / "tools.json").read_text())]


class Rejected(Exception):
    """A record that cannot become a training example; `reason` is the stats key."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def estimate_tokens(text: str) -> int:
    return round(len(text) / CHARS_PER_TOKEN)


def copies_for(weight: float | None, decision_id: str) -> int:
    if weight is None:
        return 1
    if type(weight) not in (int, float) or weight != weight or weight < 0:
        raise Rejected("weight", repr(weight))
    whole = int(weight)
    fraction = weight - whole
    draw = int(hashlib.sha256(f"weight:{decision_id}".encode()).hexdigest()[:8], 16) / 0x1_0000_0000
    return min(MAX_COPIES, whole + (1 if fraction > draw else 0))


def _prompt_items(messages) -> list[Item]:
    if not isinstance(messages, list) or not messages:
        raise Rejected("format", "messages must be a non-empty list")
    raw = [{"type": "message", **m} if isinstance(m, dict) and "type" not in m else m for m in messages]
    try:
        items = ITEMS.validate_python(raw)
    except ValidationError as error:
        raise Rejected("format", str(error)[:200]) from error
    if not isinstance(items[0], Message) or items[0].role != "system":
        raise Rejected("format", "first message must be the system prompt")
    last = items[-1]
    if isinstance(last, (Reasoning, FunctionCall)) or (isinstance(last, Message) and last.role == "assistant"):
        raise Rejected("format", "messages must end before the assistant turn")
    return items


def _arguments(call) -> dict:
    if not isinstance(call, dict) or call.get("name") != SUBMIT or not isinstance(call.get("arguments"), dict):
        raise Rejected("format", "target.tool_call must be one submit_action call with object arguments")
    args = call["arguments"]
    if set(args) - ARG_KEYS or args.get("action") not in ("fold", "check", "call", "bet", "raise"):
        raise Rejected("arguments", json.dumps(args)[:100])
    if ("amount" in args) != (args["action"] in ("bet", "raise")) or (
        "amount" in args and type(args["amount"]) is not int
    ):
        raise Rejected("arguments", "amount is required for bet/raise (an int) and forbidden otherwise")
    if "say" in args and not isinstance(args["say"], str):
        raise Rejected("arguments", "say must be a string")
    return args


def record_to_task(record: dict, *, max_reasoning_tokens: int = DEFAULT_MAX_REASONING_TOKENS,
                   strict: bool = False) -> tuple[Task, int]:
    """Validate one record; return (task, copies). Raises `Rejected` with a stats reason."""
    if not isinstance(record, dict) or not isinstance(record.get("target"), dict):
        raise Rejected("format", "record needs messages and target")
    decision_id = str(record.get("decision_id", ""))
    target = record["target"]
    prompt = _prompt_items(record.get("messages"))
    args = _arguments(target.get("tool_call"))
    reasoning, content = target.get("reasoning"), target.get("content") or ""
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise Rejected("no_reasoning")
    if not isinstance(content, str):
        raise Rejected("format", "content must be a string")
    if estimate_tokens(reasoning) > max_reasoning_tokens:
        raise Rejected("too_long", f"~{estimate_tokens(reasoning)} tokens")
    call = FunctionCall(call_id="call_" + hashlib.sha256(decision_id.encode()).hexdigest()[:12], name=SUBMIT,
                        arguments=json.dumps(args, separators=(",", ":")))
    # Legality against the prompt's own "Legal actions" line, using the same parser the audits use.
    prompt_json = [i.model_dump(mode="json") for i in prompt]
    if parse_prompt(prompt_json) is None:
        if strict:
            raise Rejected("parse")
    else:
        error = legal_reply(prompt_json, [call.model_dump(mode="json")])
        if error:
            raise Rejected(error, json.dumps(args))
    copies = copies_for(record.get("weight"), decision_id)
    reply: list[Item] = [Reasoning(content=reasoning)]
    if content.strip():
        reply.append(Message(role="assistant", content=content))
    reply.append(call)
    env = {"tools": TOOLS, "source": str(record.get("source", "")), "decision_id": decision_id}
    if record.get("weight") is not None:
        env["weight"] = record["weight"]
    return Task(input=prompt + reply, env_params=env), copies


def load_records(path: Path) -> Iterator[dict | None]:
    """One dict per non-blank line; None for a line that is not JSON."""
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError:
                    yield None


def build_tasks(records: Iterable[dict | None], **options) -> tuple[list[Task], dict[str, int]]:
    stats = {"records": 0, "kept": 0, "copies": 0, "duplicate": 0, "weight_zero": 0}
    tasks: list[Task] = []
    seen: set[str] = set()
    for record in records:
        stats["records"] += 1
        try:
            if record is None:
                raise Rejected("bad_json")
            decision_id = record.get("decision_id") if isinstance(record, dict) else None
            if decision_id is not None and decision_id in seen:
                raise Rejected("duplicate")
            task, copies = record_to_task(record, **options)
            if decision_id is not None:
                seen.add(decision_id)
        except Rejected as rejected:
            stats[rejected.reason] = stats.get(rejected.reason, 0) + 1
            continue
        if copies == 0:
            stats["weight_zero"] += 1
            continue
        stats["kept"] += 1
        stats["copies"] += copies
        tasks.extend(task.model_copy(deep=True) for _ in range(copies))
    return tasks, stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default="vals-hackathon")
    p.add_argument("--records", required=True, type=Path)
    p.add_argument("--dataset", help="dataset name; required with --upload")
    p.add_argument("--upload", action="store_true", help="create the dataset and upload; without it nothing is sent")
    p.add_argument("--max-reasoning-tokens", type=int, default=DEFAULT_MAX_REASONING_TOKENS,
                   help="estimated tokens; longer reasoning is dropped (default 2500; the event cap is 5000 per reply)")
    p.add_argument("--strict", action="store_true", help="also drop records whose prompt cannot be parsed for legality")
    p.add_argument("--seed", type=int, default=0, help="shuffle seed")
    args = p.parse_args()
    if args.upload and not args.dataset:
        p.error("--upload needs --dataset NAME")
    import random
    tasks, stats = build_tasks(load_records(args.records), max_reasoning_tokens=args.max_reasoning_tokens,
                               strict=args.strict)
    random.Random(args.seed).shuffle(tasks)
    print(json.dumps(stats | {"tasks": len(tasks), "mode": "upload" if args.upload else "dry-run"}))
    if tasks:
        print("One rendered example (tools elided):")
        shown = tasks[0].model_dump(mode="json")
        shown["env_params"] = {**shown["env_params"], "tools": [t["name"] for t in shown["env_params"]["tools"]]}
        print(json.dumps(shown, indent=1))
    if not args.upload:
        print("Dry run: nothing uploaded. Pass --upload --dataset NAME to upload.")
        return
    if not tasks:
        p.error("no valid tasks to upload")
    from ac2.sdk import Client
    client = Client(project=args.project)
    client.datasets.create(args.dataset, required_types=["traces"], exist_ok=False)
    for start in range(0, len(tasks), 500):
        client.datasets.add_tasks(args.dataset, tasks=tasks[start:start + 500])
    print(f"uploaded {len(tasks)} tasks to {args.dataset}")


if __name__ == "__main__":
    main()
