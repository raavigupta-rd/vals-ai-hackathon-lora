"""Audit public decision legality, truncation, token use and complete task traces."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re

from poker.hints import parse_prompt, text_content


def items(value):
    return json.loads(value) if isinstance(value, str) else value


def attributes(span):
    value = span.get("attributes") or {}
    return json.loads(value) if isinstance(value, str) else value


def is_cleanup(prompt):
    last = next((text_content(i) for i in reversed(prompt) if text_content(i)), "")
    return bool(re.search(r"^Now call update_scratchpad|^Rejected:.*Call update_scratchpad", last, re.M))


def cutoff(span):
    data = attributes(span)
    return bool(data.get("truncated") or data.get("finish_reason") in ("length", "max_tokens") or span.get("finish_reason") == "length")


def legal_reply(prompt, reply):
    state = parse_prompt(prompt)
    if state is None:
        return "parse"
    calls = [i for i in reply if i.get("type") == "function_call"]
    if len(calls) != 1 or calls[0].get("name") != "submit_action":
        return "format"
    try:
        args = items(calls[0]["arguments"])
    except (ValueError, TypeError, KeyError):
        return "arguments"
    if not isinstance(args, dict) or args.get("action") not in ("fold", "check", "call", "bet", "raise"):
        return "arguments"
    kind = args["action"]
    if kind == "fold":
        return None if state["to_call"] > 0 else "illegal"
    if kind == "check":
        return None if state["to_call"] == 0 else "illegal"
    if kind == "call":
        return None if state["to_call"] > 0 else "illegal"
    amount = args.get("amount")
    if type(amount) is not int:
        return "arguments"
    return None if kind == state["raise_type"] and state["raise_min"] <= amount <= state["raise_max"] else "illegal"


def quantile(values, p):
    if not values:
        return None
    values = sorted(values)
    point = (len(values) - 1) * p
    lo, hi = math.floor(point), math.ceil(point)
    return values[lo] + (values[hi] - values[lo]) * (point - lo)


def token_summary(values):
    return {"count": len(values), "sum": sum(values), "median": quantile(values, .5), "p99": quantile(values, .99), "max": max(values) if values else None}


def audit(traces):
    counts = Counter()
    lengths = {k: [] for k in ("decision_output", "cleanup_output", "task_output", "request_input", "task_all_request_tokens_upper_bound")}
    task_reports = []
    for trace in traces:
        spans = sorted([s for s in trace["spans"] if s["name"] == "get_completion"], key=lambda s: s["start_time"])
        output_tokens, all_tokens, failed = 0, 0, False
        for span in spans:
            try:
                prompt, reply = items(span["input"]), items(span["output"])
            except (ValueError, TypeError):
                counts["unreadable"] += 1
                failed = True
                continue
            kind = "cleanup" if is_cleanup(prompt) else "decision"
            counts[kind + "s"] += 1
            tokens = span.get("tokens_out")
            if not isinstance(tokens, int) or tokens <= 0:
                counts[kind + "_missing_tokens"] += 1
                failed = True
            else:
                lengths[kind + "_output"].append(tokens)
                output_tokens += tokens
            cap = 10_000 if kind == "cleanup" else 5_000
            truncated = cutoff(span) or (isinstance(tokens, int) and tokens >= cap)
            if truncated:
                counts[kind + "_cutoffs"] += 1
            if kind == "decision":
                error = legal_reply(prompt, reply)
            else:
                calls = [i for i in reply if i.get("type") == "function_call"]
                try:
                    args = items(calls[0]["arguments"]) if len(calls) == 1 and calls[0].get("name") == "update_scratchpad" else None
                    error = None if isinstance(args, dict) and isinstance(args.get("contents"), str) and len(args["contents"]) <= 5000 else "format"
                except (ValueError, TypeError, KeyError):
                    error = "arguments"
            if truncated or error or span["status"] != "OK":
                counts[kind + "_failures"] += 1
                if error:
                    counts[kind + "_" + error] += 1
                failed = True
            inp = span.get("tokens_in")
            if isinstance(inp, int):
                lengths["request_input"].append(inp)
                all_tokens += inp
            all_tokens += tokens or 0
        counts["tasks"] += 1
        counts["tasks_with_failures"] += failed
        lengths["task_output"].append(output_tokens)
        lengths["task_all_request_tokens_upper_bound"].append(all_tokens)
        task_reports.append({"trace_id": trace["trace_id"], "output_tokens": output_tokens,
                             "all_request_tokens_upper_bound": all_tokens,
                             "export_bytes": len(json.dumps(trace).encode()), "has_failure": failed})
    return {"counts": dict(counts), "decision_cutoff_rate": counts["decision_cutoffs"] / counts["decisions"] if counts["decisions"] else None,
            "decision_failure_rate": counts["decision_failures"] / counts["decisions"] if counts["decisions"] else None,
            "failed_task_rate": counts["tasks_with_failures"] / counts["tasks"] if counts["tasks"] else None,
            "tokens": {k: token_summary(v) for k, v in lengths.items()}, "tasks": task_reports,
            "budget_note": "All-request token sum deliberately overcounts repeated prompt prefixes. It is an observed upper bound, not the trainer's exact accumulated transcript length. Do not choose long-game limits from it without a measured training pilot. Export bytes are not the connector collect_outputs payload size."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit([json.loads(line) for line in args.traces.open(encoding="utf-8")])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "tasks"}, indent=2))
