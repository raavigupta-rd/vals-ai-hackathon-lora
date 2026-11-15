"""Pull our deployment's AC2 gateway spans (the arena's exact prompts and replies) into normalized JSONL.

Spans come from `client.analytics.query` with source="spans" and job_ids=[deployment id]. Each
`inference.request` span holds the request messages in `input`, the response items in `output` and
gateway metadata in `attributes` (conversation_id, finish_reason, sampling_params). Read-only on AC2.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re

from poker.trace_quality import attributes, is_cleanup, items

COLUMNS = ("span_id", "trace_id", "name", "start_time", "end_time", "duration_ms", "status", "error_message",
           "model", "tokens_in", "tokens_out", "cost", "input", "output", "attributes")
CONVERSATION = re.compile(r"^vals-arena/(?P<match>[^/]+)/hand-(?P<hand>\d+)/seat(?P<seat>\d+)")


def parse_when(value: str) -> str:
    """Accept `3h`, `45m`, `2d` (relative to now) or an ISO time; return naive UTC ISO seconds."""
    rel = re.fullmatch(r"(\d+)([mhd])", value.strip())
    if rel:
        delta = timedelta(**{{"m": "minutes", "h": "hours", "d": "days"}[rel[2]]: int(rel[1])})
        return (datetime.now(timezone.utc) - delta).strftime("%Y-%m-%dT%H:%M:%S")
    when = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if when.tzinfo:
        when = when.astimezone(timezone.utc)
    return when.strftime("%Y-%m-%dT%H:%M:%S")


def _load(value):
    try:
        return items(value) if value else None
    except (ValueError, TypeError):
        return None


def normalize_span(row: dict, deployment: str) -> dict:
    messages = _load(row.get("input"))
    reply = _load(row.get("output")) or []
    attrs = attributes(row) if row.get("attributes") else {}
    conversation = attrs.get("conversation_id") or ""
    found = CONVERSATION.match(conversation)
    reasoning, content, calls = [], [], []
    for item in reply if isinstance(reply, list) else []:
        if item.get("type") == "reasoning":
            reasoning.append(item.get("content") or "")
        elif item.get("type") == "function_call":
            calls.append({"call_id": item.get("call_id"), "name": item.get("name"), "arguments": item.get("arguments")})
        elif item.get("type") == "message":
            content.append(item.get("content") if isinstance(item.get("content"), str) else json.dumps(item.get("content")))
    finish = attrs.get("finish_reason")
    start = row.get("start_time") or ""
    return {
        "span_id": row["span_id"], "trace_id": row.get("trace_id"),
        "ts": start + "Z" if start and not start.endswith("Z") else start,
        "duration_ms": row.get("duration_ms"), "deployment": deployment,
        "conversation_id": conversation or None, "source": "arena" if found else "other",
        "arena": {"match_id": found["match"], "hand": int(found["hand"]), "seat": int(found["seat"])} if found else None,
        "messages": messages, "tools": attrs.get("tools"),
        "reasoning": "".join(reasoning), "content": "".join(content), "tool_calls": calls,
        "usage": {"tokens_in": row.get("tokens_in"), "tokens_out": row.get("tokens_out"), "cost": row.get("cost")},
        "finish_reason": finish, "cutoff": finish in ("length", "max_tokens"),
        "status": row.get("status"), "error": row.get("error_message") or None,
        "sampling_params": _load(attrs.get("sampling_params")),
        "kind": "cleanup" if isinstance(messages, list) and is_cleanup(messages) else "action",
    }


def _client_query(project):
    from ac2.sdk import Client
    analytics = Client(project=project).analytics
    return lambda spec: analytics.query(spec=spec)


def pull(deployment, since, until, out: Path, query=None, page=100, project="vals-hackathon"):
    from ac2.sdk.analytics.types import ColumnSpec, FieldFilter, QuerySpec
    query = query or _client_query(project)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            try:
                seen.add(json.loads(line)["span_id"])
            except (ValueError, KeyError):
                pass
    filters = [FieldFilter(field="name", op="eq", value="inference.request")]
    if since:
        filters.append(FieldFilter(field="start_time", op="ge", value=since))
    if until:
        filters.append(FieldFilter(field="start_time", op="lt", value=until))
    written = skipped = offset = total = 0
    with out.open("a", encoding="utf-8") as stream:
        while True:
            spec = QuerySpec(source="spans", projects=[project], job_ids=[deployment],
                             columns=[ColumnSpec(field=f) for f in COLUMNS], filters=filters,
                             sort_by="start_time", sort_order="asc", limit=page, offset=offset)
            response = query(spec)
            total = response.total
            for row in response.rows:
                if row["span_id"] in seen:
                    skipped += 1
                    continue
                seen.add(row["span_id"])
                stream.write(json.dumps(normalize_span(row, deployment), ensure_ascii=False) + "\n")
                written += 1
            offset += len(response.rows)
            if len(response.rows) < page:
                break
    return {"written": written, "skipped": skipped, "fetched": offset, "total": total, "out": str(out)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", required=True, help="deployment id, e.g. infer_d3d2b6b92171")
    parser.add_argument("--since", default="6h", help="ISO time or 3h/2d relative (default 6h)")
    parser.add_argument("--until")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--project", default="vals-hackathon")
    parser.add_argument("--page", type=int, default=100)
    args = parser.parse_args()
    out = args.out or Path(f"data/arena/gateway/{args.deployment}.jsonl")
    print(json.dumps(pull(args.deployment, parse_when(args.since), parse_when(args.until) if args.until else None,
                          out, page=args.page, project=args.project)))
