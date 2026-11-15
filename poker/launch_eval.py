"""Launch or resume a uniquely named eval and record its immutable dataset manifest in the shared ledger."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import httpx
from ac2.sdk import Client
from ac2.sdk.task_sources import DatasetSource
from vals_tools.eval import eval_config


def record_run(client, run_id, kind, config=None):
    info = client.eval.get(run_id)
    record = {"id": info.id, "kind": kind, "status": info.status, "datasets": [asdict(t) for t in info.tasks],
              "project_version_id": info.project_version_id, "created_at": info.created_at,
              "observed_at": datetime.now(timezone.utc).isoformat(), "tags": info.tags}
    if config:
        record["settings"] = {"max_parallel": config.max_parallel,
                              "grading_timeout_seconds": config.custom_harness.grading_timeout_seconds}
    path = Path("poker/run_ledger.json")
    ledger = json.loads(path.read_text(encoding="utf-8"))
    previous = next((r for r in ledger["runs"] if r["id"] == run_id), {})
    ledger["runs"] = [r for r in ledger["runs"] if r["id"] != run_id] + [previous | record]
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(record, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--deployment", default="vals-hackathon/base")
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--grading-timeout", type=float, default=900)
    parser.add_argument("--num-tasks", type=int, help="first N dataset tasks using the SDK's DatasetSource")
    args = parser.parse_args()
    client = Client(project="vals-hackathon")
    existing = [r for r in client.eval.list() if r.name == args.name]
    if existing:
        record_run(client, existing[0].id, args.name)
        return
    config = eval_config(args.dataset, deployment=args.deployment, max_parallel=args.max_parallel,
                         grading_timeout=args.grading_timeout)
    if args.num_tasks:
        config.dataset = DatasetSource(args.dataset, num_tasks=args.num_tasks)
    config.name = args.name
    config.tags |= {"ledger_key": args.name, "source_dataset": args.dataset}
    run = asyncio.run(client.eval.run(config))
    record_run(client, run.eval_id, args.name, config)


if __name__ == "__main__":
    try:
        main()
    except httpx.HTTPStatusError as error:
        # SDK upload errors may include signed storage URLs. Never echo those credentials into a report.
        print(f"Eval request failed with HTTP {error.response.status_code}; inspect live eval list before retrying.")
        raise SystemExit(1) from None
