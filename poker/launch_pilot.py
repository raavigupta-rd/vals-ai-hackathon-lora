"""Launch one bounded single-hand GRPO pilot after checking the shared training slot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import httpx
from ac2.sdk import Client
from rl.train import parser as train_parser, run_config

TERMINAL = {"succeeded", "failed", "cancelled", "canceled", "stopped", "completed", "terminated"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="poker-fast-grpo-pilot-20261114")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    settings = train_parser().parse_args(["--train-dataset", "pokerbots-train", "--steps", "4", "--hands-per-step", "4",
                                         "--samples-per-hand", "4", "--checkpoint-every", "2", "--no-eval",
                                         "--concurrency", "4", "--grading-timeout", "1200"])
    config = run_config(settings, args.name)
    config.tags.update({"experiment": "decide-fast-single-hand-pilot"})
    if args.dry_run:
        print(json.dumps(config.to_json(), indent=2))
        return
    client = Client(project="vals-hackathon")
    runs = list(client.train.list())
    same = [r for r in runs if r.name == args.name]
    if same:
        print(json.dumps({"existing_train_id": same[0].id, "status": same[0].status}))
        return
    active = [r for r in runs if r.status not in TERMINAL]
    if active:
        raise SystemExit("Shared training slot occupied: " + ", ".join(f"{r.id} ({r.status})" for r in active))
    record = {"kind": "single_hand_grpo_pilot", "name": args.name, "dataset": "pokerbots-train",
              "settings": config.to_json(), "submitted_at": datetime.now(timezone.utc).isoformat(),
              "grader_revision": None, "grader_evidence": "Completed base smoke has place artifacts and zero policy format errors; exact loaded hash unverified.",
              "gpu_estimate": 6, "long_game_training": False}
    path = Path("out/runs")
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{args.name}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    result = client.train.run(config)
    record["id"] = result.train_id
    record["status"] = client.train.get(result.train_id).status
    ledger_path = Path("poker/run_ledger.json")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["runs"].append(record)
    ledger["training_slot"] = {"owner": "integration session", "train_id": result.train_id, "rule": "Do not launch another training job until terminal."}
    ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    (path / f"{args.name}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"train_id": result.train_id, "status": record["status"], "name": args.name}))


if __name__ == "__main__":
    try:
        main()
    except httpx.HTTPStatusError as error:
        print(f"Pilot request failed with HTTP {error.response.status_code}; inspect live train list before retrying.")
        print(error.response.text[:1000] if "appliedcompute.com/v1/" in str(error.request.url) else "Storage upload failed; signed URL omitted.")
        raise SystemExit(1) from None
