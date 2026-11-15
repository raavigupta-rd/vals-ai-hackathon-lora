"""Make the GRPO task datasets (scripted opponents, no opponent endpoint needed).

Writes <prefix>-grpo-spot, -grpo-shove, -grpo-short-game and -grpo-cleanup. Dry run is the default.

    uv run python -m poker.grpo.make_datasets
    uv run python -m poker.grpo.make_datasets --upload
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from poker.grpo.tasks import build_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="vals-hackathon")
    parser.add_argument("--prefix", default="pokerbots")
    parser.add_argument("--upload", action="store_true", help="create the datasets (default: dry run)")
    parser.add_argument("--dry-run", action="store_true", help="print counts and one task per kind (the default)")
    args = parser.parse_args()
    if args.upload and args.dry_run:
        parser.error("--upload and --dry-run conflict")

    built = build_all()
    if not args.upload:
        for kind, tasks in built.items():
            seats = sorted(Counter(t.env_params["policy_seats"][0] for t in tasks).items())
            print(f"{kind}: {len(tasks)} tasks, seats {seats}")
            print(json.dumps({"env_params": tasks[0].env_params, "grader_params": tasks[0].grader_params}, indent=1))
        return

    from ac2.sdk import Client  # imported late: a dry run needs no AC2 login

    client = Client(project=args.project)
    for kind, tasks in built.items():
        name = f"{args.prefix}-grpo-{kind.replace('_', '-')}"
        client.datasets.create(name, description=f"{len(tasks)} GRPO {kind} tasks, scripted opponents.", exist_ok=False)
        for start in range(0, len(tasks), 100):
            client.datasets.add_tasks(name, tasks=tasks[start : start + 100])
        print(f"{name}: {len(tasks)} tasks")


if __name__ == "__main__":
    main()
