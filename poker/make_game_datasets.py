"""Make the multi-hand game datasets scored by final placement (grader_params reward "placement").

Writes <prefix>-train-game (400 games of 6 to 10 hands), -select-game (100 x 30 hands), -eval-game (200 x 30 hands).

    uv run python poker/make_game_datasets.py --dry-run
    uv run python poker/make_game_datasets.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ac2.sdk import Client

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poker.game_tasks import game_datasets  # noqa: E402
from vals_tools.tasks import Opponents  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="vals-hackathon")
    parser.add_argument("--prefix", default="pokerbots")
    parser.add_argument("--opponent-model", default="vals-hackathon/ac-vals-model")
    parser.add_argument("--opponent-api", choices=["responses", "chat"], default="chat")
    parser.add_argument("--opponent-effort", default="none", help="reasoning effort for the opponents, e.g. medium")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    effort = None if args.opponent_effort == "none" else args.opponent_effort
    built = game_datasets(args.prefix, Opponents(args.opponent_model, args.opponent_api, effort))
    if args.dry_run:
        for name, tasks in built.items():
            print(name, len(tasks), tasks[0].env_params, tasks[0].grader_params)
        return
    client = Client(project=args.project)
    for name, tasks in built.items():
        client.datasets.create(name, description=f"{len(tasks)} 6-max poker games, placement reward.", exist_ok=True)
        for start in range(0, len(tasks), 100):
            client.datasets.add_tasks(name, tasks=tasks[start : start + 100])
        print(f"{name}: {len(tasks)} tasks")


if __name__ == "__main__":
    main()
