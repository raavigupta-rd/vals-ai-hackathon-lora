"""Run many short local games of a deployment against shover-heavy scripted bots, in parallel.

Every decision of the deployment goes through the AC2 inference API, so it is recorded as a gateway span that
poker/synth/judge_spots.py pulls afterwards. The harness (play_local.py) is run unmodified under the event rules;
only the opponents are local scripted bots.

    uv run python -m poker.synth.run_games --model vals-hackathon/base --games 40 --hands 10 --concurrency 8 \
        --out data/synth/run1.json

AC2_API_KEY comes from the environment or ~/.ac2_api_key and is never printed. play_local retries 429s itself
with backoff; a game that still exits non-zero is retried after a pause, and the pause doubles for the whole run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[2] / "harness"
DEFAULT_BOTS = "shover,shover,nit,calling_station,shover"
SUMMARY = re.compile(r"(\d+) hands, (\d+) decisions by .*?, (\d+) failed")


def api_key() -> str:
    key = os.environ.get("AC2_API_KEY")
    path = Path.home() / ".ac2_api_key"
    if not key and path.exists():
        key = path.read_text().strip()
    if not key:
        raise SystemExit("set AC2_API_KEY or put the key in ~/.ac2_api_key")
    return key


def game_command(model: str, seed: int, hands: int, bots: str, out_dir: Path) -> list[str]:
    return ["uv", "run", "python", "play_local.py", "--model", model, "--opponents", "bots", "--bots", bots,
            "--hands", str(hands), "--seed", str(seed), "--out", str(out_dir)]


def parse_summary(output: str) -> dict:
    found = SUMMARY.search(output)
    return {"hands": int(found[1]), "decisions": int(found[2]), "failed": int(found[3])} if found else {}


def subprocess_runner(cmd: list[str], env: dict) -> tuple[int, str]:
    done = subprocess.run(cmd, cwd=HARNESS, env=env, capture_output=True, text=True)
    return done.returncode, (done.stdout or "") + (done.stderr or "")


def run_games(model: str, *, games: int, hands: int, concurrency: int, bots: str, seed_base: int, out_dir: Path,
              retries: int = 2, runner=subprocess_runner, sleep=time.sleep, key: str | None = None) -> dict:
    env = os.environ | {"AC2_API_KEY": key or api_key()}
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pause = {"s": 5.0}
    lock = threading.Lock()

    def one(i: int) -> dict:
        seed = seed_base + i
        for attempt in range(retries + 1):
            code, output = runner(game_command(model, seed, hands, bots, out_dir), env)
            info = parse_summary(output)
            if code == 0 and info:
                return {"seed": seed, "attempts": attempt + 1, "rc": 0} | info
            limited = "429" in output
            with lock:
                if limited:
                    pause["s"] = min(pause["s"] * 2, 120.0)
                wait = pause["s"]
            if attempt < retries:
                sleep(wait)
        tail = output.strip().splitlines()[-1:] or [""]
        return {"seed": seed, "attempts": retries + 1, "rc": code, "error": tail[0][:200].replace(key or "", "")}

    started = datetime.now(timezone.utc)
    clock = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one, range(games)))
    ended = datetime.now(timezone.utc)
    iso = "%Y-%m-%dT%H:%M:%S"
    return {"model": model, "bots": bots, "hands": hands, "seed_base": seed_base,
            "since": started.strftime(iso), "until": (ended + timedelta(seconds=10)).strftime(iso),
            "wall_seconds": round(time.monotonic() - clock, 1),
            "games_played": sum(1 for r in results if r["rc"] == 0), "games": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="deployment, e.g. vals-hackathon/base")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--hands", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--bots", default=DEFAULT_BOTS)
    parser.add_argument("--seed-base", type=int, default=7000)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--games-dir", type=Path, default=Path("data/synth/games"))
    parser.add_argument("--out", type=Path, required=True, help="manifest JSON (carries the since/until window)")
    args = parser.parse_args()
    manifest = run_games(args.model, games=args.games, hands=args.hands, concurrency=args.concurrency,
                         bots=args.bots, seed_base=args.seed_base, out_dir=args.games_dir, retries=args.retries)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=1))
    failed = [g for g in manifest["games"] if g["rc"] != 0]
    print(json.dumps({k: manifest[k] for k in ("games_played", "wall_seconds", "since", "until")}
                     | {"failed_games": len(failed), "out": str(args.out)}))
    if failed:
        print(f"{len(failed)} games failed; first: {failed[0]}", file=sys.stderr)


if __name__ == "__main__":
    main()
