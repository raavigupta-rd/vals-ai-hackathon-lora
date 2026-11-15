"""Paired local evaluation of deployments against a fixed, seeded field of scripted bots, independent of AC2 training.

Every deployment plays the same K games (same seeds, so the same deals, bot behaviour and seat order) of the event's
rules (6 seats, scratchpad, cleanup turn, 30 hands) in seat 0, through the harness game loop in-process (the same code
as harness/play_local.py, no subprocesses). Per game it records final place, bust, net chips, all-in call rate,
cleanup validity, format errors and tokens. The report gives each deployment's mean with a bootstrap CI over games and
the paired difference against a baseline (the first deployment by default) with a bootstrap CI over the paired games.

    uv run python -m poker.evalbots.run --deployments p/base p/c4 --games 40 --dry-run
    AC2_API_KEY=... uv run python -m poker.evalbots.run --deployments p/base p/c4 --games 40 --out out/evalbots

The default field mimics the final's tight-aggressive LLM-like opponents with a mix of archetypes; `--field` changes
it. Each game rotates the field around seats 1.. so no archetype always sits in one spot (`--no-rotate` to disable).
Only the model seat calls an endpoint; the bots never do.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "harness/src"))  # the harness is not an installed package

from poker_harness.completions import GATEWAY_URL, cleanup_params, relay_completion, table_rules, with_preset
from poker_harness.engine import ScriptedPlayer
from poker_harness.engine.scripted import ARCHETYPES
from poker_harness.game import play_session
from poker_harness.prompts import talk_fields
from poker_harness.seat import ModelSeat

DEFAULT_FIELD = ["nit", "calling_station", "shover", "maniac", "random"]
SEATS = 6
HERO = "seat0"

# metric name -> ("mean", per-game key) or ("ratio", numerator key, denominator key; pooled over games)
METRICS: dict[str, tuple] = {
    "place": ("mean", "place"),
    "bust": ("mean", "bust"),
    "net_chips": ("mean", "net_chips"),
    "allin_call_rate": ("ratio", "allin_calls", "allin_faced"),
    "cleanup_valid_rate": ("ratio", "cleanup_first_ok", "cleanup_turns"),
    "format_error_rate": ("ratio", "format_errors", "decisions"),
    "format_errors": ("mean", "format_errors"),
    "in_tokens": ("mean", "in_tokens"),
    "out_tokens": ("mean", "out_tokens"),
}

Factory = Callable[[str, dict[str, Any]], tuple[Any, Any]]  # (deployment, params) -> (complete, cleanup_complete)


def build_plan(
    deployments: list[str], games: int, hands: int, seed_base: int, field: list[str], rotate: bool = True
) -> list[dict[str, Any]]:
    """One entry per (deployment, game). Game g uses seed seed_base+g and the same bots for every deployment."""
    bad = [b for b in field if b not in ARCHETYPES]
    if bad or not field:
        raise ValueError(f"field needs archetypes from {ARCHETYPES}; got {field}")
    n = SEATS - 1
    plan = []
    for dep in deployments:
        for g in range(games):
            shift = g % len(field) if rotate else 0
            rotated = field[shift:] + field[:shift]
            plan.append(
                {"deployment": dep, "game": g, "seed": seed_base + g, "hands": hands,
                 "bots": [rotated[i % len(rotated)] for i in range(n)]}
            )
    return plan


def relay_factory(client: Any, key: str) -> Factory:
    def make(deployment: str, params: dict[str, Any]):
        mine = params | {"model_label": deployment}
        return (relay_completion(client, GATEWAY_URL, key, mine),
                relay_completion(client, GATEWAY_URL, key, cleanup_params(mine)))

    return make


async def play_game(entry: dict[str, Any], factory: Factory) -> dict[str, Any]:
    """Play one game under the event rules with the deployment in seat 0 and `entry["bots"]` in seats 1.."""
    seed = entry["seed"]
    params = with_preset({"preset": "event", "seed": seed, "submit_fields": ["action", "amount"]})
    params["total_hands"] = entry["hands"]
    rules = table_rules(params)
    if rules.table_talk:
        params["submit_fields"] = talk_fields(params["submit_fields"])
    complete, cleanup_complete = factory(entry["deployment"], params)
    players: list[Any] = [
        ModelSeat(HERO, complete, rules, model=entry["deployment"], cleanup_complete=cleanup_complete)
    ]
    for i, kind in enumerate(entry["bots"], start=1):
        players.append(ScriptedPlayer(f"seat{i}", kind, seed))
    return await play_session(f"evalbots-{entry['deployment'].replace('/', '_')}-{seed}", seed, players, rules)


def game_metrics(log: dict[str, Any]) -> dict[str, Any]:
    """Raw per-game numbers for the policy seat (seat0). Ratios are kept as numerator and denominator."""
    mine_actions: list[tuple[dict, dict]] = []  # (hand, action)
    for hand in log["hands"]:
        mine_actions += [(hand, a) for a in hand["actions"] if a["player"] == HERO]
    format_errors = sum(1 for _, a in mine_actions if a["model_response"].get("format_error"))
    in_tokens = sum(a["model_response"].get("in_tokens") or 0 for _, a in mine_actions)
    out_tokens = sum(a["model_response"].get("out_tokens") or 0 for _, a in mine_actions)

    faced = calls = 0
    for hand in log["hands"]:
        remaining = (hand.get("stacks_before") or {}).get(HERO)
        if remaining is None:
            continue
        remaining -= (hand.get("blinds_posted") or {}).get(HERO, 0)
        for a in hand["actions"]:
            if a["player"] != HERO:
                continue
            if a["call_amount"] > 0 and a["call_amount"] >= remaining:
                faced += 1
                calls += a["action"]["type"] == "call"
            remaining -= a.get("put_in", 0)

    turns = first_ok = 0
    for rec in ((log.get("private") or {}).get(HERO) or {}).get("hands", {}).values():
        cleanup = rec.get("cleanup")
        if not cleanup:
            continue
        turns += 1
        attempts = cleanup.get("attempts") or [{}]
        first_ok += attempts[0].get("outcome") == "ok" and cleanup.get("result") == "updated"
        out_tokens += sum(x.get("out_tokens") or 0 for x in cleanup.get("attempts") or [])

    return {
        "place": log["ranking"]["places"][HERO],
        "bust": int(HERO in log.get("eliminated_on_hand", {})),
        "net_chips": log["net_chips"][HERO],
        "hands_played": len(log["hands"]),
        "decisions": len(mine_actions),
        "format_errors": format_errors,
        "allin_faced": faced,
        "allin_calls": calls,
        "cleanup_turns": turns,
        "cleanup_first_ok": first_ok,
        "in_tokens": in_tokens,
        "out_tokens": out_tokens,
    }


async def evaluate(
    deployments: list[str],
    games: int,
    hands: int,
    seed_base: int,
    field: list[str],
    factory_for: Callable[[str], Factory],
    rotate: bool = True,
    concurrency: int = 4,
    on_game: Callable[[dict[str, Any], dict[str, Any] | None, str | None], None] | None = None,
) -> dict[str, Any]:
    """Play the whole plan. Returns {"games": {dep: [row, ...]}, "errors": [...], ...}; a row has seed and metrics."""
    plan = build_plan(deployments, games, hands, seed_base, field, rotate)
    sem = asyncio.Semaphore(concurrency)
    rows: dict[str, list[dict[str, Any]]] = {d: [] for d in deployments}
    errors: list[dict[str, Any]] = []

    async def one(entry: dict[str, Any]) -> None:
        async with sem:
            try:
                log = await play_game(entry, factory_for(entry["deployment"]))
                row = {"seed": entry["seed"], "bots": entry["bots"]} | game_metrics(log)
                rows[entry["deployment"]].append(row)
                if on_game:
                    on_game(entry, row, None)
            except Exception as exc:  # noqa: BLE001 - one failed game must not stop the run
                errors.append({"deployment": entry["deployment"], "seed": entry["seed"], "error": f"{type(exc).__name__}: {exc}"[:300]})
                if on_game:
                    on_game(entry, None, str(exc)[:300])

    await asyncio.gather(*(one(e) for e in plan))
    for d in rows:
        rows[d].sort(key=lambda r: r["seed"])
    return {"games": rows, "errors": errors, "hands": hands, "seed_base": seed_base, "field": field, "rotate": rotate}


# ---- statistics -------------------------------------------------------------------------------------------------


def _stat(spec: tuple, rows: list[dict[str, Any]], idx: list[int]) -> float:
    if spec[0] == "mean":
        return sum(rows[i][spec[1]] for i in idx) / len(idx)
    den = sum(rows[i][spec[2]] for i in idx)
    return sum(rows[i][spec[1]] for i in idx) / den if den else math.nan


def bootstrap_ci(values: list, stat: Callable[[list[int]], float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile CI of `stat(indices)` over resamples of range(len(values)); NaN resamples are skipped."""
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(s for s in (stat([rng.randrange(n) for _ in range(n)]) for _ in range(n_boot)) if not math.isnan(s))
    if not draws:
        return math.nan, math.nan
    return draws[int(0.025 * (len(draws) - 1))], draws[math.ceil(0.975 * (len(draws) - 1))]


def summarize(rows: list[dict[str, Any]], n_boot: int = 2000, seed: int = 0) -> dict[str, dict[str, float]]:
    out = {}
    for name, spec in METRICS.items():
        mean = _stat(spec, rows, list(range(len(rows)))) if rows else math.nan
        lo, hi = bootstrap_ci(rows, lambda idx, s=spec: _stat(s, rows, idx), n_boot, seed) if rows else (math.nan, math.nan)
        out[name] = {"mean": mean, "ci_low": lo, "ci_high": hi, "n": len(rows)}
    return out


def paired(rows: list[dict[str, Any]], base: list[dict[str, Any]], n_boot: int = 2000, seed: int = 0) -> dict[str, dict[str, float]]:
    """Per-metric difference (rows minus base) over the games both played (matched by seed), bootstrap over games."""
    base_by_seed = {r["seed"]: r for r in base}
    pairs = [(r, base_by_seed[r["seed"]]) for r in rows if r["seed"] in base_by_seed]
    a, b = [p[0] for p in pairs], [p[1] for p in pairs]
    out = {}
    for name, spec in METRICS.items():
        if not pairs:
            out[name] = {"mean": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": 0}
            continue
        def diff(idx: list[int], s: tuple = spec) -> float:
            return _stat(s, a, idx) - _stat(s, b, idx)

        lo, hi = bootstrap_ci(a, diff, n_boot, seed)
        out[name] = {"mean": diff(list(range(len(a)))), "ci_low": lo, "ci_high": hi, "n": len(a)}
    return out


def report(result: dict[str, Any], baseline: str | None = None, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    games = result["games"]
    names = list(games)
    baseline = baseline or names[0]
    return {
        "baseline": baseline,
        "games": max((len(v) for v in games.values()), default=0),
        "hands": result.get("hands"),
        "seed_base": result.get("seed_base"),
        "field": result.get("field"),
        "errors": result.get("errors", []),
        "deployments": {
            d: {
                "summary": summarize(games[d], n_boot, seed),
                "paired_vs_baseline": None if d == baseline else paired(games[d], games[baseline], n_boot, seed),
            }
            for d in names
        },
    }


def _fmt(s: dict[str, float], pct: bool = False) -> str:
    if s["n"] == 0 or math.isnan(s["mean"]):
        return "n/a"
    k = 100 if pct else 1
    return f"{s['mean'] * k:+.2f} [{s['ci_low'] * k:+.2f}, {s['ci_high'] * k:+.2f}]"


def format_report(rep: dict[str, Any]) -> str:
    pct = {"bust", "allin_call_rate", "cleanup_valid_rate", "format_error_rate"}
    lines = [f"baseline {rep['baseline']}; {rep['games']} games x {rep['hands']} hands; field {rep['field']}; 95% bootstrap CI over games"]
    for d, body in rep["deployments"].items():
        lines.append(f"\n{d}" + ("  (baseline)" if d == rep["baseline"] else ""))
        for name in METRICS:
            suffix = " %" if name in pct else ""
            line = f"  {name:<20} {_fmt(body['summary'][name], name in pct).replace('+', '')}{suffix}"
            if body["paired_vs_baseline"]:
                line += f"   diff {_fmt(body['paired_vs_baseline'][name], name in pct)}{suffix}"
            lines.append(line)
    if rep["errors"]:
        lines.append(f"\n{len(rep['errors'])} games failed: {rep['errors'][:3]}")
    return "\n".join(lines)


# ---- CLI --------------------------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--deployments", nargs="+", required=True, help="<project>/<name> ...; the first is the baseline")
    p.add_argument("--baseline", help="deployment to compare against (default: the first)")
    p.add_argument("--games", type=int, default=20, help="games per deployment (same seeds for all)")
    p.add_argument("--hands", type=int, default=30)
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--field", default=",".join(DEFAULT_FIELD), help=f"comma list of archetypes {ARCHETYPES} for seats 1..5")
    p.add_argument("--no-rotate", action="store_true", help="keep the field in the same seats every game")
    p.add_argument("--concurrency", type=int, default=4, help="games in flight")
    p.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    p.add_argument("--out", type=Path, help="write results.json and report.txt here")
    p.add_argument("--dry-run", action="store_true", help="list the plan; no endpoint calls, no key needed")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    field = [f for f in args.field.split(",") if f]
    plan = build_plan(args.deployments, args.games, args.hands, args.seed_base, field, not args.no_rotate)
    if args.baseline and args.baseline not in args.deployments:
        raise SystemExit(f"--baseline {args.baseline} is not in --deployments")
    if args.dry_run:
        print(f"{len(plan)} games ({len(args.deployments)} deployments x {args.games}), {args.hands} hands each, field {field}")
        for e in plan:
            print(f"  {e['deployment']}  game {e['game']}  seed {e['seed']}  bots {','.join(e['bots'])}")
        return 0
    key = os.environ.get("AC2_API_KEY")
    if not key:
        raise SystemExit("export AC2_API_KEY first (or use --dry-run)")
    import httpx

    async def go() -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
            def progress(entry, row, err):
                tail = f"place {row['place']} net {row['net_chips']:+d}" if row else f"FAILED {err}"
                print(f"{entry['deployment']} seed {entry['seed']}: {tail}", file=sys.stderr, flush=True)

            return await evaluate(args.deployments, args.games, args.hands, args.seed_base, field,
                                  lambda dep: relay_factory(client, key), not args.no_rotate, args.concurrency, progress)

    result = asyncio.run(go())
    rep = report(result, args.baseline, args.boot)
    text = format_report(rep)
    print(text)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "results.json").write_text(json.dumps({"result": result, "report": rep}, indent=1))
        (args.out / "report.txt").write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
