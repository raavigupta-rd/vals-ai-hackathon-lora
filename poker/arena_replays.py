"""Download publicly linked replay JSON and adapt the observed vals-arena.match/1 format."""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from poker.profile_logs import profile


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href", "")
            if href:
                self.links.append(href)


def links(html):
    parser = Links()
    parser.feed(html)
    return list(dict.fromkeys(parser.links))


def adapt(replay: dict) -> tuple[dict, dict]:
    if replay.get("schema") != "vals-arena.match/1":
        raise ValueError("unsupported arena replay schema")
    names, identities = {}, {}
    for item in replay["seats"]:
        seat, entry = item["seat"], item["entry"]
        version = (entry.get("deployment") or {}).get("version")
        name = f"{entry['id']}:{entry.get('deployment_id')}:{version}"
        names[seat] = name
        if entry["kind"] == "team":
            identities[name] = {"team": entry["id"], "team_name": entry["name"],
                                "submission": f"{entry.get('submission_name')}:{entry.get('deployment_id')}:v{version}", "probe": False}
    hands = []
    for source in replay["hands"]:
        seats = sorted(p["seat"] for p in source["players"] if p["starting_stack"] > 0)
        button = source["button_seat"]
        order = [s for s in seats if s > button] + [s for s in seats if s <= button]
        actions = []
        for event in source["actions"]:
            failed = any(event.get(k) for k in ("fallback", "coerced", "format_error", "afk", "timed_out", "token_cap", "sat_out", "error"))
            move = {"type": event["action"]}
            if event.get("amount") is not None:
                move["amount"] = event["amount"]
            actions.append(dict(player=names[event["seat"]], street=event["street"], action=move,
                                pot_before=event["pot_before"], call_amount=event["to_call"], stack_after=event["stack_after"],
                                model_response={"error": event.get("error"), "fallback_used": failed}))
        hands.append(dict(hand_number=source["hand_number"],
                          starting_state={names[p["seat"]]: {"starting_stack": p["starting_stack"], "position": order.index(p["seat"]) if p["seat"] in order else -1} for p in source["players"]},
                          blinds={names[int(s)]: amount for s, amount in source["blinds"].items()}, actions=actions))
    log = dict(game_id=replay["id"], complete=replay["status"] == "complete", hands=hands,
               final_stacks={names[r["seat"]]: r["final_chips"] for r in replay["results"]},
               session_info={k: replay["format"][k] for k in ("small_blind", "big_blind", "reset_stacks")})
    return log, identities


def fetch(limit: int, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    listing = urlopen("https://www.pokerbots.ac/matches", timeout=30).read().decode()
    match_links = [url for url in links(listing) if url.startswith("/matches/m_")][:limit]
    paths = []
    for route in match_links:
        page = urlopen("https://www.pokerbots.ac" + route, timeout=30).read().decode()
        raw_urls = [url for url in links(page) if urlparse(url).hostname == "hcgxhgpocwwu7ggt.public.blob.vercel-storage.com" and urlparse(url).path.endswith(".json")]
        if not raw_urls:
            continue  # live/incomplete matches need not have a published replay yet
        raw = urlopen(raw_urls[0], timeout=30).read()
        replay = json.loads(raw)
        path = directory / (route.rsplit("/", 1)[-1] + ".json")
        path.write_bytes(raw)
        paths.append(path)
        print(f"Downloaded {replay['id']}: {len(replay['hands'])} hands", flush=True)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--directory", type=Path, default=Path("data/replays"))
    parser.add_argument("--out", type=Path, default=Path("out/scouting/arena_profiles.json"))
    parser.add_argument("--exclude-probes", nargs="*", default=[], help="entry IDs known to be probes")
    args = parser.parse_args()
    if args.fetch:
        fetch(args.limit, args.directory)
    logs, identities, names = [], {}, {}
    for path in sorted(args.directory.glob("*.json")):
        log, mapping = adapt(json.loads(path.read_text(encoding="utf-8")))
        for identity in mapping.values():
            names[identity["team"]] = identity["team_name"]
            identity["probe"] = identity["team"] in args.exclude_probes
        logs.append(log)
        identities.update(mapping)
    result = profile(logs, identities)
    for team, value in result["teams"].items():
        value["name"] = names[team]
    result["source"] = "https://www.pokerbots.ac/matches"
    result["notes"].append("Probe identity flags require human review; styles are not classified automatically. These are early ladder observations, not forecasts of final tournament entries.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
