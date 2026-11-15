"""Capture public arena availability. No opponent metrics are invented before replays exist."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.request import urlopen


class Scripts(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_script = False
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        self.in_script = tag == "script"

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_script = False

    def handle_data(self, data):
        if self.in_script:
            self.scripts.append(data)


def snapshot(html):
    parser = Scripts()
    parser.feed(html)
    for script in parser.scripts:
        prefix = "self.__next_f.push("
        if not script.startswith(prefix):
            continue
        payload, _ = json.JSONDecoder().raw_decode(script[len(prefix):])
        if len(payload) < 2 or not isinstance(payload[1], str):
            continue
        match = re.search(r'"lb":', payload[1])
        if match:
            leaderboard, _ = json.JSONDecoder().raw_decode(payload[1][match.end():])
            rows = leaderboard["rows"]
            return {"fetched_at": datetime.now(timezone.utc).isoformat(), "source": "https://www.pokerbots.ac/",
                    "computed_at": leaderboard.get("computed_at"),
                    "team_entries": sum(r["kind"] != "house" for r in rows),
                    "entries_with_matches": sum(r["matches"] > 0 for r in rows),
                    "profiles_available": any(r["matches"] > 0 and r["kind"] != "house" for r in rows),
                    "leaderboard": leaderboard}
    raise ValueError("Arena page format changed; no leaderboard found")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, help="reuse a previously fetched public homepage")
    parser.add_argument("--out", type=Path, default=Path("out/scouting/arena_status.json"))
    args = parser.parse_args()
    html = args.html.read_text(encoding="utf-8") if args.html else urlopen("https://www.pokerbots.ac/", timeout=30).read().decode()
    result = snapshot(html)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "leaderboard"}, indent=2))


if __name__ == "__main__":
    main()
