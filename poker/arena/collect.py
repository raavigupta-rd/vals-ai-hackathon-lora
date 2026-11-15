"""Collect public arena matches from pokerbots.ac into data/arena/raw/ and data/arena/arena.db.

  uv run python -m poker.arena.collect --once
  uv run python -m poker.arena.collect --loop 600

Polite: generic User-Agent, about 1 request per second, retries with backoff. Idempotent:
complete matches are downloaded once; incomplete ones are retried on later passes.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from poker.arena import db

BASE = "https://www.pokerbots.ac"
USER_AGENT = "arena-research/0.1 (public match data, ~1 req/s)"
BLOB_HOST = "hcgxhgpocwwu7ggt.public.blob.vercel-storage.com"
INDEX_PAGES = ["/matches", "/matches?kind=ranked", "/matches?kind=tournament", "/matches?kind=exhibition",
               "/", "/leaderboard", "/tournaments"]
MATCH_RE = re.compile(r"/matches/(m_[0-9a-z_]+)")
TEAM_RE = re.compile(r'href="/teams/([^"?#/]+)"')


class FetchError(Exception):
    pass


class Fetcher:
    def __init__(self, min_interval: float = 1.0, retries: int = 4, backoff: float = 2.0, timeout: float = 30,
                 sleep=time.sleep, opener=urllib.request.urlopen):
        self.min_interval, self.retries, self.backoff, self.timeout = min_interval, retries, backoff, timeout
        self.sleep, self.opener, self._last = sleep, opener, 0.0

    def __call__(self, url: str) -> str:
        err = None
        for attempt in range(self.retries):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                self.sleep(wait)
            self._last = time.monotonic()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with self.opener(req, timeout=self.timeout) as r:
                    return r.read().decode("utf-8", "replace")
            except urllib.error.HTTPError as e:
                err = e
                if e.code in (400, 401, 403, 404, 410):
                    break  # not worth retrying
            except Exception as e:  # network errors, timeouts
                err = e
            if attempt < self.retries - 1:
                self.sleep(self.backoff * 2 ** attempt)
        raise FetchError(f"{url}: {err}")


def extract_match_ids(html: str) -> list[str]:
    return list(dict.fromkeys(MATCH_RE.findall(html)))


def extract_entry_slugs(html: str) -> list[str]:
    return list(dict.fromkeys(urllib.parse.unquote(s) for s in TEAM_RE.findall(html)))


def extract_blob_url(html: str, match_id: str) -> str | None:
    m = re.search(rf"https://{re.escape(BLOB_HOST)}/matches/{re.escape(match_id)}-[A-Za-z0-9_\-]+\.json", html)
    return m.group(0) if m else None


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Collector:
    def __init__(self, con: sqlite3.Connection, raw_dir: Path = db.RAW_DIR, fetch=None, sleep=time.sleep):
        self.con, self.raw_dir = con, Path(raw_dir)
        self.fetch = fetch or Fetcher(sleep=sleep)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _try(self, url: str) -> str | None:
        try:
            return self.fetch(url)
        except FetchError as e:
            print(f"  skip {e}")
            return None

    def load_raw_files(self) -> int:
        known = {r[0] for r in self.con.execute("SELECT match_id FROM matches")}
        n = 0
        for p in sorted(self.raw_dir.glob("m_*.json")):
            if p.stem in known:
                continue
            try:
                n += db.load_match(self.con, json.loads(p.read_text()))
            except (ValueError, KeyError) as e:
                print(f"  bad raw file {p.name}: {e}")
        return n

    def discover(self) -> set[str]:
        ids: set[str] = set()
        slugs: dict[str, None] = {}
        for page in INDEX_PAGES:
            html = self._try(BASE + page)
            if html:
                ids.update(extract_match_ids(html))
                slugs.update(dict.fromkeys(extract_entry_slugs(html)))
        for r in self.con.execute("SELECT DISTINCT entry_id FROM entries WHERE entry_id IS NOT NULL"):
            slugs.setdefault(r[0], None)
        for slug in list(slugs):
            q = urllib.parse.quote(slug, safe="")
            for url in (f"{BASE}/matches?entry={q}", f"{BASE}/teams/{q}"):
                html = self._try(url)
                if html:
                    ids.update(extract_match_ids(html))
        return ids

    def _defer(self, mid: str, status: str) -> None:
        with self.con:
            self.con.execute(
                "INSERT INTO pending(match_id,last_status,attempts,last_try) VALUES(?,?,1,?) "
                "ON CONFLICT(match_id) DO UPDATE SET last_status=excluded.last_status, attempts=attempts+1, "
                "last_try=excluded.last_try", (mid, status, now()))

    def fetch_match(self, mid: str) -> str:
        """Returns 'downloaded', 'deferred' or 'failed'."""
        page = self._try(f"{BASE}/matches/{mid}")
        url = extract_blob_url(page, mid) if page else None
        if not url:
            self._defer(mid, "no_raw_link")
            return "deferred"
        body = self._try(url)
        try:
            raw = json.loads(body) if body else None
        except ValueError:
            raw = None
        if not raw:
            self._defer(mid, "fetch_failed")
            return "failed"
        if raw.get("status") != "complete":
            self._defer(mid, str(raw.get("status")))
            return "deferred"
        (self.raw_dir / f"{mid}.json").write_text(json.dumps(raw, separators=(",", ":")))
        db.load_match(self.con, raw)
        return "downloaded"

    def run_once(self) -> dict:
        reloaded = self.load_raw_files()
        found = self.discover()
        have = {r[0] for r in self.con.execute("SELECT match_id FROM matches")}
        todo = sorted((found | {r[0] for r in self.con.execute("SELECT match_id FROM pending")}) - have)
        stats = dict(discovered=len(found), reloaded=reloaded, downloaded=0, deferred=0, failed=0)
        for mid in todo:
            stats[self.fetch_match(mid)] += 1
        stats["matches_total"] = self.con.execute("SELECT count(*) FROM matches").fetchone()[0]
        return stats


def counts(con) -> dict:
    return {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            for t in ("matches", "entries", "seats", "hands", "hole_cards", "actions", "pending")}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--once", action="store_true", help="one discovery and download pass (default)")
    g.add_argument("--loop", type=float, nargs="?", const=600, metavar="SECONDS", help="repeat; default 600")
    ap.add_argument("--db", default=str(db.DB_PATH))
    ap.add_argument("--raw-dir", default=str(db.RAW_DIR))
    args = ap.parse_args(argv)
    con = db.connect(args.db)
    c = Collector(con, Path(args.raw_dir))
    while True:
        print(f"[{now()}] pass:", c.run_once(), counts(con), flush=True)
        if args.loop is None:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
