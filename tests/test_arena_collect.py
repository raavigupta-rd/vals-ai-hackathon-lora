import json
from pathlib import Path

from poker.arena import collect, db

FIX = Path(__file__).parent / "arena_fixtures" / "match.json"
BLOB = "https://hcgxhgpocwwu7ggt.public.blob.vercel-storage.com/matches/{}-AbC123xyz.json"


def test_extractors():
    html = ('<a href="/matches/m_20261114t231117_fd8e48">x</a><a href="/matches/m_20261114t231117_fd8e48">y</a>'
            '<a href="/teams/org_3814ae0e9e26">t</a><a href="/teams/house%3Anit">h</a>'
            '<a href="/teams/house-bob-the-buster">b</a><a href="/matches?kind=ranked">k</a>')
    assert collect.extract_match_ids(html) == ["m_20261114t231117_fd8e48"]
    assert collect.extract_entry_slugs(html) == ["org_3814ae0e9e26", "house:nit", "house-bob-the-buster"]
    url = BLOB.format("m_1")
    assert collect.extract_blob_url(f'..."{url}"...', "m_1") == url
    assert collect.extract_blob_url("nothing", "m_1") is None
    assert collect.extract_blob_url(f'"{BLOB.format("m_2")}"', "m_1") is None


class FakeSite:
    def __init__(self, raws, live=()):
        self.raws, self.live, self.calls = raws, set(live), []

    def __call__(self, url):
        self.calls.append(url)
        if url.endswith("/matches") or "kind=" in url or "entry=" in url or url.endswith("/") or url.endswith("/tournaments"):
            ids = "".join(f'<a href="/matches/{i}">m</a>' for i in self.raws)
            return ids + '<a href="/teams/org_aaa">t</a>' if "entry=" not in url else ids
        if ".blob." in url:
            mid = url.rsplit("/", 1)[1].rsplit("-", 1)[0]
            return json.dumps(self.raws[mid])
        if "/matches/m_" in url:
            mid = url.rsplit("/", 1)[1]
            return f'<a href="{BLOB.format(mid)}">raw</a>'
        if url.endswith("/leaderboard"):
            raise collect.FetchError("404")
        return ""


def test_run_once_downloads_complete_and_defers_incomplete(tmp_path):
    ok = json.loads(FIX.read_text())
    live = ok | {"id": "m_live", "status": "running"}
    site = FakeSite({ok["id"]: ok, "m_live": live})
    con = db.connect(":memory:")
    c = collect.Collector(con, tmp_path, fetch=site, sleep=lambda s: None)
    s = c.run_once()
    assert s["downloaded"] == 1 and s["deferred"] == 1
    assert (tmp_path / f"{ok['id']}.json").exists() and not (tmp_path / "m_live.json").exists()
    assert con.execute("select count(*) from matches").fetchone()[0] == 1
    assert con.execute("select attempts from pending where match_id='m_live'").fetchone()[0] == 1
    # second pass: nothing new downloaded, live match retried, completed later
    n_blob_calls = sum(".blob." in u for u in site.calls)
    live["status"] = "complete"
    s2 = c.run_once()
    assert s2["downloaded"] == 1
    assert sum(".blob." in u for u in site.calls) == n_blob_calls + 1
    assert con.execute("select count(*) from pending").fetchone()[0] == 0
    s3 = c.run_once()
    assert s3["downloaded"] == 0


def test_reload_from_raw_files(tmp_path):
    ok = json.loads(FIX.read_text())
    (tmp_path / f"{ok['id']}.json").write_text(json.dumps(ok))
    con = db.connect(":memory:")
    c = collect.Collector(con, tmp_path, fetch=FakeSite({}), sleep=lambda s: None)
    assert c.load_raw_files() == 1 and c.load_raw_files() == 0


def test_fetcher_retries_with_backoff():
    sleeps, attempts = [], []

    def opener(req, timeout):
        attempts.append(1)
        if len(attempts) < 3:
            raise OSError("boom")

        class R:
            def read(self): return b"hello"
            def __enter__(self): return self
            def __exit__(self, *a): pass
        return R()

    f = collect.Fetcher(min_interval=0, sleep=sleeps.append, opener=opener, retries=4)
    assert f("https://x/y") == "hello" and len(attempts) == 3
    assert sleeps and sleeps[1] > sleeps[0]


def test_fetcher_gives_up():
    def opener(req, timeout):
        raise OSError("down")
    f = collect.Fetcher(min_interval=0, sleep=lambda s: None, opener=opener, retries=2)
    try:
        f("https://x")
    except collect.FetchError:
        return
    raise AssertionError("expected FetchError")
