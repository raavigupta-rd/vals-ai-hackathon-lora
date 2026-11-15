"""Parse 'vals-arena.match/1' JSON into rows and store them in SQLite (data/arena/arena.db)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "arena"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "arena.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
  match_id TEXT PRIMARY KEY, kind TEXT, tournament TEXT, started_at TEXT, finished_at TEXT,
  status TEXT, seed TEXT, total_hands INTEGER, small_blind INTEGER, big_blind INTEGER,
  starting_stack INTEGER, ranking TEXT);
CREATE TABLE IF NOT EXISTS entries (
  sub_key TEXT PRIMARY KEY, entry_id TEXT, name TEXT, submission_name TEXT, deployment_id TEXT,
  model TEXT, kind TEXT, checkpoint_id TEXT, version INTEGER);
CREATE TABLE IF NOT EXISTS seats (
  match_id TEXT, seat INTEGER, sub_key TEXT, entry_id TEXT,
  net_chips REAL, net_bb REAL, ev_net_bb REAL, final_chips REAL, hands INTEGER, decisions INTEGER,
  fallbacks INTEGER, coerced INTEGER, afk INTEGER, timeouts INTEGER, mean_latency_s REAL,
  place INTEGER, eliminated_on_hand INTEGER,
  PRIMARY KEY (match_id, seat));
CREATE TABLE IF NOT EXISTS hands (
  match_id TEXT, hand_number INTEGER, button_seat INTEGER, small_blind INTEGER, big_blind INTEGER,
  board TEXT, showdown INTEGER, n_players INTEGER, deltas TEXT, eliminated TEXT,
  PRIMARY KEY (match_id, hand_number));
CREATE TABLE IF NOT EXISTS hole_cards (
  match_id TEXT, hand_number INTEGER, seat INTEGER, card1 TEXT, card2 TEXT, starting_stack REAL,
  position TEXT, shown INTEGER,
  PRIMARY KEY (match_id, hand_number, seat));
CREATE TABLE IF NOT EXISTS actions (
  match_id TEXT, hand_number INTEGER, idx INTEGER, seat INTEGER, sub_key TEXT, entry_id TEXT,
  street TEXT, action TEXT, amount REAL, pot_before REAL, to_call REAL, chip_change REAL,
  stack_after REAL, card1 TEXT, card2 TEXT, hand_code TEXT, position TEXT,
  players_remaining INTEGER, effective_stack REAL, raises_before INTEGER,
  bet_frac REAL, facing_frac REAL, facing_allin INTEGER, preflop_aggressor INTEGER,
  fallback INTEGER, coerced INTEGER, format_error INTEGER, afk INTEGER, timed_out INTEGER,
  token_cap INTEGER, sat_out INTEGER, scripted INTEGER, error TEXT,
  tok_in INTEGER, tok_out INTEGER, tok_reasoning INTEGER, latency_s REAL,
  PRIMARY KEY (match_id, hand_number, idx));
CREATE INDEX IF NOT EXISTS actions_sub ON actions (sub_key);
CREATE INDEX IF NOT EXISTS actions_entry ON actions (entry_id);
CREATE TABLE IF NOT EXISTS pending (
  match_id TEXT PRIMARY KEY, last_status TEXT, attempts INTEGER, last_try TEXT);
"""

RANKS = "23456789TJQKA"


def sub_key(entry: dict) -> str:
    return f"{entry.get('id')}|{entry.get('submission_name') or ''}|{entry.get('deployment_id') or ''}"


def position_labels(seats_in_hand: list[int], button: int, blinds: dict) -> dict[int, str]:
    """Position name per seat. Heads-up: button posts the small blind (HU_SB), the other is HU_BB."""
    n = len(seats_in_hand)
    if n < 2:
        return {s: "BTN" for s in seats_in_hand}
    order = sorted(seats_in_hand)
    if button in order:
        i = order.index(button)
    else:  # button seat busted: use the last seat before it
        i = max((k for k, s in enumerate(order) if s < button), default=len(order) - 1)
    clock = order[i + 1:] + order[:i + 1]  # starts at SB, ends at button
    if n == 2:
        return {clock[1]: "HU_SB", clock[0]: "HU_BB"}  # button acts first preflop as SB
    labels = {clock[0]: "SB", clock[1]: "BB", clock[-1]: "BTN"}
    middle = clock[2:-1]
    names = ["CO", "HJ", "UTG"][:len(middle)]
    for seat, name in zip(reversed(middle), names):
        labels[seat] = name
    return labels


def parse_match(raw: dict) -> dict:
    """Flatten one raw match into row dicts. Pure function, no IO."""
    fmt = raw.get("format") or {}
    match = dict(match_id=raw["id"], kind=raw.get("kind"), tournament=raw.get("tournament"),
                 started_at=raw.get("started_at"), finished_at=raw.get("finished_at"),
                 status=raw.get("status"), seed=None if raw.get("seed") is None else str(raw["seed"]),
                 total_hands=fmt.get("total_hands"), small_blind=fmt.get("small_blind"),
                 big_blind=fmt.get("big_blind"), starting_stack=fmt.get("starting_stack"),
                 ranking=fmt.get("ranking"))
    entries, seat_key, seat_entry = {}, {}, {}
    for s in raw.get("seats", []):
        e = s["entry"]
        k = sub_key(e)
        dep = e.get("deployment") or {}
        entries[k] = dict(sub_key=k, entry_id=e.get("id"), name=e.get("name"),
                          submission_name=e.get("submission_name"), deployment_id=e.get("deployment_id"),
                          model=e.get("model"), kind=e.get("kind"), checkpoint_id=dep.get("checkpoint_id"),
                          version=dep.get("version"))
        seat_key[s["seat"]], seat_entry[s["seat"]] = k, e.get("id")
    seats = []
    for r in raw.get("results", []):
        seats.append(dict(match_id=raw["id"], seat=r["seat"], sub_key=seat_key.get(r["seat"]),
                          entry_id=r.get("entry_id") or seat_entry.get(r["seat"]),
                          net_chips=r.get("net_chips"), net_bb=r.get("net_bb"), ev_net_bb=r.get("ev_net_bb"),
                          final_chips=r.get("final_chips"), hands=r.get("hands"), decisions=r.get("decisions"),
                          fallbacks=r.get("fallbacks"), coerced=r.get("coerced"), afk=r.get("afk"),
                          timeouts=r.get("timeouts"), mean_latency_s=r.get("mean_latency_s"),
                          place=r.get("place"), eliminated_on_hand=r.get("eliminated_on_hand")))
    hands, hole, actions = [], [], []
    mid = raw["id"]
    for h in raw.get("hands", []):
        hn = h["hand_number"]
        players = {p["seat"]: p for p in h.get("players", [])}
        pos = position_labels(list(players), h.get("button_seat"), h.get("blinds") or {})
        shown = {int(k) for k in (h.get("shown") or {})}
        hands.append(dict(match_id=mid, hand_number=hn, button_seat=h.get("button_seat"),
                          small_blind=h.get("small_blind"), big_blind=h.get("big_blind"),
                          board=" ".join(h.get("board") or []), showdown=int(bool(h.get("showdown"))),
                          n_players=len(players), deltas=json.dumps(h.get("deltas")),
                          eliminated=json.dumps(h.get("eliminated"))))
        for seat, p in players.items():
            c = p.get("hole_cards") or [None, None]
            hole.append(dict(match_id=mid, hand_number=hn, seat=seat, card1=c[0], card2=c[1],
                             starting_stack=p.get("starting_stack"), position=pos.get(seat),
                             shown=int(seat in shown)))
        live = set(players)
        start = {s: p.get("starting_stack") or 0 for s, p in players.items()}
        raises = 0            # raises so far this street (preflop: blinds are not counted)
        street = None
        last_aggr, last_aggr_allin = None, False
        pf_aggr = None
        for idx, a in enumerate(h.get("actions", [])):
            seat, st = a["seat"], a["street"]
            if st != street:
                street, raises, last_aggr, last_aggr_allin = st, 0, None, False
            pot, to_call = a.get("pot_before") or 0, a.get("to_call") or 0
            chips_in = -(a.get("chip_change") or 0)
            act = a["action"]
            bet_frac = None
            if act in ("bet", "raise") and pot + to_call > 0:
                bet_frac = (chips_in - to_call) / (pot + to_call)
            facing_frac = None
            if to_call > 0 and pot - to_call > 0 and st != "preflop":
                facing_frac = to_call / (pot - to_call)
            facing_allin = int(to_call > 0 and last_aggr_allin)
            opps = [s for s in live if s != seat]
            eff = min(start.get(seat, 0), max((start[s] for s in opps), default=0))
            c = (players.get(seat) or {}).get("hole_cards") or [None, None]
            tok = a.get("tokens") or {}
            actions.append(dict(
                match_id=mid, hand_number=hn, idx=idx, seat=seat, sub_key=seat_key.get(seat),
                entry_id=seat_entry.get(seat), street=st, action=act, amount=a.get("amount"),
                pot_before=pot, to_call=to_call, chip_change=a.get("chip_change"),
                stack_after=a.get("stack_after"), card1=c[0], card2=c[1],
                hand_code=hand_code(c[0], c[1]), position=pos.get(seat), players_remaining=len(live),
                effective_stack=eff, raises_before=raises, bet_frac=bet_frac, facing_frac=facing_frac,
                facing_allin=facing_allin, preflop_aggressor=int(st != "preflop" and pf_aggr == seat),
                fallback=int(bool(a.get("fallback"))), coerced=int(bool(a.get("coerced"))),
                format_error=int(bool(a.get("format_error"))), afk=int(bool(a.get("afk"))),
                timed_out=int(bool(a.get("timed_out"))), token_cap=int(bool(a.get("token_cap"))),
                sat_out=int(bool(a.get("sat_out"))), scripted=int(bool(a.get("scripted"))),
                error=a.get("error"), tok_in=tok.get("in"), tok_out=tok.get("out"),
                tok_reasoning=tok.get("reasoning"), latency_s=a.get("latency_s")))
            if act == "fold":
                live.discard(seat)
            elif act in ("bet", "raise"):
                raises += 1
                last_aggr, last_aggr_allin = seat, a.get("stack_after") == 0
                if st == "preflop":
                    pf_aggr = seat
    return dict(match=match, entries=list(entries.values()), seats=seats, hands=hands,
                hole_cards=hole, actions=actions)


def hand_code(c1: str | None, c2: str | None) -> str | None:
    """Canonical 169-class code: 'AA', 'AKs', 'T9o'."""
    if not c1 or not c2:
        return None
    r1, r2 = sorted((c1[0], c2[0]), key=RANKS.index, reverse=True)
    if r1 == r2:
        return r1 + r2
    return r1 + r2 + ("s" if c1[1] == c2[1] else "o")


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def _insert(con, table: str, rows: list[dict], replace: bool = True) -> None:
    if not rows:
        return
    cols = list(rows[0])
    sql = f"INSERT OR {'REPLACE' if replace else 'IGNORE'} INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
    con.executemany(sql, [[r[c] for c in cols] for r in rows])


def load_match(con: sqlite3.Connection, raw: dict) -> bool:
    """Insert one complete match. Idempotent (replaces prior rows). Returns False if not complete."""
    if raw.get("status") != "complete":
        return False
    p = parse_match(raw)
    mid = raw["id"]
    with con:
        for t in ("seats", "hands", "hole_cards", "actions"):
            con.execute(f"DELETE FROM {t} WHERE match_id=?", (mid,))
        _insert(con, "matches", [p["match"]])
        _insert(con, "entries", p["entries"])
        _insert(con, "seats", p["seats"])
        _insert(con, "hands", p["hands"])
        _insert(con, "hole_cards", p["hole_cards"])
        _insert(con, "actions", p["actions"])
        con.execute("DELETE FROM pending WHERE match_id=?", (mid,))
    return True
