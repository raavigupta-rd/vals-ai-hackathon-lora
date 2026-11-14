"""Shared record validation, source provenance and deterministic hand identities."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMA = json.loads((Path(__file__).parent / "schemas/decision_record.schema.json").read_text())
VALIDATOR = Draft202012Validator(SCHEMA)
POSITIONS = ("SB", "BB", "UTG", "HJ", "CO", "BTN")
STREETS = ("preflop", "flop", "turn", "river")


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate(record: dict) -> None:
    VALIDATOR.validate(record)
    state = record["state"]
    cards = state["hole"] + state["board"]
    if len(cards) != len(set(cards)):
        raise ValueError("duplicate visible card")
    if len(state["board"]) != {"preflop": 0, "flop": 3, "turn": 4, "river": 5}[state["street"]]:
        raise ValueError("board length does not match decision street")
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite number")
        if isinstance(value, dict):
            for v in value.values():
                finite(v)
        elif isinstance(value, list):
            for v in value:
                finite(v)
    finite(record)


def legal_action(record: dict) -> bool:
    state, action = record["state"], record["action"]
    legal, kind = state["legal"], action["type"]
    if kind == "fold":
        return legal["fold"] is True
    if kind == "check":
        return legal["check"] is True
    if kind == "call":
        return legal["call"] is not None and legal["call"] > 0
    if legal.get("raise_type") != kind or action["to"] is None:
        return False
    low, high = legal["raise_min"], legal["raise_max"]
    return low is not None and high is not None and low <= action["to"] <= high


def fingerprint(record: dict) -> str:
    # Excludes source IDs and labels: the same decision cannot leak through another source or a different label.
    return digest({"format": record["format"], "state": record["state"]})


def hand_key(record: dict) -> str:
    return f"{record['source']}:{record['meta']['hand_id']}"
