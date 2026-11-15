"""Refresh a live trace export instead of reusing the SDK's cached first snapshot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import httpx
from ac2.sdk import Client
from ac2.sdk.traces import decompress_into


def export(client, run_id, path):
    state = client.traces.start_export(run_id, refresh=True)
    while state.status in ("queued", "running"):
        time.sleep(2)
        state = client.traces.export(run_id, state.artifact_key)
    if state.status != "ready" or not state.download_url:
        raise RuntimeError("Trace export did not complete")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with httpx.stream("GET", state.download_url, timeout=300) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            decompress_into(response.iter_bytes(1 << 20), stream)
    temporary.replace(path)
    metadata = {"run_id": run_id, "refreshed_at": datetime.now(timezone.utc).isoformat(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    path.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        export(Client(project="vals-hackathon"), args.run_id, args.out or Path(f"traces/{args.run_id}.jsonl"))
    except httpx.HTTPStatusError as error:
        print(f"Trace export/download failed with HTTP {error.response.status_code}; signed URL omitted.")
        raise SystemExit(1) from None
