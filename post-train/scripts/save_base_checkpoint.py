"""Save an untrained LoRA on the base model as a sampler checkpoint (= base model behaviour).

LoRA's B matrices are zero-initialised, so this checkpoint samples exactly like Qwen3.5-9B.
Use the printed tinker:// path as a baseline submission or to test the submit flow.

    uv run python scripts/save_base_checkpoint.py [--name base-qwen35-9b] [--rank 32]
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

import tinker

from config import BASE_MODEL, PROJECT_ID

LOG = Path(__file__).resolve().parent.parent / "checkpoints.jsonl"


async def main(name: str, rank: int) -> None:
    service = tinker.ServiceClient(project_id=PROJECT_ID, display_name=f"save-{name}")
    training_client = await service.create_lora_training_client(base_model=BASE_MODEL, rank=rank, seed=0)
    resp = await training_client.save_weights_for_sampler(name=name)
    path = resp.path
    print("sampler checkpoint:", path)
    with LOG.open("a") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "name": name,
                            "base_model": BASE_MODEL, "rank": rank, "trained": False, "path": path}) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="base-qwen35-9b")
    ap.add_argument("--rank", type=int, default=32)
    a = ap.parse_args()
    asyncio.run(main(a.name, a.rank))
