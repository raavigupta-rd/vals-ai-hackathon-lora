"""LoRA SFT of Qwen3.5-9B on Tinker, thinking-off tool-loop conversations.

Usage:
  python sft/train_sft.py --data data/sft_r1_thinkoff.jsonl --name sft_b_v1 \
      [--max-examples 2000] [--epochs 1] [--lr 2e-4] [--batch-size 32] [--dry-run]

--dry-run prints the example count, token totals, and the estimated cost, then exits.
Checkpoints (with tinker:// sampler paths) go to runs/<name>/checkpoints.jsonl.
"""

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

import chz
import datasets
import tinker

from tinker_cookbook import renderers
from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.renderers import ToolCall, TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset, conversation_to_datum
from tinker_cookbook.supervised.types import SupervisedDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen3.5-9B"
RENDERER = "qwen3_5_disable_thinking"
TRAIN_PRICE_PER_M = 1.463  # USD per 1M tokens, Tinker pricing page, Qwen3.5-9B
MAX_LENGTH = 16384


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def to_messages(raw: list[dict]) -> list[dict]:
    msgs = []
    for m in raw:
        m = {k: v for k, v in m.items() if v is not None}
        if "tool_calls" in m:
            m["tool_calls"] = [ToolCall.model_validate(tc) for tc in m["tool_calls"]]
        msgs.append(m)
    return msgs


@chz.chz
class JsonlToolConversationBuilder(SupervisedDatasetBuilder):
    """Rows: {"messages": [...]} with tool_calls stored as dicts."""

    file_path: str  # rows already selected and ordered by main()
    batch_size: int
    keep_order: bool = False  # curriculum: never reshuffle
    renderer_name: str = RENDERER
    per_turn: bool = False  # thinking renderers: one example per assistant turn

    def __call__(self):
        rows = [json.loads(l) for l in open(self.file_path)]
        tok = get_tokenizer(MODEL)
        renderer = renderers.get_renderer(self.renderer_name, tok, get_image_processor(MODEL))
        ds = datasets.Dataset.from_list([{"messages_json": json.dumps(r["messages"])} for r in rows])

        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(to_messages(json.loads(row["messages_json"])), renderer,
                                         MAX_LENGTH, TrainOnWhat.ALL_ASSISTANT_MESSAGES)

        def flatmap_fn(row: dict) -> list[tinker.Datum]:
            # The qwen3_5 template drops earlier turns' <think> text from history, so a single
            # sequence would lose turn 1's reasoning. One example per assistant turn keeps it and
            # gives each turn exactly the context the grader shows at inference.
            examples = renderer.build_supervised_examples(
                to_messages(json.loads(row["messages_json"])),
                train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES)
            return [datum_from_model_input_weights(mi, w, MAX_LENGTH, reduction="mean") for mi, w in examples]

        cls = OrderedDataset if self.keep_order else SupervisedDatasetFromHFDataset
        if self.per_turn:
            return cls(ds, batch_size=self.batch_size, flatmap_fn=flatmap_fn), None
        return cls(ds, batch_size=self.batch_size, map_fn=map_fn), None


class OrderedDataset(SupervisedDatasetFromHFDataset):
    """Keeps the file order every epoch (the cookbook reshuffles by default)."""

    def set_epoch(self, seed: int = 0):
        self.shuffle_dataset = self.hf_dataset


def band(rating: int | None) -> str:
    if rating is None or rating < 1400:
        return "easy"
    return "med" if rating < 2000 else "hard"


def select_rows(files: list[str], mix: dict[str, int], curriculum: bool) -> list[dict]:
    """Dedupe, keep <= MAX_LENGTH, take `mix` examples per band (shortfall filled from other
    bands), then order easy -> med -> hard by rating (curriculum) or shuffle."""
    seen, rows = set(), []
    for f in files:
        if not Path(f).exists():
            print(f"skip missing {f}")
            continue
        for line in open(f):
            r = json.loads(line)
            key = (r["id"], json.dumps(r["messages"][2]["content"])[:300])
            if r["n_tokens"] <= MAX_LENGTH and key not in seen:
                seen.add(key)
                rows.append(r)
    rng = random.Random(0)
    rng.shuffle(rows)
    pools = {b: [r for r in rows if band(r["rating"]) == b] for b in ("easy", "med", "hard")}
    picked = {b: pools[b][: mix[b]] for b in pools}
    short = sum(mix.values()) - sum(map(len, picked.values()))
    for b in ("med", "easy", "hard"):  # fill any shortfall, prefer medium
        extra = pools[b][len(picked[b]): len(picked[b]) + short]
        picked[b] += extra
        short -= len(extra)
    out = [r for b in ("easy", "med", "hard") for r in picked[b]]
    if curriculum:
        out.sort(key=lambda r: (r["rating"] or 0))
    else:
        rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--name", required=True)
    ap.add_argument("--mix", default="1000,1300,700", help="easy,med,hard example counts")
    ap.add_argument("--curriculum", action="store_true", help="order easy -> hard, no reshuffle")
    ap.add_argument("--lr-schedule", default="linear", choices=["linear", "cosine", "constant"])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--renderer", default=RENDERER,
                    choices=["qwen3_5_disable_thinking", "qwen3_5"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    per_turn = args.renderer == "qwen3_5"

    mix = dict(zip(("easy", "med", "hard"), map(int, args.mix.split(","))))
    rows = select_rows(args.data, mix, args.curriculum)
    tokens = sum(r["n_tokens"] for r in rows) * args.epochs
    steps = len(rows) // args.batch_size * args.epochs
    counts = {b: sum(band(r["rating"]) == b for r in rows) for b in ("easy", "med", "hard")}
    print(f"examples: {len(rows)} {counts} ({len({r['id'] for r in rows})} problems), "
          f"epochs: {args.epochs}, steps: {steps}, tokens: {tokens / 1e6:.1f}M, "
          f"estimated cost: ${tokens / 1e6 * TRAIN_PRICE_PER_M:.0f}")
    log_path = ROOT / "runs" / args.name
    log_path.mkdir(parents=True, exist_ok=True)
    train_file = log_path / "train.jsonl"
    with open(train_file, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if args.dry_run:
        return

    load_env()
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not set. Put it in post-train/.env")

    config = train.Config(
        log_path=str(log_path),
        model_name=MODEL,
        recipe_name="sft_thinkon_r1" if per_turn else "sft_b_thinkoff_r1",
        renderer_name=args.renderer,
        dataset_builder=JsonlToolConversationBuilder(
            file_path=str(train_file), batch_size=args.batch_size, keep_order=args.curriculum,
            renderer_name=args.renderer, per_turn=per_turn),
        learning_rate=args.lr,
        lr_schedule=args.lr_schedule,
        num_epochs=args.epochs,
        lora_rank=args.lora_rank,
        save_every=max(10, steps // 4),
    )
    asyncio.run(train.main(config))
    print(f"done; checkpoints in {log_path / 'checkpoints.jsonl'}")


if __name__ == "__main__":
    sys.exit(main())
