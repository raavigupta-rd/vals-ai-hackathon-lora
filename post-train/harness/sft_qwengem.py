"""SFT "qwengem": Qwen3.5-9B LoRA on the Gemini reference solutions of the synth_ioi TRAIN split.

Each training conversation is the routine we want the model to learn, in the exact harness format
(same system prompt, tools, tool-call syntax and tool responses as harness.env):

    user:       problem statement
    assistant:  <think> short approach (Gemini's notes) </think>  -> submit(fast_cpp)
    tool:       real submit() result (samples AC), produced by running the harness tool
    assistant:  EXIT

Held-out problems are never used. Note: Gemini is not an open-source model; check with the
organizers before submitting a model trained on its outputs.

    uv run python -m harness.sft_qwengem --dry-run        # build + inspect data, no Tinker spend
    uv run python -m harness.sft_qwengem                  # train and save sampler weights "qwengem"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import tinker
import torch
from tinker_cookbook import renderers
from tinker_cookbook.renderers.base import ToolCall, TrainOnWhat
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.tool_use.tools import handle_tool_call

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from config import BASE_MODEL, PROJECT_ID  # noqa: E402

from harness import sandbox  # noqa: E402
from harness.env import MAX_TURNS, CPTools, initial_messages  # noqa: E402
from harness.problems import load  # noqa: E402

MASK = "[MASK]"


def masked_hint(code: str, mask_frac: float, rng: random.Random) -> str:
    """Masking hint (Hinted Scaling Laws, arXiv 2611.02270): the reference solution with a random
    `mask_frac` of its words replaced by [MASK]. mask_frac >= 1 means no hint at all."""
    if mask_frac >= 1.0:
        return ""
    lines = []
    for line in code.rstrip().splitlines():
        indent = line[: len(line) - len(line.lstrip())]
        words = [MASK if rng.random() < mask_frac else w for w in line.split()]
        lines.append(indent + " ".join(words))
    return (f"\n\nHint: a reference solution with {round(mask_frac * 100)}% of its words replaced by {MASK}:\n"
            "```cpp\n" + "\n".join(lines) + "\n```")


async def build_conversation(problem: dict, sol: dict, renderer, hint: str = "") -> list | None:
    tools = CPTools(problem)
    msgs = initial_messages(problem, renderer, tools, hint=hint, max_turns=MAX_TURNS)
    think = (f"{sol['notes'].strip()}\n\nThis fits the constraints and the {problem['time_limit']:g} s limit, "
             f"so I'll submit it.")
    call = ToolCall(function=ToolCall.FunctionBody(name="submit", arguments=json.dumps({"code": sol["fast_cpp"]})),
                    id="call_0")
    msgs.append({"role": "assistant", "content": [{"type": "thinking", "thinking": think}, {"type": "text", "text": ""}],
                 "tool_calls": [call]})
    res = await handle_tool_call({"submit": tools.submit}, call)
    tool_msgs = res.messages
    body = json.loads(tool_msgs[0]["content"])
    n_samples = len(problem["samples"])
    if body.get("samples_passed") != f"{n_samples}/{n_samples}":
        print(f"skip {problem['id']}: reference failed samples ({body.get('samples_passed')})", flush=True)
        return None
    msgs += tool_msgs
    msgs.append({"role": "assistant", "content": [{"type": "thinking", "thinking": "All samples pass. Done."},
                                                  {"type": "text", "text": "EXIT"}]})
    return msgs


async def main(a):
    sandbox.ensure_container()
    tokenizer = get_tokenizer(BASE_MODEL)
    renderer = renderers.get_renderer(a.renderer, tokenizer)
    problems = {p["id"]: p for p in load(a.problems)}
    held = {p["id"] for p in load(a.heldout)}
    assert not (held & problems.keys()), "train/held-out overlap"
    sols = {s["id"]: s for s in map(json.loads, open(a.solutions))}
    # mask schedule: one mask fraction per epoch (1.0 = no hint); plain SFT = all 1.0
    schedule = [float(x) for x in a.mask_schedule.split(",")] if a.mask_schedule else [1.0] * a.epochs
    a.epochs = len(schedule)
    hint_rng = random.Random(1)

    async def epoch_data(mask_frac: float):
        convs = await asyncio.gather(*(
            build_conversation(p, sols[i], renderer, masked_hint(sols[i]["fast_cpp"], mask_frac, hint_rng))
            for i, p in problems.items() if i in sols))
        convs = [c for c in convs if c]
        out = []
        for c in convs:
            for mi, w in renderer.build_supervised_examples(c, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES):
                out.append(datum_from_model_input_weights(mi, w, max_length=a.max_length))
        return convs, out

    epochs = [await epoch_data(m) for m in schedule]
    convs, data = epochs[0]
    n_tok = sum(d.model_input.length for _, ds in epochs for d in ds) // len(epochs)
    n_train = sum(float(d.loss_fn_inputs["weights"].to_torch().sum()) for d in data)
    print(f"{len(convs)} conversations -> {len(data)} datums, {n_tok} tokens ({int(n_train)} trained) per epoch; "
          f"est. cost ${n_tok * a.epochs * 1.463e-6:.2f}; mask schedule {schedule}", flush=True)

    if a.dry_run:
        mi, w = renderer.build_supervised_examples(convs[0], train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES)[-1]
        toks = mi.to_ints()
        trained = tokenizer.decode([t for t, x in zip(toks, w.tolist()) if x > 0])
        print("---- full text (tail) ----\n" + tokenizer.decode(toks)[-1500:])
        print("---- trained tokens ----\n" + trained[-1200:])
        return

    log_dir = ROOT / "runs" / "sft" / a.name
    log_dir.mkdir(parents=True, exist_ok=True)
    mlog = (log_dir / "metrics.jsonl").open("w")
    (log_dir / "config.json").write_text(json.dumps(vars(a) | {"datums": len(data), "tokens_per_epoch": n_tok}, indent=2))
    service = tinker.ServiceClient(project_id=PROJECT_ID, display_name=a.name)
    tc = await service.create_lora_training_client(base_model=BASE_MODEL, rank=a.rank, seed=0)
    rng = random.Random(0)
    total = sum((len(ds) + a.batch_size - 1) // a.batch_size for _, ds in epochs)
    step, cum_tokens = 0, 0
    for ep in range(a.epochs):
        data = list(epochs[ep][1])
        rng.shuffle(data)
        for i in range(0, len(data), a.batch_size):
            batch = data[i:i + a.batch_size]
            cum_tokens += sum(d.model_input.length for d in batch)
            lr = a.lr * (1 - step / total)  # linear decay
            out = await tc.forward_backward(batch, loss_fn="cross_entropy")
            await tc.optim_step(tinker.AdamParams(learning_rate=lr))
            logprobs = torch.cat([o["logprobs"].to_torch() for o in out.loss_fn_outputs])
            weights = torch.cat([d.loss_fn_inputs["weights"].to_torch() for d in batch])
            nll = float(-(logprobs * weights).sum() / weights.sum())
            print(f"epoch {ep} step {step}/{total} lr {lr:.2e} nll/token {nll:.4f}", flush=True)
            mlog.write(json.dumps({"step": step, "epoch": ep, "mask_frac": schedule[ep], "lr": lr, "nll": nll,
                                   "total_steps": total,
                                   "cost_usd": cum_tokens * 1.463e-6}) + "\n")
            mlog.flush()
            step += 1
    resp = await tc.save_weights_for_sampler(name=a.name)
    path = resp.path
    print("sampler checkpoint:", path)
    (log_dir / "checkpoint.json").write_text(json.dumps({"sampler_path": path}))
    with (ROOT / "checkpoints.jsonl").open("a") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "name": a.name, "base_model": BASE_MODEL,
                            "rank": a.rank, "trained": "sft on gemini synth_ioi train solutions", "epochs": a.epochs,
                            "mask_schedule": schedule,
                            "lr": a.lr, "renderer": a.renderer, "path": path}) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default=str(ROOT / "data" / "synth_ioi" / "train.jsonl"))
    ap.add_argument("--heldout", default=str(ROOT / "data" / "synth_ioi" / "heldout.jsonl"))
    ap.add_argument("--solutions", default=str(ROOT / "data" / "synth_ioi" / "solutions.jsonl"))
    ap.add_argument("--renderer", default="qwen3_5")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--name", default="qwengem")
    ap.add_argument("--mask-schedule", default="",
                    help="comma-separated mask fraction per epoch, e.g. 0.3,0.6,0.9,1.0 (overrides --epochs)")
    asyncio.run(main(ap.parse_args()))
