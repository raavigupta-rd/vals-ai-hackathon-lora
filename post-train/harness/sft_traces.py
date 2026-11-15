"""SFT Qwen3.5-9B on verified MiMo teacher traces (from harness/mimo_traces.py), re-rendered in our harness format.

Keeps episodes that scored 100 on hidden tests (optionally also 40-point ones), never held-out problems, and
(by default) drops episodes with turns cut off by the token cap. Each kept episode becomes the exact
conversation the student will see in harness.env: same system prompt (5-turn limit), tools, Qwen3.5 tool-call
syntax and tool responses; the loss is on every assistant turn (reasoning + tool call / EXIT).

    uv run python -m harness.sft_traces --traces data/mimo_traces/*.jsonl --name lenz-mimo-sft --max-examples 250
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from config import BASE_MODEL, PROJECT_ID  # noqa: E402

from harness.env import MAX_TURNS, CPTools, initial_messages  # noqa: E402
from harness.problems import load  # noqa: E402


def select(traces: list[dict], held: set[str], min_score: float, allow_truncated: bool, per_problem: int,
           max_examples: int | None, seed: int) -> list[dict]:
    ok = [t for t in traces if t["id"] not in held and t["score"] >= min_score
          and (allow_truncated or all(x.get("finish") != "length" for x in t["turns_data"]))
          and t["turns"] >= 1]
    # prefer full solves that ended with EXIT and used fewer output tokens; cap per problem for diversity
    ok.sort(key=lambda t: (-t["score"], t["stop"] != "exit", t["out_tokens"]))
    by_pid: dict[str, list[dict]] = {}
    for t in ok:
        by_pid.setdefault(t["id"], []).append(t)
    picked = []
    for rnd in range(per_problem):  # round-robin over problems
        for pid in sorted(by_pid):
            if rnd < len(by_pid[pid]):
                picked.append(by_pid[pid][rnd])
    random.Random(seed).shuffle(picked)
    return picked[:max_examples] if max_examples else picked


def to_conversation(trace: dict, problem: dict, renderer) -> list:
    msgs = initial_messages(problem, renderer, CPTools(problem), max_turns=MAX_TURNS)
    for t, turn in enumerate(trace["turns_data"]):
        calls = [ToolCall(function=ToolCall.FunctionBody(name=c["name"], arguments=json.dumps(c["arguments"])),
                          id=f"call_{t}_{i}") for i, c in enumerate(turn["tool_calls"])]
        msg = {"role": "assistant",
               "content": [{"type": "thinking", "thinking": turn["reasoning"]}, {"type": "text", "text": turn["content"]}]}
        if calls:
            msg["tool_calls"] = calls
        msgs.append(msg)
        for i, (c, result) in enumerate(zip(turn["tool_calls"], turn["tool_results"])):
            msgs.append({"role": "tool", "content": result, "tool_call_id": f"call_{t}_{i}", "name": c["name"]})
    return msgs


async def main(a):
    tokenizer = get_tokenizer(BASE_MODEL)
    renderer = renderers.get_renderer(a.renderer, tokenizer)
    problems = {}
    for f in a.problems:
        problems.update({p["id"]: p for p in load(f)})
    held = {p["id"] for p in load(a.heldout)}
    traces = []
    for f in a.traces:
        traces += [json.loads(line) for line in open(f)]
    traces = [t for t in traces if t["id"] in problems]
    picked = select(traces, held, a.min_score, a.allow_truncated, a.per_problem, a.max_examples, a.seed)
    print(f"{len(traces)} traces -> {len(picked)} selected from {len({t['id'] for t in picked})} problems "
          f"(score>={a.min_score}, held-out excluded)", flush=True)
    data = []
    for t in picked:
        conv = to_conversation(t, problems[t["id"]], renderer)
        for mi, w in renderer.build_supervised_examples(conv, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES):
            data.append(datum_from_model_input_weights(mi, w, max_length=a.max_length))
    n_tok = sum(d.model_input.length for d in data)
    n_train = sum(float(d.loss_fn_inputs["weights"].to_torch().sum()) for d in data)
    print(f"{len(data)} datums, {n_tok} tokens ({int(n_train)} trained) per epoch; "
          f"est. cost ${n_tok * a.epochs * 1.463e-6:.2f} for {a.epochs} epochs", flush=True)
    if a.dry_run:
        mi, w = renderer.build_supervised_examples(to_conversation(picked[0], problems[picked[0]["id"]], renderer),
                                                   train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES)[0]
        toks = mi.to_ints()
        print("---- trained text of first example (head) ----\n" +
              tokenizer.decode([x for x, wt in zip(toks, w.tolist()) if wt > 0])[:1500])
        return

    log_dir = ROOT / "runs" / "sft" / a.name
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "config.json").write_text(json.dumps(vars(a) | {"datums": len(data), "traces": len(picked),
                                                               "tokens_per_epoch": n_tok}, indent=2))
    mlog = (log_dir / "metrics.jsonl").open("w")
    service = tinker.ServiceClient(project_id=PROJECT_ID, display_name=a.name)
    tc = await service.create_lora_training_client(base_model=BASE_MODEL, rank=a.rank, seed=0)
    rng = random.Random(a.seed)
    total = a.epochs * ((len(data) + a.batch_size - 1) // a.batch_size)
    step, cum = 0, 0
    for ep in range(a.epochs):
        rng.shuffle(data)
        for i in range(0, len(data), a.batch_size):
            batch = data[i:i + a.batch_size]
            cum += sum(d.model_input.length for d in batch)
            lr = a.lr * (1 - step / total)
            out = await tc.forward_backward(batch, loss_fn="cross_entropy")
            await tc.optim_step(tinker.AdamParams(learning_rate=lr))
            lp = torch.cat([o["logprobs"].to_torch() for o in out.loss_fn_outputs])
            wt = torch.cat([d.loss_fn_inputs["weights"].to_torch() for d in batch])
            nll = float(-(lp * wt).sum() / wt.sum())
            print(f"epoch {ep} step {step + 1}/{total} lr {lr:.2e} nll/token {nll:.4f}", flush=True)
            mlog.write(json.dumps({"step": step, "epoch": ep, "lr": lr, "nll": nll, "total_steps": total,
                                   "cost_usd": cum * 1.463e-6}) + "\n")
            mlog.flush()
            step += 1
    path = (await tc.save_weights_for_sampler(name=a.name)).path
    print("sampler checkpoint:", path, flush=True)
    (log_dir / "checkpoint.json").write_text(json.dumps({"sampler_path": path}))
    with (ROOT / "checkpoints.jsonl").open("a") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "name": a.name,
                            "base_model": BASE_MODEL, "trained": "sft on verified MiMo-V2.6-Distill traces",
                            "traces": len(picked), "epochs": a.epochs, "lr": a.lr, "path": path}) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--problems", nargs="+", default=[str(ROOT / "data" / "synth_ioi" / "problems.jsonl"),
                                                      str(ROOT / "data" / "cc_train" / "problems.jsonl")])
    ap.add_argument("--heldout", default=str(ROOT / "data" / "synth_ioi" / "heldout.jsonl"))
    ap.add_argument("--name", default="lenz-mimo-sft")
    ap.add_argument("--renderer", default="qwen3_5")
    ap.add_argument("--min-score", type=float, default=100.0)
    ap.add_argument("--allow-truncated", action="store_true")
    ap.add_argument("--per-problem", type=int, default=4, help="max traces per problem")
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=65536)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    a.problems = [p for p in a.problems if Path(p).exists()]
    asyncio.run(main(a))
