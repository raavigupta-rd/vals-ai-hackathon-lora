"""Figures for docs/results_summary.pdf, computed from run files (run from post-train/)."""
import json, statistics as st, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def rows(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
def ev(tag): return [r for r in rows(f"runs/{tag}/results.jsonl") if not r.get("infra_error")]
def mimo(f): return rows(f"data/mimo_traces/{f}.jsonl")
out = "docs/figures"
C = {"base": "#2563eb", "off": "#93c5fd", "sft": "#d97706", "mimo": "#7c3aed"}

# 1) held-out scores (40/60 all-or-nothing), synthetic held-out set
bars = [
    ("Base think on\n(5 turns)", st.mean(r["score"] for r in ev("ab5-think-on")), C["base"]),
    ("Base think off\n(5 turns)", st.mean(r["score"] for r in ev("ab5-think-off")), C["off"]),
    ("Base high\n(real settings)", st.mean(r["score"] for r in ev("real-base-high")), C["base"]),
    ("Base none\n(real settings)", st.mean(r["score"] for r in ev("real-base-none")), C["off"]),
    ("SFT qwengem", st.mean(r["score"] for r in ev("eval-qwengem")), C["sft"]),
    ("SFT qwengem-\nmasked", st.mean(r["score"] for r in ev("eval-qwengemmasked")), C["sft"]),
    ("MiMo think off\n(real)", st.mean(r["score"] for r in mimo("real_mimo_nothink")), C["mimo"]),
    ("MiMo think on\n(real)", st.mean(r["score"] for r in mimo("real_mimo_think")), C["mimo"]),
]
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(len(bars)), [b[1] for b in bars], color=[b[2] for b in bars])
for i, b in enumerate(bars): ax.text(i, b[1] + 0.8, f"{b[1]:.1f}", ha="center", fontsize=9)
ax.set_xticks(range(len(bars))); ax.set_xticklabels([b[0] for b in bars], fontsize=8)
ax.set_ylabel("held-out score (0-100)"); ax.set_title("Synthetic held-out set (34 problems x 2 samples)")
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(f"{out}/heldout_scores.png", dpi=150)

# 2) behaviour at real settings: turns and output tokens per task
arms = [("Base high", ev("real-base-high"), "sampled_tokens", C["base"]), ("Base none", ev("real-base-none"), "sampled_tokens", C["off"]),
        ("MiMo off", mimo("real_mimo_nothink"), "out_tokens", C["mimo"]), ("MiMo on", mimo("real_mimo_think"), "out_tokens", C["mimo"])]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
for i, (n, R, k, c) in enumerate(arms):
    axs[0].bar(i, st.mean(r["turns"] for r in R), color=c); axs[1].bar(i, st.mean(r[k] for r in R) / 1000, color=c)
for a, t in zip(axs, ["turns per task", "output tokens per task (K)"]):
    a.set_xticks(range(len(arms))); a.set_xticklabels([x[0] for x in arms]); a.set_title(t); a.spines[["top", "right"]].set_visible(False)
fig.suptitle("Behaviour under real-grader settings (100 turns, 16K output/turn)"); fig.tight_layout(); fig.savefig(f"{out}/behaviour_real.png", dpi=150)

# 3) leaderboard (read off the Vals charts, assuming 100 tasks) -- estimates
lb = [("PITA #3", 40.8, 140, 7.2), ("chicken #6", 35.9, 116, 4.0), ("chicken #7", 31, 73, 2.5),
      ("cluster", 26, 240, 9.0), ("REVAL", 27.8, 440, 10.0), ("Glasses", 0.6, 40, 1.6)]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for n, s, tok, tu in lb:
    axs[0].scatter(tok, s, color=C["base"]); axs[0].annotate(n, (tok, s), fontsize=8, xytext=(4, 3), textcoords="offset points")
    axs[1].scatter(tu, s, color=C["base"]); axs[1].annotate(n, (tu, s), fontsize=8, xytext=(4, 3), textcoords="offset points")
axs[0].axhline(19.84, ls="--", color="gray"); axs[1].axhline(19.84, ls="--", color="gray")
axs[0].text(300, 21, "Axe #1 = 19.84 (base, thinking off)", fontsize=8, color="gray")
axs[0].set_xlabel("total tokens per task (K, input+cached+output)"); axs[1].set_xlabel("turns per task")
for a in axs: a.set_ylabel("leaderboard score"); a.spines[["top", "right"]].set_visible(False)
fig.suptitle("Leaderboard, Nov 14 ~19:30 PT (estimates read from charts, 100 tasks assumed)"); fig.tight_layout(); fig.savefig(f"{out}/leaderboard.png", dpi=150)
print({b[0].replace(chr(10), " "): round(b[1], 1) for b in bars})
