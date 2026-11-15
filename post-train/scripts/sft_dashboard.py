"""Dashboard for SFT runs (runs/sft/<name>/metrics.jsonl) and held-out evals (runs/<tag>/results.jsonl).

Focus: does the model finish within the 5-turn limit (EXIT, not cut off) and keep its output short?

    uv run python scripts/sft_dashboard.py [--port 8766]      -> http://localhost:8766 (refreshes every 20 s)
"""

import argparse
import json
import statistics as st
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "runs"
DEFAULT_EVALS = ["ab5-think-on", "eval-lenz-mimo-sft", "eval-qwengem", "eval-qwengemmasked"]
TRACES = RUNS.parent / "data" / "mimo_traces"
HELDOUT = RUNS.parent / "data" / "synth_ioi" / "heldout.jsonl"
TARGET_TRACES = 250
MAX_TURNS = 5


def jsonl(p: Path) -> list[dict]:
    out = []
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def sft_runs() -> list[dict]:
    out = []
    for d in sorted((RUNS / "sft").glob("*")) if (RUNS / "sft").exists() else []:
        m = jsonl(d / "metrics.jsonl")
        ck = json.loads((d / "checkpoint.json").read_text()) if (d / "checkpoint.json").exists() else {}
        out.append({"name": d.name, "steps": m, "total": m[-1]["total_steps"] if m else None,
                    "sampler_path": ck.get("sampler_path"), "done": bool(ck)})
    return out


def eval_summary(tag: str) -> dict | None:
    rows = [r for r in jsonl(RUNS / tag / "results.jsonl") if not r.get("infra_error")]
    if not rows:
        return {"tag": tag, "n": 0}
    turns = [r["turns"] for r in rows]
    toks = [r["sampled_tokens"] for r in rows]
    per_turn = [r["sampled_tokens"] / max(1, r["turns"]) for r in rows]
    stops = [r["stop_reason"] for r in rows]
    n = len(rows)
    return {
        "tag": tag, "n": n,
        "score": st.mean(r["score"] for r in rows),
        "score_se": (st.pstdev([r["score"] for r in rows]) / n ** 0.5) if n > 1 else 0,
        "solved": sum(r["score"] == 100 for r in rows) / n,
        "small_only": sum(r["score"] == 40 for r in rows) / n,
        "submitted": sum(r["submitted"] for r in rows) / n,
        "exit": stops.count("completed") / n,
        "turn_limit": sum(s == "max_turns" or (t >= MAX_TURNS and s != "completed") for s, t in zip(stops, turns)) / n,
        "budget": sum(s in ("max_tokens", "max_sampled_tokens", "context_overflow") for s in stops) / n,
        "turns_hist": [turns.count(k) for k in range(1, MAX_TURNS + 1)],
        "tok_mean": st.mean(toks), "tok_median": st.median(toks),
        "tok_per_turn_median": st.median(per_turn),
        "tok_hist": hist(toks, [0, 5_000, 10_000, 20_000, 40_000, 80_000, 1e12]),
    }


def traces_summary() -> dict:
    """MiMo teacher episodes (data/mimo_traces/*.jsonl): progress toward verified SFT traces + MiMo's own score."""
    held = {json.loads(l)["id"] for l in HELDOUT.read_text().splitlines()} if HELDOUT.exists() else set()
    rows = []
    for f in sorted(TRACES.glob("*.jsonl")) if TRACES.exists() else []:
        rows += [r | {"file": f.stem} for r in jsonl(f)]
    if not rows:
        return {"n": 0}
    train = [r for r in rows if r["id"] not in held]
    hrows = [r for r in rows if r["id"] in held]
    verified = [r for r in train if r["score"] == 100 and all(t.get("finish") != "length" for t in r["turns_data"])]
    per_problem = {}
    for r in verified:
        per_problem[r["id"]] = per_problem.get(r["id"], 0) + 1
    usable = sum(min(4, v) for v in per_problem.values())  # sft_traces.py caps 4 per problem
    stops = {}
    for r in rows:
        stops[r["stop"]] = stops.get(r["stop"], 0) + 1
    by_file = {}
    for r in rows:
        d = by_file.setdefault(r["file"], {"n": 0, "full": 0})
        d["n"] += 1
        d["full"] += r["score"] == 100
    n = len(rows)
    return {
        "n": n, "train_n": len(train), "verified": len(verified), "usable": usable, "target": TARGET_TRACES,
        "problems_with_trace": len(per_problem),
        "solve_rate": sum(r["score"] == 100 for r in rows) / n,
        "small_only_rate": sum(r["score"] == 40 for r in rows) / n,
        "exit_rate": stops.get("exit", 0) / n, "stops": stops, "by_file": by_file,
        "heldout_n": len(hrows),
        "heldout_score": st.mean(r["score"] for r in hrows) if hrows else None,
        "heldout_se": (st.pstdev([r["score"] for r in hrows]) / len(hrows) ** 0.5) if len(hrows) > 1 else None,
        "train_score": st.mean(r["score"] for r in train) if train else None,
        "out_tok_median": st.median(r["out_tokens"] for r in rows),
        "turns_hist": [sum(r["turns"] == k for r in rows) for k in range(1, MAX_TURNS + 1)],
        "out_hist": hist([r["out_tokens"] for r in rows], [0, 4_000, 8_000, 16_000, 24_000, 32_001]),
    }


def hist(xs, edges):
    return [sum(edges[i] <= x < edges[i + 1] for x in xs) for i in range(len(edges) - 1)]


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SFT Eval Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--bg:#fafaf9;--card:#fff;--fg:#1c1917;--mut:#78716c;--line:#e7e5e4;--good:#16a34a;--bad:#dc2626;--c0:#78716c;--c1:#2563eb;--c2:#d97706;--c3:#7c3aed;--c4:#0891b2}
@media (prefers-color-scheme:dark){:root{--bg:#0c0a09;--card:#1c1917;--fg:#f5f5f4;--mut:#a8a29e;--line:#292524;--good:#4ade80;--bad:#f87171;--c0:#a8a29e;--c1:#60a5fa;--c2:#fbbf24;--c3:#a78bfa;--c4:#22d3ee}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
main{max-width:1150px;margin:0 auto;padding:20px 16px} h1{font-size:20px;margin:0 0 4px} h2{font-size:15px;margin:22px 0 8px}
.mut{color:var(--mut)} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;overflow-x:auto}
.card h3{margin:0 0 8px;font-size:13px;color:var(--mut);font-weight:600} canvas{max-height:240px}
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:5px 8px;border-top:1px solid var(--line);text-align:right;white-space:nowrap} th:first-child,td:first-child{text-align:left}
th{color:var(--mut);font-weight:600;border-top:none} .good{color:var(--good);font-weight:600} .bad{color:var(--bad);font-weight:600}
code{font-size:11px;word-break:break-all;white-space:normal}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:0 0 12px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.kpi b{display:block;font-size:22px;font-variant-numeric:tabular-nums} .kpi span{color:var(--mut);font-size:12px}
.bar{height:14px;background:var(--line);border-radius:7px;overflow:hidden;margin:6px 0} .bar div{height:100%;background:var(--c1)}
</style></head><body><main>
<h1>SFT + eval dashboard</h1>
<div class="mut" id="status">loading…</div>
<h2>MiMo teacher traces (5 turns, ≤32K output tokens)</h2>
<div class="kpis" id="tkpis"></div>
<div class="grid">
<div class="card"><h3>Verified traces for SFT (score 100, no cut-off turns, ≤4 per problem)</h3><div class="bar"><div id="tbar"></div></div><div class="mut" id="tbartxt"></div></div>
<div class="card"><h3>Episodes by source</h3><table id="tfiles"></table></div>
<div class="card"><h3>Turns per MiMo episode</h3><canvas id="c_tturns"></canvas></div>
<div class="card"><h3>Output tokens per MiMo episode</h3><canvas id="c_tout"></canvas></div>
</div>
<h2>Held-out eval (5 turns, thinking on): finishes in time? short output?</h2>
<div class="card"><table id="evtab"></table></div>
<div class="grid" style="margin-top:12px">
<div class="card"><h3>Turns used per rollout (5 = hit the limit unless it said EXIT)</h3><canvas id="c_turns"></canvas></div>
<div class="card"><h3>Sampled tokens per rollout</h3><canvas id="c_tok"></canvas></div>
</div>
<h2>SFT training</h2>
<div class="grid">
<div class="card"><h3>Loss (NLL per trained token)</h3><canvas id="c_loss"></canvas></div>
<div class="card"><h3>Runs</h3><table id="sfttab"></table></div>
</div>
</main><script>
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const COL=['--c0','--c1','--c2','--c3','--c4'];
const charts={};
function chart(id,type,labels,datasets,opts={}){
  datasets.forEach((d,i)=>{const c=css(COL[i%COL.length]);d.borderColor=c;d.backgroundColor=c;});
  if(charts[id]){charts[id].data.labels=labels;charts[id].data.datasets=datasets;charts[id].update();return}
  charts[id]=new Chart(document.getElementById(id),{type,data:{labels,datasets},options:Object.assign({animation:false,
    plugins:{legend:{labels:{color:css('--mut')}}},scales:{x:{ticks:{color:css('--mut')},grid:{color:css('--line')}},
    y:{beginAtZero:true,ticks:{color:css('--mut')},grid:{color:css('--line')}}}},opts)});
}
const pct=v=>v==null?'–':(100*v).toFixed(0)+'%', num=(v,d=1)=>v==null?'–':Number(v).toFixed(d), k=v=>v==null?'–':(v/1000).toFixed(1)+'K';
async function refresh(){
  const d=await (await fetch('/api/data')).json();
  const t=d.traces||{n:0};
  if(t.n){
    document.getElementById('tkpis').innerHTML=[
      ['Episodes',t.n,`${t.train_n} on train problems`],['Verified traces',t.verified,`${t.problems_with_trace} problems`],
      ['Usable for SFT',t.usable,`target ${t.target}`],['MiMo full-solve rate',pct(t.solve_rate),`40-only ${pct(t.small_only_rate)}`],
      ['MiMo held-out score',t.heldout_score==null?'–':num(t.heldout_score)+(t.heldout_se==null?'':' ± '+num(t.heldout_se)),`n=${t.heldout_n} · base 40.3`],
      ['EXIT in time',pct(t.exit_rate),Object.entries(t.stops).map(([k,v])=>k+' '+v).join(' · ')],
      ['Output / episode',k(t.out_tok_median),'median tokens']
    ].map(([a,b,c])=>`<div class="kpi"><span>${a}</span><b>${b}</b><span>${c}</span></div>`).join('');
    document.getElementById('tbar').style.width=Math.min(100,100*t.usable/t.target)+'%';
    document.getElementById('tbartxt').textContent=`${t.usable} / ${t.target} usable (${t.verified} verified in total)`;
    document.getElementById('tfiles').innerHTML='<tr><th>file</th><th>episodes</th><th>full solves</th></tr>'+Object.entries(t.by_file).map(([f,v])=>`<tr><td>${f}</td><td>${v.n}</td><td>${v.full}</td></tr>`).join('');
    chart('c_tturns','bar',['1','2','3','4','5'],[{label:'MiMo',data:t.turns_hist}]);
    chart('c_tout','bar',['<4K','4–8K','8–16K','16–24K','24–32K'],[{label:'MiMo',data:t.out_hist}]);
  } else { document.getElementById('tkpis').innerHTML='<div class="mut">no MiMo episodes yet</div>'; }
  const ev=d.evals.filter(e=>e.n>0), base=ev.find(e=>e.tag==='ab5-think-on');
  const cls=(v,b,lowerBetter)=>b==null||v==null?'':((lowerBetter?v<b:v>b)?'good':(v===b?'':'bad'));
  document.getElementById('evtab').innerHTML='<tr><th>run</th><th>n</th><th>score</th><th>solved</th><th>40-only</th><th>submitted</th><th>EXIT in time</th><th>hit 5-turn limit</th><th>ran out of tokens</th><th>tokens / rollout (median)</th><th>tokens / turn (median)</th></tr>'+
    d.evals.map(e=>e.n?`<tr><td>${e.tag}</td><td>${e.n}</td><td class="${cls(e.score,base?.score)}">${num(e.score)} ± ${num(e.score_se)}</td><td>${pct(e.solved)}</td><td>${pct(e.small_only)}</td><td>${pct(e.submitted)}</td>
      <td class="${cls(e.exit,base?.exit)}">${pct(e.exit)}</td><td class="${cls(e.turn_limit,base?.turn_limit,true)}">${pct(e.turn_limit)}</td><td class="${cls(e.budget,base?.budget,true)}">${pct(e.budget)}</td>
      <td class="${cls(e.tok_median,base?.tok_median,true)}">${k(e.tok_median)}</td><td>${k(e.tok_per_turn_median)}</td></tr>`:`<tr><td>${e.tag}</td><td colspan="10" class="mut">no results yet</td></tr>`).join('');
  chart('c_turns','bar',['1','2','3','4','5'],ev.map(e=>({label:e.tag,data:e.turns_hist.map(x=>x/e.n)})),
    {scales:{x:{title:{display:true,text:'turns',color:css('--mut')},ticks:{color:css('--mut')}},y:{beginAtZero:true,ticks:{color:css('--mut'),callback:v=>Math.round(v*100)+'%'},grid:{color:css('--line')}}}});
  chart('c_tok','bar',['<5K','5–10K','10–20K','20–40K','40–80K','≥80K'],ev.map(e=>({label:e.tag,data:e.tok_hist.map(x=>x/e.n)})),
    {scales:{x:{ticks:{color:css('--mut')}},y:{beginAtZero:true,ticks:{color:css('--mut'),callback:v=>Math.round(v*100)+'%'},grid:{color:css('--line')}}}});
  const maxSteps=Math.max(1,...d.sft.map(r=>r.steps.length));
  chart('c_loss','line',[...Array(maxSteps).keys()],d.sft.map(r=>({label:r.name,data:r.steps.map(s=>s.nll),tension:.2,pointRadius:2})));
  document.getElementById('sfttab').innerHTML='<tr><th>run</th><th>progress</th><th>mask (epoch)</th><th>last loss</th><th>cost</th></tr>'+
    d.sft.map(r=>{const l=r.steps[r.steps.length-1]||{};return `<tr><td>${r.name}</td><td>${r.done?'done':(r.steps.length+'/'+(r.total??'?'))}</td><td>${l.mask_frac==null?'–':pct(l.mask_frac)}</td><td>${num(l.nll,3)}</td><td>${l.cost_usd==null?'–':'$'+num(l.cost_usd,2)}</td></tr>`+(r.sampler_path?`<tr><td colspan="5"><code>${r.sampler_path}</code></td></tr>`:'')}).join('');
  document.getElementById('status').textContent='updated '+new Date().toLocaleTimeString()+' · refreshes every 20 s · green = better than base (ab5-think-on)';
}
refresh(); setInterval(refresh,20000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/data"):
            tags = list(DEFAULT_EVALS) + sorted(p.parent.name for p in RUNS.glob("eval-*/results.jsonl")
                                                if p.parent.name not in DEFAULT_EVALS)
            body = json.dumps({"evals": [eval_summary(t) for t in tags], "sft": sft_runs(),
                               "traces": traces_summary()}).encode()
            ctype = "application/json"
        else:
            body, ctype = PAGE.encode(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    print(f"SFT dashboard on http://localhost:{a.port}")
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
