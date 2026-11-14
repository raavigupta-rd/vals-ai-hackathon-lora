"""Live dashboard for RL runs under runs/rl/<name>/ (metrics.jsonl + checkpoints.jsonl).

    uv run python scripts/rl_dashboard.py [--port 8765]
    open http://localhost:8765            (auto-refreshes every 15 s)
"""

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "runs" / "rl"
KEYS = {
    "score": "env/all/score", "solved": "env/all/solved", "small_only": "env/all/small_only",
    "submitted": "env/all/submitted", "turns": "env/all/turns", "reward": "env/all/reward/total",
    "frac_mixed": "env/all/by_group/frac_mixed", "frac_all_bad": "env/all/by_group/frac_all_bad",
    "tokens_per_turn": "env/all/ac_tokens_per_turn", "cost": "cost/cumulative_usd",
    "step_seconds": "time/total",
}


def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def run_data(name: str) -> dict:
    d = RUNS / name
    by_step: dict[int, dict] = {}  # metrics.jsonl writes several partial rows per step; merge them
    for m in _jsonl(d / "metrics.jsonl"):
        row = by_step.setdefault(int(m.get("step", 0)), {"step": int(m.get("step", 0))})
        row.update({k: m[v] for k, v in KEYS.items() if m.get(v) is not None})
    steps = [r for _, r in sorted(by_step.items()) if r.get("score") is not None]
    cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
    ds = cfg.get("dataset_builder", {})
    started = (d / "config.json").stat().st_mtime if (d / "config.json").exists() else None
    log = (d / "logs.log").read_text(errors="replace").splitlines()[-200:] if (d / "logs.log").exists() else []
    n_batches = next((int(line.rsplit(" ", 2)[-2]) for line in log if "Will train on" in line), None)
    return {
        "name": name, "steps": steps, "n_batches": n_batches,
        "checkpoints": _jsonl(d / "checkpoints.jsonl"),
        "config": {k: cfg.get(k) for k in ("model_name", "renderer_name", "learning_rate", "lora_rank",
                                           "max_tokens", "temperature", "load_checkpoint_path")}
        | {k: ds.get(k) for k in ("problems_path", "group_size", "batch_size", "max_trajectory_tokens", "turn_bonus")},
        "elapsed_min": round((time.time() - started) / 60, 1) if started else None,
        "updated": max((p.stat().st_mtime for p in d.glob("*") if p.is_file()), default=0),
    }


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RL Run Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--bg:#fafaf9;--card:#fff;--fg:#1c1917;--mut:#78716c;--line:#e7e5e4;--a:#2563eb;--b:#16a34a;--c:#d97706;--d:#dc2626}
@media (prefers-color-scheme:dark){:root{--bg:#0c0a09;--card:#1c1917;--fg:#f5f5f4;--mut:#a8a29e;--line:#292524;--a:#60a5fa;--b:#4ade80;--c:#fbbf24;--d:#f87171}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:20px 16px}
h1{font-size:20px;margin:0 0 4px} .mut{color:var(--mut)} select{font:inherit;padding:4px 8px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:16px 0}
.kpi,.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.kpi b{display:block;font-size:24px;font-variant-numeric:tabular-nums} .kpi span{color:var(--mut);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
.card h3{margin:0 0 8px;font-size:13px;color:var(--mut);font-weight:600} canvas{max-height:220px}
code{font-size:12px;word-break:break-all} table{width:100%;border-collapse:collapse;font-size:12px} td{padding:3px 6px;border-top:1px solid var(--line);vertical-align:top}
</style></head><body><main>
<h1>RL run <select id="run"></select></h1>
<div class="mut" id="status">loading…</div>
<div class="kpis" id="kpis"></div>
<div class="grid">
<div class="card"><h3>Mean score (0–100)</h3><canvas id="c_score"></canvas></div>
<div class="card"><h3>Outcome rates</h3><canvas id="c_rates"></canvas></div>
<div class="card"><h3>Turns per episode</h3><canvas id="c_turns"></canvas></div>
<div class="card"><h3>Cumulative cost (USD)</h3><canvas id="c_cost"></canvas></div>
</div>
<div class="card" style="margin-top:12px"><h3>Checkpoints (sampler paths are submittable)</h3><table id="ckpts"></table></div>
<div class="card" style="margin-top:12px"><h3>Config</h3><table id="cfg"></table></div>
</main><script>
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const charts={};
function line(id,labels,sets,max){
  const ds=sets.map(([label,data,color])=>({label,data,borderColor:css(color),backgroundColor:css(color),tension:.2,pointRadius:3}));
  if(charts[id]){charts[id].data.labels=labels;charts[id].data.datasets=ds;charts[id].update();return}
  charts[id]=new Chart(document.getElementById(id),{type:'line',data:{labels,datasets:ds},
    options:{animation:false,plugins:{legend:{display:sets.length>1,labels:{color:css('--mut')}}},
    scales:{x:{title:{display:true,text:'step',color:css('--mut')},ticks:{color:css('--mut')},grid:{color:css('--line')}},
            y:{min:0,max,ticks:{color:css('--mut')},grid:{color:css('--line')}}}}});
}
const pct=v=>v==null?'–':(100*v).toFixed(0)+'%', num=(v,d=1)=>v==null?'–':Number(v).toFixed(d);
async function load(){
  const runs=await (await fetch('/api/runs')).json(); const sel=document.getElementById('run');
  if(!sel.options.length){runs.forEach(r=>sel.add(new Option(r,r)));sel.onchange=refresh}
  refresh();
}
async function refresh(){
  const name=document.getElementById('run').value; if(!name)return;
  const d=await (await fetch('/api/run?name='+encodeURIComponent(name))).json();
  const s=d.steps, last=s[s.length-1]||{}, first=s[0]||{};
  document.getElementById('status').textContent=`${s.length}${d.n_batches?'/'+d.n_batches:''} steps done · running for ${d.elapsed_min} min · last update ${new Date(d.updated*1000).toLocaleTimeString()} · refreshes every 15 s`;
  document.getElementById('kpis').innerHTML=[
    ['Score (latest)',num(last.score),s.length>1?`step 0: ${num(first.score)}`:''],
    ['Solved 100',pct(last.solved),''],['Small-only 40',pct(last.small_only),''],
    ['Submitted',pct(last.submitted),''],['Turns',num(last.turns),''],
    ['Groups w/ signal',pct(last.frac_mixed),'mixed rewards in group'],['Cost',last.cost==null?'–':'$'+num(last.cost,2),'cumulative']
  ].map(([k,v,sub])=>`<div class="kpi"><span>${k}</span><b>${v}</b><span>${sub}</span></div>`).join('');
  const L=s.map(r=>r.step);
  line('c_score',L,[['score',s.map(r=>r.score),'--a']],100);
  line('c_rates',L,[['solved',s.map(r=>r.solved),'--b'],['submitted',s.map(r=>r.submitted),'--a'],['small-only',s.map(r=>r.small_only),'--c'],['all-zero groups',s.map(r=>r.frac_all_bad),'--d']],1);
  line('c_turns',L,[['turns',s.map(r=>r.turns),'--c']]);
  line('c_cost',L,[['USD',s.map(r=>r.cost),'--d']]);
  document.getElementById('ckpts').innerHTML=d.checkpoints.length?d.checkpoints.map(c=>`<tr><td>${c.name??''}</td><td><code>${c.sampler_path??''}</code></td></tr>`).join(''):'<tr><td class="mut">none yet</td></tr>';
  document.getElementById('cfg').innerHTML=Object.entries(d.config).map(([k,v])=>`<tr><td>${k}</td><td><code>${v??''}</code></td></tr>`).join('');
}
load(); setInterval(refresh,15000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/runs"):
            runs = sorted((p for p in RUNS.iterdir() if p.is_dir()), key=lambda p: -p.stat().st_mtime) if RUNS.exists() else []
            return self._send(json.dumps([p.name for p in runs]).encode(), "application/json")
        if self.path.startswith("/api/run"):
            from urllib.parse import parse_qs, urlparse
            name = parse_qs(urlparse(self.path).query).get("name", [""])[0]
            if not name or "/" in name or not (RUNS / name).is_dir():
                self.send_error(404)
                return
            return self._send(json.dumps(run_data(name)).encode(), "application/json")
        return self._send(PAGE.encode(), "text/html; charset=utf-8")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    print(f"RL dashboard on http://localhost:{a.port}")
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
