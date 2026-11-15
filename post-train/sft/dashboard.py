"""Live local dashboard for Tinker SFT runs.

Reads runs/<name>/metrics.jsonl and runs/<name>/checkpoints.jsonl (written by the cookbook
trainer) and charts loss, learning rate, tokens, cost, and step time. The page refreshes every 15 s.

Usage: python sft/dashboard.py [--port 8765]   then open http://localhost:8765
"""

import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

RUNS = Path(__file__).resolve().parent.parent / "runs"
TRAIN_PRICE_PER_M = 1.463  # USD per 1M training tokens, Qwen3.5-9B (Tinker pricing page)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a line still being written
    return out


def run_list() -> list[dict]:
    runs = []
    for d in sorted(RUNS.glob("*/"), key=lambda p: p.stat().st_mtime, reverse=True):
        m = d / "metrics.jsonl"
        n_train = sum(1 for _ in open(d / "train.jsonl")) if (d / "train.jsonl").exists() else None
        runs.append({"name": d.name, "has_metrics": m.exists(),
                     "updated": (m if m.exists() else d).stat().st_mtime, "examples": n_train})
    return runs


def clean(v):
    return v if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)) else None


def run_data(name: str) -> dict:
    d = RUNS / name
    rows = read_jsonl(d / "metrics.jsonl")
    keys = sorted({k for r in rows for k, v in r.items() if clean(v) is not None})
    series = {k: [clean(r.get(k)) for r in rows] for k in keys}
    return {"name": name, "n": len(rows), "series": series,
            "checkpoints": read_jsonl(d / "checkpoints.jsonl"),
            "price_per_m": TRAIN_PRICE_PER_M}


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SFT Training Monitor</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--bg:#f7f7f5;--card:#fff;--fg:#1d1d1b;--muted:#6b6b66;--line:#e4e4df;--accent:#2f6fde;--accent2:#d9822b;--good:#2e8b57}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#151514;--card:#1e1e1c;--fg:#ececea;--muted:#9a9a94;--line:#2e2e2b;--accent:#6b9cf0;--accent2:#e8a35c;--good:#5cbf8a}}
:root[data-theme="dark"]{--bg:#151514;--card:#1e1e1c;--fg:#ececea;--muted:#9a9a94;--line:#2e2e2b;--accent:#6b9cf0;--accent2:#e8a35c;--good:#5cbf8a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;padding:16px;border-bottom:1px solid var(--line)}
h1{font-size:17px;margin:0 12px 0 0}select{font:inherit;padding:6px 8px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
.muted{color:var(--muted)}main{padding:16px;max-width:1300px;margin:0 auto}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:16px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.tile .k{font-size:12px;color:var(--muted)}.tile .v{font-size:22px;font-weight:600;margin-top:4px;font-variant-numeric:tabular-nums}
.bar{height:6px;background:var(--line);border-radius:3px;margin-top:8px;overflow:hidden}.bar>div{height:100%;background:var(--good)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.card h2{font-size:13px;margin:0 0 8px;color:var(--muted);font-weight:600}.chart{position:relative;height:240px}
table{width:100%;border-collapse:collapse;font-size:12px}td,th{padding:6px;border-bottom:1px solid var(--line);text-align:left;word-break:break-all}
@media (max-width:480px){.grid{grid-template-columns:1fr}}
</style></head><body>
<header><h1>SFT Training Monitor</h1><select id="run"></select>
<span class="muted" id="status">loading…</span></header>
<main>
<div class="tiles" id="tiles"></div>
<div class="grid">
 <div class="card"><h2>Training loss (train_mean_nll) — raw and smoothed</h2><div class="chart"><canvas id="loss"></canvas></div></div>
 <div class="card"><h2>Learning rate</h2><div class="chart"><canvas id="lr"></canvas></div></div>
 <div class="card"><h2>Tokens per step</h2><div class="chart"><canvas id="tok"></canvas></div></div>
 <div class="card"><h2>Seconds per step</h2><div class="chart"><canvas id="time"></canvas></div></div>
 <div class="card"><h2>Any metric</h2><select id="anykey"></select><div class="chart"><canvas id="any"></canvas></div></div>
 <div class="card"><h2>Checkpoints</h2><div id="ckpts" class="muted">none yet</div></div>
</div></main>
<script>
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const charts={};let current=null,anyKey=null,last=null;
function ema(a,alpha=0.15){let s=null;return a.map(v=>v==null?s:(s=s==null?v:alpha*v+(1-alpha)*s))}
function pick(series,...names){for(const n of names){if(series[n])return series[n]}
 const k=Object.keys(series).find(k=>names.some(n=>k.endsWith(n)));return k?series[k]:null}
function draw(id,labels,sets){const c=document.getElementById(id);if(!sets.length||!sets[0].data){if(charts[id]){charts[id].destroy();delete charts[id]}return}
 const grid=css('--line'),fg=css('--muted');
 const cfg={type:'line',data:{labels,datasets:sets.map(s=>({label:s.label,data:s.data,borderColor:s.color,borderWidth:s.width||2,pointRadius:0,spanGaps:true,tension:0.2}))},
  options:{animation:false,maintainAspectRatio:false,interaction:{mode:'index',intersect:false},
   scales:{x:{title:{display:true,text:'step',color:fg},ticks:{color:fg,maxTicksLimit:8},grid:{color:grid}},y:{ticks:{color:fg},grid:{color:grid}}},
   plugins:{legend:{display:sets.length>1,labels:{color:fg}}}}};
 if(charts[id]){charts[id].data=cfg.data;charts[id].options=cfg.options;charts[id].update()}else charts[id]=new Chart(c,cfg)}
function fmt(v,d=3){return v==null?'–':(Math.abs(v)>=1000?v.toLocaleString(undefined,{maximumFractionDigits:0}):v.toFixed(d))}
async function loadRuns(){const runs=await (await fetch('/api/runs')).json();const sel=document.getElementById('run');
 const prev=sel.value;sel.innerHTML=runs.map(r=>`<option value="${r.name}">${r.name}${r.has_metrics?'':' (no metrics yet)'}</option>`).join('');
 if(prev&&runs.some(r=>r.name===prev))sel.value=prev;current=sel.value;return runs}
async function refresh(){const runs=await loadRuns();if(!current){document.getElementById('status').textContent='no runs in runs/ yet';return}
 const d=await (await fetch('/api/run?name='+encodeURIComponent(current))).json();const s=d.series;
 const steps=s.step||[...Array(d.n).keys()];const loss=pick(s,'train_mean_nll','loss');
 const prog=pick(s,'progress');const toks=pick(s,'num_tokens','tokens','elapsed_tokens');
 const tsec=pick(s,'time/step','time/total','step_time');
 const lastOf=a=>a?[...a].reverse().find(v=>v!=null):null;
 const cumTok=s.elapsed_tokens?lastOf(s.elapsed_tokens):(toks?toks.reduce((a,b)=>a+(b||0),0):null);
 const p=lastOf(prog);const avg=tsec?tsec.filter(v=>v!=null).reduce((a,b)=>a+b,0)/Math.max(1,tsec.filter(v=>v!=null).length):null;
 const eta=(p&&avg&&d.n)?(d.n/p-d.n)*avg:null;const info=runs.find(r=>r.name===current);
 document.getElementById('tiles').innerHTML=[
  ['Steps logged',d.n],['Latest loss',fmt(lastOf(loss))],['Smoothed loss',fmt(lastOf(ema(loss||[])))],
  ['Progress',p==null?'–':(p*100).toFixed(1)+'%',p],['Tokens trained',cumTok==null?'–':(cumTok/1e6).toFixed(2)+'M'],
  ['Cost so far',cumTok==null?'–':'$'+(cumTok/1e6*d.price_per_m).toFixed(2)],
  ['ETA',eta==null?'–':Math.round(eta/60)+' min'],['Examples',info&&info.examples!=null?info.examples:'–']]
  .map(([k,v,bar])=>`<div class="tile"><div class="k">${k}</div><div class="v">${v}</div>${bar!=null?`<div class="bar"><div style="width:${Math.min(100,bar*100)}%"></div></div>`:''}</div>`).join('');
 draw('loss',steps,[{label:'raw',data:loss,color:css('--muted'),width:1},{label:'EMA',data:loss&&ema(loss),color:css('--accent')}]);
 draw('lr',steps,[{label:'lr',data:pick(s,'learning_rate'),color:css('--accent2')}]);
 draw('tok',steps,[{label:'tokens',data:pick(s,'num_tokens','tokens'),color:css('--accent')}]);
 draw('time',steps,[{label:'sec',data:tsec,color:css('--accent2')}]);
 const ks=Object.keys(s).filter(k=>k!=='step');const ak=document.getElementById('anykey');
 if(ak.options.length!==ks.length){ak.innerHTML=ks.map(k=>`<option>${k}</option>`).join('');if(anyKey&&ks.includes(anyKey))ak.value=anyKey}
 anyKey=ak.value;draw('any',steps,[{label:anyKey,data:s[anyKey],color:css('--good')}]);
 document.getElementById('ckpts').innerHTML=d.checkpoints.length?'<table><tr><th>name</th><th>sampler path</th></tr>'+
  d.checkpoints.map(c=>`<tr><td>${c.name??c.batch??''}</td><td>${c.sampler_path??c.state_path??''}</td></tr>`).join('')+'</table>':'none yet';
 document.getElementById('status').textContent='updated '+new Date().toLocaleTimeString()+' · refreshes every 15 s'}
document.getElementById('run').onchange=e=>{current=e.target.value;refresh()};
document.getElementById('anykey').onchange=e=>{anyKey=e.target.value;refresh()};
refresh();setInterval(refresh,15000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        if url.path == "/api/runs":
            self._send(json.dumps(run_list()).encode(), "application/json")
        elif url.path == "/api/run":
            name = parse_qs(url.query).get("name", [""])[0]
            if not name or "/" in name or ".." in name or not (RUNS / name).is_dir():
                self.send_error(404)
                return
            self._send(json.dumps(run_data(name)).encode(), "application/json")
        else:
            self._send(PAGE.encode(), "text/html; charset=utf-8")

    def log_message(self, *args) -> None:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    print(f"dashboard on http://localhost:{args.port}  (runs dir: {RUNS})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
