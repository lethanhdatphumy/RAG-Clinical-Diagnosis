"""
Generates dashboard.html comparing the RAG pipeline vs plain LLM baseline.
Reads results.json (pipeline) and baseline_results.json (direct LLM).
"""

import json
import os
from collections import defaultdict
from statistics import mean, median

RESULTS_PATH  = os.path.join(os.path.dirname(__file__), "results.json")
BASELINE_PATH = os.path.join(os.path.dirname(__file__), "baseline_results.json")
DASH_PATH     = os.path.join(os.path.dirname(__file__), "dashboard.html")

TYPES_ORDER = ["vague", "specific", "general"]


def stats(results, baseline):
    n = len(results)
    b_by_id = {r["id"]: r for r in baseline}

    ok = [r for r in results if not r.get("error")]
    route_acc = round(sum(1 for r in ok if r.get("route_correct")) / max(len(ok), 1) * 100, 1)
    dis_hit   = round(sum(1 for r in ok if r.get("disease_hit"))   / max(len(ok), 1) * 100, 1)
    avg_lat   = round(mean(r["latency_s"] for r in results), 2)
    med_lat   = round(median(r["latency_s"] for r in results), 2)
    max_lat   = round(max(r["latency_s"] for r in results), 2)

    b_ok = [r for r in baseline if not r.get("error")]
    b_dis_hit = round(sum(1 for r in b_ok if r.get("disease_hit")) / max(len(b_ok), 1) * 100, 1)
    b_avg_lat = round(mean(r["latency_s"] for r in baseline), 2)

    # per-type
    p_type = defaultdict(lambda: {"dis": 0, "route": 0, "total": 0, "lat": []})
    b_type = defaultdict(lambda: {"dis": 0, "total": 0, "lat": []})
    for r in results:
        qt = r.get("query_type", "?")
        p_type[qt]["total"] += 1
        if r.get("disease_hit"):   p_type[qt]["dis"] += 1
        if r.get("route_correct"): p_type[qt]["route"] += 1
        p_type[qt]["lat"].append(r["latency_s"])
    for r in baseline:
        qt = r.get("query_type", "?")
        b_type[qt]["total"] += 1
        if r.get("disease_hit"): b_type[qt]["dis"] += 1
        b_type[qt]["lat"].append(r["latency_s"])

    p_dis_by_type = [round(p_type[t]["dis"]  / max(p_type[t]["total"], 1) * 100, 1) for t in TYPES_ORDER]
    b_dis_by_type = [round(b_type[t]["dis"]  / max(b_type[t]["total"], 1) * 100, 1) for t in TYPES_ORDER]
    p_lat_by_type = [round(mean(p_type[t]["lat"]), 2) if p_type[t]["lat"] else 0 for t in TYPES_ORDER]
    b_lat_by_type = [round(mean(b_type[t]["lat"]), 2) if b_type[t]["lat"] else 0 for t in TYPES_ORDER]
    p_route_by_type = [round(p_type[t]["route"] / max(p_type[t]["total"], 1) * 100, 1) for t in TYPES_ORDER]

    # route split
    routes = defaultdict(int)
    for r in results:
        routes[r.get("route") or "unknown"] += 1

    # per-query latency series
    lats     = [r["latency_s"] for r in results]
    b_lats   = [b_by_id.get(r["id"], {}).get("latency_s", 0) for r in results]
    lat_cols = ["#10b981" if not r.get("error") else "#ef4444" for r in results]
    ids      = [r["id"] for r in results]

    # table rows
    rows = []
    for r in results:
        br = b_by_id.get(r["id"], {})
        rows.append({
            "id": r["id"],
            "qtype": r.get("query_type", "?"),
            "query": r["query"],
            "disease": r.get("primary_disease", ""),
            "route": r.get("route"),
            "route_ok": r.get("route_correct"),
            "docs_rel": r.get("docs_relevant"),
            "p_hit": r.get("disease_hit"),
            "b_hit": br.get("disease_hit"),
            "p_lat": r["latency_s"],
            "b_lat": br.get("latency_s", "—"),
            "p_ans": (r.get("answer_snippet") or r.get("error") or "")[:80],
            "b_ans": (br.get("answer_snippet") or br.get("error") or "")[:80],
        })

    return dict(
        n=n, errors=len([r for r in results if r.get("error")]),
        route_acc=route_acc, dis_hit=dis_hit,
        avg_lat=avg_lat, med_lat=med_lat, max_lat=max_lat,
        b_dis_hit=b_dis_hit, b_avg_lat=b_avg_lat,
        p_dis_by_type=p_dis_by_type, b_dis_by_type=b_dis_by_type,
        p_lat_by_type=p_lat_by_type, b_lat_by_type=b_lat_by_type,
        p_route_by_type=p_route_by_type,
        route_labels=list(routes.keys()), route_vals=list(routes.values()),
        ids=ids, lats=lats, b_lats=b_lats, lat_cols=lat_cols,
        rows=rows,
    )


def _esc(s):
    return str(s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

def _icon(v):
    if v is True:  return '<span class="ok">✓</span>'
    if v is False: return '<span class="no">✗</span>'
    return '<span class="na">—</span>'


def render_rows(rows):
    out = []
    for r in rows:
        out.append(f"""<tr>
<td class="mono">{r['id']}</td>
<td><span class="badge b-{r['qtype']}">{r['qtype']}</span></td>
<td class="q-cell" title="{_esc(r['query'])}">{_esc(r['query'][:55])}{'…' if len(r['query'])>55 else ''}</td>
<td>{_esc(r['disease'][:28])}</td>
<td><span class="badge b-r-{r['route'] or 'null'}">{r['route'] or '—'}</span></td>
<td class="c">{_icon(r['route_ok'])}</td>
<td class="c">{_icon(r['docs_rel'])}</td>
<td class="c hl-p">{_icon(r['p_hit'])}</td>
<td class="c hl-b">{_icon(r['b_hit'])}</td>
<td class="mono">{r['p_lat']}s</td>
<td class="mono">{r['b_lat']}{'s' if isinstance(r['b_lat'], float) else ''}</td>
</tr>""")
    return "\n".join(out)


def build(results, baseline):
    s = stats(results, baseline)
    data_js = f"""
const S = {json.dumps({k: v for k, v in s.items() if k != 'rows'})};
"""
    rows_html = render_rows(s["rows"])
    return HTML.replace("{{DATA_JS}}", data_js).replace("{{ROWS}}", rows_html)


HTML = r"""<title>Clinical RAG — Pipeline vs Baseline</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root {
  --bg:#0e1117; --surf:#161b27; --surf2:#1e2536; --bdr:#2a3347;
  --fg:#e2e8f0; --muted:#8895a7;
  --cyan:#22d3ee; --purple:#a78bfa; --green:#10b981;
  --amber:#f59e0b; --red:#ef4444; --rose:#fb7185;
  color-scheme: dark;
}
@media(prefers-color-scheme:light){
  :root:not([data-theme="dark"]){
    --bg:#f1f5f9;--surf:#fff;--surf2:#f8fafc;--bdr:#e2e8f0;
    --fg:#0f172a;--muted:#64748b;color-scheme:light;
  }
}
:root[data-theme="light"]{
  --bg:#f1f5f9;--surf:#fff;--surf2:#f8fafc;--bdr:#e2e8f0;
  --fg:#0f172a;--muted:#64748b;color-scheme:light;
}
*,*::before,*::after{box-sizing:border-box}
body{
  margin:0;background:var(--bg);color:var(--fg);
  font-family:'IBM Plex Sans',system-ui,sans-serif;font-size:14px;line-height:1.6;
  padding-left:max(16px,env(safe-area-inset-left,0px));
  padding-right:max(16px,env(safe-area-inset-right,0px));
  padding-block:0;
}
/* header */
.hdr{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;
  padding:20px 0 16px;border-bottom:1px solid var(--bdr);margin-bottom:24px;}
.hdr h1{font-size:18px;font-weight:600;margin:0;letter-spacing:-.01em;}
.hdr h1 .hl{color:var(--cyan);}
.hdr .sub{color:var(--muted);font-size:12px;}
.hdr button{background:var(--surf2);border:1px solid var(--bdr);border-radius:6px;
  padding:4px 10px;color:var(--muted);cursor:pointer;font-size:12px;}
.hdr button:hover{color:var(--fg);}
/* legend pills */
.legend-row{display:flex;gap:12px;margin-bottom:18px;flex-wrap:wrap;}
.leg{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--muted);}
.dot{width:10px;height:10px;border-radius:50%;}
/* stat grid */
.stat-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:24px;}
.tile{background:var(--surf);border:1px solid var(--bdr);border-radius:10px;padding:16px;}
.tile .lbl{font-size:11px;font-weight:500;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;}
.tile .val{font-family:'IBM Plex Mono',monospace;font-size:28px;font-weight:500;line-height:1;}
.tile .sub{font-size:11px;color:var(--muted);margin-top:4px;}
.tile.cyan  .val{color:var(--cyan);}
.tile.green .val{color:var(--green);}
.tile.amber .val{color:var(--amber);}
.tile.red   .val{color:var(--red);}
.tile.purple .val{color:var(--purple);}
/* improvement chip */
.imp{display:inline-block;padding:1px 6px;border-radius:99px;font-size:11px;font-weight:600;margin-left:6px;}
.imp.pos{background:rgba(16,185,129,.18);color:var(--green);}
.imp.neg{background:rgba(239,68,68,.18);color:var(--red);}
/* chart grid */
.chart-grid{display:grid;grid-template-columns:200px 1fr;gap:16px;margin-bottom:20px;align-items:start;}
@media(max-width:600px){.chart-grid{grid-template-columns:1fr;}}
.chart-row2{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px;}
@media(max-width:600px){.chart-row2{grid-template-columns:1fr;}}
.card{background:var(--surf);border:1px solid var(--bdr);border-radius:10px;padding:16px;}
.card h3{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 14px;}
.ch-wrap{position:relative;}
.h150{height:150px;} .h180{height:180px;} .h170{height:170px;}
/* table */
.sec-lbl{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 12px;}
.tbl-wrap{overflow-x:auto;background:var(--surf);border:1px solid var(--bdr);border-radius:10px;margin-bottom:32px;}
table{border-collapse:collapse;width:100%;min-width:760px;font-size:13px;}
thead th{padding:9px 11px;text-align:left;font-size:11px;font-weight:600;text-transform:uppercase;
  letter-spacing:.05em;color:var(--muted);border-bottom:1px solid var(--bdr);
  background:var(--surf2);white-space:nowrap;}
tbody tr{border-bottom:1px solid var(--bdr);}
tbody tr:last-child{border-bottom:none;}
tbody tr:hover{background:var(--surf2);}
tbody td{padding:8px 11px;vertical-align:middle;}
.q-cell{max-width:220px;}
.mono{font-family:'IBM Plex Mono',monospace;font-variant-numeric:tabular-nums;}
.c{text-align:center;}
.hl-p{background:rgba(34,211,238,.05);}
.hl-b{background:rgba(251,113,133,.05);}
.ok{color:var(--green);font-weight:700;}
.no{color:var(--red);font-weight:700;}
.na{color:var(--muted);}
/* badges */
.badge{display:inline-block;padding:2px 7px;border-radius:999px;font-size:11px;font-weight:500;}
.b-vague   {background:rgba(245,158,11,.15);color:var(--amber);}
.b-specific{background:rgba(34,211,238,.12);color:var(--cyan);}
.b-general {background:rgba(167,139,250,.15);color:var(--purple);}
.b-r-rag      {background:rgba(34,211,238,.12);color:var(--cyan);}
.b-r-finetuned{background:rgba(167,139,250,.15);color:var(--purple);}
.b-r-null,.b-r-unknown{background:rgba(239,68,68,.15);color:var(--red);}
</style>

<div class="hdr">
  <div>
    <h1>Clinical RAG — <span class="hl">Pipeline vs Baseline</span></h1>
    <div class="sub">Agentic RAG · LangGraph · qwen2.5-med (local Ollama) · 24 queries</div>
  </div>
  <button onclick="toggleTheme()">Toggle theme</button>
</div>

<div class="legend-row">
  <div class="leg"><div class="dot" style="background:var(--cyan)"></div>RAG Pipeline</div>
  <div class="leg"><div class="dot" style="background:var(--rose)"></div>Plain LLM Baseline</div>
</div>

<div class="stat-row" id="stat-row"></div>

<div class="chart-grid">
  <div class="card">
    <h3>Route split</h3>
    <div class="ch-wrap h150"><canvas id="donut"></canvas></div>
  </div>
  <div class="card">
    <h3>Latency per query — Pipeline (cyan) vs Baseline (rose)</h3>
    <div class="ch-wrap h170"><canvas id="latcmp"></canvas></div>
  </div>
</div>

<div class="chart-row2">
  <div class="card">
    <h3>Disease hit rate by query type</h3>
    <div class="ch-wrap h180"><canvas id="discmp"></canvas></div>
  </div>
  <div class="card">
    <h3>Avg latency by query type (seconds)</h3>
    <div class="ch-wrap h180"><canvas id="lattype"></canvas></div>
  </div>
</div>

<div class="sec-lbl">Query-level results</div>
<div class="tbl-wrap">
  <table>
    <thead>
      <tr>
        <th>#</th><th>Type</th><th>Query</th><th>Ground truth</th>
        <th>Route</th><th>Rt✓</th><th>Docs</th>
        <th style="color:var(--cyan)">RAG hit</th>
        <th style="color:var(--rose)">LLM hit</th>
        <th style="color:var(--cyan)">RAG lat</th>
        <th style="color:var(--rose)">LLM lat</th>
      </tr>
    </thead>
    <tbody>{{ROWS}}</tbody>
  </table>
</div>

<script>
{{DATA_JS}}

const CYAN="#22d3ee", ROSE="#fb7185", GREEN="#10b981", PURPLE="#a78bfa", AMBER="#f59e0b";

/* stat tiles */
const diff_dis = (S.dis_hit - S.b_dis_hit).toFixed(1);
const diff_lat = (S.b_avg_lat - S.avg_lat).toFixed(1);
const tiles = [
  {lbl:"Queries", val: S.n, sub:"24 total", cls:"cyan"},
  {lbl:"Route accuracy", val: S.route_acc+"%", sub:"vs expected route", cls:"green"},
  {lbl:"RAG disease hit",
   val: S.dis_hit+"%",
   sub: `Baseline: ${S.b_dis_hit}%`+
        `<span class="imp ${+diff_dis>=0?'pos':'neg'}">${+diff_dis>=0?'+':''}${diff_dis}pp</span>`,
   cls:"green"},
  {lbl:"Avg latency",
   val: S.avg_lat+"s",
   sub:`Baseline: ${S.b_avg_lat}s`+
       `<span class="imp ${+diff_lat>=0?'pos':'neg'}">${+diff_lat>=0?'-':'+'}${Math.abs(diff_lat)}s</span>`,
   cls:"amber"},
  {lbl:"Errors", val: S.errors, sub:"pipeline failures", cls: S.errors>0?"red":""},
];
document.getElementById("stat-row").innerHTML = tiles.map(t =>
  `<div class="tile ${t.cls}">
    <div class="lbl">${t.lbl}</div>
    <div class="val">${t.val}</div>
    <div class="sub">${t.sub}</div>
  </div>`
).join("");

Chart.defaults.color = getComputedStyle(document.documentElement).getPropertyValue("--muted").trim();
Chart.defaults.borderColor = getComputedStyle(document.documentElement).getPropertyValue("--bdr").trim();
const SURF = getComputedStyle(document.documentElement).getPropertyValue("--surf").trim();

/* donut */
new Chart(document.getElementById("donut"),{
  type:"doughnut",
  data:{labels:S.route_labels,datasets:[{
    data:S.route_vals, backgroundColor:[CYAN,PURPLE,"#ef4444"],
    borderWidth:2, borderColor:SURF
  }]},
  options:{cutout:"68%",plugins:{legend:{position:"bottom",labels:{boxWidth:10,padding:8,font:{size:11}}}}}
});

/* latency comparison bar */
new Chart(document.getElementById("latcmp"),{
  type:"bar",
  data:{
    labels: S.ids.map(i=>"#"+i),
    datasets:[
      {label:"Pipeline", data:S.lats, backgroundColor:CYAN+"55", borderColor:CYAN, borderWidth:1, borderRadius:3, borderSkipped:false},
      {label:"Baseline", data:S.b_lats, backgroundColor:ROSE+"55", borderColor:ROSE, borderWidth:1, borderRadius:3, borderSkipped:false},
    ]
  },
  options:{
    plugins:{legend:{position:"top",labels:{boxWidth:10,font:{size:11}}},
      tooltip:{callbacks:{label:ctx=>ctx.dataset.label+": "+ctx.parsed.y+"s"}}},
    scales:{
      x:{grid:{display:false},ticks:{font:{size:10}}},
      y:{title:{display:true,text:"seconds",font:{size:10}},ticks:{font:{size:10}}}
    }
  }
});

const TYPE_LABELS=["vague","specific","general"];
const groupOpts = (title, unit="%") => ({
  plugins:{
    legend:{position:"top",labels:{boxWidth:10,font:{size:11}}},
    tooltip:{callbacks:{label:ctx=>ctx.dataset.label+": "+ctx.parsed.y+unit}}
  },
  scales:{
    x:{grid:{display:false}},
    y:{min:0, ticks:{callback:v=>v+unit}}
  }
});

/* disease hit by type */
new Chart(document.getElementById("discmp"),{
  type:"bar",
  data:{
    labels:TYPE_LABELS,
    datasets:[
      {label:"Pipeline",  data:S.p_dis_by_type, backgroundColor:CYAN+"66", borderColor:CYAN, borderWidth:1.5, borderRadius:4, borderSkipped:false},
      {label:"Baseline",  data:S.b_dis_by_type, backgroundColor:ROSE+"66", borderColor:ROSE, borderWidth:1.5, borderRadius:4, borderSkipped:false},
    ]
  },
  options:{...groupOpts("% disease hit"),scales:{...groupOpts().scales,y:{...groupOpts().scales.y,max:100}}}
});

/* latency by type */
new Chart(document.getElementById("lattype"),{
  type:"bar",
  data:{
    labels:TYPE_LABELS,
    datasets:[
      {label:"Pipeline", data:S.p_lat_by_type, backgroundColor:CYAN+"66", borderColor:CYAN, borderWidth:1.5, borderRadius:4, borderSkipped:false},
      {label:"Baseline", data:S.b_lat_by_type, backgroundColor:ROSE+"66", borderColor:ROSE, borderWidth:1.5, borderRadius:4, borderSkipped:false},
    ]
  },
  options:{...groupOpts("s latency","s")}
});

function toggleTheme(){
  const r=document.documentElement,c=r.getAttribute("data-theme");
  const n=c==="dark"?"light":c==="light"?null:"dark";
  n?r.setAttribute("data-theme",n):r.removeAttribute("data-theme");
}
</script>
"""


if __name__ == "__main__":
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    with open(BASELINE_PATH) as f:
        baseline = json.load(f)
    html = build(results, baseline)
    with open(DASH_PATH, "w") as f:
        f.write(html)
    print(f"Dashboard written → {DASH_PATH}")
