#!/usr/bin/env python3
"""
店铺零售 Web 仪表盘（纯标准库，无需 Flask）

运行：python3 dashboard.py
访问：http://127.0.0.1:5050
"""
import csv
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

LOG_DIR = Path(__file__).resolve().parent / "analytics_logs"
LIVE_STATE = LOG_DIR / "live_state.json"

HTML = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"/>
<title>店铺零售仪表盘</title>
<style>
body{font-family:-apple-system,sans-serif;background:#0f1419;color:#e7e9ea;padding:20px}
h1{font-size:1.4rem}.meta{color:#71767b;font-size:.85rem;margin-bottom:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
.card{background:#16181c;border:1px solid #2f3336;border-radius:12px;padding:16px}
.card h2{font-size:1rem;color:#1d9bf0;margin-bottom:10px}
.stat{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #2f3336}
.val{color:#00ba7c;font-weight:600}.alerts{color:#f4212e;font-size:.85rem}
table{width:100%;border-collapse:collapse;font-size:.8rem;margin-top:16px}
th,td{padding:6px;border-bottom:1px solid #2f3336;text-align:left}th{color:#71767b}
</style></head><body>
<h1>店铺零售 BI 仪表盘</h1>
<p class="meta">最后更新：<span id="u">-</span> · <a href="/" style="color:#1d9bf0">刷新</a></p>
<div class="grid" id="g"></div>
<h2 style="margin-top:20px;font-size:1rem">CSV 历史（最近20条）</h2>
<table id="t"><thead><tr><th>时间</th><th>摄像头</th><th>人数</th><th>进</th><th>出</th><th>站/坐</th><th>排队</th></tr></thead><tbody></tbody></table>
<script>
fetch('/api/all').then(r=>r.json()).then(d=>{
  document.getElementById('u').textContent=d.updated||'-';
  let h='';
  for(const [cam,v] of Object.entries(d.cameras||{})){
    let zs=''; for(const [k,n] of Object.entries(v.zone_current||{})) zs+=`<div class="stat"><span>${k}</span><span class="val">${n}</span></div>`;
    let al=(v.alerts||[]).map(a=>`<div>${a}</div>`).join('')||'无';
    h+=`<div class="card"><h2>${cam} FPS ${v.fps||0}</h2>
      <div class="stat"><span>累计客流</span><span class="val">${v.customers_total||0}</span></div>
      <div class="stat"><span>进店/离店</span><span class="val">${v.line_in||0}/${v.line_out||0}</span></div>
      <div class="stat"><span>峰值</span><span class="val">${v.peak||0}</span></div>
      <div class="stat"><span>站/坐/排队</span><span class="val">${v.standing||0}/${v.sitting||0}/${v.queue||0}</span></div>${zs}
      <div class="alerts">${al}</div></div>`;
  }
  document.getElementById('g').innerHTML=h||'<p>等待 78.py 导出 live_state.json...</p>';
  document.querySelector('#t tbody').innerHTML=(d.history||[]).map(r=>
    `<tr><td>${r.time}</td><td>${r.cam}</td><td>${r.persons}</td><td>${r.in}</td><td>${r.out}</td><td>${r.stand}/${r.sit}</td><td>${r.queue}</td></tr>`).join('');
});
setInterval(()=>location.reload(),5000);
</script></body></html>"""


def read_live():
    if not LIVE_STATE.exists():
        return {"updated": None, "cameras": {}}
    try:
        return json.loads(LIVE_STATE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"updated": None, "cameras": {}}


def read_history(limit=20):
    rows = []
    if not LOG_DIR.exists():
        return rows
    for p in sorted(LOG_DIR.glob("*.csv"), reverse=True):
        try:
            with open(p, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rows.append({
                        "time": row.get("time", ""), "cam": row.get("cam", ""),
                        "persons": row.get("persons", ""), "in": row.get("in", ""),
                        "out": row.get("out", ""), "stand": row.get("standing", ""),
                        "sit": row.get("sitting", ""), "queue": row.get("queue", ""),
                    })
        except OSError:
            pass
    rows.sort(key=lambda r: r["time"], reverse=True)
    return rows[:limit]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/all":
            data = read_live()
            data["history"] = read_history()
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode("utf-8"))


if __name__ == "__main__":
    LOG_DIR.mkdir(exist_ok=True)
    port = 5050
    print(f"仪表盘启动: http://127.0.0.1:{port}")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
