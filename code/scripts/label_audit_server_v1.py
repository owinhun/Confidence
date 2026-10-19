#!/usr/bin/env python3
"""Blind listening audit for the sound-event clips whose labels the reviews question.

An independent CLAP checkpoint separates nine of the twelve query classes well and the two
eating classes and sneezing poorly. `label_noise_sensitivity_v1.py` bounds how much the result
could depend on those clips, but it cannot settle whether the labels are wrong: CLAP is also the
detector in Sec. 7, so using it to flag suspect labels is not an independent audit.

This serves the clips to a human instead. The manifest label is deliberately NOT shown -- the
annotator answers "is this sound present?" and the agreement is computed afterwards. Showing the
label first would turn the audit into a confirmation task and it would tell us nothing.

Stdlib only, resumable, one JSON object per decision appended to outputs/label_audit_v1.jsonl.
Default set is the three weakly-separated classes (98 clips); --all serves all 400.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path("/workspace/svhalluc")
MANIFEST = ROOT / "claude_track2/data/held_attr_soundevent_v1.jsonl"
OUT = ROOT / "claude_track2/outputs/label_audit_v1.jsonl"
WEAK = {"people eating apple", "people eating crisps", "people sneezing"}

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>label audit</title><style>
:root{--bg:#faf9f7;--fg:#1a1a19;--mut:#6b6a67;--line:#e0ded9;--acc:#2f6f4e;--no:#9b3b32}
@media(prefers-color-scheme:dark){:root{--bg:#191918;--fg:#eeede9;--mut:#9d9b96;--line:#333230}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
.wrap{max-width:760px;margin:0 auto;padding:20px 16px 60px}
h1{font-size:15px;font-weight:600;margin:0 0 2px}
.sub{color:var(--mut);font-size:13px;margin-bottom:14px}
.bar{height:5px;background:var(--line);border-radius:3px;overflow:hidden;margin-bottom:18px}
.bar i{display:block;height:100%;background:var(--acc);transition:width .2s}
.q{font-size:20px;font-weight:600;margin:16px 0 4px}
.q b{color:var(--acc)}
video{width:100%;border-radius:8px;background:#000;display:block}
.btns{display:flex;gap:8px;margin:16px 0 10px;flex-wrap:wrap}
button{flex:1;min-width:110px;padding:13px 10px;font-size:15px;font-weight:600;border:1px solid var(--line);
 border-radius:8px;background:var(--bg);color:var(--fg);cursor:pointer}
button:hover{border-color:var(--mut)}
button.y{border-color:var(--acc);color:var(--acc)}
button.n{border-color:var(--no);color:var(--no)}
button.s{flex:0 0 auto;min-width:80px;font-weight:400;color:var(--mut)}
kbd{font:11px monospace;border:1px solid var(--line);border-radius:3px;padding:1px 4px;color:var(--mut);margin-left:5px}
.meta{color:var(--mut);font-size:12px;font-family:monospace;word-break:break-all;margin-top:12px}
.done{text-align:center;padding:60px 20px}
.done h2{font-size:19px}
table{border-collapse:collapse;width:100%;margin-top:18px;font-size:14px}
td,th{border-bottom:1px solid var(--line);padding:7px 6px;text-align:left}
th{color:var(--mut);font-weight:500;font-size:12px}
td.n{text-align:right;font-variant-numeric:tabular-nums}
.hint{color:var(--mut);font-size:12px;margin-top:16px}
</style></head><body><div class="wrap" id="app">불러오는 중…</div>
<script>
let items=[],i=0,total=0,doneN=0;
async function boot(){
  const r=await (await fetch('/api/items')).json();
  items=r.items; total=r.total; doneN=r.done; i=0; render();
}
function render(){
  const app=document.getElementById('app');
  if(i>=items.length){ finish(); return; }
  const it=items[i];
  const pct=total?((doneN)/total*100):0;
  app.innerHTML=`
   <h1>라벨 청취 검증</h1>
   <div class="sub">${doneN} / ${total} 완료 · 남은 ${items.length-i}개</div>
   <div class="bar"><i style="width:${pct}%"></i></div>
   <video id="v" src="/clip?p=${encodeURIComponent(it.clip)}" controls autoplay></video>
   <div class="q">이 클립에 <b>${it.q}</b> 소리가 들리나요?</div>
   <div class="btns">
     <button class="y" onclick="send('yes')">들린다<kbd>1</kbd></button>
     <button class="n" onclick="send('no')">안 들린다<kbd>2</kbd></button>
     <button class="s" onclick="send('unsure')">모르겠음<kbd>3</kbd></button>
   </div>
   <div class="hint">판정 기준: 질문한 소리가 <b>클립 안에서 실제로 들리면</b> "들린다". 화면에 보이지만
     소리가 없으면 "안 들린다". 정답은 표시하지 않습니다 — 일치율은 나중에 계산합니다.</div>
   <div class="meta">${it.clip.split('/').pop()}</div>`;
}
async function send(v){
  const it=items[i];
  await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({clip:it.clip,query_class:it.q,answer:v})});
  doneN++; i++; render();
}
async function finish(){
  const s=await (await fetch('/api/summary')).json();
  let rows=s.per_class.map(c=>`<tr><td>${c.cls}</td><td class="n">${c.n}</td>
    <td class="n">${c.agree}</td><td class="n">${c.pct}%</td><td class="n">${c.unsure}</td></tr>`).join('');
  document.getElementById('app').innerHTML=`<div class="done"><h2>이 묶음 완료</h2>
    <p class="sub">${s.total_labelled}개 판정 · 매니페스트와 일치 ${s.agree_pct}%</p></div>
    <table><tr><th>클래스</th><th class="n">판정</th><th class="n">일치</th><th class="n">일치율</th><th class="n">모르겠음</th></tr>
    ${rows}</table>
    <div class="hint">결과: <code>outputs/label_audit_v1.jsonl</code> · 창을 닫아도 이어서 할 수 있습니다.</div>`;
}
document.addEventListener('keydown',e=>{
  if(i>=items.length)return;
  if(e.key==='1')send('yes'); else if(e.key==='2')send('no'); else if(e.key==='3')send('unsure');
  else if(e.key===' '){const v=document.getElementById('v');if(v){e.preventDefault();v.paused?v.play():v.pause();}}
});
boot();
</script></body></html>"""


def load_items(all_classes):
    rows = [json.loads(l) for l in MANIFEST.open() if l.strip()]
    if not all_classes:
        rows = [r for r in rows if r["query_class"] in WEAK]
    done = set()
    if OUT.exists():
        for l in OUT.read_text().splitlines():
            if l.strip():
                done.add(json.loads(l)["clip"])
    return rows, done


class H(BaseHTTPRequestHandler):
    all_classes = False

    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if u.path == "/api/items":
            rows, done = load_items(self.all_classes)
            todo = [{"clip": r["clip"], "q": r["query_class"]} for r in rows
                    if r["clip"] not in done]
            return self._send(200, json.dumps(
                {"items": todo, "total": len(rows), "done": len(rows) - len(todo)}))
        if u.path == "/api/summary":
            return self._send(200, json.dumps(summary(self.all_classes)))
        if u.path == "/clip":
            rel = parse_qs(u.query).get("p", [""])[0]
            # only ever serve files the manifest names, never an arbitrary path
            rows, _ = load_items(True)
            if rel not in {r["clip"] for r in rows}:
                return self._send(403, b"no", "text/plain")
            f = ROOT / rel
            if not f.exists():
                return self._send(404, b"missing", "text/plain")
            data = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Accept-Ranges", "none")
            self.end_headers()
            self.wfile.write(data)
            return
        self._send(404, b"no", "text/plain")

    def do_POST(self):
        if urlparse(self.path).path != "/api/label":
            return self._send(404, b"no", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        d = json.loads(self.rfile.read(n) or b"{}")
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"clip": d["clip"], "query_class": d["query_class"],
                                "answer": d["answer"]}, ensure_ascii=False) + "\n")
        self._send(200, json.dumps({"ok": True}))

    def log_message(self, *a):
        pass


def summary(all_classes):
    rows, _ = load_items(True)
    lab = {r["clip"]: r["label"] for r in rows}
    cls = {r["clip"]: r["query_class"] for r in rows}
    got = []
    if OUT.exists():
        for l in OUT.read_text().splitlines():
            if l.strip():
                got.append(json.loads(l))
    seen, per = {}, {}
    for g in got:
        seen[g["clip"]] = g["answer"]          # last decision wins
    n_ag = n_tot = 0
    for c, a in seen.items():
        k = cls.get(c, "?")
        p = per.setdefault(k, {"n": 0, "agree": 0, "unsure": 0})
        p["n"] += 1
        if a == "unsure":
            p["unsure"] += 1
            continue
        # manifest label A means the event is present
        ag = (a == "yes") == (lab.get(c) == "A")
        p["agree"] += int(ag)
        n_ag += int(ag)
        n_tot += 1
    return {"total_labelled": len(seen),
            "agree_pct": round(n_ag / n_tot * 100, 1) if n_tot else 0.0,
            "per_class": [{"cls": k, "n": v["n"], "agree": v["agree"], "unsure": v["unsure"],
                           "pct": round(v["agree"] / max(1, v["n"] - v["unsure"]) * 100, 1)}
                          for k, v in sorted(per.items())]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--all", action="store_true", help="serve all 400 clips, not just the three weak classes")
    a = ap.parse_args()
    H.all_classes = a.all
    rows, done = load_items(a.all)
    print(f"[audit] {len(rows)} clips ({'all classes' if a.all else '3 weak classes'}), "
          f"{len(done)} already labelled", flush=True)
    print(f"[audit] http://localhost:{a.port}/", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
