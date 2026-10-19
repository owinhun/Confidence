#!/usr/bin/env python3
"""E4 dispatcher: 4 cascade cells x 2 orders of cascadex_warn. One job per GPU (0-7, session permission),
only on a card with enough FREE memory so other tenants are never crowded out. Watchdog: >=10 per-clip
skips -> kill + requeue (half-loaded model). Resume is the runner's own, by clip."""
import subprocess, time, os, datetime, pathlib
os.chdir("/workspace/svhalluc/claude_track2")
R=pathlib.Path("research_reports/20260907_E4"); LOGS=R/"logs"; LOGS.mkdir(exist_ok=True); S=R/"e4_status.txt"
def log(m):
    line=f"[{datetime.datetime.utcnow():%H:%M}Z] {m}"; print(line,flush=True); S.open("a").write(line+"\n")
CELLS=[("q2a","outputs/nm_qwen2audio_sound_cascadex_warn{sw}_v1.jsonl",26000,
        ["python3",str(R/"eval_e4_nm_v1.py"),"--model","qwen2audio","--attr","sound","--cond","cascadex_warn"]),
       ("q25","outputs/s2_sound_cascadex_warn{sw}_v1.jsonl",30000,
        ["python3",str(R/"eval_e4_s2_v1.py"),"--attr","sound","--cond","cascadex_warn"]),
       ("q3","outputs/s2q3_sound_cascadex_warn{sw}_v1.jsonl",71000,
        ["python3",str(R/"eval_e4_q3_v1.py"),"--attr","sound","--cond","cascadex_warn"]),
       ("nemo","outputs/nm_nemotron_sound_cascadex_warn{sw}_v1.jsonl",71000,
        ["bash",str(R/"run_e4_nemotron.sh"),"--attr","sound","--cond","cascadex_warn"])]
JOBS=[]
for name,out,need,cmd in CELLS:
    for sw in ("","_swap"):
        JOBS.append(dict(label=f"{name}{sw}",out=out.format(sw=sw),need=need,tries=0,
                         cmd=cmd+(["--swap_options"] if sw else [])))
def rows(p): return sum(1 for l in open(p) if l.strip()) if os.path.exists(p) else 0
def gpus():
    o=subprocess.run(["nvidia-smi","--query-gpu=index,memory.used,memory.total,utilization.gpu",
                      "--format=csv,noheader,nounits"],capture_output=True,text=True).stdout
    d={}
    for l in o.strip().splitlines():
        i,u,t,ut=[int(x) for x in l.split(",")]
        d[i]=(t-u,ut)
    return d
pending=[j for j in JOBS if rows(j["out"])<400]
for j in JOBS:
    if rows(j["out"])>=400: log(f"skip {j['label']} ({rows(j['out'])})")
running={}
while pending or running:
    for g,(j,p,lp,t0) in list(running.items()):
        sk=sum(1 for l in open(lp) if l.startswith("[skip]") or l.startswith("SKIP")) if os.path.exists(lp) else 0
        if p.poll() is None and sk>=10:
            log(f"{j['label']} poisoned ({sk} skips) -> kill"); p.kill(); time.sleep(3)
        if p.poll() is not None:
            n=rows(j["out"]); del running[g]
            if n>=400: log(f"{j['label']} OK ({n}) GPU {g} {int(time.time()-t0)}s")
            else:
                j["tries"]+=1
                if j["tries"]<4: log(f"{j['label']} incomplete ({n}), requeue"); pending.append(j)
                else: log(f"{j['label']} FAILED")
    st=gpus()
    for g in sorted(st):
        if g in running or not pending: continue
        free,ut=st[g]
        idx=next((i for i,j in enumerate(pending) if free>j["need"]+2000 and (j["need"]<60000 or ut<5)),None)
        if idx is None: continue
        j=pending.pop(idx); lp=LOGS/f"{j['label']}.log"
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        p=subprocess.Popen(j["cmd"],stdout=open(lp,"w"),stderr=subprocess.STDOUT,env=env)
        running[g]=(j,p,lp,time.time()); log(f"{j['label']} attempt {j['tries']+1} GPU {g}"); time.sleep(45)
    time.sleep(30)
log("E4 CHAIN DONE")
