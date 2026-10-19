#!/usr/bin/env python3
"""Parallel E1 dispatcher: one job per GPU among 0-3 (never 4-7), launched only where the card has enough
free memory. Watchdog kills a run with >=10 per-clip skips (half-loaded model / co-tenant). Resume is the
runner's own (by clip). Status lines go to e1_status.txt (same format as the chain)."""
import subprocess, time, os, sys, datetime, pathlib
os.chdir("/workspace/svhalluc/claude_track2")
R=pathlib.Path("research_reports/20260906_E1"); LOGS=R/"logs"; LOGS.mkdir(exist_ok=True)
S=R/"e1_beta_status.txt"
def log(m):
    line=f"[{datetime.datetime.utcnow():%H:%M}Z] {m}"; print(line,flush=True); S.open("a").write(line+"\n")
CONDS=[f"{k}_{l}" for k in ("claimconf","detconf","readerconf") for l in ("high","low")]
JOBS=[]
for c in CONDS:
    for sw in ("","_swap"):
        JOBS.append(dict(label=f"q2aT_{c}{sw}",need=26000,out=f"outputs/nm_qwen2audio_sound_{c}_T{sw}_v1.jsonl",
             cmd=["python3",str(R/"eval_e1_nm_v1.py"),"--model","qwen2audio","--attr","sound","--cond",c,"--claim","true"]+(["--swap_options"] if sw else []),tries=0))
def rows(p): return sum(1 for l in open(p) if l.strip()) if os.path.exists(p) else 0
def gpu_free():
    o=subprocess.run(["nvidia-smi","--query-gpu=index,memory.used,memory.total","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout
    d={}
    for l in o.strip().splitlines():
        i,u,t=[int(x) for x in l.split(",")]
        if i<8: d[i]=t-u
    return d
pending=[j for j in JOBS if rows(j["out"])<400]
for j in JOBS:
    if rows(j["out"])>=400: log(f"skip {j['label']} ({rows(j['out'])})")
running={}   # gpu -> (job, Popen, logpath, t0)
while pending or running:
    # reap
    for g,(j,p,lp,t0) in list(running.items()):
        sk=sum(1 for l in open(lp) if l.startswith("[skip]") or l.startswith("SKIP")) if os.path.exists(lp) else 0
        if p.poll() is None and sk>=10:
            log(f"{j['label']} poisoned ({sk} skips) -> kill"); p.kill(); time.sleep(3)
        if p.poll() is not None:
            n=rows(j["out"]); del running[g]
            if n>=400: log(f"{j['label']} OK ({n}) GPU {g} {int(time.time()-t0)}s")
            else:
                j["tries"]+=1
                if j["tries"]<3: log(f"{j['label']} incomplete ({n}), requeue"); pending.append(j)
                else: log(f"{j['label']} FAILED")
    # launch
    free=gpu_free()
    for g in range(8):
        if g in running or not pending: continue
        idx=next((i for i,j in enumerate(pending) if free.get(g,0)>j["need"]+2000),None)
        if idx is None: continue
        j=pending.pop(idx); lp=LOGS/f"{j['label']}.log"
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        p=subprocess.Popen(j["cmd"],stdout=open(lp,"w"),stderr=subprocess.STDOUT,env=env)
        running[g]=(j,p,lp,time.time()); log(f"{j['label']} attempt {j['tries']+1} GPU {g}")
        time.sleep(20)   # let the load claim its memory before the next placement decision
    time.sleep(30)
log("E1 BETA CHAIN DONE")
