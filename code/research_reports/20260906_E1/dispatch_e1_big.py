#!/usr/bin/env python3
"""E1 opportunistic cells (Nemotron/sound, Qwen3-Omni/sound): 65 GB models, one per card, only on a card in 0-7 (user permission 2026-09-06 20:4x KST) with
>71 GB free and <5% util. Watchdog: >=10 per-clip skips -> kill + requeue (co-tenant during load). Status -> e1_big_status.txt."""
import subprocess, time, os, datetime, pathlib
os.chdir("/workspace/svhalluc/claude_track2")
R=pathlib.Path("research_reports/20260906_E1"); LOGS=R/"logs"; LOGS.mkdir(exist_ok=True); S=R/"e1_big_status.txt"
def log(m):
    line=f"[{datetime.datetime.utcnow():%H:%M}Z] {m}"; print(line,flush=True); S.open("a").write(line+"\n")
CONDS=[f"{k}_{l}" for k in ("claimconf","detconf","readerconf") for l in ("high","low")]
JOBS=[]
for c in CONDS:
    for sw in ("","_swap"):
        JOBS.append(dict(label=f"nemo_{c}{sw}",out=f"outputs/nm_nemotron_sound_{c}{sw}_v1.jsonl",tries=0,
            cmd=["bash",str(R/"run_e1_nemotron.sh"),"--attr","sound","--cond",c]+(["--swap_options"] if sw else [])))
        JOBS.append(dict(label=f"q3_{c}{sw}",out=f"outputs/s2q3_sound_{c}{sw}_v1.jsonl",tries=0,
            cmd=["python3",str(R/"eval_e1_q3_v1.py"),"--attr","sound","--cond",c]+(["--swap_options"] if sw else [])))
def rows(p): return sum(1 for l in open(p) if l.strip()) if os.path.exists(p) else 0
def gpus():
    o=subprocess.run(["nvidia-smi","--query-gpu=index,memory.used,memory.total,utilization.gpu","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout
    d={}
    for l in o.strip().splitlines():
        i,u,t,ut=[int(x) for x in l.split(",")]
        if i<8: d[i]=(t-u,ut)
    return d
pending=[j for j in JOBS if rows(j["out"])<400]
for j in JOBS:
    if rows(j["out"])>=400: log(f"skip {j['label']}")
running={}
while pending or running:
    for g,(j,p,lp,t0) in list(running.items()):
        sk=sum(1 for l in open(lp) if l.startswith("[skip]") or l.startswith("SKIP")) if os.path.exists(lp) else 0
        if p.poll() is None and sk>=10: log(f"{j['label']} poisoned ({sk} skips) -> kill"); p.kill(); time.sleep(3)
        if p.poll() is not None:
            n=rows(j["out"]); del running[g]
            if n>=400: log(f"{j['label']} OK ({n}) GPU {g} {int(time.time()-t0)}s")
            else:
                j["tries"]+=1
                if j["tries"]<4: log(f"{j['label']} incomplete ({n}), requeue"); pending.append(j)
                else: log(f"{j['label']} FAILED")
    st=gpus()
    for g in range(8):
        if g in running or not pending: continue
        free,ut=st.get(g,(0,100))
        if free>71000 and ut<5:
            j=pending.pop(0); lp=LOGS/f"{j['label']}.log"
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            p=subprocess.Popen(j["cmd"],stdout=open(lp,"w"),stderr=subprocess.STDOUT,env=env)
            running[g]=(j,p,lp,time.time()); log(f"{j['label']} attempt {j['tries']+1} GPU {g}"); time.sleep(90)
    time.sleep(45)
log("E1 BIG CHAIN DONE")
