#!/usr/bin/env python3
"""Post-hoc directional audit of the bare-forwarding loss (Table 4 'plain' column).

Splits every clip by (ground truth, detector decision):
    FN = false absence  (sound present, detector says absent  -> claim 'does not contain')
    FP = false presence (sound absent,  detector says present -> claim 'contains')
    TP / TN = true claims.
For each cell and each detector: per-direction n, detector error rate, omission baseline A0,
conditional forwarding effect (plain - none, accuracy points) with a source-video cluster
bootstrap CI, and the direction's contribution to the total loss (n_dir/N * conditional).
Order-averaged over the answer orders that exist. Reads only; writes nothing outside this folder.
Clusters use the true 11-char YouTube id (the paper's split('_')[0] misparses ids containing '_';
the total is also reported under that clustering so the discrepancy is visible).
"""
import json, os, re, sys
import numpy as np
OUT = "outputs"; BOOTS = 2000
CELLS = [("Qwen2.5-Omni","s2"),("Qwen3-Omni","s2q3"),("Qwen2-Audio","nm_qwen2audio"),
         ("Nemotron/sound","nm_nemotron"),("Audio Flamingo 3 (sens.)","nm_af3")]
def L(name):
    p=f"{OUT}/{name}_v1.jsonl"
    if not os.path.exists(p): return None
    d={json.loads(l)["clip"]:json.loads(l) for l in open(p) if l.strip()}
    return d if len(d)>=400 else None
def vid(c):
    b=os.path.basename(c); m=re.match(r"^(.{11})_\d{6}",b); return m.group(1) if m else b
def vid_paper(c): return c.split("/")[-1].split("_")[0]
def boot(diffs_by_order, groups_by_order, vkey):
    """order-averaged mean of diffs, cluster bootstrap over videos."""
    vids=sorted({v for g in groups_by_order for v in g})
    vi={v:i for i,v in enumerate(vids)}
    per=[]
    for d,g in zip(diffs_by_order,groups_by_order):
        d=np.asarray(d,float); gi=np.array([vi[v] for v in g])
        per.append((d,[np.where(gi==i)[0] for i in range(len(vids))]))
    rng=np.random.default_rng(0); bs=[]
    for _ in range(BOOTS):
        sel=rng.integers(0,len(vids),len(vids)); vals=[]
        for d,b in per:
            idx=np.concatenate([b[i] for i in sel if b[i].size]) if any(b[i].size for i in sel) else np.array([],int)
            if idx.size: vals.append(d[idx].mean()*100)
        if vals: bs.append(np.mean(vals))
    pt=float(np.mean([d.mean()*100 for d,_ in per]))
    return pt,float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))
def run(det):
    detf={"x":"clap_cascade","2x":"clap2_cascade"}[det]
    D=L(detf); plain=f"cascade{det}_plain"
    print(f"\n=== detector {det} ({detf}) ===")
    res={}
    for name,pfx in CELLS:
        orders=[]
        for tag in ("","_swap"):
            a,b=L(f"{pfx}_sound_{plain}{tag}"),L(f"{pfx}_sound_none{tag}")
            if a and b: orders.append((a,b))
        if not orders: print(f"{name}: no data"); continue
        keys=sorted(set.intersection(*(set(a)&set(b) for a,b in orders))&set(D))
        cat={}
        for k in keys:
            y,dp=D[k]["label"],D[k]["det_pred"]
            cat[k]=("TP" if dp=="A" else "FN") if y=="A" else ("FP" if dp=="A" else "TN")
        N=len(keys); rows={}
        for c in ("FN","FP","TP","TN","ALL","FALSE","TRUE"):
            ks=[k for k in keys if c=="ALL" or cat[k]==c or (c=="FALSE" and cat[k] in("FN","FP")) or (c=="TRUE" and cat[k] in("TP","TN"))]
            if not ks: continue
            diffs=[[a[k]["correct"]-b[k]["correct"] for k in ks] for a,b in orders]
            a0=np.mean([np.mean([b[k]["correct"] for k in ks]) for a,b in orders])*100
            pt,lo,hi=boot(diffs,[[vid(k) for k in ks]]*len(orders),vid)
            rows[c]=dict(n=len(ks),share=len(ks)/N,A0=a0,eff=pt,lo=lo,hi=hi,contrib=pt*len(ks)/N)
        # paper-style clustering for the total, to show the parsing discrepancy
        diffs=[[a[k]["correct"]-b[k]["correct"] for k in keys] for a,b in orders]
        pt2,lo2,hi2=boot(diffs,[[vid_paper(k) for k in keys]]*len(orders),vid_paper)
        res[name]=rows
        print(f"\n{name}  N={N} orders={len(orders)}  total {rows['ALL']['eff']:+.1f} [{rows['ALL']['lo']:+.1f},{rows['ALL']['hi']:+.1f}]"
              f"  (paper clustering CI [{lo2:+.1f},{hi2:+.1f}])")
        print(f"  {'dir':6}{'n':>5}{'share':>7}{'A0':>7}{'cond.eff':>10}{'95% CI':>18}{'contrib':>9}")
        for c in ("FN","FP","TP","TN","FALSE","TRUE"):
            r=rows.get(c); 
            if r: print(f"  {c:6}{r['n']:5d}{r['share']*100:6.1f}%{r['A0']:7.1f}{r['eff']:+10.1f}   [{r['lo']:+6.1f},{r['hi']:+6.1f}]{r['contrib']:+9.2f}")
    return res
if __name__=="__main__":
    allres={d:run(d) for d in ("x","2x")}
    json.dump(allres,open(os.path.join(os.path.dirname(__file__),"direction_split_results.json"),"w"),indent=1)
