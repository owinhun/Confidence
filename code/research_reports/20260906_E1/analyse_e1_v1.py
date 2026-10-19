#!/usr/bin/env python3
"""E1 (preregistered 2026-09-06 18:20 KST). Per cell: false-claim block swing S_F = m(F_high) - m(F_low) on
unaided-correct items, order-averaged where both orders exist, for Confidence (anon_), Priority (priority_),
Claim confidence, Detector confidence, Reader confidence. Paired differences with 11-char source-video
cluster bootstrap CIs. Equivalence margin +-0.3 log-odds (prereg). Reads outputs/ only."""
import json, math, os, re, sys
import numpy as np
OUT="outputs"; EPS=1e-6; BOOTS=2000; MARGIN=0.3
CELLS=[("Qwen2-Audio/sound","nm_qwen2audio_sound"),("Qwen2.5-Omni/sound","s2_sound"),("Nemotron/sound","nm_nemotron_sound"),("Qwen3-Omni/sound","s2q3_sound")]
FIELDS=[("Confidence","anon"),("Priority","priority"),("Claim conf.","claimconf"),("Detector conf.","detconf"),("Reader conf.","readerconf")]
def L(name):
    p=f"{OUT}/{name}_v1.jsonl"
    if not os.path.exists(p): return None
    d={json.loads(l)["clip"]:json.loads(l) for l in open(p) if l.strip()}
    return d if len(d)>=400 else None
def marg(r,sw):
    pa=min(max(r["p_A"],EPS),1-EPS); lo=np.log(pa)-np.log(1-pa); v=lo if r["label"]=="A" else -lo
    return -v if sw else v
def vid(c):
    b=os.path.basename(c); m=re.match(r"^(.{11})_\d{6}",b); return m.group(1) if m else b
def swings(pfx):
    """Paper-style order averaging: within each order, the mean over clips unaided-correct in THAT order;
    then the mean of the order means. Returns field -> ((pt,lo,hi), n_first+n_swap, per-order clip dicts)."""
    per={}; have={}
    for tag in ("","_swap"):
        b=L(f"{pfx}_none{tag}")
        if b is None: continue
        sw=(tag=="_swap")
        for label,key in FIELDS:
            H,Lo=L(f"{pfx}_{key}_high{tag}"),L(f"{pfx}_{key}_low{tag}")
            if H is None or Lo is None: continue
            have.setdefault(label,set()).add(tag or "_first")
            per.setdefault(label,{})[tag]={k:marg(H[k],sw)-marg(Lo[k],sw) for k in b if b[k]["correct"] and k in H and k in Lo}
    out={}
    for label,orders in per.items():
        out[label]=(boot_orders([orders[t] for t in orders]),sum(len(v) for v in orders.values()),orders)
    return out,have
def boot_orders(dicts):
    """mean of per-order means; cluster bootstrap over source videos shared across orders."""
    vids=sorted({vid(k) for d in dicts for k in d}); vi={v:i for i,v in enumerate(vids)}
    per=[]
    for d in dicts:
        ks=sorted(d); vals=np.array([d[k] for k in ks]); g=np.array([vi[vid(k)] for k in ks])
        per.append((vals,[np.where(g==i)[0] for i in range(len(vids))]))
    pt=float(np.mean([v.mean() for v,_ in per])); rng=np.random.default_rng(0); bs=[]
    for _ in range(BOOTS):
        sel=rng.integers(0,len(vids),len(vids)); ms=[]
        for v,b in per:
            idx=np.concatenate([b[i] for i in sel if b[i].size]) if any(b[i].size for i in sel) else np.array([],int)
            if idx.size: ms.append(v[idx].mean())
        bs.append(float(np.mean(ms)))
    return pt,float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))
def paired(oA,oB):
    dicts=[]; n=0
    for t in oA:
        if t in oB:
            ks=set(oA[t])&set(oB[t]); dicts.append({k:oA[t][k]-oB[t][k] for k in ks}); n+=len(ks)
    return boot_orders(dicts),n
def f(t): return f"{t[0]:+.2f} [{t[1]:+.2f},{t[2]:+.2f}]"
if __name__=="__main__":
    ALL={}
    for name,pfx in CELLS:
        res,have=swings(pfx); ALL[name]=res
        print(f"\n=== {name} ===  orders: "+", ".join(f"{k}:{sorted(v)}" for k,v in have.items()))
        for label,_ in FIELDS:
            if label in res: print(f"  S_F {label:15} {f(res[label][0])}  n={res[label][1]}")
        need=["Claim conf.","Detector conf.","Reader conf."]
        if all(x in res for x in need) and "Priority" in res and "Confidence" in res:
            d=lambda a,b: paired(res[a][2],res[b][2])
            rd,n=d("Reader conf.","Detector conf."); cd,_=d("Claim conf.","Detector conf."); rp,_=d("Reader conf.","Priority"); cc,_=d("Claim conf.","Confidence"); dc,_=d("Detector conf.","Confidence")
            print(f"  paired: reader-detector {f(rd)} (n={n}) | claim-detector {f(cd)} | reader-priority {f(rp)} | claim-conf {f(cc)} | detector-conf {f(dc)}")
            a = rd[1]>-MARGIN and rd[2]<MARGIN
            b = res["Reader conf."][0][2] <= res["Priority"][0][0] and abs(cc[0])<MARGIN and abs(dc[0])<MARGIN
            print("  prereg grid:", "(a) reader~detector: instruction reading NOT excluded" if a else ("(b) reader at control level, claim/detector track Confidence: upstream reading strengthened" if b else "(c) mixed"))
        else:
            print("  (incomplete: waiting for "+", ".join(x for x in need+["Priority","Confidence"] if x not in res)+")")

    if "--macros" in sys.argv:
        need=["Reader conf.","Detector conf.","Priority","Confidence"]
        cells=[n for n,r in ALL.items() if all(x in r for x in need)]
        # application condition (prereg note 23:20 KST): both relabelled fields' CI upper bound (i.e. less negative end) below Priority point
        ok=all(ALL[n][f][0][2] < ALL[n]["Priority"][0][0] for n in cells for f in ("Reader conf.","Detector conf."))
        rel=[-ALL[n][f][0][0] for n in cells for f in ("Reader conf.","Detector conf.")]
        pri=[-ALL[n]["Priority"][0][0] for n in cells]
        shrink=[ALL[n]["Confidence"][0][0]-ALL[n][f][0][0] for n in cells for f in ("Reader conf.","Detector conf.")]  # negative-negative: Confidence more negative => value negative
        M={"EOneCells":str(len(cells)),"EOneRelMin":f"{min(rel):.2f}","EOneRelMax":f"{max(rel):.2f}","EOnePrioMin":f"{min(pri):.2f}","EOnePrioMax":f"{max(pri):.2f}",
           # a range endpoint must not round toward the claim: the smallest shrink is 0.26, and
           # printing 0.3 would assert every relabelling costs at least 0.3. Floor the low end.
           "EOneShrinkMin":f"{math.floor(min(-x for x in shrink)*10)/10:.1f}",
           "EOneShrinkMax":f"{math.ceil(max(-x for x in shrink)*10)/10:.1f}","EOneBeatsAll":"yes" if ok else "no"}
        with open("paper/auto_e1.tex","w") as fh:
            fh.write("% Generated by research_reports/20260906_E1/analyse_e1_v1.py --macros -- do not edit by hand.\n")
            for k,v in M.items(): fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
        print("macros:",M)
