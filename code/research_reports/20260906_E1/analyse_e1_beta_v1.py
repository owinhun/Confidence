#!/usr/bin/env python3
"""E1 secondary (prereg): claim-aligned contrast swing beta_high - beta_low per field on Qwen2-Audio, with
beta_c = 0.5*(m(T_c) - m(F_c)) on unaided-correct items (m0 cancels), order-averaged like Table 3 (beta_C / beta_P)."""
import sys, os, json, re
sys.path.insert(0, "research_reports/20260906_E1")
from analyse_e1_v1 import L, marg, boot_orders, vid
PFX="nm_qwen2audio_sound"
FIELDS=[("Confidence",("anon","true")),("Priority",("priority","priority_true")),
        ("Claim conf.",("claimconf","claimconf_T")),("Detector conf.",("detconf","detconf_T")),("Reader conf.",("readerconf","readerconf_T"))]
def files(key,lvl,tag):
    # false-claim file and true-claim file naming: anon_high / true_high ; priority_high / priority_true_high ; claimconf_high / claimconf_high_T
    fk,tk=key
    F=L(f"{PFX}_{fk}_{lvl}{tag}")
    if tk.endswith("_T"): T=L(f"{PFX}_{tk[:-2]}_{lvl}_T{tag}")
    else: T=L(f"{PFX}_{tk}_{lvl}{tag}") or L(f"{PFX}_{tk}_{lvl}_T{tag}")
    return F,T
rows={}
for label,key in FIELDS:
    dicts=[]
    for tag in ("","_swap"):
        b=L(f"{PFX}_none{tag}"); sw=(tag=="_swap")
        Fh,Th=files(key,"high",tag); Fl,Tl=files(key,"low",tag)
        if None in (b,Fh,Th,Fl,Tl): continue
        d={k:0.5*((marg(Th[k],sw)-marg(Fh[k],sw))-(marg(Tl[k],sw)-marg(Fl[k],sw))) for k in b if b[k]["correct"] and all(k in x for x in (Fh,Th,Fl,Tl))}
        dicts.append(d)
    if dicts: rows[label]=(boot_orders(dicts),sum(len(d) for d in dicts),len(dicts))
    else: rows[label]=None
print("Qwen2-Audio/sound  beta swing (high-low), unaided-correct, order-averaged")
for label,_ in FIELDS:
    r=rows[label]
    print(f"  {label:15} "+("--" if r is None else f"{r[0][0]:+.2f} [{r[0][1]:+.2f},{r[0][2]:+.2f}]  n={r[1]} orders={r[2]}"))
