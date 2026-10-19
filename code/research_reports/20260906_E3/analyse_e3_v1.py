#!/usr/bin/env python3
"""E3 (preregistered 2026-09-06 13:19 KST): swapped-order 5-rung ladder on Qwen2-Audio/sound and
Nemotron/sound. For each order: beta per rung on the complement (unaided-INCORRECT in that order)
and on unaided-correct items; endpoint swing beta(vhigh)-beta(vlow); linear trend slope over rung
index 0..4; swap-minus-first difference of the swing. Cluster bootstrap over 11-char source ids.
Reads outputs/ only."""
import json, os, re, sys
import numpy as np
OUT="outputs"; EPS=1e-6; BOOTS=2000
LAD=["v_vlow","v_low","v_mod","v_high","v_vhigh"]
NEMO=[("v_vlow","v_vlow_T"),("anon_low","true_low"),("v_mod","v_mod_T"),("anon_high","true_high"),("v_vhigh","v_vhigh_T")]
CELLS=[("Qwen2-Audio/sound","nm_qwen2audio",[(c,c+"_T") for c in LAD]),("Nemotron/sound","nm_nemotron",NEMO)]
def L(name):
    p=f"{OUT}/{name}_v1.jsonl"
    if not os.path.exists(p): return None
    d={json.loads(l)["clip"]:json.loads(l) for l in open(p) if l.strip()}
    return d if len(d)>=400 else None
def LT(base,name,tag):   # true-claim file; Nemotron's true_high exists only as true_high_T in the first order
    return L(f"{base}_sound_{name}{tag}") or L(f"{base}_sound_{name}_T{tag}")
def marg(r,swapped=False):
    # p_A is the probability of option LETTER A; in the swapped order letter A means "No", so the
    # truth-aligned margin flips sign (same convention as gen_all_numbers_v1.marg)
    pa=min(max(r["p_A"],EPS),1-EPS); lo=np.log(pa)-np.log(1-pa); v=lo if r["label"]=="A" else -lo
    return -v if swapped else v
def vid(c):
    b=os.path.basename(c); m=re.match(r"^(.{11})_\d{6}",b); return m.group(1) if m else b
def boot_stat(items, fn):
    """items: list of (vid, payload); fn(list of payload)->scalar. cluster bootstrap CI."""
    vids=sorted({v for v,_ in items}); by={v:[p for vv,p in items if vv==v] for v in vids}
    pt=fn([p for _,p in items]); rng=np.random.default_rng(0); bs=[]
    for _ in range(BOOTS):
        sel=rng.integers(0,len(vids),len(vids)); pool=[p for i in sel for p in by[vids[i]]]
        bs.append(fn(pool))
    return pt,float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))
def beta_rows(base,pairs,tag):
    b=L(f"{base}_sound_none{tag}")
    if b is None: return None
    rows=[]   # (vid, clip, correct0, [beta_r for r in 0..4] or None)
    sw=(tag=="_swap")
    F=[L(f"{base}_sound_{f}{tag}") for f,_ in pairs]; T=[LT(base,t,tag) for _,t in pairs]
    have=[f is not None and t is not None for f,t in zip(F,T)]
    for k in b:
        m0=marg(b[k],sw); br=[]
        for i in range(5):
            if have[i] and k in F[i] and k in T[i]: br.append(0.5*((marg(T[i][k],sw)-m0)-(marg(F[i][k],sw)-m0)))
            else: br.append(None)
        rows.append((vid(k),k,b[k]["correct"],br))
    return rows,have
def summarise(rows,have,subset):
    R=[r for r in rows if (r[2] if subset=="correct" else not r[2])]
    out={"n":len(R)}
    for i,c in enumerate(LAD):
        if have[i]:
            it=[(r[0],r[3][i]) for r in R if r[3][i] is not None]
            out[c]=boot_stat(it,lambda xs: float(np.mean(xs)))
    if have[0] and have[4]:
        it=[(r[0],(r[3][4],r[3][0])) for r in R if r[3][4] is not None and r[3][0] is not None]
        out["swing_top_bottom"]=boot_stat(it,lambda xs: float(np.mean([a-b for a,b in xs])))
    if all(have):
        it=[(r[0],r[3]) for r in R if all(x is not None for x in r[3])]
        def slope(xs):
            Y=np.array(xs); x=np.arange(5)-2.0; return float(np.mean(Y,axis=0)@x/(x@x))
        out["trend_slope"]=boot_stat(it,slope)
    return out
def fmt(t): return "--" if t is None else f"{t[0]:+.2f} [{t[1]:+.2f},{t[2]:+.2f}]"
if __name__=="__main__":
    res={}
    for name,base,pairs in CELLS:
        print(f"\n=== {name} ===")
        for tag in ("","_swap"):
            got=beta_rows(base,pairs,tag)
            if got is None: print(f" order{tag or '_first'}: no baseline"); continue
            rows,have=got; print(f" order{tag or '_first'}: rungs available {['Y' if h else '-' for h in have]}")
            for subset in ("incorrect","correct"):
                o=summarise(rows,have,subset); res[(name,tag,subset)]=o
                print(f"  {subset:9} n={o['n']:3d} "+" ".join(f"{c}:{fmt(o.get(c))}" for c in LAD if c in o))
                print(f"            swing(top-bottom) {fmt(o.get('swing_top_bottom'))}   trend slope/rung {fmt(o.get('trend_slope'))}")
        # secondary: swap - first swing on the complement, if both exist
        a=res.get((name,"","incorrect"),{}).get("swing_top_bottom"); b=res.get((name,"_swap","incorrect"),{}).get("swing_top_bottom")
        if a and b: print(f"  complement swing: first {a[0]:+.2f}  swap {b[0]:+.2f}  diff {b[0]-a[0]:+.2f} (equivalence margin ±0.5)")
    # --- macros for the manuscript, only once both cells have every swapped rung
    M={}; short={"Qwen2-Audio/sound":"QTwoA","Nemotron/sound":"Nemo"}
    ok=all(("swing_top_bottom" in res.get((n,"_swap","incorrect"),{})) and ("trend_slope" in res.get((n,"_swap","incorrect"),{})) for n,_,_ in CELLS)
    if ok and "--macros" in sys.argv:
        for n,_,_ in CELLS:
            k=short[n]; c=res[(n,"_swap","incorrect")]; u=res[(n,"_swap","correct")]; f=res[(n,"","incorrect")]
            M[f"EThreeSwing{k}"]=f"{c['swing_top_bottom'][0]:+.2f}"; M[f"EThreeSwing{k}Lo"]=f"{c['swing_top_bottom'][1]:+.2f}"; M[f"EThreeSwing{k}Hi"]=f"{c['swing_top_bottom'][2]:+.2f}"
            M[f"EThreeSlope{k}"]=f"{c['trend_slope'][0]:+.2f}"; M[f"EThreeSlope{k}Lo"]=f"{c['trend_slope'][1]:+.2f}"; M[f"EThreeSlope{k}Hi"]=f"{c['trend_slope'][2]:+.2f}"
            M[f"EThreeN{k}"]=str(c["n"]); M[f"EThreeSwingCorr{k}"]=f"{u['swing_top_bottom'][0]:+.2f}"
            M[f"EThreeDiff{k}"]=f"{c['swing_top_bottom'][0]-f['swing_top_bottom'][0]:+.2f}"
        with open("paper/auto_e3.tex","w") as fh:
            fh.write("% Generated by research_reports/20260906_E3/analyse_e3_v1.py --macros -- do not edit by hand.\n")
            for k,v in M.items(): fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
        print("wrote paper/auto_e3.tex", len(M), "macros")
    elif "--macros" in sys.argv: print("macros NOT written: swapped rungs incomplete")
