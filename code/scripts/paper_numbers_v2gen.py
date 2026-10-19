#!/usr/bin/env python3
"""Master numbers for manuscript v2. Differs from v1 in three ways: cells that fail the
option-order control are withdrawn, cells whose swapped read-out is the sound one are quoted
from the swap, and the robustness battery is tabulated."""
import json, os, numpy as np
from math import comb
OUT="outputs"
def L(n):
    p=f"{OUT}/{n}_v1.jsonl"
    return {json.loads(l)['clip']: json.loads(l) for l in open(p).read().splitlines() if l.strip()} if os.path.exists(p) else None
MOD=[('s2','Qwen2.5-Omni'),('s2q3','Qwen3-Omni'),('s2vl','VideoLLaMA2.1'),
     ('nm_qwen2audio','Qwen2-Audio'),('nm_af3','Audio-Flamingo-3'),('nm_phi4mm','Phi-4-mm')]
# cells withdrawn after the option-order control, and cells to quote from the swapped run
WITHDRAWN={('s2vl','sound')}
QUOTE_SWAP={('s2vl','colour')}
def stat(b):
    A=[i for i in b if b[i]['label']=='A']; B=[i for i in b if b[i]['label']=='B']
    return (np.mean([b[i]['correct'] for i in b])*100,
            np.mean([b[i]['correct'] for i in A])*100,
            np.mean([b[i]['correct'] for i in B])*100)
def macro(b,w,ok):
    sub=[i for i in ok if i in w]; ad=np.array([w[i]['pred']!=w[i]['label'] for i in sub])
    a=[k for k,i in enumerate(sub) if w[i]['label']=='A']; bb=[k for k,i in enumerate(sub) if w[i]['label']=='B']
    return (ad[a].mean()+ad[bb].mean())/2*100, dict(zip(sub,ad))
def mcn(d1,d2):
    com=sorted(set(d1)&set(d2))
    n01=sum(1 for i in com if d1[i] and not d2[i]); n10=sum(1 for i in com if not d1[i] and d2[i]); m=n01+n10
    return n01,n10,(min(1.,sum(comb(m,k) for k in range(min(n01,n10)+1))*2/2**m) if m else 1.)

print("== 1. ELIGIBILITY / WITHDRAWAL ==")
for pfx,nm in MOD:
    for a in ['sound','colour','noise','lowpass','reverb']:
        b=L(f'{pfx}_{a}_none')
        if not b or len(b)<400: continue
        a0,rA,rB=stat(b)
        v='not evaluable' if (a0<60 or rA<60 or rB<60) else ('WITHDRAWN (option order)' if (pfx,a) in WITHDRAWN else ('eligible (quote swap)' if (pfx,a) in QUOTE_SWAP else 'eligible'))
        print(f"{nm}|{a}|A0={a0:.1f}|rec={rA:.0f}/{rB:.0f}|{v}")

print("\n== 2. CONFIDENCE CONTRASTS (v2 set) ==")
for pfx,nm in MOD:
    for a in ['sound','colour','noise']:
        if (pfx,a) in WITHDRAWN: continue
        sw='_swap' if (pfx,a) in QUOTE_SWAP else ''
        b=L(f'{pfx}_{a}_none{sw}')
        if not b or len(b)<400: continue
        a0,rA,rB=stat(b)
        if a0<60 or rA<60 or rB<60: continue
        ok=[i for i in b if b[i]['correct']]
        for s in ['anon','expert']:
            hi=L(f'{pfx}_{a}_{s}_high{sw}'); lo=L(f'{pfx}_{a}_{s}_low{sw}')
            if not hi or not lo or len(hi)<400 or len(lo)<400: continue
            mh,dh=macro(b,hi,ok); ml,dl=macro(b,lo,ok); n01,n10,p=mcn(dh,dl)
            print(f"{nm}|{a}|{s}|hi={mh:.1f}|lo={ml:.1f}|disc={n01}/{n10}|p={p:.1e}|read={'swap' if sw else 'orig'}")

print("\n== 3. ROBUSTNESS BATTERY (sound) ==")
for pfx,nm in MOD:
    b0=L(f'{pfx}_sound_none')
    if not b0 or len(b0)<400: continue
    row=[]
    for tag,lab in [('','orig'),('_swap','swap'),('_post','post')]:
        b=L(f'{pfx}_sound_none{tag}')
        if not b or len(b)<400: row.append(f"{lab}:-"); continue
        ok=[i for i in b if b[i]['correct']]
        hi=L(f'{pfx}_sound_anon_high{tag}'); lo=L(f'{pfx}_sound_anon_low{tag}')
        if not hi or not lo or len(hi)<400 or len(lo)<400: row.append(f"{lab}:-"); continue
        mh,dh=macro(b,hi,ok); ml,dl=macro(b,lo,ok); n01,n10,p=mcn(dh,dl)
        row.append(f"{lab}: A0={np.mean([r['correct'] for r in b.values()])*100:.1f} hi={mh:.1f} lo={ml:.1f} {n01}/{n10}")
    print(f"{nm}|"+" | ".join(row))
    ok=[i for i in b0 if b0[i]['correct']]
    extra=[]
    for c in ['hedge_plain','reliab_high','reliab_low','certain_high','certain_low']:
        w=L(f'{pfx}_sound_{c}')
        if w and len(w)>=400:
            m,_=macro(b0,w,ok); extra.append(f"{c}={m:.1f}")
    if extra: print(f"   {nm} 조건별 macro-POR: "+"  ".join(extra))

print("\n== 4. CROSS-PROPOSITION ==")
for pfx,nm in MOD:
    b=L(f'{pfx}_aux_noise_none')
    if not b or len(b)<400: continue
    a0=np.mean([r['correct'] for r in b.values()])*100
    def A(c,attr='aux_noise'):
        w=L(f'{pfx}_{attr}_{c}')
        return np.mean([r['correct'] for r in w.values()])*100 if w and len(w)>=400 else None
    sb=L(f'{pfx}_sound_none'); sa0=np.mean([r['correct'] for r in sb.values()])*100
    same=A('anon_low','sound'); irr=A('irrelevant_low','sound'); other=A('anon_low')
    f=lambda x: f"{x:.1f}" if x is not None else "-"
    print(f"{nm}|aux_A0={a0:.1f}|other_prop_cost={f(a0-other) if other else '-'}|"
          f"same_prop_cost={f(sa0-same) if same else '-'}|irrelevant_cost={f(sa0-irr) if irr else '-'}")
