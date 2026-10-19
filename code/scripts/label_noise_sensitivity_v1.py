#!/usr/bin/env python3
"""Does the effect survive dropping the sound-event classes whose labels look noisiest?

The paper reports that an independent CLAP check separates nine of twelve query classes well but
only weakly for the two eating classes and sneezing. That is honest but incomplete: a reviewer
will ask whether the headline contrast depends on those clips. This recomputes the claim-aligned
ladder on the nine-class subset.

The check has a known limitation worth stating rather than hiding. CLAP is also the detector in
Sec. 7, so using it to flag suspect labels is not an independent audit -- a class where CLAP
separates poorly may have noisy labels or may simply be a class CLAP is bad at. The subset
result therefore bounds how much the conclusion could depend on those clips; it does not
establish that the labels are wrong.
"""

from __future__ import annotations

import json
import os

import numpy as np

EPS = 1e-6
WEAK = {"people eating apple", "people eating crisps", "people sneezing"}
QL = ["v_vlow", "v_low", "v_mod", "v_high", "v_vhigh"]
NL = [("v_vlow", "v_vlow_T"), ("anon_low", "true_low"), ("v_mod", "v_mod_T"),
      ("anon_high", "true_high_T"), ("v_vhigh", "v_vhigh_T")]
SER = [("Qwen2.5", "s2", "cs_qwen25", None), ("Qwen3", "s2q3", "cs_qwen3", None),
       ("Qwen2-Audio", "nm_qwen2audio", "nm_qwen2audio", None),
       ("AF3", "nm_af3", "nm_af3", None),
       ("Nemotron", "nm_nemotron", "nm_nemotron", NL)]

CLS = {json.loads(l)["clip"]: json.loads(l)["query_class"]
       for l in open("data/held_attr_soundevent_v1.jsonl") if l.strip()}


def L(n):
    p = f"outputs/{n}_v1.jsonl"
    if not os.path.exists(p):
        return None
    d = {}
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            d[r["clip"]] = r
    return d if len(d) >= 400 else None


def marg(r):
    pa = min(max(r["p_A"], EPS), 1 - EPS)
    return (np.log(pa) - np.log(1 - pa)) * (1 if r["label"] == "A" else -1)


if __name__ == "__main__":
    n_drop = sum(1 for c in CLS.values() if c in WEAK)
    print(f"약한 3클래스 클립 {n_drop}/{len(CLS)} 제외 후 재계산\n")
    print(f"{'cell':13}{'subset':9}" + "".join(f"{l:>9}" for l in
                                               ("v.low", "low", "mod", "high", "v.high"))
          + "   단조?")
    for name, pfx, lad, nl in SER:
        b = L(f"{pfx}_sound_none")
        if b is None:
            continue
        for keep_all in (True, False):
            C = [k for k in b if b[k]["correct"]
                 and (keep_all or CLS.get(k) not in WEAK)]
            ys = []
            for i in range(5):
                if nl:
                    fs, ts = f"{pfx}_sound_{nl[i][0]}", f"{pfx}_sound_{nl[i][1]}"
                else:
                    fs, ts = f"{lad}_sound_{QL[i]}", f"{lad}_sound_{QL[i]}_T"
                F, T = L(fs), L(ts)
                if F is None or T is None:
                    ys.append(None)
                    continue
                ks = [k for k in C if k in F and k in T]
                ys.append(float(np.mean([0.5 * (marg(T[k]) - marg(F[k])) for k in ks]))
                          if ks else None)
            good = [y for y in ys if y is not None]
            mono = all(a <= b_ for a, b_ in zip(good, good[1:])) if len(good) > 1 else None
            lab = "전체" if keep_all else "9클래스"
            print(f"{name:13}{lab:9}"
                  + "".join(f"{y:+9.2f}" if y is not None else f"{'--':>9}" for y in ys)
                  + f"   {'예' if mono else '아니오'}  n={len(C)}")
        print()
