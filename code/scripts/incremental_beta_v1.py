#!/usr/bin/env python3
"""How much of the claim-aligned contrast belongs to DECLARING CONFIDENCE?

beta is the paper's identified result, but until now it was measured against a signal-only
baseline, so it carried everything the block and the claim do as well. The new
priority_true_{high,low} runs close that: a non-epistemic field can now be given a true claim,
so the same claim-aligned contrast can be formed for a field that declares no confidence.

    beta_conf(c)  = 1/2 [ m(Confidence: c, true) - m(Confidence: c, false) ]
    beta_prio(c)  = 1/2 [ m(Priority: c,   true) - m(Priority: c,   false) ]
    delta_beta(c) = beta_conf(c) - beta_prio(c)

delta_beta is the part of the claim-aligned contrast that survives holding the block, the
source, the claim and a filled third field fixed. If it were near zero, the ladder would be
measuring "a field with a graded word in it" rather than declared confidence.

Both terms are differences between two conditions of identical structure, so a multiplicative
flattening of the output distribution cancels in each of them separately and again in their
difference.
"""

from __future__ import annotations

import json
import os

import numpy as np

OUT = "outputs"
# model -> (stage2 prefix, ladder prefix, attribute). VideoLLaMA sits on colour because its
# sound cell was withdrawn when the option swap put unaided accuracy at chance.
MODELS = [("Qwen2.5-Omni", "s2", "cs_qwen25", "sound"),
          ("Qwen3-Omni", "s2q3", "cs_qwen3", "sound"),
          ("VideoLLaMA2.1", "s2vl", "cs_vllama", "colour"),
          ("Qwen2-Audio", "nm_qwen2audio", "nm_qwen2audio", "sound"),
          ("Audio-Flamingo-3", "nm_af3", "nm_af3", "sound")]
LEVELS = [("high", "v_high"), ("low", "v_low")]
EPS = 1e-6
BOOTS = 1500


def L(name):
    p = f"{OUT}/{name}_v1.jsonl"
    if not os.path.exists(p):
        return None
    d = {}
    for line in open(p).read().splitlines():
        if line.strip():
            r = json.loads(line)
            d[r["clip"]] = r
    return d if len(d) >= 400 else None


def margin(rec):
    pa = min(max(rec["p_A"], EPS), 1 - EPS)
    lo = np.log(pa) - np.log(1 - pa)
    return lo if rec["label"] == "A" else -lo


def video(clip):
    return clip.split("/")[-1].split("_")[0]


def betas(base, lad, attr, lvl, ladlvl):
    """beta_conf, beta_prio and their difference at one declared level."""
    b = L(f"{base}_{attr}_none")
    cT, cF = L(f"{lad}_{attr}_{ladlvl}_T"), L(f"{lad}_{attr}_{ladlvl}")
    pT, pF = L(f"{base}_{attr}_priority_true_{lvl}"), L(f"{base}_{attr}_priority_{lvl}")
    if any(x is None for x in (b, cT, cF, pT, pF)):
        return None
    keys = [k for k in b if b[k]["correct"] and all(k in d for d in (cT, cF, pT, pF))]
    if len(keys) < 40:
        return None
    per = {k: (0.5 * (margin(cT[k]) - margin(cF[k])),
               0.5 * (margin(pT[k]) - margin(pF[k]))) for k in keys}
    vids = sorted({video(k) for k in keys})
    rng = np.random.default_rng(0)

    def stat(sel):
        kk = [k for k in keys if video(k) in sel]
        if not kk:
            return None
        bc = float(np.mean([per[k][0] for k in kk]))
        bp = float(np.mean([per[k][1] for k in kk]))
        return bc, bp, bc - bp

    pt = stat(set(vids))
    bs = [s for s in (stat(set(rng.choice(vids, len(vids), replace=True)))
                      for _ in range(BOOTS)) if s]
    lo = float(np.percentile([x[2] for x in bs], 2.5))
    hi = float(np.percentile([x[2] for x in bs], 97.5))
    return pt[0], pt[1], pt[2], lo, hi, len(keys)


if __name__ == "__main__":
    print("Claim-aligned contrast attributable to the confidence field specifically.")
    print("beta_conf uses Confidence: c; beta_prio uses Priority: c with the same claim.\n")
    for lvl, ladlvl in LEVELS:
        print(f"=== declared level = {lvl} ===")
        print(f"{'model':18}{'beta_conf':>11}{'beta_prio':>11}"
              f"{'delta_beta':>12}{'95% CI':>20}{'n':>6}")
        for name, base, lad, attr in MODELS:
            r = betas(base, lad, attr, lvl, ladlvl)
            if r is None:
                print(f"{name:18}{'-- pending run --':>11}")
                continue
            bc, bp, db, lo, hi, n = r
            share = "" if abs(bc) < 1e-9 else f"  ({100 * db / bc:.0f}% of beta_conf)"
            print(f"{name:18}{bc:11.3f}{bp:11.3f}{db:12.3f}"
                  f"{f'[{lo:.2f}, {hi:.2f}]':>20}{n:>6}{share}")
        print()
