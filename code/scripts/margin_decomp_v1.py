#!/usr/bin/env python3
"""Identify the two components on a truth-aligned logit margin instead of on accuracy.

The accuracy-difference version in the manuscript is not identified: A0 - T mixes a
direction-free disruption with whatever helpful pull a *true* claim exerts, and accuracy is
bounded so differences are not additive near the ceiling. Every run stored p_A, so the margin
can be reconstructed with no new inference.

For an item with ground-truth answer y in {A, B} define the truth-aligned margin

    m = logit p(correct option)  =  log p_A - log p_B   if y = A,   negated if y = B

and, for a condition with declared confidence c and claim polarity r (+1 true, -1 false),

    delta(c, r) = m(c, r) - m0

Fitting  delta = alpha_c + beta_c * r  gives

    alpha_c = ( E[delta | r=+1] + E[delta | r=-1] ) / 2     common to both polarities
    beta_c  = ( E[delta | r=+1] - E[delta | r=-1] ) / 2     claim-aligned

alpha is the component that does not depend on which way the claim points; beta is the one
that does. Unlike the accuracy version these are estimated jointly from both polarities, so
neither absorbs the other. Confidence intervals are bootstrapped over source videos.
"""

from __future__ import annotations

import json
import os

import numpy as np

OUT = "outputs"
LAD = ["v_vlow", "v_low", "v_mod", "v_high", "v_vhigh"]
MODELS = [("s2", "cs_qwen25", "Qwen2.5-Omni"),
          ("s2q3", "cs_qwen3", "Qwen3-Omni"),
          ("s2vl", "cs_vllama", "VideoLLaMA2.1"),
          ("nm_qwen2audio", "nm_qwen2audio", "Qwen2-Audio"),
          ("nm_af3", "nm_af3", "Audio-Flamingo-3")]
EPS = 1e-6


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
    """log-odds of the correct option."""
    pa = min(max(rec["p_A"], EPS), 1 - EPS)
    lo = np.log(pa) - np.log(1 - pa)          # log-odds of answering A
    return lo if rec["label"] == "A" else -lo


def video(clip):
    return clip.split("/")[-1].split("_")[0]


def decompose(base_pfx, lad_pfx, boots=2000, seed=0):
    b = L(f"{base_pfx}_sound_none")
    if b is None:
        return None
    rows = []
    for c in LAD:
        t, f = L(f"{lad_pfx}_sound_{c}_T"), L(f"{lad_pfx}_sound_{c}")
        if t is None or f is None:
            return None
        for clip in b:
            if clip in t and clip in f:
                m0 = margin(b[clip])
                rows.append((video(clip), c,
                             margin(t[clip]) - m0, margin(f[clip]) - m0))
    if not rows:
        return None
    vids = sorted({r[0] for r in rows})
    rng = np.random.default_rng(seed)

    def fit(sel):
        out = {}
        for c in LAD:
            d = [(dt, df) for v, cc, dt, df in rows if cc == c and v in sel]
            if not d:
                return None
            dt = np.mean([x[0] for x in d])
            df = np.mean([x[1] for x in d])
            out[c] = ((dt + df) / 2, (dt - df) / 2)   # alpha, beta
        return out

    point = fit(set(vids))
    boot = []
    for _ in range(boots):
        s = set(rng.choice(vids, len(vids), replace=True))
        r = fit(s)
        if r:
            boot.append(r)
    ci = {}
    for c in LAD:
        a = np.array([x[c][0] for x in boot])
        bb = np.array([x[c][1] for x in boot])
        ci[c] = ((np.percentile(a, 2.5), np.percentile(a, 97.5)),
                 (np.percentile(bb, 2.5), np.percentile(bb, 97.5)))
    return point, ci, len({r[0] for r in rows})


if __name__ == "__main__":
    print("Truth-aligned margin decomposition (log-odds of the correct option).")
    print("alpha = component common to true and false claims (direction-free)")
    print("beta  = claim-aligned component\n")
    for base, lad, name in MODELS:
        res = decompose(base, lad)
        if res is None:
            print(f"{name}: ladder incomplete\n")
            continue
        point, ci, nv = res
        print(f"{name}  ({nv} videos)")
        print(f"  {'level':9}{'alpha':>9}{'95% CI':>20}{'beta':>9}{'95% CI':>20}")
        for c in LAD:
            a, b_ = point[c]
            (al, ah), (bl, bh) = ci[c]
            print(f"  {c:9}{a:9.3f}{f'[{al:.2f}, {ah:.2f}]':>20}"
                  f"{b_:9.3f}{f'[{bl:.2f}, {bh:.2f}]':>20}")
        print()
