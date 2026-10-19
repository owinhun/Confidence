#!/usr/bin/env python3
"""Evaluate interface policies with numbers, instead of asserting that none worked.

Both reviews flagged the same sentence: "no confidence-translation policy we tried beat simply
omitting uncertain metadata" is unverifiable without a table. Everything needed is already on
disk. For each item we have the truth-aligned margin with no metadata (m0), with a TRUE claim
at each declared confidence, and with a FALSE claim at each level. A detector whose claims are
correct with probability pi produces the true branch with probability pi and the false branch
otherwise, so any policy that is a function of (m0, m_with, declared c) can be scored exactly
by mixing the two branches -- no new inference, and no simulation of the model itself.

Policies compared, all deployable:
  omit           ignore the metadata entirely (answer from the signal)
  passthrough    forward the claim at the detector's own confidence
  always-high    forward it as "very high" (an aggressive constant)
  always-low     forward it as "very low" (a conservative constant)
  threshold(t)   forward at high if pi >= t, otherwise omit
  two-pass       run with and without the block; if they disagree AND the declared confidence
                 is below "high", keep the signal-only answer
  oracle         know whether the claim is true (upper bound, not deployable)
"""

from __future__ import annotations

import json
import os

import numpy as np

OUT = "outputs"
LAD = ["v_vlow", "v_low", "v_mod", "v_high", "v_vhigh"]
MODELS = [("s2", "cs_qwen25", "Qwen2.5-Omni"),
          ("s2q3", "cs_qwen3", "Qwen3-Omni"),
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


def corr(rec):
    return 1.0 if rec["correct"] else 0.0


def build(base_pfx, lad_pfx):
    b = L(f"{base_pfx}_sound_none")
    if b is None:
        return None
    T, F = {}, {}
    for c in LAD:
        t, f = L(f"{lad_pfx}_sound_{c}_T"), L(f"{lad_pfx}_sound_{c}")
        if t is None or f is None:
            return None
        T[c], F[c] = t, f
    clips = [k for k in b if all(k in T[c] and k in F[c] for c in LAD)]
    return b, T, F, clips


def score(b, T, F, clips, policy, pi):
    """Expected accuracy of a policy when the claim is true with probability pi."""
    tot = 0.0
    for k in clips:
        c0 = corr(b[k])
        if policy == "omit":
            tot += c0
            continue
        if policy == "oracle":
            # keep the metadata only when it is true; the upper bound
            tot += pi * corr(T["v_vhigh"][k]) + (1 - pi) * c0
            continue
        if policy == "always-high":
            lvl = "v_vhigh"
        elif policy == "always-low":
            lvl = "v_vlow"
        elif policy == "passthrough":
            # map the detector's precision onto the nearest verbal level
            lvl = LAD[min(4, max(0, int(round((pi - 0.5) / 0.125))))]
        elif policy.startswith("threshold"):
            t = float(policy.split("@")[1])
            if pi < t:
                tot += c0
                continue
            lvl = "v_vhigh"
        elif policy == "two-pass":
            # forward at the detector's own level, but if the metadata flips the answer and
            # the declared level is below "high", fall back to the signal-only answer
            lvl = LAD[min(4, max(0, int(round((pi - 0.5) / 0.125))))]
            keep = lvl in ("v_high", "v_vhigh")
            for br, w in ((T, pi), (F, 1 - pi)):
                flipped = br[lvl][k]["pred"] != b[k]["pred"]
                tot += w * (corr(br[lvl][k]) if (keep or not flipped) else c0)
            continue
        else:
            raise ValueError(policy)
        tot += pi * corr(T[lvl][k]) + (1 - pi) * corr(F[lvl][k])
    return tot / len(clips) * 100


POLICIES = ["omit", "passthrough", "always-high", "always-low",
            "threshold@0.8", "threshold@0.95", "two-pass", "oracle"]

if __name__ == "__main__":
    PIS = [0.6, 0.7, 0.8, 0.9, 0.95]
    for base, lad, name in MODELS:
        d = build(base, lad)
        if d is None:
            print(f"{name}: ladder incomplete\n")
            continue
        b, T, F, clips = d
        print(f"{name}   n={len(clips)}   A0={np.mean([corr(b[k]) for k in clips])*100:.1f}")
        print(f"  {'policy':16}" + "".join(f"pi={p:<7.2f}" for p in PIS))
        best_non_oracle = {p: -1 for p in PIS}
        rows = {}
        for pol in POLICIES:
            vals = [score(b, T, F, clips, pol, p) for p in PIS]
            rows[pol] = vals
            if pol != "oracle":
                for p, v in zip(PIS, vals):
                    best_non_oracle[p] = max(best_non_oracle[p], v)
        for pol in POLICIES:
            marks = ["*" if (pol != "oracle" and abs(v - best_non_oracle[p]) < 1e-9) else " "
                     for p, v in zip(PIS, rows[pol])]
            print(f"  {pol:16}" + "".join(f"{v:7.1f}{m} " for v, m in zip(rows[pol], marks)))
        print()
