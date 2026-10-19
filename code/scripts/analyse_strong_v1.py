#!/usr/bin/env python3
"""Tier-T judgment (wave22, pre-registered).

Order-averaged accuracy change of bare forwarding of the strong probe's decisions
(cascademixstr_plain) vs omitting, per order-robust cell, with the same source-video
cluster bootstrap as the sweep. Judged against the pre-registered sign predictions:
gain where the probe's OOF accuracy (0.897) exceeds the cell's bare pi*, loss where it
falls short. Refutations are reported as-is.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import analyse_cascade_v2 as C

CELLS = [("Qwen2.5", "s2", 0.902), ("Qwen3", "s2q3", 0.921),
         ("Qwen2-Audio", "nm_qwen2audio", 0.806), ("Nemotron", "nm_nemotron", 0.890)]
AHAT = 0.897

def delta(pfx, cond):
    per, keys = [], []
    for sfx in ("", "_swap"):
        A, B = C.L(f"{pfx}_sound_{cond}{sfx}"), C.L(f"{pfx}_sound_none{sfx}")
        if A is None or B is None:
            return None
        ks = sorted(set(A) & set(B))
        per.append(np.array([A[k]["correct"] - B[k]["correct"] for k in ks], float))
        keys.append(ks)
    pt = 100 * np.mean([d.mean() for d in per])
    vids = sorted({C.video(k) for ks in keys for k in ks})
    vi = {v: i for i, v in enumerate(vids)}
    buckets = [[np.where(np.array([vi[C.video(k)] for k in ks]) == i)[0]
                for i in range(len(vids))] for ks in keys]
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        sel = rng.integers(0, len(vids), len(vids))
        vals = []
        for d, bk in zip(per, buckets):
            idx = np.concatenate([bk[i] for i in sel if bk[i].size])
            vals.append(d[idx].mean())
        bs.append(100 * np.mean(vals))
    return pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

print(f"probe OOF = {AHAT}")
print(f"{'cell':13}{'pi*':>7}{'예측':>6}{'Δ(pp)':>8}{'95% CI':>18}{'판정':>8}")
match = 0
for name, pfx, pistar in CELLS:
    r = delta(pfx, "cascademixstr_plain")
    if r is None:
        print(f"{name:13} 데이터 없음"); continue
    pt, lo, hi = r
    pred = "+" if AHAT > pistar else "-"
    obs = "+" if pt > 0 else "-"
    ok = pred == obs
    match += ok
    print(f"{name:13}{pistar:7.3f}{pred:>6}{pt:+8.1f}  [{lo:+6.1f},{hi:+6.1f}]{'일치' if ok else '반증':>8}")
print(f"\n부호 일치 {match}/4")
