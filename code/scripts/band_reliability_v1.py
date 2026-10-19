#!/usr/bin/env python3
"""Are the detector's confidence bands monotone out-of-fold, or only by construction?

Isotonic regression makes the fitted score monotone in the training data by definition, so a
monotone band ordering measured on the fit is not evidence. The cross-fitted file assigns every
clip a band from a rule fitted on the other four folds, so the correctness measured here is
genuinely held-out. Wilson intervals, ECE and Brier are what a reviewer would ask for.
"""

from __future__ import annotations

import json
from collections import defaultdict
from math import sqrt

import numpy as np

LEVELS = ["very low", "low", "moderate", "high", "very high"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def report(path, tag):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    hit, tot, qs = defaultdict(int), defaultdict(int), defaultdict(list)
    for r in rows:
        lv = r["det_level"]
        ok = int(r["det_pred"] == r["label"])
        tot[lv] += 1
        hit[lv] += ok
        qs[lv].append(r["q"])
    print(f"\n=== {tag} (out-of-fold, n={len(rows)}) ===")
    print(f"  {'band':12}{'n':>5}{'correct':>9}{'95% CI':>18}{'mean q':>9}{'gap':>8}")
    v, ece = [], 0.0
    for lv in LEVELS:
        if not tot[lv]:
            continue
        p = hit[lv] / tot[lv]
        lo, hi = wilson(hit[lv], tot[lv])
        mq = float(np.mean(qs[lv]))
        ece += tot[lv] / len(rows) * abs(mq - p)
        v.append(p)
        print(f"  {lv:12}{tot[lv]:5}{p*100:8.1f}%{f'[{lo*100:.0f}, {hi*100:.0f}]':>18}"
              f"{mq*100:8.1f}%{(mq-p)*100:+8.1f}")
    mono = all(a <= b for a, b in zip(v, v[1:]))
    inv = max(max(v[:i+1]) - v[i] for i in range(len(v))) * 100
    brier = float(np.mean([(r["q"] - int(r["det_pred"] == r["label"])) ** 2 for r in rows]))
    print(f"  단조 {mono}   최대 역전 {inv:.1f}pt   ECE {ece*100:.1f}%   Brier {brier:.3f}")
    # is the ordering better than chance? Spearman between band index and correctness
    idx = np.array([LEVELS.index(r["det_level"]) for r in rows])
    ok = np.array([int(r["det_pred"] == r["label"]) for r in rows])
    rng = np.random.default_rng(0)
    obs = np.corrcoef(idx, ok)[0, 1]
    null = [np.corrcoef(rng.permutation(idx), ok)[0, 1] for _ in range(2000)]
    print(f"  band-correctness 상관 {obs:+.3f}, 순열검정 p = {(sum(1 for x in null if x >= obs)+1)/2001:.4f}")


if __name__ == "__main__":
    report("outputs/clapx_cascade_v1.jsonl", "CLAP-unfused, 5-fold cross-fitted")
    report("outputs/clap2x_cascade_v1.jsonl", "larger-CLAP, 5-fold cross-fitted")
