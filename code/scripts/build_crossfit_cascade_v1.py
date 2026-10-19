#!/usr/bin/env python3
"""Cross-fitted cascade files, so every clip's forwarded message is out-of-fold.

The half-split version had a leak the reviews flagged: the decision threshold, the isotonic
calibration and the band precisions were fitted on one half of the source videos, and Sec. 7
then reported over all 400 clips -- including the half the rule had seen. Restricting to the
clean half fixes the leak but halves n, and the number-versus-word contrast loses significance
in four of five cells at n=196 even though its point estimate grows.

Five-fold cross-fitting resolves both. Folds are over source videos, never clips, so the two
segments that share a video cannot land on opposite sides. For each fold the threshold,
calibration and band precisions are fitted on the other four and applied to the held-out fold;
every clip therefore carries a message produced by a rule that never saw it, and all 400 are
usable.

Written for both detectors, since the second one exists to test whether the downstream findings
are about the interface or about CLAP.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/workspace/svhalluc/claude_track2")
LEVELS = ["very low", "low", "moderate", "high", "very high"]
K = 5


def video(clip):
    return clip.split("/")[-1].split("_")[0]


def crossfit(src, out, tag):
    rows = [json.loads(l) for l in src.open() if l.strip()]
    vids = sorted({video(r["clip"]) for r in rows})
    # deterministic assignment of source videos to folds; no clip-level split anywhere
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(vids))
    fold_of = {vids[perm[i]]: i % K for i in range(len(vids))}

    for r in rows:
        r["fold"] = fold_of[video(r["clip"])]

    per_fold_acc = []
    for f in range(K):
        tr = [r for r in rows if r["fold"] != f]
        te = [r for r in rows if r["fold"] == f]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit([r["p_present"] for r in tr], [int(r["label"] == "A") for r in tr])

        def q_of(r):
            p = float(np.clip(iso.predict([r["p_present"]])[0], 1e-4, 1 - 1e-4))
            pred = "A" if p >= 0.5 else "B"
            return p, pred, (p if pred == "A" else 1 - p)

        qs_tr = [q_of(r)[2] for r in tr]
        cuts = np.quantile(qs_tr, [0.2, 0.4, 0.6, 0.8])
        hit, tot = defaultdict(int), defaultdict(int)
        for r in tr:
            _, pred, q = q_of(r)
            lv = LEVELS[int(np.searchsorted(cuts, q))]
            tot[lv] += 1
            hit[lv] += int(pred == r["label"])
        prec = {lv: hit[lv] / tot[lv] for lv in LEVELS if tot[lv]}

        for r in te:
            p, pred, q = q_of(r)
            lv = LEVELS[int(np.searchsorted(cuts, q))]
            r["p_cal"] = round(p, 6)
            r["det_pred"] = pred
            r["q"] = round(q, 6)
            r["det_level"] = lv
            r["det_precision"] = round(prec.get(lv, 0.5), 6)
            r["cal_pct"] = round(prec.get(lv, 0.5) * 100)
            r["mono_level"] = lv
        per_fold_acc.append(np.mean([r["det_pred"] == r["label"] for r in te]) * 100)

    acc = np.mean([r["det_pred"] == r["label"] for r in rows]) * 100
    hit, tot = defaultdict(int), defaultdict(int)
    for r in rows:
        tot[r["det_level"]] += 1
        hit[r["det_level"]] += int(r["det_pred"] == r["label"])
    prec = {lv: hit[lv] / tot[lv] * 100 for lv in LEVELS if tot[lv]}
    v = [prec[l] for l in LEVELS if l in prec]

    print(f"\n=== {tag} ===")
    print(f"  out-of-fold 정확도 {acc:.1f}%   fold별 {np.round(per_fold_acc,1).tolist()}"
          f"  (sd {np.std(per_fold_acc):.1f})")
    for lv in LEVELS:
        if lv in prec:
            print(f"  {lv:12}{tot[lv]:5}{prec[lv]:8.1f}%")
    print(f"  단조: {all(a <= b for a, b in zip(v, v[1:]))}"
          f"   최대 역전 {max(max(v[:i+1])-v[i] for i in range(len(v))):.1f}pt")
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  -> {out.name}")


if __name__ == "__main__":
    crossfit(ROOT / "outputs/clap_cascade_v1.jsonl",
             ROOT / "outputs/clapx_cascade_v1.jsonl", "CLAP-unfused, 5-fold cross-fitted")
    crossfit(ROOT / "outputs/clap2_cascade_v1.jsonl",
             ROOT / "outputs/clap2x_cascade_v1.jsonl", "larger-CLAP, 5-fold cross-fitted")
