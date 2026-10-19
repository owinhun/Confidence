#!/usr/bin/env python3
"""Rebuild the forwarded confidence so it describes the decision that was forwarded.

The first construction was wrong. CLAP's score s is a similarity to the *positive* phrasing, and
the decision threshold chosen on the calibration split is 0.876, not 0.5 -- but the confidence
was taken as max(s, 1-s), which is confidence relative to 0.5. For a clip with s = 0.71 the
detector therefore says "absent" while declaring high confidence in a quantity that measures
belief in *presence*. On 131 of 400 clips (32.8%) the declared confidence pointed at the
opposite of the claim being forwarded.

That is a defect in how the message was built, not a property of the detector, and it plausibly
manufactured the non-monotone precision bands the paper then interpreted.

The correct construction, in three steps, all fitted on the calibration half of the source
videos:

  1. map the raw score to a probability, p_i = P(y=1 | s_i), by isotonic regression;
  2. take the confidence in the decision actually made, q_i = p_i if the decision is "present"
     else 1 - p_i, which lies in [0.5, 1] by construction and always refers to the forwarded
     claim;
  3. band q by calibration-split quintiles, and measure each band's precision there.

The decision itself now comes from the calibrated probability at 0.5, which is the same rule as
a threshold on s but stated on a scale where the confidence is well defined.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/workspace/svhalluc/claude_track2")
LEVELS = ["very low", "low", "moderate", "high", "very high"]


def rebuild(src, out, tag):
    rows = [json.loads(l) for l in src.open() if l.strip()]
    cal = [r for r in rows if r["split"] == "cal"]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit([r["p_present"] for r in cal], [int(r["label"] == "A") for r in cal])

    for r in rows:
        p = float(np.clip(iso.predict([r["p_present"]])[0], 1e-4, 1 - 1e-4))
        r["p_cal"] = round(p, 6)
        r["det_pred"] = "A" if p >= 0.5 else "B"
        # confidence in the decision that is actually being forwarded
        r["q"] = round(p if r["det_pred"] == "A" else 1 - p, 6)

    cal = [r for r in rows if r["split"] == "cal"]
    qs = np.quantile([r["q"] for r in cal], [0.2, 0.4, 0.6, 0.8])
    for r in rows:
        r["det_level"] = LEVELS[int(np.searchsorted(qs, r["q"]))]

    hit, tot = defaultdict(int), defaultdict(int)
    for r in cal:
        tot[r["det_level"]] += 1
        hit[r["det_level"]] += int(r["det_pred"] == r["label"])
    prec = {lv: hit[lv] / tot[lv] for lv in LEVELS if tot[lv]}

    order = sorted(prec, key=prec.get)
    mono = {lv: LEVELS[i] for i, lv in enumerate(order)}
    for r in rows:
        lv = r["det_level"]
        r["det_precision"] = round(prec.get(lv, 0.5), 6)
        r["mono_level"] = mono.get(lv, lv)
        r["cal_pct"] = round(prec.get(lv, 0.5) * 100)

    test = [r for r in rows if r["split"] == "test"]
    acc = np.mean([r["det_pred"] == r["label"] for r in test]) * 100
    n_flip = sum(r["det_pred"] == "B" and r["p_present"] > 0.5 for r in rows)

    print(f"\n=== {tag} ===")
    print(f"  held-out 정확도 {acc:.1f}%   (재보정 전 값과 비교할 것)")
    print(f"  {'band':12}{'n(cal)':>8}{'정밀도':>10}   -> 단조 재배정")
    v = []
    for lv in LEVELS:
        if lv in prec:
            print(f"  {lv:12}{tot[lv]:8}{prec[lv]*100:9.1f}%   {mono[lv]}")
            v.append(prec[lv])
    is_mono = all(a <= b for a, b in zip(v, v[1:]))
    inv = max(max(v[:i+1]) - v[i] for i in range(len(v))) * 100
    print(f"  단조인가: {is_mono}   최대 역전폭 {inv:.1f}pt")
    print(f"  (참고) 여전히 p_present>0.5 인데 'B' 판정: {n_flip}개 — 임계값이 0.5가 아니어서가"
          f" 아니라 보정 곡선 때문이며, q 는 항상 전달된 주장을 가리킨다")

    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  -> {out.name}")
    return prec, acc


if __name__ == "__main__":
    src = ROOT / "outputs/clap_cascade_v1.jsonl"
    bak = ROOT / "outputs/clap_cascade_BUGGY_v1.jsonl"
    if not bak.exists():
        bak.write_text(src.read_text())
        print(f"원본 보존: {bak.name}")
    rebuild(src, src, "CLAP-unfused (재구성)")
