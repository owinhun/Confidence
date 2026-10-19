#!/usr/bin/env python3
"""Second detector's cascade file, built exactly like the first.

The first CLAP checkpoint's bands were monotone once the confidence was defined about the
decision rather than about presence. That could still be a property of one checkpoint. Running
the whole cascade -- not just the band check -- on `laion/larger_clap_general` says whether the
downstream findings (claim forwarding loses; a number beats a graded word) survive a change of
detector. Same isotonic calibration, same split by source video, same quintile bands.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/workspace/svhalluc/claude_track2")
LEVELS = ["very low", "low", "moderate", "high", "very high"]

split = {json.loads(l)["clip"]: json.loads(l)["split"]
         for l in (ROOT / "outputs/clap_cascade_v1.jsonl").open() if l.strip()}
rows = [json.loads(l) for l in (ROOT / "outputs/clap2_detector_v1.jsonl").open() if l.strip()]
for r in rows:
    r["split"] = split.get(r["clip"], "test")

cal = [r for r in rows if r["split"] == "cal"]
iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
iso.fit([r["p_present"] for r in cal], [int(r["label"] == "A") for r in cal])
for r in rows:
    p = float(np.clip(iso.predict([r["p_present"]])[0], 1e-4, 1 - 1e-4))
    r["p_cal"] = round(p, 6)
    r["det_pred"] = "A" if p >= 0.5 else "B"
    r["q"] = round(p if r["det_pred"] == "A" else 1 - p, 6)

cal = [r for r in rows if r["split"] == "cal"]
qs = np.quantile([r["q"] for r in cal], [0.2, 0.4, 0.6, 0.8])
for r in rows:
    r["det_level"] = LEVELS[int(np.searchsorted(qs, r["q"]))]

cal = [r for r in rows if r["split"] == "cal"]
hit, tot = defaultdict(int), defaultdict(int)
for r in cal:
    tot[r["det_level"]] += 1
    hit[r["det_level"]] += int(r["det_pred"] == r["label"])
prec = {lv: hit[lv] / tot[lv] for lv in LEVELS if tot[lv]}
for r in rows:
    r["det_precision"] = round(prec.get(r["det_level"], 0.5), 6)
    r["cal_pct"] = round(prec.get(r["det_level"], 0.5) * 100)
    r["mono_level"] = r["det_level"]

test = [r for r in rows if r["split"] == "test"]
acc = np.mean([r["det_pred"] == r["label"] for r in test]) * 100
v = [prec[l] for l in LEVELS if l in prec]
print(f"larger-CLAP  held-out 정확도 {acc:.1f}%")
for lv in LEVELS:
    if lv in prec:
        print(f"  {lv:12}{tot[lv]:4}{prec[lv]*100:8.1f}%")
print(f"  단조: {all(a <= b for a, b in zip(v, v[1:]))}")

out = ROOT / "outputs/clap2_cascade_v1.jsonl"
with out.open("w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"-> {out.name} ({len(rows)}행)")
