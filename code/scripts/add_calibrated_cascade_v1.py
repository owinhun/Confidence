#!/usr/bin/env python3
"""Two repaired versions of the forwarded confidence, added to the cascade file.

Forwarding the detector's own word-level confidence made every downstream model worse. The
cascade table says why: the detector's bands are not monotone in precision -- clips it calls
"high" are right 65.1% of the time and clips it calls "moderate" 78.6%. A model that reads the
word and moves toward the claim is therefore being steered by a signal that is, in part,
anti-correlated with correctness.

That diagnosis makes two repairs testable, and they separate two different explanations.

  mono  re-labels the bands so the word ordering matches measured precision, keeping the same
        five words and the same clip partition. If the harm was the non-monotonicity, this fixes
        it; if forwarding hurts regardless of what the word means, it will not.

  pct   replaces the word with the band's measured precision as a percentage. This is what an
        honestly calibrated pipeline would actually emit, and it also asks whether a number is
        read differently from a word when both carry the same information.

Both are calibrated on the SAME half of the source videos the threshold was chosen on, so the
test half stays untouched. Precision is measured there, not on the clips being scored.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/workspace/svhalluc/claude_track2")
SRC = ROOT / "outputs/clap_cascade_v1.jsonl"
LEVELS = ["very low", "low", "moderate", "high", "very high"]

rows = [json.loads(l) for l in SRC.open() if l.strip()]

# precision per band, measured on the calibration half only
hit, tot = defaultdict(int), defaultdict(int)
for r in rows:
    if r["split"] == "cal":
        tot[r["det_level"]] += 1
        hit[r["det_level"]] += int(r["det_pred"] == r["label"])
prec = {lv: hit[lv] / tot[lv] for lv in LEVELS if tot[lv]}

# the monotone repair: sort the bands by measured precision and hand out the words in order,
# so "very high" always names the band that is in fact most often right
order = sorted(prec, key=prec.get)
mono = {lv: LEVELS[i] for i, lv in enumerate(order)}

print("band          n   precision   -> monotone word")
for lv in LEVELS:
    if lv in prec:
        print(f"  {lv:11}{tot[lv]:4}   {prec[lv]*100:5.1f}%      {mono[lv]}")

for r in rows:
    lv = r["det_level"]
    r["mono_level"] = mono.get(lv, lv)
    r["cal_pct"] = round(prec.get(lv, 0.5) * 100)

with SRC.open("w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"\n{SRC} 갱신: mono_level, cal_pct 추가 (n={len(rows)})")
