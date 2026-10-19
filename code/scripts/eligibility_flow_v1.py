#!/usr/bin/env python3
"""Every model x attribute cell we ever ran, and why each one is in or out.

The gate was pre-registered but it was reported in the Limitations section, which invites the
reading that it was chosen after seeing which cells gave the largest effect. The answer to that
is not an argument, it is the full flow: how many cells were attempted, how many left at each
step, and whether the conclusion survives moving the thresholds.

Gate (pre-registered, on the NO-METADATA pass only -- no condition with metadata in it is read
before a cell is admitted):
    unaided accuracy A0 in [0.60, 0.95]      below -> the model cannot judge the property
                                             above -> ceiling; retained for override, but there
                                                      is almost no headroom to measure
    both-direction recall >= 0.60            below -> the model answers one way regardless, so
                                                      "correct" items are correct by position

Sensitivity: the same counts recomputed at accuracy lower bounds 0.55/0.60/0.65 and recall
thresholds 0.55/0.60/0.65, so the reader can see the gate is not load-bearing.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict

OUT = "outputs"
MODELS = OrderedDict([("s2", "Qwen2.5-Omni"), ("s2q3", "Qwen3-Omni"),
                      ("s2vl", "VideoLLaMA2.1"), ("nm_qwen2audio", "Qwen2-Audio"),
                      ("nm_af3", "Audio-Flamingo-3"), ("nm_phi4mm", "Phi-4-multimodal")])
ATTRS = ["sound", "colour", "noise", "lowpass", "reverb"] + \
        [f"grad_{a}_{g}" for a in ("noise", "lowpass", "reverb") for g in ("g1", "g2", "g3", "g4")]


def load(base, attr):
    p = f"{OUT}/{base}_{attr}_none_v1.jsonl"
    if not os.path.exists(p):
        return None
    rs = [json.loads(l) for l in open(p).read().splitlines() if l.strip()]
    return rs if len(rs) >= 400 else None


def stats(rs):
    acc = sum(r["correct"] for r in rs) / len(rs)
    rec = []
    for lab in ("A", "B"):
        sub = [r for r in rs if r["label"] == lab]
        rec.append(sum(r["correct"] for r in sub) / len(sub) if sub else 0.0)
    return acc, min(rec), max(rec)


def classify(acc, rmin, alo=0.60, ahi=0.95, rthr=0.60):
    if acc < alo:
        return "floor"
    if rmin < rthr:
        return "one-sided"
    if acc > ahi:
        return "ceiling"
    return "eligible"


if __name__ == "__main__":
    rows = []
    for base, name in MODELS.items():
        for attr in ATTRS:
            rs = load(base, attr)
            if rs is None:
                continue
            acc, rmin, rmax = stats(rs)
            rows.append((name, attr, acc, rmin, rmax, classify(acc, rmin)))

    print(f"{'model':18}{'attribute':18}{'A0':>7}{'rec-':>7}{'rec+':>7}  verdict")
    for r in rows:
        print(f"{r[0]:18}{r[1]:18}{r[2]*100:7.1f}{r[3]*100:7.1f}{r[4]*100:7.1f}  {r[5]}")

    print(f"\n--- flow over {len(rows)} model x attribute cells actually run ---")
    order = ["floor", "one-sided", "ceiling", "eligible"]
    why = {"floor": "unaided accuracy below 0.60: property not judgeable",
           "one-sided": "one-direction recall below 0.60: answers by position",
           "ceiling": "unaided accuracy above 0.95: retained, but no headroom",
           "eligible": "carried into the confidence contrast"}
    for k in order:
        n = sum(r[5] == k for r in rows)
        print(f"{k:12}{n:4}   {why[k]}")

    print("\n--- sensitivity: eligible / ceiling counts as the gate moves ---")
    print(f"{'A0 lower':>9}{'recall':>9}{'eligible':>10}{'ceiling':>9}{'floor':>8}{'one-sided':>11}")
    for alo in (0.55, 0.60, 0.65):
        for rthr in (0.55, 0.60, 0.65):
            c = {k: 0 for k in order}
            for r in rows:
                c[classify(r[2], r[3], alo=alo, rthr=rthr)] += 1
            print(f"{alo:9.2f}{rthr:9.2f}{c['eligible']:10}{c['ceiling']:9}"
                  f"{c['floor']:8}{c['one-sided']:11}")
