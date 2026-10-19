#!/usr/bin/env python3
"""The claim-aligned contrast on every cell we ran, gate-passing or not.

The reviews both land on the same worry: only five of thirty-two model-by-attribute cells pass
the perceptual-eligibility gate, so the effect might live in a narrow operating band that the
gate quietly selects for.

The gate exists for the override rate, which conditions on items answered correctly without
metadata and is meaningless when that set is tiny or is correct by position. The margin has no
such requirement -- beta is defined wherever a true and a false claim were both scored. So the
worry is testable: compute beta on the cells the gate rejected and see whether the sign and the
ordering survive.

A floor cell is the interesting case. If a model cannot judge the property at all, its unaided
answers are near chance, and yet a claim can still move its margin toward itself. That would say
the confidence field acts on the decision variable rather than on perception, which is a
stronger statement than the gated analysis can make.
"""

from __future__ import annotations

import json
import os

import numpy as np

OUT = "outputs"
EPS = 1e-6
BOOTS = 1200

MODELS = [("Qwen2.5-Omni", "s2"), ("Qwen3-Omni", "s2q3"), ("VideoLLaMA2.1", "s2vl"),
          ("Qwen2-Audio", "nm_qwen2audio"), ("Audio-Flamingo-3", "nm_af3"),
          ("Nemotron-Omni", "nm_nemotron"), ("Phi-4-mm", "nm_phi4mm")]
ATTRS = ["sound", "colour", "noise", "lowpass", "reverb", "silence"] + \
        [f"grad_{a}_{g}" for a in ("noise", "lowpass", "reverb") for g in ("g1", "g2", "g3", "g4")]


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


def gate(b):
    acc = np.mean([r["correct"] for r in b.values()])
    rec = [np.mean([r["correct"] for r in b.values() if r["label"] == Lb]) for Lb in ("A", "B")]
    if acc < 0.60:
        return "floor", acc, rec
    if min(rec) < 0.60:
        return "one-sided", acc, rec
    if acc > 0.95:
        return "ceiling", acc, rec
    return "eligible", acc, rec


def beta_at(pfx, attr, level):
    """beta on ALL items, not just baseline-correct: the gate is what we are testing."""
    F = L(f"{pfx}_{attr}_anon_{level}")
    T = L(f"{pfx}_{attr}_true_{level}") or L(f"{pfx}_{attr}_true_{level}_T")
    if F is None or T is None:
        return None
    ks = sorted(set(F) & set(T))
    if len(ks) < 100:
        return None
    per = {k: 0.5 * (margin(T[k]) - margin(F[k])) for k in ks}
    vids = sorted({video(k) for k in ks})
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(BOOTS):
        sel = set(rng.choice(vids, len(vids), replace=True))
        v = [per[k] for k in ks if video(k) in sel]
        if v:
            bs.append(np.mean(v))
    return (float(np.mean(list(per.values()))),
            float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(ks))


if __name__ == "__main__":
    print("beta computed on every cell with both claim polarities, whatever the gate said.")
    print("beta_high should exceed beta_low if the ordering is not an artefact of the gate.\n")
    print(f"{'model':17}{'attribute':16}{'gate':11}{'A0':>6}"
          f"{'beta(low)':>22}{'beta(high)':>22}{'n':>6}")
    rows = []
    for name, pfx in MODELS:
        for attr in ATTRS:
            b = L(f"{pfx}_{attr}_none")
            if b is None:
                continue
            status, acc, rec = gate(b)
            lo_, hi_ = beta_at(pfx, attr, "low"), beta_at(pfx, attr, "high")
            if lo_ is None and hi_ is None:
                continue
            f = lambda r: "--".rjust(22) if r is None else \
                f"{r[0]:6.2f} [{r[1]:5.2f},{r[2]:5.2f}]".rjust(22)
            n = (hi_ or lo_)[3]
            print(f"{name:17}{attr:16}{status:11}{acc * 100:6.1f}{f(lo_)}{f(hi_)}{n:>6}")
            if lo_ and hi_:
                rows.append((status, hi_[0] - lo_[0], hi_[0], lo_[0]))

    print("\n--- does the ordering hold outside the gate? ---")
    for st in ("eligible", "ceiling", "one-sided", "floor"):
        sub = [r for r in rows if r[0] == st]
        if not sub:
            continue
        d = [r[1] for r in sub]
        pos = sum(x > 0 for x in d)
        print(f"  {st:11} cells={len(sub):3}  beta(high)-beta(low) > 0 in {pos}/{len(sub)}"
              f"   median {np.median(d):+.2f}")
