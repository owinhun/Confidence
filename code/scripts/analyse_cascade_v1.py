#!/usr/bin/env python3
"""Does repairing the forwarded confidence recover what forwarding it cost?

Forwarding the detector's own word made every model worse, and the diagnosis was that its bands
are not monotone in precision. Two repairs test that diagnosis. `mono` keeps the same five words
and the same clip partition and only re-orders which band gets which word; `pct` sends the
band's measured precision as a number. Both are fitted on the calibration half of the source
videos, so nothing here is read off the clips being scored.

Read the columns against `claim`, not against `omit`. The question is not whether forwarding
helps -- Table 4 already says it usually does not -- but whether, having decided to forward a
claim, adding a confidence field helps or hurts. `conf` is the uncalibrated field; if the
non-monotonicity was the whole problem, `mono` returns to `claim` and `pct` does at least as
well.
"""

from __future__ import annotations

import json
import os

import numpy as np

CELLS = [("Qwen2.5-Omni", "s2"), ("Qwen3-Omni", "s2q3"), ("Qwen2-Audio", "nm_qwen2audio"),
         ("Audio-Flamingo-3", "nm_af3"), ("Nemotron/sound", "nm_nemotron")]
CONDS = ["none", "cascade_bare", "cascade_conf", "cascade_mono", "cascade_pct"]
HEAD = {"none": "omit", "cascade_bare": "claim", "cascade_conf": "+conf",
        "cascade_mono": "+mono", "cascade_pct": "+pct"}


def L(name, need=400):
    p = f"outputs/{name}_v1.jsonl"
    if not os.path.exists(p):
        return None
    d = {}
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            d[r["clip"]] = r
    return d if len(d) >= need else None


def video(clip):
    return clip.split("/")[-1].split("_")[0]


if __name__ == "__main__":
    print("Accuracy (%) on the clips every listed condition scored. Read against `claim`.\n")
    print(f"{'model':18}" + "".join(f"{HEAD[c]:>8}" for c in CONDS)
          + f"{'mono-conf':>11}{'pct-conf':>10}{'n':>6}")
    for name, pfx in CELLS:
        got = {c: L(f"{pfx}_sound_{c}") for c in CONDS}
        have = [c for c in CONDS if got[c] is not None]
        if "cascade_bare" not in have or "cascade_conf" not in have:
            missing = [c for c in CONDS if got[c] is None]
            print(f"{name:18}{'대기중: ' + ','.join(HEAD[m] for m in missing)}")
            continue
        ks = sorted(set.intersection(*(set(got[c]) for c in have)))
        acc = {c: np.mean([got[c][k]["correct"] for k in ks]) * 100 for c in have}
        row = "".join(f"{acc[c]:8.1f}" if c in acc else f"{'--':>8}" for c in CONDS)
        dm = f"{acc['cascade_mono'] - acc['cascade_conf']:+11.1f}" if "cascade_mono" in acc else f"{'--':>11}"
        dp = f"{acc['cascade_pct'] - acc['cascade_conf']:+10.1f}" if "cascade_pct" in acc else f"{'--':>10}"
        print(f"{name:18}{row}{dm}{dp}{len(ks):6}")

    # a one-point gap on 400 clips is four clips; the interesting contrasts get an interval
    # before any of them is quoted as a finding. Paired over source videos, since clips from one
    # video are not independent.
    print("\n--- paired contrasts, bootstrap over source videos (95% CI) ---\n")
    print(f"{'model':18}{'mono - conf':>22}{'pct - conf':>22}{'pct - claim':>22}")
    rng_seed = 0
    for name, pfx in CELLS:
        got = {c: L(f"{pfx}_sound_{c}") for c in CONDS}
        if got["cascade_bare"] is None or got["cascade_conf"] is None:
            continue
        have = [c for c in CONDS if got[c] is not None]
        ks = sorted(set.intersection(*(set(got[c]) for c in have)))
        vids = sorted({video(k) for k in ks})
        by = {v: [k for k in ks if video(k) == v] for v in vids}
        out = []
        for a, b in (("cascade_mono", "cascade_conf"), ("cascade_pct", "cascade_conf"),
                     ("cascade_pct", "cascade_bare")):
            if got[a] is None or got[b] is None:
                out.append(f"{'--':>22}")
                continue
            d = {k: got[a][k]["correct"] - got[b][k]["correct"] for k in ks}
            rng = np.random.default_rng(rng_seed)
            bs = []
            for _ in range(2000):
                sel = rng.choice(vids, len(vids), replace=True)
                idx = [k for v in sel for k in by[v]]
                bs.append(np.mean([d[k] for k in idx]) * 100)
            lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
            pt = np.mean(list(d.values())) * 100
            mark = "" if lo > 0 or hi < 0 else "*"
            out.append(f"{f'{pt:+.1f} [{lo:+.1f},{hi:+.1f}]{mark}':>22}")
        print(f"{name:18}" + "".join(out))
    print("  * interval spans zero.\n")

    print("\n--- readback: was the block read at all? ---")
    print("Scored against the CLAIM's polarity, not the manifest: a false claim is answered")
    print("correctly by contradicting the ground truth.\n")
    print(f"{'model':18}{'false claim':>13}{'true claim':>12}{'n':>6}")
    for name, pfx in [("Audio-Flamingo-3", "nm_af3"), ("Qwen2-Audio", "nm_qwen2audio")]:
        out = []
        for cond, claim_true in (("anon_high", False), ("true_high", True)):
            d = L(f"{pfx}_sound_{cond}_readback")
            if d is None:
                out.append(None)
                continue
            # `pred` is the model's letter; `label` is the manifest. A false claim asserts the
            # negation, so echoing it means predicting the opposite of the manifest.
            ok = [(r["pred"] == r["label"]) == claim_true for r in d.values()]
            out.append((np.mean(ok) * 100, len(ok)))
        cells = "".join(f"{o[0]:12.1f}%" if o else f"{'대기중':>13}" for o in out)
        n = next((o[1] for o in out if o), 0)
        print(f"{name:18}{cells}{n:6}")
    print("\nA model that never reads the block cannot beat chance here.")
