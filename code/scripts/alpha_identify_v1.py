#!/usr/bin/env python3
"""Is alpha a truth-aligned cost, or just entropy flattening on a selected subset?

The objection is exact. The analysis conditions on items answered correctly unaided, so their
truth-aligned margin m0 is positive by construction. If a hedged claim merely flattens the
logit distribution, |m| shrinks, m moves toward zero, and alpha < 0 falls out mechanically --
and the fact that alpha is most negative at LOW declared confidence fits that story better
than ours.

Two tests separate them, both from data already on disk.

  1. Sign reversal. Refit alpha on items answered INCORRECTLY unaided, where m0 < 0. Flattening
     moves those margins UP toward zero, so it predicts alpha > 0 there. A genuine cost in
     truth-aligned evidence predicts alpha < 0 in both subsets.

  2. Empty block. Compute the same margin change under a block with no claim in it. Flattening
     from the mere presence of an uninformative structured field would show up here; a cost
     that requires a claim would not.

Also reports alpha stratified by baseline margin tertile: pure flattening scales with |m0|, so
it should grow steeply across tertiles.
"""

from __future__ import annotations

import json
import os

import numpy as np

OUT = "outputs"
LAD = ["v_vlow", "v_low", "v_mod", "v_high", "v_vhigh"]
MODELS = [("s2", "cs_qwen25", "Qwen2.5-Omni"),
          ("s2q3", "cs_qwen3", "Qwen3-Omni"),
          ("s2vl", "cs_vllama", "VideoLLaMA2.1"),
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


def margin(rec):
    pa = min(max(rec["p_A"], EPS), 1 - EPS)
    lo = np.log(pa) - np.log(1 - pa)
    return lo if rec["label"] == "A" else -lo


def video(clip):
    return clip.split("/")[-1].split("_")[0]


def alpha_on(base, lad, subset, boots=1500, seed=0):
    """subset in {'correct','incorrect','all'} -- returns alpha per level with CI."""
    b = L(f"{base}_sound_none")
    if b is None:
        return None
    keys = [k for k in b if (subset == "all"
                             or (subset == "correct") == bool(b[k]["correct"]))]
    rows = []
    for c in LAD:
        t, f = L(f"{lad}_sound_{c}_T"), L(f"{lad}_sound_{c}")
        if t is None or f is None:
            return None
        for k in keys:
            if k in t and k in f:
                m0 = margin(b[k])
                rows.append((video(k), c, margin(t[k]) - m0, margin(f[k]) - m0))
    if len(rows) < 40:
        return None
    vids = sorted({r[0] for r in rows})
    rng = np.random.default_rng(seed)

    def fit(sel):
        out = {}
        for c in LAD:
            d = [(dt, df) for v, cc, dt, df in rows if cc == c and v in sel]
            if not d:
                return None
            out[c] = (np.mean([x[0] for x in d]) + np.mean([x[1] for x in d])) / 2
        return out

    point = fit(set(vids))
    boot = [r for r in (fit(set(rng.choice(vids, len(vids), replace=True)))
                        for _ in range(boots)) if r]
    ci = {c: (np.percentile([x[c] for x in boot], 2.5),
              np.percentile([x[c] for x in boot], 97.5)) for c in LAD}
    return point, ci, len(rows) // len(LAD)


def empty_block_margin(base):
    """Margin change under a block that contains no claim."""
    b, e = L(f"{base}_sound_none"), L(f"{base}_sound_null_block")
    if b is None or e is None:
        return None
    ok = [k for k in b if b[k]["correct"] and k in e]
    d = [margin(e[k]) - margin(b[k]) for k in ok]
    vids = sorted({video(k) for k in ok})
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(1500):
        sel = set(rng.choice(vids, len(vids), replace=True))
        vals = [margin(e[k]) - margin(b[k]) for k in ok if video(k) in sel]
        if vals:
            bs.append(np.mean(vals))
    return np.mean(d), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(ok)


if __name__ == "__main__":
    print("TEST 1 -- alpha on unaided-CORRECT vs unaided-INCORRECT items")
    print("flattening predicts alpha<0 on correct and alpha>0 on incorrect;")
    print("a truth-aligned cost predicts alpha<0 in both.\n")
    print(f"{'model':18}{'subset':11}{'n/level':>8}"
          + "".join(f"{c.replace('v_',''):>16}" for c in LAD))
    for base, lad, name in MODELS:
        for sub in ("correct", "incorrect"):
            r = alpha_on(base, lad, sub)
            if r is None:
                print(f"{name:18}{sub:11}{'--':>8}")
                continue
            pt, ci, n = r
            cells = "".join(f"{pt[c]:7.2f}[{ci[c][0]:.2f},{ci[c][1]:.2f}]".rjust(16)
                            for c in LAD)
            print(f"{name:18}{sub:11}{n:>8}{cells}")
    print("\nTEST 2 -- margin change under an EMPTY block (no claim)")
    print(f"{'model':18}{'delta m':>10}{'95% CI':>20}{'n':>7}")
    for base, lad, name in MODELS:
        r = empty_block_margin(base)
        if r is None:
            print(f"{name:18}{'--':>10}")
            continue
        m, lo, hi, n = r
        print(f"{name:18}{m:10.3f}{f'[{lo:.2f}, {hi:.2f}]':>20}{n:>7}")
