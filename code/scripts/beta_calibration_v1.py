#!/usr/bin/env python3
"""Two analyses the reviews asked for, on data already on disk. No GPU.

E. Confidence vs Priority as a difference-in-differences. The paper currently quotes a ratio,
   beta_confidence / beta_Priority. A ratio is unstable when its denominator is near zero, and
   two of the seven cells have a small one. The difference beta_conf - beta_prio has no such
   problem and carries a confidence interval, so it is the honest way to say the same thing.
   Both terms come from the SAME items, so the interval is a paired bootstrap over source
   videos -- clips from one video are not independent of each other.

F. Is the integration calibrated? beta rises with the declared level, but a model that used the
   declaration correctly would move by an amount tied to what the level asserts. Reading the
   five words as the probabilities they name (0.1 ... 0.9), a calibrated reader's margin shift
   would be linear in logit(c) with slope 1. Regressing beta on logit(c) gives the slope the
   models actually use. A slope far from 1 says the direction is right and the magnitude is not,
   which is a sharper claim than "confidence moves the model".
"""

from __future__ import annotations

import json
import os

import numpy as np

OUT = "outputs"
EPS = 1e-6
BOOTS = 2000
LEVELS = ["very low", "low", "moderate", "high", "very high"]
# the probabilities the five words name, as the prompt's own ladder defines them
PROB = {"very_low": 0.1, "low": 0.3, "moderate": 0.5, "high": 0.7, "very_high": 0.9}

CELLS = [("Qwen2.5-Omni", "s2", "sound"), ("Qwen3-Omni", "s2q3", "sound"),
         ("VideoLLaMA2.1", "s2vl", "colour"), ("Qwen2-Audio", "nm_qwen2audio", "sound"),
         ("Audio-Flamingo-3", "nm_af3", "sound"), ("Nemotron/sound", "nm_nemotron", "sound"),
         ("Nemotron/colour", "nm_nemotron", "colour")]


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


def margin(rec, swapped):
    pa = min(max(rec["p_A"], EPS), 1 - EPS)
    lo = np.log(pa) - np.log(1 - pa)
    v = lo if rec["label"] == "A" else -lo
    return -v if swapped else v


def video(clip):
    return clip.split("/")[-1].split("_")[0]


def paired(pfx, attr, pairs, tag):
    """Per-item values for several (true, false) condition pairs, on the same item set."""
    sw = tag == "_swap"
    b = L(f"{pfx}_{attr}_none{tag}")
    if b is None:
        return None
    got = {}
    for name, hi, lo in pairs:
        A, B = L(f"{pfx}_{attr}_{hi}{tag}"), L(f"{pfx}_{attr}_{lo}{tag}")
        if A is None or B is None:
            return None
        got[name] = (A, B)
    ks = [k for k in b if b[k]["correct"] and all(k in A and k in B for A, B in got.values())]
    if len(ks) < 30:
        return None
    return ks, {n: {k: margin(A[k], sw) - margin(B[k], sw) for k in ks} for n, (A, B) in got.items()}


def boot_diff(ks, per_a, per_b):
    """Paired bootstrap over source videos of mean(a) - mean(b)."""
    vids = sorted({video(k) for k in ks})
    by = {v: [k for k in ks if video(k) == v] for v in vids}
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(BOOTS):
        sel = rng.choice(vids, len(vids), replace=True)
        idx = [k for v in sel for k in by[v]]
        bs.append(np.mean([per_a[k] for k in idx]) - np.mean([per_b[k] for k in idx]))
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


if __name__ == "__main__":
    print("E. Confidence minus Priority, paired on the same items, bootstrapped over videos.")
    print("   A ratio blows up when its denominator is small; a difference does not.")
    print("   The resample is shared across answer orders, so the interval covers exactly the")
    print("   order-averaged quantity the point estimate reports.\n")
    print(f"{'cell':18}{'beta_conf':>10}{'beta_prio':>10}{'difference':>12}{'95% CI':>18}{'n':>6}")
    PAIRS = [("conf", "anon_high", "anon_low"), ("prio", "priority_high", "priority_low")]
    for name, pfx, attr in CELLS:
        orders = [r for r in (paired(pfx, attr, PAIRS, tag) for tag in ("", "_swap")) if r]
        if not orders:
            print(f"{name:18}{'--':>10}")
            continue

        def om(field):
            return float(np.mean([np.mean(list(per[field].values())) for _ks, per in orders]))

        c, p_ = om("conf"), om("prio")
        vids = sorted({video(k) for _ks, _p in orders for k in _ks})
        by = [{v: [k for k in ks if video(k) == v] for v in vids} for ks, _p in orders]
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(BOOTS):
            sel = rng.choice(vids, len(vids), replace=True)
            vals = []
            for (ks, per), idxmap in zip(orders, by):
                idx = [k for v in sel for k in idxmap[v]]
                if idx:
                    vals.append(np.mean([per["conf"][k] for k in idx])
                                - np.mean([per["prio"][k] for k in idx]))
            if vals:
                bs.append(np.mean(vals))
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        star = "" if lo > 0 or hi < 0 else "  (spans 0)"
        n = len(orders[0][0])
        print(f"{name:18}{c:10.2f}{p_:10.2f}{c - p_:12.2f}"
              f"{f'[{lo:.2f}, {hi:.2f}]':>18}{n:6}{star}")

    print("\nF. Is the integration calibrated? beta at each rung against the logit of the")
    print("   probability that rung's word names. A calibrated reader would use slope 1.\n")
    print(f"{'cell':18}{'slope':>8}{'intercept':>11}{'R^2':>7}   beta by rung")
    QWEN_LAD = ["v_vlow", "v_low", "v_mod", "v_high", "v_vhigh"]
    NEMO_LAD = [("v_vlow", "v_vlow_T"), ("anon_low", "true_low"), ("v_mod", "v_mod_T"),
                ("anon_high", "true_high_T"), ("v_vhigh", "v_vhigh_T")]
    LADS = {"s2": "cs_qwen25", "s2q3": "cs_qwen3", "s2vl": "cs_vllama",
            "nm_qwen2audio": "nm_qwen2audio", "nm_af3": "nm_af3", "nm_nemotron": "nm_nemotron"}
    XS = [np.log(p / (1 - p)) for p in (0.1, 0.3, 0.5, 0.7, 0.9)]
    for name, pfx, attr in CELLS:
        b = L(f"{pfx}_{attr}_none")
        if b is None:
            print(f"{name:18}{'--':>8}")
            continue
        keys = [k for k in b if b[k]["correct"]]
        nemo = NEMO_LAD if pfx == "nm_nemotron" else None
        lad = LADS[pfx]
        ys, row = [], []
        for i in range(5):
            if nemo:
                fs, ts = f"{pfx}_{attr}_{nemo[i][0]}", f"{pfx}_{attr}_{nemo[i][1]}"
            else:
                fs, ts = f"{lad}_{attr}_{QWEN_LAD[i]}", f"{lad}_{attr}_{QWEN_LAD[i]}_T"
            F, T = L(fs), L(ts)
            if F is None or T is None:
                row.append("   --")
                ys.append(None)
                continue
            ks = [k for k in keys if k in F and k in T]
            v = float(np.mean([0.5 * (margin(T[k], False) - margin(F[k], False)) for k in ks]))
            ys.append(v)
            row.append(f"{v:+5.2f}")
        pts = [(x, y) for x, y in zip(XS, ys) if y is not None]
        if len(pts) < 3:
            print(f"{name:18}{'--':>8}{'':11}{'':7}   {' '.join(row)}")
            continue
        xa = np.array([p[0] for p in pts])
        ya = np.array([p[1] for p in pts])
        A = np.vstack([xa, np.ones(len(xa))]).T
        (m, b_), *_ = np.linalg.lstsq(A, ya, rcond=None)
        r2 = 1 - np.sum((ya - A @ [m, b_]) ** 2) / max(1e-12, np.sum((ya - ya.mean()) ** 2))
        print(f"{name:18}{m:8.2f}{b_:11.2f}{r2:7.2f}   {' '.join(row)}")
    print("\n   slope 1 = the model moves exactly as much as the declaration warrants.")
