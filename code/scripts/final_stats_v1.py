#!/usr/bin/env python3
"""The three analyses the manuscript still owes, all on the complete six-model dataset.

  A. Holm-corrected family of confidence contrasts. Seventeen exact McNemar tests were run;
     reporting them uncorrected invites the obvious objection.
  B. One item-level model over every evaluable cell, rather than the two-model GEE the draft
     currently cites, with model and attribute as fixed effects and the video as the cluster.
  C. The interaction the draft demoted to "heterogeneous": does the model's own baseline
     margin predict how much a high-confidence false claim moves it? Fitted per model so a
     single dominant family cannot carry it.
"""

from __future__ import annotations

import json
import os
import warnings
from math import comb

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
OUT = "/workspace/svhalluc/claude_track2/outputs"

MODELS = [("s2", "Qwen2.5-Omni"), ("s2q3", "Qwen3-Omni"), ("s2vl", "VideoLLaMA2.1"),
          ("nm_qwen2audio", "Qwen2-Audio"), ("nm_af3", "Audio-Flamingo-3")]
ATTRS = ["sound", "colour", "noise"]


def L(name):
    p = f"{OUT}/{name}_v1.jsonl"
    if not os.path.exists(p):
        return None
    return {json.loads(l)["clip"]: json.loads(l)
            for l in open(p).read().splitlines() if l.strip()}


def evaluable(pfx, attr):
    b = L(f"{pfx}_{attr}_none")
    if not b or len(b) < 400:
        return None
    A = [i for i in b if b[i]["label"] == "A"]
    B = [i for i in b if b[i]["label"] == "B"]
    a0 = np.mean([b[i]["correct"] for i in b])
    rA = np.mean([b[i]["correct"] for i in A])
    rB = np.mean([b[i]["correct"] for i in B])
    return b if (a0 >= 0.60 and rA >= 0.60 and rB >= 0.60) else None


def mcnemar(hi, lo, ok):
    n01 = sum(1 for i in ok if (hi[i]["pred"] != hi[i]["label"])
              and not (lo[i]["pred"] != lo[i]["label"]))
    n10 = sum(1 for i in ok if not (hi[i]["pred"] != hi[i]["label"])
              and (lo[i]["pred"] != lo[i]["label"]))
    m = n01 + n10
    p = (min(1.0, sum(comb(m, k) for k in range(min(n01, n10) + 1)) * 2 / 2 ** m)
         if m else 1.0)
    return n01, n10, p


# ------------------------------------------------------------------ A. Holm
def part_a():
    print("=" * 74)
    print("A. Confidence contrasts with Holm-Bonferroni correction")
    print("=" * 74)
    rows = []
    for pfx, nm in MODELS:
        for attr in ATTRS:
            b = evaluable(pfx, attr)
            if b is None:
                continue
            ok = [i for i in b if b[i]["correct"]]
            for src in ["anon", "expert"]:
                hi, lo = L(f"{pfx}_{attr}_{src}_high"), L(f"{pfx}_{attr}_{src}_low")
                if not hi or not lo or len(hi) < 400 or len(lo) < 400:
                    continue
                sub = [i for i in ok if i in hi and i in lo]
                n01, n10, p = mcnemar(hi, lo, sub)
                rows.append((nm, attr, src, n01, n10, p))
    rows.sort(key=lambda r: r[5])
    m = len(rows)
    print(f"{'model':16}{'attr':8}{'src':8}{'disc':>10}{'p':>11}{'Holm p':>11}  sig")
    running = 0.0
    n_sig = 0
    for k, (nm, attr, src, n01, n10, p) in enumerate(rows):
        adj = min(1.0, max(running, (m - k) * p))
        running = adj
        sig = "yes" if adj < 0.05 else "no"
        n_sig += adj < 0.05
        print(f"{nm:16}{attr:8}{src:8}{f'{n01}/{n10}':>10}{p:11.1e}{adj:11.1e}  {sig}")
    print(f"\n  {n_sig} of {m} survive Holm at 0.05; "
          f"{sum(1 for r in rows if r[4] == 0)} of {m} have no reversed item.")
    return rows


# --------------------------------------------------------- B. pooled item model
def part_b():
    print("\n" + "=" * 74)
    print("B. Pooled item-level model over every evaluable cell")
    print("=" * 74)
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    recs = []
    for pfx, nm in MODELS:
        for attr in ATTRS:
            b = evaluable(pfx, attr)
            if b is None:
                continue
            for src in ["anon", "expert"]:
                for force in ["high", "low"]:
                    w = L(f"{pfx}_{attr}_{src}_{force}")
                    if not w or len(w) < 400:
                        continue
                    for clip, r in w.items():
                        base = b.get(clip)
                        if base is None or not base["correct"]:
                            continue
                        p_true = (base["p_A"] if base["label"] == "A"
                                  else 1.0 - base["p_A"])
                        recs.append(dict(
                            video=clip.split("/")[-1].split("_")[0],
                            model=nm, attr=attr,
                            force=1 if force == "high" else 0,
                            src=1 if src == "expert" else 0,
                            dirA=1 if base["label"] == "A" else 0,
                            conf=p_true,
                            adopt=int(r["pred"] != r["label"])))
    df = pd.DataFrame(recs)
    print(f"  {len(df)} item-condition rows, {df.video.nunique()} videos, "
          f"{df.model.nunique()} models, {df.attr.nunique()} attributes")
    df["conf_c"] = df.conf - df.conf.mean()
    g = smf.gee("adopt ~ force + src + dirA + conf_c + C(model) + C(attr)",
                "video", df, family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()).fit()
    for k in ["force", "src", "dirA", "conf_c"]:
        lo, hi = np.exp(g.conf_int().loc[k])
        print(f"    {k:8} OR={np.exp(g.params[k]):8.3f}  95% CI [{lo:.3f}, {hi:.3f}]"
              f"  p={g.pvalues[k]:.3g}")
    return df


# ------------------------------------------- C. baseline margin x confidence
def part_c():
    print("\n" + "=" * 74)
    print("C. Does the model's own baseline margin predict override? (per model)")
    print("=" * 74)
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    print(f"{'model':16}{'attr':8}{'n':>7}{'OR(margin)':>12}{'95% CI':>22}{'p':>10}")
    for pfx, nm in MODELS:
        for attr in ATTRS:
            b = evaluable(pfx, attr)
            if b is None:
                continue
            w = L(f"{pfx}_{attr}_anon_high")
            if not w or len(w) < 400:
                continue
            rows = []
            for clip, r in w.items():
                base = b.get(clip)
                if base is None or not base["correct"]:
                    continue
                # scaled to units of 0.1 so the odds ratio is readable; the raw 0-1 range
                # drives the coefficient so negative that exp() underflows to 0.000
                margin = abs(base["p_A"] - 0.5) * 20         # 0 = knife-edge, 10 = certain
                rows.append(dict(video=clip.split("/")[-1].split("_")[0],
                                 margin=margin,
                                 dirA=1 if base["label"] == "A" else 0,
                                 adopt=int(r["pred"] != r["label"])))
            d = pd.DataFrame(rows)
            if d.adopt.nunique() < 2 or len(d) < 50:
                print(f"{nm:16}{attr:8}{len(d):>7}   (no variation)")
                continue
            try:
                g = smf.gee("adopt ~ margin + dirA", "video", d,
                            family=sm.families.Binomial(),
                            cov_struct=sm.cov_struct.Exchangeable()).fit()
                lo, hi = np.exp(g.conf_int().loc["margin"])
                print(f"{nm:16}{attr:8}{len(d):>7}{np.exp(g.params['margin']):12.3f}"
                      f"{f'[{lo:.3f}, {hi:.3f}]':>22}{g.pvalues['margin']:10.3g}")
            except Exception as e:  # noqa: BLE001
                print(f"{nm:16}{attr:8}{len(d):>7}   fit failed: {type(e).__name__}")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
