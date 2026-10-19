#!/usr/bin/env python3
"""Formal per-item model of claim adoption, replacing the per-cell rate comparisons.

The reviewer's objection to the cell-wise tables is fair: each POR number is a
marginal rate, so the confidence contrast is never adjusted for the covariates
that plausibly drive it (which video, which claim direction, how confident the
model was without any statement). This fits the pre-registered model

    ClaimAdoption ~ AssertionForce + Source + ClaimDirection + BaselineConfidence
                    + (1 | Video)

on the Stage-2 held-out factorial, with attribute as a fixed effect and the video
as the grouping factor, so the confidence coefficient is read off adjusted for
everything else. A GEE with an exchangeable working correlation over video is
reported alongside, because a binomial GLMM with a near-singular random intercept
can understate the standard error and GEE's robust sandwich errors do not.

Also runs the pre-registered TOST equivalence test on the SOURCE factor: the
correct question for a factor we expect to be inert is not "is it significant"
but "is it inside a margin we declared uninteresting" (+/- 5 percentage points).
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

OUT = Path("/workspace/svhalluc/claude_track2/outputs")
ATTRS = ["colour", "silence", "sound"]
CELLS = ["anon_high", "anon_low", "expert_high", "expert_low"]
MARGIN = 0.05  # pre-registered equivalence margin on the source factor


def load(attr, cond, prefix="s2"):
    p = OUT / f"{prefix}_{attr}_{cond}_v1.jsonl"
    if not p.exists():
        return None
    return {json.loads(l)["clip"]: json.loads(l)
            for l in p.read_text().splitlines() if l.strip()}


def build(prefix="s2"):
    """One row per (item, cell) among items the model got right WITHOUT a statement."""
    recs = []
    for attr in ATTRS:
        base = load(attr, "none", prefix)
        if not base:
            continue
        for cond in CELLS:
            d = load(attr, cond, prefix)
            if not d:
                continue
            for clip, r in d.items():
                b = base.get(clip)
                if b is None or not b["correct"]:
                    continue  # POR is conditioned on being correct unaided
                # p_A is P(answer "A"); baseline confidence in the TRUE answer
                p_true = b["p_A"] if b["label"] == "A" else 1.0 - b["p_A"]
                recs.append(dict(
                    video=clip.split("/")[-1].split("_")[0],
                    attr=attr,
                    source="expert" if cond.startswith("expert") else "anon",
                    force="high" if cond.endswith("high") else "low",
                    direction=b["label"],
                    base_conf=p_true,
                    adopt=int(r["pred"] != r["label"]),
                ))
    return pd.DataFrame(recs)


def por_by_cell(df):
    print("\n--- direction-balanced macro-POR per cell (for reference) ---")
    for attr in sorted(df.attr.unique()):
        s = df[df.attr == attr]
        row = []
        for src, f in product(["anon", "expert"], ["high", "low"]):
            c = s[(s.source == src) & (s.force == f)]
            m = np.mean([c[c.direction == d].adopt.mean() for d in ["A", "B"]])
            row.append(f"{src}_{f} {m*100:5.1f}")
        print(f"  {attr:8} " + "  ".join(row) + f"   (n={len(s)//4})")


def fit(df, label):
    print(f"\n=== {label}: adoption ~ force + source + direction + base_conf "
          f"+ attr, video random intercept ===")
    d = df.copy()
    d["force_hi"] = (d.force == "high").astype(int)
    d["src_exp"] = (d.source == "expert").astype(int)
    d["dir_A"] = (d.direction == "A").astype(int)
    d["conf_c"] = d.base_conf - d.base_conf.mean()
    formula = "adopt ~ force_hi + src_exp + dir_A + conf_c + C(attr)"

    try:
        m = smf.mixedlm(formula, d, groups=d["video"]).fit(reml=False)
        print("  [LMM on the 0/1 outcome, video random intercept]")
        for k in ["force_hi", "src_exp", "dir_A", "conf_c"]:
            print(f"    {k:10} beta={m.params[k]:+.4f}  se={m.bse[k]:.4f}  "
                  f"p={m.pvalues[k]:.3g}")
    except Exception as e:  # noqa: BLE001
        print(f"  LMM failed: {e}")

    fam = sm.families.Binomial()
    g = smf.gee(formula, "video", d, family=fam,
                cov_struct=sm.cov_struct.Exchangeable()).fit()
    print("  [logistic GEE, exchangeable within video, robust SE]")
    for k in ["force_hi", "src_exp", "dir_A", "conf_c"]:
        or_ = np.exp(g.params[k])
        lo, hi = np.exp(g.conf_int().loc[k])
        print(f"    {k:10} OR={or_:6.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
              f"p={g.pvalues[k]:.3g}")
    return g


def tost_source(df, label):
    """Equivalence, not a null test: is the source effect inside +/- 5 pp?"""
    print(f"\n=== {label}: TOST equivalence on SOURCE (margin +/- {MARGIN*100:.0f} pp) ===")
    for attr in sorted(df.attr.unique()) + ["ALL"]:
        s = df if attr == "ALL" else df[df.attr == attr]
        # paired at the item level, averaged over force, direction-balanced
        piv = s.pivot_table(index=["video", "attr", "direction"], columns="source",
                            values="adopt", aggfunc="mean")
        piv = piv.dropna()
        if len(piv) < 10:
            continue
        d = (piv["expert"] - piv["anon"]).values
        n, mean, se = len(d), d.mean(), stats.sem(d)
        if se == 0:
            print(f"  {attr:8} diff={mean*100:+.2f} pp  (zero variance)")
            continue
        t_lo = (mean + MARGIN) / se
        t_hi = (mean - MARGIN) / se
        p_lo = 1 - stats.t.cdf(t_lo, n - 1)   # H0: diff <= -margin
        p_hi = stats.t.cdf(t_hi, n - 1)       # H0: diff >= +margin
        p = max(p_lo, p_hi)
        ci = stats.t.interval(0.90, n - 1, mean, se)
        verdict = "EQUIVALENT" if p < 0.05 else "not shown equivalent"
        print(f"  {attr:8} diff={mean*100:+5.2f} pp  90% CI "
              f"[{ci[0]*100:+5.2f}, {ci[1]*100:+5.2f}]  TOST p={p:.3g}  -> {verdict}")


def main():
    for prefix, label in (("s2", "Qwen2.5-Omni"), ("s2q3", "Qwen3-Omni-30B")):
        df = build(prefix)
        if df.empty:
            print(f"\n### {label}: no data yet")
            continue
        print(f"\n{'='*70}\n### {label}   items={df.video.nunique()} rows={len(df)}")
        por_by_cell(df)
        fit(df, label)
        tost_source(df, label)


if __name__ == "__main__":
    main()
