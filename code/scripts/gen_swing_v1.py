#!/usr/bin/env python3
"""The claim-aligned contrast swing under Confidence and under a non-epistemic field.

Two defects motivated this. The body quoted the swing from the unswapped pass while every table
around it is order-averaged, and the sentence immediately after quoted a *different* quantity ---
the Confidence-minus-Priority difference on the false branch alone --- as though it were the
paired version of the same number. One quantity, order-averaged, with an interval.
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np

EPS, BOOTS = 1e-6, 2000
CELLS = [("Qwen2.5", "s2", "sound"), ("Qwen3", "s2q3", "sound"),
         ("Qwen3/noise", "s2q3", "noise"),
         ("VideoLLaMA2.1", "s2vl", "colour"), ("Qwen2-Audio", "nm_qwen2audio", "sound"),
         ("AF3", "nm_af3", "sound"), ("Nemotron/snd", "nm_nemotron", "sound"),
         ("Nemotron/col", "nm_nemotron", "colour"),
         # wave14 gave the battery to the last three admitted cells, so the field-name result
         # now covers every cell that clears the gate -- there is no selection left to state
         ("Qwen2.5/col", "s2", "colour"), ("Qwen2.5/gnoise", "s2", "grad_noise_g4"),
         ("Qwen3/col", "s2q3", "colour")]


def Lb(n):
    return L(n)


def L(n):
    p = f"outputs/{n}_v1.jsonl"
    if not os.path.exists(p):
        return None
    d = {}
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            d[r["clip"]] = r
    return d if len(d) >= 400 else None


def marg(r, sw):
    pa = min(max(r["p_A"], EPS), 1 - EPS)
    v = (np.log(pa) - np.log(1 - pa)) * (1 if r["label"] == "A" else -1)
    return -v if sw else v


def video(c):
    return c.split("/")[-1].split("_")[0]


def per_item(pfx, attr, t_hi, f_hi, t_lo, f_lo):
    """Per-order per-item swing of the claim-aligned contrast."""
    out = []
    for tag in ("", "_swap"):
        sw = tag == "_swap"
        b = L(f"{pfx}_{attr}_none{tag}")
        # Nemotron's unswapped true-high pass was written under a `_T` suffix by the
        # confidence-scale battery; the block text is identical, which is what makes it the
        # same condition.
        g = {k: (L(f"{pfx}_{attr}_{k}{tag}") or L(f"{pfx}_{attr}_{k}_T{tag}")
                 or L(f"{pfx}_{attr}_{k}{tag}_T")) for k in (t_hi, f_hi, t_lo, f_lo)}
        if b is None or any(v is None for v in g.values()):
            continue
        ks = [k for k in b if b[k]["correct"] and all(k in v for v in g.values())]
        if len(ks) < 30:
            continue
        out.append({k: (0.5 * (marg(g[t_hi][k], sw) - marg(g[f_hi][k], sw))
                        - 0.5 * (marg(g[t_lo][k], sw) - marg(g[f_lo][k], sw))) for k in ks})
    return out


def boot(per_a, per_b):
    """Paired difference of two swings, resampling source videos once for both."""
    ks = [sorted(set(a) & set(b)) for a, b in zip(per_a, per_b)]
    vids = sorted({video(k) for kk in ks for k in kk})
    vi = {v: i for i, v in enumerate(vids)}
    arrs = []
    for a, b, kk in zip(per_a, per_b, ks):
        d = np.array([a[k] - b[k] for k in kk])
        grp = np.array([vi[video(k)] for k in kk])
        arrs.append((d, [np.where(grp == i)[0] for i in range(len(vids))]))
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(BOOTS):
        sel = rng.integers(0, len(vids), len(vids))
        vals = []
        for d, buckets in arrs:
            idx = np.concatenate([buckets[i] for i in sel if buckets[i].size])
            if idx.size:
                vals.append(d[idx].mean())
        if vals:
            bs.append(float(np.mean(vals)))
    pt = float(np.mean([np.mean([a[k] - b[k] for k in kk]) for a, b, kk in zip(per_a, per_b, ks)]))
    return pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


if __name__ == "__main__":
    print(f"{'cell':16}{'Confidence':>12}{'Priority':>11}{'difference':>12}{'95% CI':>20}{'ratio':>8}")
    SDW = []
    C, P, D, R, IV = [], [], [], [], []
    for name, pfx, attr in CELLS:
        c = per_item(pfx, attr, "true_high", "anon_high", "true_low", "anon_low")
        p_ = per_item(pfx, attr, "priority_true_high", "priority_high",
                      "priority_true_low", "priority_low")
        if not c or not p_ or len(c) != len(p_):
            print(f"{name:16}데이터 부족")
            continue
        mc = float(np.mean([np.mean(list(d.values())) for d in c]))
        mp = float(np.mean([np.mean(list(d.values())) for d in p_]))
        pt, lo, hi = boot(c, p_)
        # The caption's s.d. must be the one a reader can reconcile with the interval.
        # Pooling both answer orders into one s.d. folds in the between-order mean gap, which
        # the two-order-averaged estimate does not carry the same way -- three separate reviews
        # tried that reconciliation and all three failed. Report the WITHIN-order s.d.
        for a, b in zip(c, p_):
            kk = sorted(set(a) & set(b))
            SDW.append(float(np.std([a[k] - b[k] for k in kk], ddof=1)))
        C.append(mc); P.append(mp); D.append(pt); R.append(mc / mp); IV.append((lo, hi))
        print(f"{name:16}{mc:12.2f}{mp:11.2f}{pt:12.2f}"
              f"{f'[{lo:.2f}, {hi:.2f}]':>20}{mc/mp:8.2f}")
    def rng_(x, nd=2):
        return f"{min(x):.{nd}f}--{max(x):.{nd}f}"
    print(f"\n  Confidence {rng_(C)}   Priority {rng_(P)}   difference {rng_(D)}   ratio {rng_(R,1)}")
    # The reviews' single largest demand: the ten headline cells, each with its own
    # Confidence swing, Priority swing, paired difference and interval, on one page. Every
    # number already existed in this loop's locals; only the table rows were missing. A_0 is
    # order-averaged, matching every other table. AF3 is emitted with its sensitivity mark.
    PRETTY = {"sound": "sound", "colour": "colour", "noise": "noise",
              "grad_noise_g4": "noise (g4)"}
    rows_tab = []
    for (nm, pfx, attr), mc, mp, pt, (lo_, hi_) in zip(CELLS, C, P, D, IV):
        b1, b2 = Lb(f"{pfx}_{attr}_none"), Lb(f"{pfx}_{attr}_none_swap")
        a1 = 100 * sum(r["correct"] for r in b1.values()) / len(b1)
        a2 = 100 * sum(r["correct"] for r in b2.values()) / len(b2)
        a0 = 0.5 * (a1 + a2)
        ceil = "$^{c}$" if max(a1, a2) > 95 else ""
        model = nm.split("/")[0].replace("Qwen2.5", "Qwen2.5-Omni").replace(
            "Qwen3", "Qwen3-Omni").replace("AF3", "Audio Flamingo 3$^{s}$")
        rows_tab.append(
            f"{model} & {PRETTY.get(attr, attr)} & {a0:.1f}{ceil} & ${mc:.2f}$ & ${mp:.2f}$ & "
            f"${pt:.2f}$ [{lo_:.2f}, {hi_:.2f}] \\\\")
    Dp = [d for (nm, *_), d in zip(CELLS, D) if "AF3" not in nm]
    Rp = [r for (nm, *_), r in zip(CELLS, R) if "AF3" not in nm]
    pathlib.Path("paper/auto_swing.tex").write_text(
        "% Generated by scripts/gen_swing_v1.py -- do not edit by hand.\n"
        f"\\newcommand{{\\SwingConfMin}}{{{min(C):.2f}}}\n"
        f"\\newcommand{{\\SwingConfMax}}{{{max(C):.2f}}}\n"
        f"\\newcommand{{\\SwingPrioMin}}{{{min(P):.2f}}}\n"
        f"\\newcommand{{\\SwingPrioMax}}{{{max(P):.2f}}}\n"
        "\\newcommand{\\HeadRows}{%\n" + "\n".join(rows_tab) + "\n}\n"
        f"\\newcommand{{\\SwingDiffMin}}{{{min(D):.2f}}}\n"
        f"\\newcommand{{\\SwingDiffMax}}{{{max(D):.2f}}}\n"
        f"\\newcommand{{\\SwingRatioMin}}{{{min(R):.1f}}}\n"
        f"\\newcommand{{\\SwingRatioMax}}{{{max(R):.1f}}}\n"
        # AF3 is the order-sensitive cell and is reported separately; a range the prose calls
        # "the cells" must not be set by it. The ratio range happens to be unaffected, but the
        # paired difference is -- its low end was AF3's 0.42.
        f"\\newcommand{{\\SwingPrimCells}}{{{len(Dp)}}}\n"
        f"\\newcommand{{\\SwingPrimDiffMin}}{{{min(Dp):.2f}}}\n"
        f"\\newcommand{{\\SwingPrimDiffMax}}{{{max(Dp):.2f}}}\n"
        f"\\newcommand{{\\SwingPrimRatioMin}}{{{min(Rp):.1f}}}\n"
        f"\\newcommand{{\\SwingPrimRatioMax}}{{{max(Rp):.1f}}}\n"
        f"\\newcommand{{\\DeltaSdWMin}}{{{min(SDW):.2f}}}\n"
        f"\\newcommand{{\\DeltaSdWMax}}{{{max(SDW):.2f}}}\n"
        # how many headline cells are at ceiling in one order -- a reviewer has to be able to
        # read this off the prose, not reconstruct it by counting superscripts in the table
        f"\\newcommand{{\\CeilHeadN}}{{{sum(1 for r in rows_tab if '^{c}' in r)}}}\n")
    print("  paper/auto_swing.tex 작성")
