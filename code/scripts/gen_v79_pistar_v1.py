#!/usr/bin/env python3
"""Sampling intervals for the bare break-even pi* and for the sweep's observed zero crossing.

Both mock reviews of v78 raised the same defect: the paper stakes its one falsifiable
quantitative prediction on pi*, says the sweep "crosses zero inside each cell's bare-pi*
bracket", and then prints no interval anywhere -- so a reader cannot check the claim, and
"bracket" is left undefined. This adds two source-clustered intervals:

  pi*      = -G_false / (G_true - G_false), resampled over source videos;
  crossing = the linear interpolation of the sweep's accuracy delta through zero, resampled
             over the SAME source-video draw so the two are paired.

Nothing here is a new estimand. pi* and the sweep are both already reported as point
estimates; this only attaches their sampling error, and reports the paired gap
(crossing - pi*), which is the quantity the prediction is actually about.
"""
from __future__ import annotations
import pathlib, sys
sys.path.insert(0, "scripts")
import numpy as np
import analyse_cascade_v2 as C

BOOTS = 2000
CELLS = [("Qwen2.5-Omni", "s2"), ("Qwen3-Omni", "s2q3"),
         ("Qwen2-Audio", "nm_qwen2audio"), ("Nemotron", "nm_nemotron")]
# the sweep grid, as printed in Table 5
LEVELS = [(0.748, "cascadex_plain"), (0.80, "cascademix80_plain"),
          (0.85, "cascademix85_plain"), (0.90, "cascademix90_plain"),
          (0.95, "cascademix95_plain"), (1.00, "bare_true")]


def load_pair(pfx, cond):
    """Per-order arrays of (per-item correctness delta vs no-metadata), plus their keys."""
    per, keys = [], []
    for sfx in ("", "_swap"):
        A, B = C.L(f"{pfx}_sound_{cond}{sfx}"), C.L(f"{pfx}_sound_none{sfx}")
        if A is None or B is None:
            return None, None
        ks = sorted(set(A) & set(B))
        per.append(np.array([A[k]["correct"] - B[k]["correct"] for k in ks], float))
        keys.append(ks)
    return per, keys


def branch_arrays(pfx):
    """Per-order per-item gains of a wholly true and a wholly false bare claim."""
    T, F, keys = [], [], []
    for sfx in ("", "_swap"):
        b = C.L(f"{pfx}_sound_none{sfx}")
        t = C.L(f"{pfx}_sound_bare_true{sfx}")
        f = C.L(f"{pfx}_sound_bare_false{sfx}")
        if None in (b, t, f):
            return None, None, None
        ks = sorted(set(b) & set(t) & set(f))
        base = np.array([b[k]["correct"] for k in ks], float)
        T.append(np.array([t[k]["correct"] for k in ks], float) - base)
        F.append(np.array([f[k]["correct"] for k in ks], float) - base)
        keys.append(ks)
    return T, F, keys


def buckets_for(keys, vi, nv):
    return [[np.where(np.array([vi[C.video(k)] for k in ks]) == i)[0] for i in range(nv)]
            for ks in keys]


def cross_from(pts):
    """Linear interpolation of the sweep through zero. None if it never crosses upward."""
    xs = [lv for lv, _ in LEVELS]
    for i in range(len(pts) - 1):
        if pts[i] < 0 <= pts[i + 1]:
            return xs[i] + (xs[i + 1] - xs[i]) * (-pts[i]) / (pts[i + 1] - pts[i])
    return None


def run(pfx):
    T, F, bkeys = branch_arrays(pfx)
    sweep = [load_pair(pfx, cond) for _, cond in LEVELS]
    if T is None or any(s[0] is None for s in sweep):
        return None
    # one video index shared by every array in this cell, so a bootstrap draw moves
    # pi* and the crossing together -- the gap is what the prediction is about
    allkeys = [k for ks in bkeys for k in ks] + [k for s in sweep for ks in s[1] for k in ks]
    vids = sorted({C.video(k) for k in allkeys})
    vi, nv = {v: i for i, v in enumerate(vids)}, len(vids)
    bT, bF = buckets_for(bkeys, vi, nv), buckets_for(bkeys, vi, nv)
    bS = [buckets_for(s[1], vi, nv) for s in sweep]

    def point(sel=None):
        def m(arrs, bks):
            out = []
            for d, bk in zip(arrs, bks):
                idx = (np.concatenate([bk[i] for i in sel if bk[i].size])
                       if sel is not None else np.arange(d.size))
                if idx.size:
                    out.append(d[idx].mean())
            return float(np.mean(out)) if out else np.nan
        gt, gf = 100 * m(T, bT), 100 * m(F, bF)
        ps = -gf / (gt - gf) if gt > gf else np.nan
        pts = [100 * m(s[0], bk) for s, bk in zip(sweep, bS)]
        return ps, cross_from(pts), pts

    ps0, cr0, pts0 = point()
    rng = np.random.default_rng(0)
    P, X, G = [], [], []
    for _ in range(BOOTS):
        sel = rng.integers(0, nv, nv)
        ps, cr, _ = point(sel)
        if np.isfinite(ps):
            P.append(ps)
        if cr is not None:
            X.append(cr)
        if np.isfinite(ps) and cr is not None:
            G.append(cr - ps)
    q = lambda a, p: float(np.percentile(a, p))
    return dict(pistar=ps0, pi_lo=q(P, 2.5), pi_hi=q(P, 97.5),
                cross=cr0, cr_lo=q(X, 2.5), cr_hi=q(X, 97.5),
                gap=cr0 - ps0, g_lo=q(G, 2.5), g_hi=q(G, 97.5),
                covered=q(X, 2.5) <= ps0 <= q(X, 97.5), pts=pts0)


if __name__ == "__main__":
    res = {}
    hdr = f"{'cell':14}{'pi*':>7}{'95% CI':>18}{'crossing':>10}{'95% CI':>18}{'gap':>8}{'95% CI':>18}  pi* in CI"
    print(hdr)
    for nm, pfx in CELLS:
        r = run(pfx)
        if r is None:
            print(f"{nm:14}데이터 부족"); continue
        res[nm] = r
        ci = "[%.3f, %.3f]" % (r["pi_lo"], r["pi_hi"])
        cx = "[%.3f, %.3f]" % (r["cr_lo"], r["cr_hi"])
        gp = "[%+.3f, %+.3f]" % (r["g_lo"], r["g_hi"])
        print("%-14s%7.3f%18s%10.3f%18s%+8.3f%18s   %s" % (
            nm, r["pistar"], ci, r["cross"], cx, r["gap"], gp,
            "Y" if r["covered"] else "N"))

    gaps = [r["gap"] for r in res.values()]
    cr = [r["cross"] for r in res.values()]
    nin = sum(r["covered"] for r in res.values())
    halfw = [0.5 * (r["pi_hi"] - r["pi_lo"]) for r in res.values()]
    rows = "\n".join(
        f"{nm} & " + " & ".join(f"${p:+.1f}$" for p in r["pts"])
        + f" & {r['pistar']:.2f} [{r['pi_lo']:.2f}, {r['pi_hi']:.2f}] \\\\"
        for nm, r in res.items())
    pathlib.Path("paper/auto_v79.tex").write_text(
        "% Generated by scripts/gen_v79_pistar_v1.py -- do not edit by hand.\n"
        "\\newcommand{\\SweepRowsCI}{%\n" + rows + "\n}\n"
        f"\\newcommand{{\\PiHalfMin}}{{{min(halfw):.2f}}}\n"
        f"\\newcommand{{\\PiHalfMax}}{{{max(halfw):.2f}}}\n"
        f"\\newcommand{{\\CrossGapMin}}{{{min(gaps):+.3f}}}\n"
        f"\\newcommand{{\\CrossGapMax}}{{{max(gaps):+.3f}}}\n"
        f"\\newcommand{{\\CrossGapMinPP}}{{{100*min(gaps):.1f}}}\n"
        f"\\newcommand{{\\CrossGapMaxPP}}{{{100*max(gaps):.1f}}}\n"
        f"\\newcommand{{\\CrossGapAbsMax}}{{{max(abs(g) for g in gaps):.2f}}}\n"
        f"\\newcommand{{\\CrossCoverN}}{{{nin}}}\n"
        f"\\newcommand{{\\GapZeroN}}{{{sum(1 for r in res.values() if r['g_lo'] <= 0 <= r['g_hi'])}}}\n"
        f"\\newcommand{{\\CrossMin}}{{{min(cr):.2f}}}\n"
        f"\\newcommand{{\\CrossMax}}{{{max(cr):.2f}}}\n"
        f"\\newcommand{{\\CrossCells}}{{{len(res)}}}\n")
    print(f"\n  gap {100*min(gaps):.1f}--{100*max(gaps):.1f} pp, pi* inside crossing CI in {nin}/{len(res)}")
    print("  paper/auto_v79.tex 작성")
