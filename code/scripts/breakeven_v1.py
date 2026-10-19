#!/usr/bin/env python3
"""The break-even detector precision, per model and per declared confidence level.

The policy section previously said which policy won at three arbitrary values of pi. The
deployable form of the same fact is a threshold: forwarding a claim declared at level c beats
omitting it exactly when the detector's precision at that level exceeds

    pi*_c = -G_false(c) / (G_true(c) - G_false(c)),

where G_true(c) and G_false(c) are the accuracy gains over omission when the forwarded claim is
respectively true and false. This is a number an engineer can compare against a measured
detector, rather than a curve to read off a plot. pi* outside [0,1] means the comparison never
flips: below 0, forwarding wins at every precision; above 1, omitting does.

The interval comes from a video-clustered bootstrap of both branches jointly, so it carries the
correlation between them rather than treating the two gains as independent.
"""

from __future__ import annotations

import json
import os

import numpy as np

OUT = "outputs"
LAD = [("v_vlow", "very low"), ("v_low", "low"), ("v_mod", "moderate"),
       ("v_high", "high"), ("v_vhigh", "very high")]
MODELS = [("s2", "cs_qwen25", "Qwen2.5-Omni"),
          ("s2q3", "cs_qwen3", "Qwen3-Omni"),
          ("s2vl", "cs_vllama", "VideoLLaMA2.1"),
          ("nm_qwen2audio", "nm_qwen2audio", "Qwen2-Audio"),
          ("nm_af3", "nm_af3", "Audio-Flamingo-3")]
ATTR = {"s2vl": "colour"}   # VideoLLaMA's sound cell is withdrawn; its ladder is on colour
BOOTS = 1500


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


def video(clip):
    return clip.split("/")[-1].split("_")[0]


def breakeven(base, lad, attr):
    b = L(f"{base}_{attr}_none")
    if b is None:
        return None
    out = {}
    for cond, _name in LAD:
        T, F = L(f"{lad}_{attr}_{cond}_T"), L(f"{lad}_{attr}_{cond}")
        if T is None or F is None:
            out[cond] = None
            continue
        ks = [k for k in b if k in T and k in F]
        if len(ks) < 100:
            out[cond] = None
            continue
        base_c = {k: bool(b[k]["correct"]) for k in ks}
        gt = {k: bool(T[k]["correct"]) - base_c[k] for k in ks}
        gf = {k: bool(F[k]["correct"]) - base_c[k] for k in ks}
        vids = sorted({video(k) for k in ks})
        rng = np.random.default_rng(0)

        def pistar(sel):
            """gain(pi) = pi*a + (1-pi)*c, so the sign of the two branch gains decides
            whether a threshold exists at all: both positive -> forward at any precision,
            both negative -> never forward, otherwise the root lies in (0,1)."""
            kk = [k for k in ks if video(k) in sel]
            if not kk:
                return None
            a = 100.0 * np.mean([gt[k] for k in kk])
            c = 100.0 * np.mean([gf[k] for k in kk])
            if a > 0 and c > 0:
                return -np.inf, a, c          # always worth forwarding
            if a <= 0 and c <= 0:
                return np.inf, a, c           # never worth forwarding
            return -c / (a - c), a, c

        pt = pistar(set(vids))
        if pt is None:
            out[cond] = None
            continue
        # cells with no interior threshold contribute +-inf; keep them out of the percentile
        # but record how often a threshold existed at all, so an interval is never quoted as
        # if it summarised resamples that had none
        raw = [r[0] for r in (pistar(set(rng.choice(vids, len(vids), replace=True)))
                              for _ in range(BOOTS)) if r]
        bs = [x for x in raw if np.isfinite(x)]
        lo, hi = ((float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
                  if len(bs) >= 20 else (np.nan, np.nan))
        out[cond] = (pt[0], lo, hi, pt[1], pt[2], len(ks),
                     len(bs) / len(raw) if raw else 0.0)
    return out


def cell(v):
    if v is None:
        return "--".rjust(19)
    p, lo, hi, _a, _c, _n, frac = v
    if p == -np.inf:
        return "always".rjust(19)
    if p == np.inf or p >= 1:
        return "never".rjust(19)
    if p <= 0:
        return "always".rjust(19)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{p:.3f}".rjust(19)
    star = "" if frac > 0.95 else "*"
    return f"{p:.3f} [{lo:.2f},{hi:.2f}]{star}".rjust(19)


if __name__ == "__main__":
    print("Break-even detector precision pi* for forwarding a claim declared at each level.")
    print("'always' = forwarding wins at every precision; 'never' = omitting always wins.\n")
    print(f"{'model':18}" + "".join(n.rjust(19) for _, n in LAD))
    tabs = {}
    for base, lad, name in MODELS:
        attr = ATTR.get(base, "sound")
        r = breakeven(base, lad, attr)
        tabs[name] = r
        if r is None:
            print(f"{name:18}" + "--".rjust(19))
            continue
        print(f"{name:18}" + "".join(cell(r[c]) for c, _ in LAD))

    print("\nUnderlying gains over omission (accuracy points): true branch / false branch")
    print(f"{'model':18}" + "".join(n.rjust(19) for _, n in LAD))
    for _base, _lad, name in MODELS:
        r = tabs.get(name)
        if r is None:
            print(f"{name:18}" + "--".rjust(19))
            continue
        row = ""
        for c, _ in LAD:
            v = r[c]
            row += ("--".rjust(19) if v is None else f"{v[3]:+.1f} / {v[4]:+.1f}".rjust(19))
        print(f"{name:18}{row}")
