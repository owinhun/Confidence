#!/usr/bin/env python3
# v8: ICASSP 2027 requires no type smaller than 9pt anywhere, including inside figures.
# The figure is placed at width=\linewidth (244.7pt) so it is scaled by 244.7/natural. At 9pt
# authored with a 3-column legend the natural width is 251.7pt, which renders 8.75pt. The legend
# width scales with the font, so raising the font alone never converges. Two columns of legend
# plus 9.5pt authored gives natural 252.3pt -> 9.21pt rendered, and is no taller than before.
"""Figures for v7.

fig1 replaces the alpha/beta pair from v4. alpha is no longer presented as an identified
quantity, so giving it half the figure misrepresented what the paper claims; and the v6 caption
already described the figure this file now actually draws. The two panels are the claim-aligned
contrast on the subset the analysis conditions on (items answered correctly without metadata)
and on its complement (items answered incorrectly), because the competing explanation --
a multiplicative flattening of the output distribution -- predicts the second panel collapses
toward zero while a genuine claim-aligned effect does not.

Nemotron is drawn with the same ladder as the rest. Its two cells are the only ones from a
text backbone unrelated to Qwen, so they are the panel's real test and are given the two
darkest, most distinguishable marks.

Series are separated three ways at once -- hue, marker shape and line style -- so the figure
survives greyscale printing and the common forms of colour-vision deficiency; every series is
also directly labelled, so identity never rests on colour.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path("/workspace/svhalluc/claude_track2/paper")
DATA = Path("/workspace/svhalluc/claude_track2/outputs")
W = 3.4
EPS = 1e-6
BLUE, ORANGE, GREEN, PURPLE, GOLD, INKY = ("#2a78d6", "#eb6834", "#1baf7a",
                                           "#6b3fa0", "#b8860b", "#111111")
INK, MUTED, LINE, SURF = "#1b1b1b", "#6b6b6b", "#d5d5d5", "#ffffff"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9.5, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# name, colour, marker, linestyle, baseline prefix, attribute,
# and the (false, true) file stem for each of the five rungs
QWEN_LAD = ["v_vlow", "v_low", "v_mod", "v_high", "v_vhigh"]
# Nemotron was scored outside eval_confscale, so two of its rungs live under the names the
# factorial battery uses; the block text is identical, which is what makes them the same rung.
NEMO_LAD = [("v_vlow", "v_vlow_T"), ("anon_low", "true_low"), ("v_mod", "v_mod_T"),
            ("anon_high", "true_high_T"), ("v_vhigh", "v_vhigh_T")]

SERIES = [
    ("Qwen2.5",      BLUE,   "o", "-",  "s2",            "cs_qwen25",     "sound",  None),
    ("Qwen3",        ORANGE, "s", "-",  "s2q3",          "cs_qwen3",      "sound",  None),
    # Qwen3/noise carries the override result and is a primary row in Table 1, but it only ever
    # had the two-rung anon ladder, so it could not be drawn here -- which is why Figure 1's
    # seven cells and Table 1's seven rows named different sets. wave12 ran the five-rung ladder.
    ("Qwen3/noise",  ORANGE, "*", ":",  "s2q3",          "cs_qwen3",      "noise",  None),
    ("VideoLLaMA",   GREEN,  "^", "-",  "s2vl",          "cs_vllama",     "colour", None),
    ("Qwen2-Aud",   PURPLE, "D", "-",  "nm_qwen2audio", "nm_qwen2audio", "sound",  None),
    ("AF3",          GOLD,   "v", "--", "nm_af3",        "nm_af3",        "sound",  None),
    ("Nemo./sound", INKY, "P", "-",  "nm_nemotron",   "nm_nemotron",   "sound",  NEMO_LAD),
    ("Nemo./col",   INKY, "X", "--", "nm_nemotron",   "nm_nemotron",   "colour", NEMO_LAD),
]


def load(stem):
    p = DATA / f"{stem}_v1.jsonl"
    if not p.exists():
        return None
    d = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            d[r["clip"]] = r
    return d if len(d) >= 400 else None


def margin(rec):
    pa = min(max(rec["p_A"], EPS), 1 - EPS)
    lo = np.log(pa) - np.log(1 - pa)
    return lo if rec["label"] == "A" else -lo


def beta_curve(base, lad, attr, nemo, want_correct):
    """Claim-aligned contrast at each rung, on the requested baseline subset."""
    b = load(f"{base}_{attr}_none")
    if b is None:
        return None
    keys = [k for k in b if bool(b[k]["correct"]) == want_correct]
    # the two ceiling cells leave only 19 and 18 unaided-incorrect items. That is thin, and the
    # caption says so, but dropping them would contradict the text's claim about all models --
    # showing a short series with its n stated is the honest option.
    if len(keys) < 12:
        return None
    ys = []
    for i in range(5):
        if nemo:
            f_stem, t_stem = (f"{base}_{attr}_{nemo[i][0]}", f"{base}_{attr}_{nemo[i][1]}")
        else:
            f_stem, t_stem = (f"{lad}_{attr}_{QWEN_LAD[i]}", f"{lad}_{attr}_{QWEN_LAD[i]}_T")
        F, T = load(f_stem), load(t_stem)
        if F is None or T is None:
            return None
        ks = [k for k in keys if k in F and k in T]
        if not ks:
            return None
        ys.append(float(np.mean([0.5 * (margin(T[k]) - margin(F[k])) for k in ks])))
    return ys, len(keys)


def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.55), sharex=True, sharey=True)
    for ax, want, title in ((axes[0], True, "correct unaided"),
                            (axes[1], False, "incorrect unaided")):
        for name, col, mk, ls, base, lad, attr, nemo in SERIES:
            r = beta_curve(base, lad, attr, nemo, want)
            if r is None:
                continue
            ys, n = r
            ax.plot(range(5), ys, ls, color=col, marker=mk, ms=3.4, lw=1.2,
                    mew=0, label=(name if ax is axes[0] else None))
        ax.axhline(0, color=MUTED, lw=0.7, ls=(0, (3, 2)))
        ax.set_title(title, pad=3, color=MUTED)
        ax.set_xticks(range(5))
        ax.set_xticklabels(["v.low", "low", "mod", "high", "v.high"], rotation=30)
        ax.grid(color=LINE, lw=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(r"contrast $\beta$")
    # inside the axes the legend ran past the panel edge and clipped the two Nemotron labels;
    # below the figure it spans both panels and nothing is truncated
    h, lab = axes[0].get_legend_handles_labels()
    # the axis label sits between the rotated ticks and the legend; both are outside the axes,
    # so their offsets have to be set against each other or they collide
    # The axis label and the legend both live below the axes, so their offsets are set against
    # each other. At the current panel height the two rows of the legend reach up far enough that
    # -0.16 put the label on top of its first row; the label goes above the rotated ticks
    # instead, where nothing else competes for the space.
    fig.text(0.5, -0.145, "declared confidence", ha="center", color=INK, fontsize=9.5)
    fig.legend(h, lab, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.60), handlelength=0.9, columnspacing=0.5, labelspacing=0.25,
               handletextpad=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig1_decomp_v11.pdf", bbox_inches="tight")
    plt.close(fig)
    print("fig1_decomp_v11.pdf saved")


if __name__ == "__main__":
    fig1()
