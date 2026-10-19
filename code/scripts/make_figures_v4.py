#!/usr/bin/env python3
"""Figures for manuscript v4. Two, both carrying results that no table states as compactly.

fig1  the identified decomposition: alpha (direction-free) and beta (claim-aligned) against
      declared confidence, five models, video-clustered bootstrap bands. This replaces the
      accuracy-panel figure, whose flat Audio-Flamingo-3 curve was an artefact of a bounded
      outcome near a decision boundary.
fig2  what a designer can do: accuracy against detector precision for each policy, per model,
      with the omission baseline as the reference line.

Palette #2a78d6 / #eb6834 / #1baf7a / #6b3fa0 / #b8860b, checked with the dataviz validator;
every series is also directly labelled so identity never rests on colour alone.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from margin_decomp_v1 import LAD, decompose  # noqa: E402
from mitigation_v1 import POLICIES, build, score  # noqa: E402

OUT = Path("/workspace/svhalluc/claude_track2/paper")
W = 3.4
BLUE, ORANGE, GREEN, PURPLE, GOLD = "#2a78d6", "#eb6834", "#1baf7a", "#6b3fa0", "#b8860b"
INK, MUTED, LINE, SURF = "#1b1b1b", "#6b6b6b", "#d5d5d5", "#ffffff"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.2,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "pdf.fonttype": 42,
})

MODELS = [("s2", "cs_qwen25", "Qwen2.5", BLUE, "o"),
          ("s2q3", "cs_qwen3", "Qwen3", ORANGE, "s"),
          ("s2vl", "cs_vllama", "VideoLLaMA", GREEN, "^"),
          ("nm_qwen2audio", "nm_qwen2audio", "Qwen2-Aud", PURPLE, "D"),
          ("nm_af3", "nm_af3", "AF3", GOLD, "v")]
XT = ["v.lo", "lo", "mod", "hi", "v.hi"]


def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.85), sharex=True)
    for base, lad, name, col, mk in MODELS:
        r = decompose(base, lad, boots=600)
        if r is None:
            continue
        point, ci, _ = r
        for ax, idx in ((axes[0], 0), (axes[1], 1)):
            y = [point[c][idx] for c in LAD]
            lo = [ci[c][idx][0] for c in LAD]
            hi = [ci[c][idx][1] for c in LAD]
            ax.fill_between(range(5), lo, hi, color=col, alpha=0.16, lw=0)
            ax.plot(range(5), y, "-", color=col, marker=mk, ms=2.8, lw=1.0,
                    label=name if idx == 1 else None)
    for ax, ttl, ylab in ((axes[0], r"$\alpha$: direction-free", "margin change (log-odds)"),
                          (axes[1], r"$\beta$: claim-aligned", None)):
        ax.axhline(0, color=MUTED, lw=0.7, ls=(0, (3, 2)))
        ax.set_title(ttl, pad=3)
        ax.set_xticks(range(5))
        ax.set_xticklabels(XT, rotation=40, ha="right")
        ax.grid(color=LINE, lw=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if ylab:
            ax.set_ylabel(ylab)
    axes[1].legend(loc="upper left", frameon=False, handletextpad=0.3, borderpad=0.1,
                   labelspacing=0.18, handlelength=1.2)
    fig.text(0.55, -0.04, "declared confidence", ha="center", fontsize=7)
    fig.tight_layout(pad=0.2, w_pad=0.7)
    fig.savefig(OUT / "fig1_decomp.pdf", bbox_inches="tight")
    plt.close(fig)


def fig2():
    """Plotted as gain over omitting the metadata, so every panel shares one scale and the
    design question -- when is forwarding worth it? -- is read off the zero line."""
    PIS = np.arange(0.55, 0.985, 0.025)
    shown = [("always-high", ORANGE, "-"), ("threshold@0.8", BLUE, "-"),
             ("two-pass", GREEN, "-"), ("oracle", INK, (0, (2, 2)))]
    cand = [m for m in MODELS if m[0] != "s2vl"]
    fig, axes = plt.subplots(1, len(cand), figsize=(W, 1.55), sharey=True)
    for ax, (base, lad, name, _, _) in zip(axes, cand):
        d = build(base, lad)
        if d is None:
            continue
        b, T, F, clips = d
        base_y = [score(b, T, F, clips, "omit", p) for p in PIS]
        for pol, col, ls in shown:
            ys = [score(b, T, F, clips, pol, p) - z for p, z in zip(PIS, base_y)]
            ax.plot(PIS, ys, ls=ls, color=col, lw=1.0)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(name, pad=2, fontsize=6.4)
        ax.set_xticks([0.6, 0.8, 0.95])
        ax.set_xticklabels(["0.6", "0.8", ".95"], fontsize=5.4)
        ax.grid(color=LINE, lw=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("gain over omitting (pts)")
    axes[0].set_ylim(-32, 20)
    axes[0].text(0.575, 2.0, "forward", fontsize=4.8, color=MUTED)
    axes[0].text(0.575, -6.0, "omit", fontsize=4.8, color=MUTED)
    for pol, col, ls in shown:
        axes[-1].plot([], [], ls=ls, color=col, lw=1.0, label=pol)
    axes[-1].legend(loc="lower right", frameon=False, fontsize=5.0, handletextpad=0.3,
                    borderpad=0.1, labelspacing=0.16, handlelength=1.3)
    fig.text(0.55, -0.06, "detector precision $\\pi$", ha="center", fontsize=7)
    fig.tight_layout(pad=0.2, w_pad=0.3)
    fig.savefig(OUT / "fig2_policy.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1(); print("fig1_decomp.pdf")
    fig2(); print("fig2_policy.pdf")
    for f in ("fig1_decomp", "fig2_policy"):
        print(f"  {f}.pdf {os.path.getsize(OUT / f'{f}.pdf')/1024:.0f} KB")
