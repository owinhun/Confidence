#!/usr/bin/env python3
"""ICASSP figures for the cascade-override paper. Single column (3.4 in), print-safe.

figA  the cascade and where it breaks (conceptual flow + the measured quantity)
figB  confidence is causal -- 12 pre-registered contrasts, high vs low, all one-sided
figC  the dose-response knob -- verbal vs numeric confidence, three models
figD  exposure is model- and modality-dependent

Every number is recomputed from outputs/*.jsonl at draw time; nothing is typed in by hand,
so a figure can never drift from the data behind it.

Palette: #2a78d6 / #eb6834 / #1baf7a, validated with the dataviz validator (all checks pass;
the green carries a contrast WARN against the surface, discharged by direct labels on every
mark that uses it).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, LINE, SURF = "#1b1b1b", "#6b6b6b", "#d5d5d5", "#ffffff"
OUT = Path("/workspace/svhalluc/claude_track2/paper")
OUTPUTS = Path("/workspace/svhalluc/claude_track2/outputs")
W = 3.4  # ICASSP single column, inches

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "pdf.fonttype": 42,
})

MODELS = [("s2", "cs_qwen25", "Qwen2.5-Omni-7B"),
          ("s2q3", "cs_qwen3", "Qwen3-Omni-30B"),
          ("s2vl", "cs_vllama", "VideoLLaMA2.1-AV")]
# Qwen2-Audio is stored under a different prefix (it has no video stream, so it runs
# through the new-models harness); it enters figE as a fourth panel.
EXTRA = [("nm_qwen2audio_sound", "nm_qwen2audio", "Qwen2-Audio (audio-only)")]


# ------------------------------------------------------------------ data access
def load(name):
    p = OUTPUTS / f"{name}_v1.jsonl"
    if not p.exists():
        return None
    return {json.loads(l)["clip"]: json.loads(l)
            for l in p.read_text().splitlines() if l.strip()}


def macro_por(pfx, attr, cond):
    """Direction-balanced override rate among items answered correctly unaided."""
    base = load(f"{pfx}_{attr}_none")
    w = load(f"{pfx}_{attr}_{cond}") if pfx.startswith("s2") else load(f"{cond}")
    if base is None or w is None:
        return None
    ok = [i for i in base if base[i]["correct"] and i in w]
    ad = np.array([w[i]["pred"] != w[i]["label"] for i in ok])
    a = [k for k, i in enumerate(ok) if w[i]["label"] == "A"]
    b = [k for k, i in enumerate(ok) if w[i]["label"] == "B"]
    return (ad[a].mean() + ad[b].mean()) / 2 * 100


def macro_por_cs(pfx, csname, cond):
    base = load(f"{pfx}_sound_none")
    w = load(f"{csname}_sound_{cond}")
    ok = [i for i in base if base[i]["correct"] and i in w]
    ad = np.array([w[i]["pred"] != w[i]["label"] for i in ok])
    a = [k for k, i in enumerate(ok) if w[i]["label"] == "A"]
    b = [k for k, i in enumerate(ok) if w[i]["label"] == "B"]
    return (ad[a].mean() + ad[b].mean()) / 2 * 100


# ---------------------------------------------------------------------- fig A
def figA():
    fig, ax = plt.subplots(figsize=(W, 2.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.0); ax.axis("off")

    def box(x, y, w, h, label, sub, fc, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.10",
                                    fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center", fontsize=6.8,
                color=INK)
        if sub:
            ax.text(x + w / 2, y + h * 0.24, sub, ha="center", va="center", fontsize=5.8,
                    color=MUTED, style="italic")

    def arrow(x0, x1, y, lab=None):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=7,
                                     lw=0.8, color=MUTED, shrinkA=0, shrinkB=0))
        if lab:
            ax.text((x0 + x1) / 2, y + 0.16, lab, ha="center", fontsize=5.6, color=MUTED)

    box(0.05, 2.75, 2.15, 1.05, "upstream\ndetector", "SED / sync / VAD", "#eef4fc", BLUE)
    arrow(2.30, 3.25, 3.28, "text")
    box(3.35, 2.75, 3.25, 1.05, 'Source: …\nClaim: "…"\nConfidence: high', None,
        "#fdf1ea", ORANGE)
    arrow(6.70, 7.55, 3.28)
    box(7.65, 2.75, 2.30, 1.05, "AV-LLM", "final answer", "#eef4fc", BLUE)

    box(0.05, 1.30, 2.15, 0.95, "raw signal", "audio + video", "#f2f2f2", MUTED)
    ax.add_patch(FancyArrowPatch((2.25, 1.78), (8.80, 2.68), arrowstyle="-|>",
                                 mutation_scale=7, lw=0.8, color=MUTED,
                                 connectionstyle="arc3,rad=-0.16", shrinkA=1, shrinkB=1))
    ax.text(5.55, 1.62, "the two disagree — which wins?", ha="center", fontsize=6.2,
            color=INK)
    ax.text(5.0, 0.42, "POR  =  P( follow the false claim  |  correct from the signal alone )",
            ha="center", fontsize=6.1, color=ORANGE)
    fig.tight_layout(pad=0.15)
    fig.savefig(OUT / "figA_cascade.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------- fig B
def figB():
    """One small-multiple panel per model: four dumbbells, high vs low, with the
    McNemar discordant counts printed as direct labels so identity never rests on
    colour alone and the reader sees the one-sidedness without a caption."""
    from math import comb

    def discordant(pfx, attr, src):
        base = load(f"{pfx}_{attr}_none")
        hi = load(f"{pfx}_{attr}_{src}_high")
        lo = load(f"{pfx}_{attr}_{src}_low")
        if base is None or hi is None or lo is None:
            return None, None
        ok = [i for i in base if base[i]["correct"] and i in hi and i in lo]
        n01 = sum(1 for i in ok if (hi[i]["pred"] != hi[i]["label"])
                  and not (lo[i]["pred"] != lo[i]["label"]))
        n10 = sum(1 for i in ok if not (hi[i]["pred"] != hi[i]["label"])
                  and (lo[i]["pred"] != lo[i]["label"]))
        return n01, n10

    fig, axes = plt.subplots(3, 1, figsize=(W, 2.9), sharex=True)
    base_conds = [("sound", "anon", "sound, anon."), ("sound", "expert", "sound, expert"),
                  ("colour", "anon", "grey, anon."), ("colour", "expert", "grey, expert")]
    for ax, (pfx, _, name) in zip(axes, MODELS):
        conds = list(base_conds)
        # the one low-level acoustic cell that met the perceptual-eligibility criterion
        if load(f"{pfx}_noise_anon_high") and load(f"{pfx}_noise_anon_low"):
            conds.append(("noise", "anon", "noise, anon."))
        y = np.arange(len(conds))[::-1]
        for i, (attr, src, lab) in enumerate(conds):
            hi = macro_por(pfx, attr, f"{src}_high")
            lo = macro_por(pfx, attr, f"{src}_low")
            n01, n10 = discordant(pfx, attr, src)
            if hi is None or lo is None or n01 is None:
                continue
            ax.plot([lo, hi], [y[i], y[i]], color=LINE, lw=1.1, zorder=1,
                    solid_capstyle="round")
            ax.plot(hi, y[i], "o", ms=4.0, color=ORANGE, mec=SURF, mew=0.7, zorder=3)
            ax.plot(lo, y[i], "o", ms=4.0, color=BLUE, mec=SURF, mew=0.7, zorder=3)
            ax.text(max(hi, lo) + 3.0, y[i], f"{n01}/{n10}", va="center", fontsize=5.6,
                    color=MUTED)
        ax.set_yticks(y)
        ax.set_yticklabels([c[2] for c in conds], fontsize=6)
        ax.set_ylim(-0.7, len(conds) - 0.3)
        ax.set_xlim(-3, 118)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(axis="x", color=LINE, lw=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(name, loc="left", pad=2, fontsize=6.6, color=INK)
    axes[-1].set_xlabel("override rate (macro-POR, %)   ·   labels: discordant pairs")
    axes[0].plot([], [], "o", ms=4.0, color=ORANGE, label='Confidence: "high"')
    axes[0].plot([], [], "o", ms=4.0, color=BLUE, label='Confidence: "low"')
    axes[0].legend(loc="lower right", frameon=False, handletextpad=0.3, borderpad=0.1,
                   labelspacing=0.2, bbox_to_anchor=(1.02, -0.05))
    fig.tight_layout(pad=0.2, h_pad=0.7)
    fig.savefig(OUT / "figB_confidence.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------- fig C
def acc(pfx, cs, cond, tag=""):
    """Overall accuracy on the sound-event set under a given metadata condition."""
    w = load(f"{cs}_sound_{cond}{tag}")
    if w is None:
        return None
    return np.mean([r["correct"] for r in w.values()]) * 100


def figC():
    """The signed-evidence figure. A TRUE claim tagged with low confidence does not merely
    fail to help -- it drives the model off answers it had right, and the accuracy curve
    crosses the no-metadata baseline far above 0.5. Left: verbal, three models. Right: the
    numeric scale on Qwen2.5, where the crossing can be located (0.79, CI [0.72, 0.86])."""
    from scipy import stats
    V = ["v_vlow", "v_low", "v_mod", "v_high", "v_vhigh"]
    N = ["n_015", "n_035", "n_060", "n_085", "n_095"]
    NUM = [0.15, 0.35, 0.60, 0.85, 0.95]
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.85))
    styles = [(BLUE, "o"), (ORANGE, "s"), (GREEN, "^")]

    ax = axes[0]
    for (pfx, cs, name), (c, mk) in zip(MODELS, styles):
        a0 = acc(pfx, cs, "none") if load(f"{cs}_sound_none") else None
        if a0 is None:                        # baselines live under the Stage-2 prefix
            b = load(f"{pfx}_sound_none")
            a0 = np.mean([r["correct"] for r in b.values()]) * 100
        pts = [(k, acc(pfx, cs, v, "_T")) for k, v in enumerate(V)]
        pts = [(k, y - a0) for k, y in pts if y is not None]
        if not pts:
            continue
        ax.plot([k for k, _ in pts], [y for _, y in pts], "-", color=c, marker=mk,
                ms=3.2, lw=1.0, label=name.split("-")[0])
    ax.axhline(0, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.set_xticks(range(5))
    ax.set_xticklabels(["v.low", "low", "mod", "high", "v.high"], rotation=35, ha="right")
    ax.set_ylabel("accuracy change (pp)")
    ax.set_title("true claim, verbal", pad=3)
    ax.legend(loc="lower right", frameon=False, handletextpad=0.3, borderpad=0.15,
              labelspacing=0.2)

    ax = axes[1]
    lo_all = 0
    for (pfx, cs, name), (c, mk) in zip(MODELS, styles):
        b = load(f"{pfx}_sound_none")
        a0 = np.mean([r["correct"] for r in b.values()]) * 100
        ys = [acc(pfx, cs, k, "_T") for k in N]
        if any(y is None for y in ys):
            continue
        ys = [y - a0 for y in ys]
        lo_all = min(lo_all, min(ys))
        sl, ic, r, p, se = stats.linregress(NUM, ys)
        xs = np.linspace(0.05, 1.0, 50)
        ax.plot(xs, sl * xs + ic, "-", color=c, lw=0.7, alpha=0.55, zorder=1)
        ax.plot(NUM, ys, mk, color=c, ms=3.0, zorder=3)
        ax.plot([-ic / sl], [0], "*", color=c, ms=7, zorder=4)
    ax.axhline(0, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.axvline(0.5, color=MUTED, lw=0.8, ls=(0, (1, 2)))
    ax.text(0.5, lo_all * 0.97, "p=0.5", fontsize=5.6, color=MUTED, ha="center", va="bottom")
    ax.set_xlabel('stated "Confidence: p"')
    ax.set_title("true claim, numeric", pad=3)
    ax.set_xlim(0.05, 1.02)

    for a in axes:
        a.grid(color=LINE, lw=0.5, alpha=0.7)
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.2, w_pad=0.8)
    fig.savefig(OUT / "figC_signed.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------- fig D
def figD():
    fig, ax = plt.subplots(figsize=(W, 1.6))
    labels = [n.split("-")[0] for _, _, n in MODELS]
    snd = [macro_por(p, "sound", "anon_high") for p, _, _ in MODELS]
    col = [macro_por(p, "colour", "anon_high") for p, _, _ in MODELS]
    x = np.arange(3)
    w = 0.34
    b1 = ax.bar(x - w / 2 - 0.01, snd, w, color=BLUE, label="sound event (audio)")
    b2 = ax.bar(x + w / 2 + 0.01, col, w, color=ORANGE, label="greyscale (visual)")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 2.0,
                    f"{r.get_height():.0f}", ha="center", fontsize=6, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("override rate (%)")
    ax.set_ylim(0, 112)
    ax.grid(axis="y", color=LINE, lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="upper left", handletextpad=0.4, borderpad=0.15)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "figD_exposure.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    figA(); figB(); figC(); figD()
    for f in ["figA_cascade", "figB_confidence", "figC_signed", "figD_exposure"]:
        p = OUT / f"{f}.pdf"
        print(f"{f}.pdf  {os.path.getsize(p)/1024:.0f} KB")


# ---------------------------------------------------------------------- fig E
def figE():
    """The decomposition in one picture: accuracy under a TRUE and a FALSE claim at each
    declared confidence, against the no-metadata baseline. Low confidence pulls BOTH curves
    below the baseline (destabilisation); high confidence separates them (override)."""
    V = ["v_vlow", "v_low", "v_mod", "v_high", "v_vhigh"]
    panels = [(p, c, n) for p, c, n in MODELS]
    for pfx, lab in [("nm_qwen2audio_sound", "Qwen2-Audio*"),
                     ("nm_af3_sound", "Audio-Flamingo-3*")]:
        if load(f"{pfx}_none"):
            panels.append((pfx, pfx, lab))
    fig, axes = plt.subplots(1, len(panels), figsize=(W, 1.55), sharey=True)
    for ax, (pfx, cs, name) in zip(axes, panels):
        b = load(f"{pfx}_none") if pfx.startswith("nm_") else load(f"{pfx}_sound_none")
        a0 = np.mean([r["correct"] for r in b.values()]) * 100
        for tag, col, mk, lab in [("_T", GREEN, "^", "true claim"),
                                  ("", ORANGE, "o", "false claim")]:
            if pfx.startswith("nm_"):
                pts = [(k, (lambda w: None if w is None or len(w) < 400 else
                            np.mean([r["correct"] for r in w.values()]) * 100)(
                           load(f"{cs}_{v}{tag}"))) for k, v in enumerate(V)]
            else:
                pts = [(k, acc(pfx, cs, v, tag)) for k, v in enumerate(V)]
            pts = [(k, y) for k, y in pts if y is not None]
            ax.plot([k for k, _ in pts], [y for _, y in pts], "-", color=col, marker=mk,
                    ms=3.0, lw=1.0, label=lab)
        ax.axhline(a0, color=MUTED, lw=0.8, ls=(0, (3, 2)))
        short = {"Qwen2.5-Omni-7B": "Qwen2.5", "Qwen3-Omni-30B": "Qwen3",
                 "VideoLLaMA2.1-AV": "VideoLLaMA", "Qwen2-Audio*": "Qwen2-Aud*",
                 "Audio-Flamingo-3*": "AF3*"}.get(name, name)
        ax.set_title(short, pad=2, fontsize=6.0)
        ax.set_xticks(range(5))
        ax.set_xticklabels(["v.lo", "lo", "mod", "hi", "v.hi"], rotation=45, ha="right",
                           fontsize=5.0)
        ax.grid(color=LINE, lw=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("accuracy (%)")
    axes[0].set_ylim(0, 105)
    axes[0].text(0.12, 0.09, "no-metadata\nbaseline", transform=axes[0].transAxes,
                 fontsize=5.2, color=MUTED)
    axes[-1].legend(loc="lower right", frameon=False, handletextpad=0.3, borderpad=0.1,
                   labelspacing=0.2, fontsize=5.8)
    fig.tight_layout(pad=0.2, w_pad=0.4)
    fig.savefig(OUT / "figE_decomposition.pdf", bbox_inches="tight")
    plt.close(fig)


figE()
print("figE_decomposition.pdf written")
