#!/usr/bin/env python3
"""ICASSP figures for the audit+interface paper (single-column, print-friendly).

fig1: the three-tier interface (conceptual flow: signal -> encoders discard ->
      expert recovers -> language interface -> tiers).
fig2: main results grouped bars (two benchmarks x four conditions) with the
      expert-only reference line.
Palette follows the dataviz skill defaults: #2a78d6 (blue), #eb6834 (orange),
#1baf7a (green), plus neutral grays. All numbers from outputs/log_v1.md.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, LINE = "#1b1b1b", "#6b6b6b", "#d0d0d0"
OUT = "/workspace/svhalluc/claude_track2/paper"

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.2,
    "axes.edgecolor": LINE, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.labelcolor": INK,
    "figure.dpi": 300,
})


def fig1():
    fig, ax = plt.subplots(figsize=(3.35, 2.15))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=6.6, tc=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    linewidth=0.7, facecolor=fc, edgecolor=ec))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, linespacing=1.25)

    def arrow(x1, y1, x2, y2, color=MUTED, lw=0.8):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=6, linewidth=lw,
                                     color=color, shrinkA=1, shrinkB=1))

    # ---- row 1: the failure path
    box(0.15, 4.75, 1.85, 1.05, "raw A/V\nclip", "#ffffff", LINE)
    box(2.50, 4.75, 2.05, 1.05, "AVLLM\nencoders", "#f4f4f4", LINE)
    box(5.05, 4.75, 1.85, 1.05, "frozen\nLLM", "#f4f4f4", LINE)
    box(7.40, 4.75, 2.45, 1.05, "chance\n49.9 / 50.8%", "#fdecec", "#e0b4b4",
        tc="#a33")
    arrow(2.00, 5.27, 2.50, 5.27)
    arrow(4.55, 5.27, 5.05, 5.27)
    arrow(6.90, 5.27, 7.40, 5.27)
    ax.text(3.52, 4.42, "synchrony discarded", ha="center", fontsize=6.0,
            color="#a33", style="italic")

    # ---- row 2: the repair path
    box(0.15, 2.55, 1.85, 1.05, "frozen sync\nexpert", "#eaf2fc", BLUE, tc=BLUE)
    box(2.50, 2.55, 2.05, 1.05, "verbalized\nevidence", "#eaf2fc", BLUE, tc=BLUE)
    arrow(1.08, 4.75, 1.08, 3.60, color=BLUE)
    arrow(2.00, 3.07, 2.50, 3.07, color=BLUE)
    arrow(4.55, 3.07, 5.97, 4.75, color=BLUE)
    ax.text(5.75, 3.35, "language interface", ha="left", fontsize=6.0,
            color=BLUE, style="italic")

    # ---- row 3: the three tiers
    box(0.15, 0.35, 3.05, 0.95, "verdict\n88.8 / 71.5%", "#eafaf3", GREEN,
        fs=6.2, tc="#127a55")
    box(3.45, 0.35, 3.05, 0.95, "probability\n82.7 / 66.6%", "#fdf0e9", ORANGE,
        fs=6.2, tc="#b4491c")
    box(6.75, 0.35, 3.10, 0.95, "+10M adapter\n88.3 / 72.1%", "#eaf2fc", BLUE,
        fs=6.2, tc=BLUE)
    ax.text(5.0, 1.55, "evidence format $\\rightarrow$ VGGSound-Sparse / SVHalluc TA accuracy",
            ha="center", fontsize=5.9, color=MUTED)

    fig.tight_layout(pad=0.15)
    fig.savefig(f"{OUT}/fig1_interface.pdf", bbox_inches="tight")
    print("fig1 written")


def fig2():
    import numpy as np
    conds = ["frozen", "verdict", "probability", "prob.+adapter"]
    vggs = [49.9, 88.8, 82.7, 88.3]
    ta = [50.8, 71.5, 66.6, 72.1]
    expert = [88.8, 71.2]
    colors = ["#bcbcbc", GREEN, ORANGE, BLUE]

    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.75), sharey=True)
    for ax, vals, ref, title in zip(axes, [vggs, ta], expert,
                                    ["VGGSound-Sparse", "SVHalluc TA"]):
        x = np.arange(len(conds))
        ax.bar(x, vals, width=0.62, color=colors, linewidth=0)
        ax.axhline(ref, color=INK, linestyle=(0, (3, 2)), linewidth=0.7)
        ax.text(len(conds) - 0.35, ref, " expert-only", fontsize=5.6,
                color=INK, ha="left", va="center")
        ax.axhline(50, color=MUTED, linewidth=0.5)
        ax.text(len(conds) - 0.35, 50, " chance", fontsize=5.6, color=MUTED,
                ha="left", va="center")
        ax.set_xlim(-0.65, len(conds) + 0.55)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 1.4, f"{v:.1f}", ha="center", fontsize=5.8,
                    color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(conds, rotation=28, ha="right", fontsize=5.9)
        ax.set_ylim(40, 100)
        ax.set_title(title, fontsize=6.8, pad=3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2, width=0.5)
    axes[0].set_ylabel("accuracy (%)", fontsize=6.6)
    fig.tight_layout(pad=0.2, w_pad=0.8)
    fig.savefig(f"{OUT}/fig2_main.pdf", bbox_inches="tight")
    print("fig2 written")




def fig3():
    import numpy as np
    deltas = [0.2, 0.4, 0.8, 1.2, 2.0]
    expert = [56.4, 72.7, 88.0, 88.9, 88.8]
    verdict = [56.4, 72.7, 88.0, 88.9, 88.8]
    adapter = [57.1, 72.8, 88.3, 88.3, 88.3]

    fig, ax = plt.subplots(figsize=(3.35, 1.65))
    ax.plot(deltas, expert, marker="s", ms=3.4, lw=1.6, color="#8a8a8a",
            label="expert-only", zorder=2)
    ax.plot(deltas, verdict, marker="o", ms=3.0, lw=1.1, color=GREEN,
            label="verdict injection", zorder=3)
    ax.plot(deltas, adapter, marker="^", ms=3.0, lw=1.1, color=BLUE,
            label="+ adapter", zorder=3)
    ax.axhline(50, color=MUTED, lw=0.5)
    ax.text(0.12, 51.0, "chance", fontsize=5.6, color=MUTED, va="bottom")
    ax.set_xlabel("audio-visual offset $\\delta$ (s)", fontsize=6.6)
    ax.set_ylabel("accuracy (%)", fontsize=6.6)
    ax.set_xlim(0.05, 2.35); ax.set_ylim(45, 95)
    ax.set_xticks(deltas); ax.set_xticklabels([str(d) for d in deltas], fontsize=6.0)
    ax.tick_params(length=2, width=0.5, labelsize=6.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=5.8, frameon=False, loc="center right",
              bbox_to_anchor=(1.0, 0.42), handlelength=1.6, borderaxespad=0.2)
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{OUT}/fig3_offset.pdf", bbox_inches="tight")
    print("fig3 written")




def fig4():
    """Override grid: native vs post-false-statement accuracy, 3 models x attributes."""
    import numpy as np
    models = ["Q2.5", "VL2.1", "Q3"]
    attrs = ["greyscale", "silence gap", "sound event"]
    native = np.array([[99.8, 75.5, 94.3], [80.8, 51.5, 94.3], [99.6, 79.2, 90.8]])
    after = np.array([[42.5, 0.0, 43.9], [15.5, 0.2, 38.5], [94.0, 0.0, 30.4]])

    fig, axes = plt.subplots(1, 3, figsize=(3.35, 1.6), sharey=True)
    x = np.arange(len(models))
    for k, ax in enumerate(axes):
        n, a = native[:, k], after[:, k]
        ax.bar(x - 0.19, n, width=0.36, color="#bcbcbc", linewidth=0, label="no evidence")
        ax.bar(x + 0.19, a, width=0.36, color=ORANGE, linewidth=0,
               label="one false statement")
        for xi in x:
            if not np.isnan(a[xi]):
                ax.annotate("", xy=(xi + 0.19, a[xi] + 2), xytext=(xi - 0.19, n[xi] + 2),
                            arrowprops=dict(arrowstyle="->", lw=0.6, color=INK,
                                            shrinkA=0, shrinkB=0))
        ax.axhline(50, color=MUTED, lw=0.5)
        ax.set_title(attrs[k], fontsize=6.6, pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=0, fontsize=6.0)
        ax.set_ylim(0, 108)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2, width=0.5, labelsize=5.8)
    axes[0].set_ylabel("accuracy (%)", fontsize=6.4)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=5.4, frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(0.55, -0.10), handlelength=1.2)
    fig.tight_layout(pad=0.2, w_pad=0.5, rect=[0, 0.04, 1, 1])
    fig.savefig(f"{OUT}/fig4_override.pdf", bbox_inches="tight")
    print("fig4 written")


def fig5():
    """Framing ablation: only hedging preserves the percept."""
    import numpy as np
    frames = ["bare\nassertion", "expert\nmodel", "emphatic", "user\nsays", "hedged"]
    vals = [31.0, 42.5, 44.0, 49.0, 99.8]
    fig, ax = plt.subplots(figsize=(3.35, 1.45))
    cols = [ORANGE] * 4 + [GREEN]
    ax.bar(np.arange(len(frames)), vals, width=0.6, color=cols, linewidth=0)
    ax.axhline(99.8, color=INK, ls=(0, (3, 2)), lw=0.7)
    ax.text(-0.5, 103.5, "accuracy with no evidence", fontsize=5.6, va="bottom",
            ha="left", color=INK)
    ax.axhline(50, color=MUTED, lw=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 2 if v < 95 else v - 8, f"{v:.1f}", ha="center",
                fontsize=5.8, color=INK if v < 95 else 'white')
    ax.set_xticks(np.arange(len(frames)))
    ax.set_xticklabels(frames, fontsize=5.8)
    ax.set_ylim(0, 122); ax.set_xlim(-0.6, len(frames) - 0.35)
    ax.set_ylabel("accuracy (%)", fontsize=6.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, width=0.5, labelsize=5.8)
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{OUT}/fig5_framing.pdf", bbox_inches="tight")
    print("fig5 written")


if __name__ == "__main__":
    fig4()
    fig5()
