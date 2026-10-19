#!/usr/bin/env python3
"""Fig. 2: the error-repair sweep as the curve it is.

Table 5 prints the accuracy change of bare forwarding (vs. omitting) at six target claim
correctness levels and the bare-sentence break-even pi*. The result is a curve crossing zero
next to a predicted line; a table of interpolated crossings is the weaker form. Same size,
fonts and palette as Fig. 1 (make_figures_v8.py). Values come from analyse_sweep_v1.py's
functions, so nothing is retyped.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import make_figures_v8 as M          # rcParams, W, palette, INK
import analyse_sweep_v1 as S         # delta(), bare_pistar(), CELLS, LEVELS  (runs its main once)

xs = [float(lv) for lv, _ in S.LEVELS]
STYLE = {"Qwen2.5": (M.BLUE, "o"), "Qwen3": (M.ORANGE, "s"),
         "Qwen2-Audio": (M.PURPLE, "D"), "Nemotron": (M.INK if hasattr(M, "INK") else "k", "P")}
LABEL = {"Qwen2.5": "Qwen2.5", "Qwen3": "Qwen3", "Qwen2-Audio": "Qwen2-Aud", "Nemotron": "Nemo./sound"}

fig, ax = plt.subplots(figsize=(M.W, 1.55))
ax.axhline(0, color="0.6", lw=0.6, ls="--", zorder=1)
for nm, pfx in S.CELLS:
    ds = [S.delta(pfx, c) for _, c in S.LEVELS]
    y = np.array([d[0] for d in ds]); lo = np.array([d[1] for d in ds]); hi = np.array([d[2] for d in ds])
    col, mk = STYLE[nm]
    ax.errorbar(xs, y, yerr=[y - lo, hi - y], color=col, marker=mk, ms=3, lw=1.1,
                elinewidth=0.5, capsize=0, label=LABEL[nm], zorder=3)
    ps = S.bare_pistar(pfx)
    if ps is not None:                       # predicted break-even: short bar straddling the zero line, white halo so it reads over error bars
        ax.plot([ps, ps], [-2.8, 2.8], color=col, lw=2.2, solid_capstyle="butt", zorder=7,
                path_effects=[pe.Stroke(linewidth=4.0, foreground="white"), pe.Normal()])
ax.set_xlim(0.735, 1.012); ax.set_xticks([0.748, 0.80, 0.85, 0.90, 0.95, 1.0])
ax.set_xticklabels(["0.748", "0.80", "0.85", "0.90", "0.95", "1.0"])
ax.set_yticks([-10, 0, 10, 20])
ax.set_xlabel(r"forwarded-claim correctness $\pi$", labelpad=1)
ax.set_ylabel("acc. change vs.\nomitting (pp)", labelpad=1)
ax.tick_params(length=2, pad=1.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
from matplotlib.lines import Line2D
h, l = ax.get_legend_handles_labels()
# legend inside, top-left; extra headroom on y so its last row clears the Qwen2-Aud curve
ax.set_ylim(-16, 30)
# legend at the same 9.5 pt as the axis text (ICASSP floor is 9 pt after the \linewidth scale);
# opaque white box so nothing shows through, no transparency
ax.legend(h, l, loc="upper left", frameon=True, facecolor="white", edgecolor="none", framealpha=1.0,
          ncol=2, columnspacing=0.9, handlelength=1.3, handletextpad=0.35,
          borderaxespad=0.0, borderpad=0.15, labelspacing=0.1, fontsize=9.5,
          bbox_to_anchor=(0.02, 1.0))
fig.savefig("paper/fig2_sweep_v1.pdf", bbox_inches="tight", pad_inches=0.01)
print("wrote paper/fig2_sweep_v1.pdf")
