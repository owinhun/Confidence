#!/usr/bin/env python3
"""The override-rate table and the eligibility flow, both computed per answer order.

Three defects motivated this. Table 3's caption said every row was order-averaged while the
printed numbers were the original order (49.2/24.6 against an average of 49.15/22.05). Figure 1
and Table 3 used the same VideoLLaMA cell in *different* orders, so its baseline appeared as
both 324 and 368. And the eligibility flow counted five ceiling cells while two more cells sit
above the 0.95 bound.

Everything here is emitted per order, and the averaged column is only written when both orders
exist and both pass the gate -- a cell that survives in one order is not evidence of the same
kind as one that survives in both, and printing it in the same column hid that.

p-values are exact McNemar, reported raw with the Holm-adjusted threshold alongside, since the
family size is what makes an adjusted decision meaningful.
"""

from __future__ import annotations

import json
import pathlib
import os
from math import comb

import numpy as np

CELLS = [("Qwen2.5", "s2", "sound"), ("Qwen3", "s2q3", "sound"), ("Qwen3", "s2q3", "noise"),
         ("VideoLLaMA2.1", "s2vl", "colour"), ("Qwen2-Aud", "nm_qwen2audio", "sound"),
         ("Audio Flamingo 3", "nm_af3", "sound"), ("Nemotron", "nm_nemotron", "sound"),
         ("Nemotron", "nm_nemotron", "colour"), ("Qwen2.5", "s2", "silence")]


def L(name):
    p = f"outputs/{name}_v1.jsonl"
    if not os.path.exists(p):
        return None
    d = {}
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            d[r["clip"]] = r
    return d if len(d) >= 400 else None


def gate(b):
    acc = float(np.mean([r["correct"] for r in b.values()]))
    rec = [float(np.mean([r["correct"] for r in b.values() if r["label"] == Lb]))
           for Lb in ("A", "B")]
    if acc < 0.60:
        st = "floor"
    elif min(rec) < 0.60:
        st = "one-sided"
    elif acc > 0.95:
        st = "ceiling"
    else:
        st = "eligible"
    return st, acc, rec


def mcnemar(a, b):
    """Exact two-sided McNemar on the discordant pairs."""
    n = a + b
    if n == 0:
        return 1.0, 0, 0
    k = min(a, b)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p), a, b


def order(pfx, attr, tag):
    b = L(f"{pfx}_{attr}_none{tag}")
    if b is None:
        return None
    st, acc, rec = gate(b)
    A, B = L(f"{pfx}_{attr}_anon_high{tag}"), L(f"{pfx}_{attr}_anon_low{tag}")
    if A is None or B is None:
        return dict(status=st, acc=acc, rec=rec, por=None)
    C = [k for k in b if b[k]["correct"] and k in A and k in B]
    hi = {k: A[k]["pred"] != A[k]["label"] for k in C}
    lo = {k: B[k]["pred"] != B[k]["label"] for k in C}

    def macro(d):
        """POR macro-averaged over the two claim directions, as the paper defines it: a cell
        with unequal direction counts must not let the larger direction set the rate."""
        per = [[d[k] for k in C if b[k]["label"] == lb] for lb in ("A", "B")]
        return float(np.mean([np.mean(v) for v in per if v])) * 100

    d10 = sum(1 for k in C if hi[k] and not lo[k])
    d01 = sum(1 for k in C if lo[k] and not hi[k])
    p, _, _ = mcnemar(d10, d01)
    return dict(status=st, acc=acc, rec=rec,
                por=(macro(hi), macro(lo), len(C), d10, d01, p))


if __name__ == "__main__":
    print("Per answer order. 'avg' is written only when BOTH orders exist and BOTH pass the gate.\n")
    print(f"{'cell':26}{'order':7}{'A0':>7}{'recall':>10}{'status':>11}"
          f"{'|C|':>6}{'high':>7}{'low':>7}{'disc':>9}{'p_raw':>10}")
    fam = []
    rows = []
    for name, pfx, attr in CELLS:
        got = {}
        for tag, lab in (("", "orig"), ("_swap", "swap")):
            r = order(pfx, attr, tag)
            if r is None:
                continue
            got[lab] = r
            rec = f"{r['rec'][0]*100:.0f}/{r['rec'][1]*100:.0f}"
            if r["por"]:
                hi, lo, n, d10, d01, p = r["por"]
                fam.append(p)
                print(f"{name+'/'+attr:26}{lab:7}{r['acc']*100:7.1f}{rec:>10}{r['status']:>11}"
                      f"{n:6}{hi:7.1f}{lo:7.1f}{f'{d10}/{d01}':>9}{p:10.1e}")
            else:
                print(f"{name+'/'+attr:26}{lab:7}{r['acc']*100:7.1f}{rec:>10}{r['status']:>11}"
                      f"{'--':>6}")
        if len(got) == 2 and all(g["por"] for g in got.values()):
            both = all(g["status"] in ("eligible", "ceiling") for g in got.values())
            hi = np.mean([g["por"][0] for g in got.values()])
            lo = np.mean([g["por"][1] for g in got.values()])
            tagm = "avg" if both else "avg*"
            print(f"{'':26}{tagm:7}{'':7}{'':10}{'':11}{'':6}{hi:7.1f}{lo:7.1f}"
                  f"{'':9}{'both-order gate: ' + ('yes' if both else 'NO'):>22}")
        rows.append((name, attr, got))
        print()

    if fam:
        K = len(fam)
        s = sorted(fam)
        print(f"Holm over the K={K} prespecified comparisons in this table:")
        rej = 0
        for i, p in enumerate(s):
            if p < 0.05 / (K - i):
                rej += 1
            else:
                break
        print(f"  rejected at 0.05: {rej}/{K}   largest rejected p = {s[rej-1]:.2e}" if rej else
              "  none rejected")
        print(f"  smallest threshold applied = {0.05/K:.2e}")
        _extra = (f"\\newcommand{{\\PorK}}{{{K}}}\n"
                  f"\\newcommand{{\\PorRejN}}{{{rej}}}\n")

    # --- LaTeX. Primary rows are cells that pass the gate in BOTH orders; a cell that survives
    # one order only is not the same kind of evidence and is emitted separately.
    prim, sens, notes = [], [], []
    for name, attr, got in rows:
        if len(got) != 2 or not all(g["por"] for g in got.values()):
            if got:
                lab, g = next(iter(got.items()))
                if g["por"]:
                    hi, lo, n, d10, d01, pv = g["por"]
                    sens.append(f"{name} & {attr} & {n} & {hi:.1f} & {lo:.1f} & "
                                f"-- & {pv:.1e} \\\\")
            continue
        both = all(g["status"] in ("eligible", "ceiling") for g in got.values())
        o, s = got["orig"], got["swap"]
        hi = (o["por"][0] + s["por"][0]) / 2
        lo = (o["por"][1] + s["por"][1]) / 2
        # the estimate averages two per-order estimates, so both baseline-correct counts are
        # the honest thing to print; min() named a set that does not exist and did not match
        # Figure 1's single-order count either
        n = str(o["por"][2]) + "/" + str(s["por"][2])
        pv = max(o["por"][5], s["por"][5])
        spread = abs(o["por"][0] - s["por"][0]), abs(o["por"][1] - s["por"][1])
        # A cell above the 0.95 bound is admitted as a ceiling cell rather than an eligible one.
        # The table defined a ^c footnote for exactly this and then never printed it, so a reader
        # dividing 381 by 400 got 0.9525 and saw a cell that looks like it violates the gate.
        ceil = [lab for lab, g in (("orig", o), ("swap", s)) if g["status"] == "ceiling"]
        # order suffix is a digit, not a letter: "s" already means "sensitivity" in Tables 1, 3, 4
        suf = {"orig": "1", "swap": "2"}[ceil[0]] if ceil else ""   # digit, not a letter: s = sensitivity elsewhere
        mark = "$^{c}$" if len(ceil) == 2 else (f"$^{{c\\text{{-}}{suf}}}$" if ceil else "")
        line = (f"{name}{mark} & {attr} & {n} & {hi:.1f} & {lo:.1f} & "
                f"{spread[0]:.1f}/{spread[1]:.1f} & {pv:.1e} \\\\")
        (prim if both else sens).append(line)
        notes.append(f"{name}/{attr}: both-order gate {'yes' if both else 'NO'}")
    out = pathlib.Path("paper/auto_por.tex")
    out.write_text(_extra +"% Generated by scripts/gen_por_table_v1.py -- do not edit by hand.\n"
                   "% Rows are averaged over BOTH answer orders; the spread column is the\n"
                   "% per-order gap, so magnitude sensitivity is visible instead of hidden.\n"
                   "\\newcommand{\\TabPOR}{%\n" + "\n".join(prim) + "\n}\n"
                   "\\newcommand{\\TabPORsens}{%\n" + "\n".join(sens) + "\n}\n")
    print(f"\n{out} 작성 — primary {len(prim)}행, sensitivity {len(sens)}행")
    for n_ in notes:
        print("   " + n_)
