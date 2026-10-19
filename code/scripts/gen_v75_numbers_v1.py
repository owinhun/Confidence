#!/usr/bin/env python3
"""v75 macros, all derived from the existing generated tables so they cannot rot.

Three review findings drive them:
 1. Sec. 7 claimed the strong probe lands "within 0.3 pp" of the sweep's pi=0.90 column. Recomputed
    from auto_strong + auto_sweep the worst cell is Nemotron (-0.4 vs +0.1), i.e. 0.5 pp with the
    sign reversed. ProbeSweepMax replaces the hand-typed 0.3.
 2. Sec. 7 called the declared field "the smallest of the three parts". On the accuracy scale of
    Table 4 that fails on Nemotron, where the field costs more than moving the claim into the block.
    FieldOverClaimN counts the cascade cells where |verbal| exceeds |in block|.
 3. The strong probe's +8.4 pp on Qwen2-Audio lands the cascade at the probe's own accuracy, so the
    gain is over the reader, not over the detector. ProbeCascAcc / ProbeAcc make that explicit.
"""
import re, pathlib
P = pathlib.Path("paper")
def macros(*names):
    """Scalar macros plus the multi-line row blocks, which a non-greedy {...} match would truncate."""
    d = {}
    for n in names:
        t = (P / n).read_text()
        for k, v in re.findall(r"\\newcommand\{\\(\w+)\}\{([^{}]*)\}", t):
            d[k] = v
        for k, v in re.findall(r"\\newcommand\{\\(\w+)\}\{%\n(.*?)\n\}", t, re.S):
            d[k] = v
    return d
def cells(row):
    return [c.strip().strip("$") for c in row.rstrip("\\ ").split("&")]
M = macros("auto_strong.tex", "auto_sweep.tex", "auto_cascade.tex", "auto_ledger.tex", "auto_all.tex")
# --- 1. probe vs the sweep's pi=0.90 column, per cell
sweep = {cells(r)[0]: cells(r)[1:] for r in M["SweepRows"].strip().splitlines()}
probe = {"Qwen2.5-Omni": M["StrongQTwoFive"], "Qwen3-Omni": M["StrongQThree"],
         "Qwen2-Audio": M["StrongQTwoA"], "Nemotron": M["StrongNemo"]}
gaps, flips = [], 0
for cell, pv in probe.items():
    col90 = float(sweep[cell][3])           # .748 .80 .85 .90 .95 1.0 pi*
    gaps.append(abs(float(pv) - col90))
    flips += (float(pv) < 0) != (col90 < 0)
# --- 2. field cost vs in-block claim cost on the accuracy scale
cas = {cells(r)[0].replace("$^{s}$", "").replace("^{s}", ""): cells(r)[1:]
       for r in M["TabCascade"].strip().splitlines()}
over = []
for cell, cols in cas.items():
    if "Flamingo" in cell:
        continue
    # cols after the model name: omit, plain, [CI], in block, verbal, numeric, pi*
    inblock, verbal = float(cols[3]), float(cols[4])
    if abs(verbal) > abs(inblock):
        over.append((cell, verbal, inblock))
# --- 3b. per-item spread of Table 3's Delta, which two reviews asked for: the intervals look
#     implausibly tight until one sees that Delta is a difference-of-differences on the same items,
#     so the item-level s.d. is under 1.2 log-odds and ~390 clusters give a 0.04-0.12 half-width.
import json, os, math
import numpy as np
def _L(n):
    q = f"outputs/{n}_v1.jsonl"
    if not os.path.exists(q):
        return None
    d = {json.loads(l)["clip"]: json.loads(l) for l in open(q) if l.strip()}
    return d if len(d) >= 400 else None
def _LT(pfx, attr, k, tag):
    return _L(f"{pfx}_{attr}_{k}{tag}") or _L(f"{pfx}_{attr}_{k}_T{tag}") or _L(f"{pfx}_{attr}_{k}{tag}_T")
def _m(r, sw):
    pa = min(max(r["p_A"], 1e-6), 1 - 1e-6)
    v = math.log(pa / (1 - pa))
    v = v if r["label"] == "A" else -v
    return -v if sw else v
_CELLS = [("s2", "sound"), ("s2q3", "sound"), ("s2q3", "noise"), ("s2vl", "colour"),
          ("nm_qwen2audio", "sound"), ("nm_nemotron", "sound"), ("nm_nemotron", "colour"),
          ("s2", "colour"), ("s2", "grad_noise_g4"), ("s2q3", "colour")]
_sds = []
for pfx, attr in _CELLS:
    vals = []
    for tag in ("", "_swap"):
        b = _L(f"{pfx}_{attr}_none{tag}")
        if b is None:
            continue
        sw = tag == "_swap"
        g = {k: _LT(pfx, attr, k, tag) for k in ("true_high", "anon_high", "true_low", "anon_low",
             "priority_true_high", "priority_high", "priority_true_low", "priority_low")}
        if any(v is None for v in g.values()):
            continue
        for k in (k for k in b if b[k]["correct"] and all(k in v for v in g.values())):
            sc = 0.5 * (_m(g["true_high"][k], sw) - _m(g["anon_high"][k], sw)) - 0.5 * (_m(g["true_low"][k], sw) - _m(g["anon_low"][k], sw))
            sp = 0.5 * (_m(g["priority_true_high"][k], sw) - _m(g["priority_high"][k], sw)) - 0.5 * (_m(g["priority_true_low"][k], sw) - _m(g["priority_low"][k], sw))
            vals.append(sc - sp)
    if vals:
        _sds.append(float(np.std(vals, ddof=1)))

# --- 3. the strong probe's cascade accuracy against the probe's own
reader_a0 = float(cas["Qwen2-Audio"][0])
casc = reader_a0 + float(M["StrongQTwoA"])
out = {
    "ProbeSweepMax": f"{max(gaps):.1f}",
    "ProbeSignFlipN": str(flips),
    "FieldOverClaimN": str(len(over)),
    "FieldOverClaimCell": over[0][0] if over else "none",
    "FieldOverClaimVal": f"{abs(over[0][1]):.1f}" if over else "0",
    "FieldOverClaimRef": f"{abs(over[0][2]):.1f}" if over else "0",
    "ProbeCascAcc": f"{casc:.1f}",
    "ProbeAcc": f"{100 * float(M['StrongAcc']):.1f}",
    "DeltaSdMin": f"{math.floor(min(_sds) * 10) / 10:.1f}",
    "DeltaSdMax": f"{math.ceil(max(_sds) * 10) / 10:.1f}",
}
with open(P / "auto_v75.tex", "w") as f:
    f.write("% Generated by scripts/gen_v75_numbers_v1.py -- do not edit by hand.\n")
    for k, v in out.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("per-cell probe-vs-sweep gaps:", [f"{g:.1f}" for g in gaps])
print("field over claim:", over)
print(open(P / "auto_v75.tex").read())
