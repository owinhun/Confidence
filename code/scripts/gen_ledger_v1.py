#!/usr/bin/env python3
"""The ten headline cells, named.

Two external reviews asked the same question from opposite directions: how many of the cells
carrying the result are sound-event cells, and which set each count refers to. The paper had
the answer spread across a funnel diagram, a sentence in Limitations and four differently-sized
"primary" counts. Naming the cells in one table is cheaper than defending the omission -- an
audio paper whose headline rests on four sound cells should say so on the page rather than let
a reviewer discover it.

Everything here is derived: membership follows from which output files exist and from the same
eligibility rule the Protocol states, so the table cannot drift from the runs the way a typed
count can.
"""
from __future__ import annotations
import json, os, collections, pathlib
import numpy as np

OUT = "outputs"
MODELS = [("Qwen2.5-Omni", "s2"), ("Qwen3-Omni", "s2q3"), ("VideoLLaMA2.1", "s2vl"),
          ("Qwen2-Audio", "nm_qwen2audio"), ("Nemotron", "nm_nemotron")]
ATTRS = ["sound", "colour", "noise", "lowpass", "reverb", "silence"] + \
        [f"grad_{a}_{g}" for a in ("noise", "lowpass", "reverb") for g in ("g1", "g2", "g3", "g4")]
PRETTY = {"sound": "sound", "colour": "colour", "noise": "noise", "grad_noise_g4": "noise (g4)"}
FIELDS = ["anon", "priority", "urgency", "reliab", "certain", "prior"]


def L(name):
    p = f"{OUT}/{name}_v1.jsonl"
    if not os.path.exists(p):
        return None
    d = {}
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            d[r["clip"]] = r
    return d if len(d) >= 400 else None


def verdict(b):
    a = np.mean([r["correct"] for r in b.values()])
    rec = [np.mean([r["correct"] for r in b.values() if r["label"] == lb]) for lb in "AB"]
    if a < 0.60:
        return "floor"
    if min(rec) < 0.60:
        return "recall"
    return "ceiling" if a > 0.95 else "eligible"


def field_cells():
    """Cells that were run on all six field names."""
    seen = collections.defaultdict(set)
    for fn in os.listdir(OUT):
        if not fn.endswith("_v1.jsonl") or "_swap" in fn:
            continue
        b = fn[: -len("_v1.jsonl")]
        for fl in FIELDS:
            for lvl in ("high", "low"):
                suf = f"_{fl}_{lvl}"
                if b.endswith(suf):
                    seen[b[: -len(suf)]].add(fl)
    return {c for c, s in seen.items() if len(s) == 6}


def main():
    fields = field_cells()
    rows = []
    for nm, pfx in MODELS:
        for a in ATTRS:
            o, s = L(f"{pfx}_{a}_none"), L(f"{pfx}_{a}_none_swap")
            if o is None or s is None:
                continue
            if verdict(o) not in ("eligible", "ceiling") or verdict(s) not in ("eligible", "ceiling"):
                continue
            # The body says every non-sensitivity number is order-averaged, and Table 5's
            # omit column is; this table quoted the first order only, so the same A_0
            # appeared twice with different values (95.2 here, 93.6 there). Display the
            # average. The gate itself stays per-order as pre-registered -- the ceiling
            # mark records that an order exceeded 0.95, which is what caps POR headroom.
            a0o = 100 * np.mean([r["correct"] for r in o.values()])
            a0s = 100 * np.mean([r["correct"] for r in s.values()])
            rows.append(dict(
                model=nm, attr=PRETTY.get(a, a),
                a0=(a0o + a0s) / 2,
                ceiling=verdict(o) == "ceiling" or verdict(s) == "ceiling",
                field=f"{pfx}_{a}" in fields,
                casc=bool(L(f"{pfx}_{a}_cascadex_plain")),
            ))
    order = {"sound": 0, "colour": 1}
    rows.sort(key=lambda r: (order.get(r["attr"], 2), -r["a0"]))
    # \checkmark lives in amssymb, which the paper does not load; a bullet needs no package
    tick = lambda b: r"$\bullet$" if b else ""

    body = "\n".join(
        f"{r['model']} & {r['attr']} & {r['a0']:.1f}{'$^{c}$' if r['ceiling'] else ''} "
        f"& {tick(r['field'])} & {tick(r['casc'])} \\\\" for r in rows)
    counts = collections.Counter(r["attr"].split()[0] for r in rows)

    # A review objected that the admission rule manufactures the regime the recommendation
    # holds in: every admitted cell reads more accurately than the detector, so "do not
    # forward" is close to arithmetic there. The rule admits from A_0 >= 0.60, well below the
    # detector, so it does not exclude that regime by construction. Find the cells that landed
    # in it, to say so with a number rather than an assertion.
    # NOTE 2026-09-07: a0 below is deliberately the FIRST order, because the gate is evaluated
    # per order -- this cell passed in that order (A_0 73.5, both recalls >= 0.60) and collapsed
    # under the swap (A_0 58.2, recall 0.21), so its order-averaged recall 0.54 never faced the
    # gate at all. A review read the bare 73.5 against the order-averaged A_0 everywhere else,
    # so Sec. 8 now says "in the order it passed". Do not order-average this one: doing so drops
    # the cell entirely and removes the paper's only concrete answer to "the gate manufactures
    # the regime".
    below = []
    for nm, pfx in MODELS:
        for a in ATTRS:
            o = L(f"{pfx}_{a}_none")
            if o is None:
                continue
            a0 = 100 * np.mean([r["correct"] for r in o.values()])
            rec = [np.mean([r["correct"] for r in o.values() if r["label"] == lb]) for lb in "AB"]
            if 60 <= a0 < 74.8 and min(rec) >= 0.60:
                below.append((nm, PRETTY.get(a, a), a0, verdict(L(f"{pfx}_{a}_none_swap") or {}) if L(f"{pfx}_{a}_none_swap") else "not run"))
    extra = ""
    if below:
        nm, at, a0, sw = below[0]
        extra = (f"\\newcommand{{\\BelowDetN}}{{{len(below)}}}\n"
                 f"\\newcommand{{\\BelowDetCell}}{{{nm}/{at}}}\n"
                 f"\\newcommand{{\\BelowDetA}}{{{a0:.1f}}}\n")
    print("  below-detector candidates:", below or "none")
    pathlib.Path("paper/auto_ledger.tex").write_text(
        "% Generated by scripts/gen_ledger_v1.py -- do not edit by hand.\n"
        f"\\newcommand{{\\LedgerRows}}{{%\n{body}\n}}\n"
        f"\\newcommand{{\\LedgerSoundN}}{{{counts['sound']}}}\n"
        f"\\newcommand{{\\LedgerColourN}}{{{counts['colour']}}}\n"
        f"\\newcommand{{\\LedgerNoiseN}}{{{counts['noise']}}}\n"
        f"\\newcommand{{\\LedgerTotN}}{{{len(rows)}}}\n"
        f"\\newcommand{{\\LedgerFieldN}}{{{sum(r['field'] for r in rows)}}}\n"
        f"\\newcommand{{\\LedgerCascN}}{{{sum(r['casc'] for r in rows)}}}\n" + extra)
    print(f"{len(rows)} cells: {dict(counts)}; fields {sum(r['field'] for r in rows)}, "
          f"cascade {sum(r['casc'] for r in rows)}")
    print("paper/auto_ledger.tex 작성")


if __name__ == "__main__":
    main()
