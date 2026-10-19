#!/usr/bin/env python3
"""The decomposition, with each step actually one step wide.

The published version walked null_block -> priority_c -> anon_c and called the middle step
"+ claim". It is not. Between those two prompts four things move at once: Source (--- ->
anonymous system), Claim ("" -> the sentence), the field name (Confidence -> Priority) and the
field value (--- -> c). Only the last step was ever a genuine one-component contrast.

srconly_c closes the gap. It carries the source line and a filled non-epistemic field with an
empty claim, so the ladder becomes

    none        (no block)
    null_block  Source: ---   Claim: ""   Confidence: ---
    srconly_c   Source: anon  Claim: ""   Priority: c
    priority_c  Source: anon  Claim: "F"  Priority: c
    anon_c      Source: anon  Claim: "F"  Confidence: c

and each adjacent difference moves exactly one thing:

    null_block - none          an empty structured block
    srconly_c  - null_block    a source, and a field with a value in it
    priority_c - srconly_c     THE CLAIM, and nothing else
    anon_c     - priority_c    the field's epistemic content, and nothing else

Truth-aligned margin, unaided-correct items, video-clustered bootstrap.
"""

from __future__ import annotations

import json
import os

import numpy as np

OUT = "outputs"
MODELS = [("s2", "Qwen2.5-Omni", "sound"), ("s2q3", "Qwen3-Omni", "sound"),
          ("s2vl", "VideoLLaMA2.1", "colour"),
          ("nm_qwen2audio", "Qwen2-Audio", "sound"), ("nm_af3", "Audio-Flamingo-3", "sound")]
EPS = 1e-6
BOOTS = 1500
STEPS = [("empty block", "null_block", "none"),
         ("+ source & filled field", "srconly_{lvl}", "null_block"),
         ("+ CLAIM", "priority_{lvl}", "srconly_{lvl}"),
         ("+ CONFIDENCE meaning", "anon_{lvl}", "priority_{lvl}")]


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


def margin(rec):
    pa = min(max(rec["p_A"], EPS), 1 - EPS)
    lo = np.log(pa) - np.log(1 - pa)
    return lo if rec["label"] == "A" else -lo


def video(clip):
    return clip.split("/")[-1].split("_")[0]


def contrast(base, attr, hi, lo, keys):
    a, b = L(f"{base}_{attr}_{hi}"), L(f"{base}_{attr}_{lo}")
    if a is None or b is None:
        return None
    ok = [k for k in keys if k in a and k in b]
    if len(ok) < 30:
        return None
    per = {k: margin(a[k]) - margin(b[k]) for k in ok}
    vids = sorted({video(k) for k in ok})
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(BOOTS):
        sel = set(rng.choice(vids, len(vids), replace=True))
        v = [per[k] for k in ok if video(k) in sel]
        if v:
            bs.append(np.mean(v))
    return (float(np.mean(list(per.values()))),
            float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(ok))


def cell(r):
    return "--".rjust(22) if r is None else f"{r[0]:7.2f} [{r[1]:6.2f},{r[2]:6.2f}]".rjust(22)


if __name__ == "__main__":
    print("Each column adds exactly one component to the column before it.\n")
    for lvl in ("high", "low"):
        print(f"=== declared level = {lvl} ===")
        print(f"{'model':18}" + "".join(n.rjust(22) for n, _, _ in STEPS) + f"{'n':>6}")
        for base, name, attr in MODELS:
            b = L(f"{base}_{attr}_none")
            if b is None:
                print(f"{name:18}{'--':>22}")
                continue
            keys = [k for k in b if b[k]["correct"]]
            rs = [contrast(base, attr, hi.format(lvl=lvl), lo.format(lvl=lvl), keys)
                  for _, hi, lo in STEPS]
            n = next((r[3] for r in rs if r), 0)
            print(f"{name:18}" + "".join(cell(r) for r in rs) + f"{n:>6}")
        print()

    print("=== what the earlier version called '+ claim', against what it is ===")
    print("old = priority_c - null_block (four components move); new = priority_c - srconly_c\n")
    print(f"{'model':18}{'old (4 moves)':>22}{'new (claim only)':>22}{'share that was':>18}")
    for base, name, attr in MODELS:
        b = L(f"{base}_{attr}_none")
        if b is None:
            continue
        keys = [k for k in b if b[k]["correct"]]
        old = contrast(base, attr, "priority_high", "null_block", keys)
        new = contrast(base, attr, "priority_high", "srconly_high", keys)
        if old is None or new is None:
            print(f"{name:18}{'--':>22}")
            continue
        share = 100 * new[0] / old[0] if abs(old[0]) > 1e-9 else float("nan")
        print(f"{name:18}{cell(old)}{cell(new)}{share:17.0f}%")
