#!/usr/bin/env python3
"""What is attributable to the CONFIDENCE FIELD, as opposed to the claim it is attached to?

The objection is that the paper contrasts a signal-only baseline against a prompt carrying a
structured block, a source, a claim, and a confidence level all at once, then names the result
after the confidence field. Everything between those two prompts is confounded.

Three conditions already on disk cut the difference into parts that are each one step wide.
Every step holds the prompt fixed except for the named component:

    none        signal only
    null_block  Source: ---   Claim: ""   Confidence: ---      block present, nothing in it
    priority_c  Source: anon  Claim: "F"  Priority: c          claim present, field carries
                                                               no epistemic content
    anon_c      Source: anon  Claim: "F"  Confidence: c        field carries declared confidence

so that

    d_structure  = m(null_block) - m(none)        the block itself
    d_claim      = m(priority_c) - m(null_block)  inserting the claim
    d_confidence = m(anon_c)     - m(priority_c)  DECLARING CONFIDENCE, and nothing else

priority_c is a better control than a bare claim with no block, because it keeps the block, the
source line, the quoting and a filled third field; only the epistemic content of that field
moves. reliab_c and certain_c are the converse check -- epistemic synonyms that should behave
like confidence rather than like priority.

Margins are truth-aligned, computed on items answered correctly without metadata, with a
video-clustered bootstrap.
"""

from __future__ import annotations

import json
import os

import numpy as np

OUT = "outputs"
# VideoLLaMA is scored on colour: its sound cell was withdrawn when the option swap put
# unaided accuracy at chance, so the one-step contrasts have to sit on the surviving cell.
MODELS = [("s2", "Qwen2.5-Omni", "sound"), ("s2q3", "Qwen3-Omni", "sound"),
          ("s2vl", "VideoLLaMA2.1", "colour"),
          ("nm_qwen2audio", "Qwen2-Audio", "sound"), ("nm_af3", "Audio-Flamingo-3", "sound")]
EPS = 1e-6
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


def margin(rec):
    pa = min(max(rec["p_A"], EPS), 1 - EPS)
    lo = np.log(pa) - np.log(1 - pa)
    return lo if rec["label"] == "A" else -lo


def video(clip):
    return clip.split("/")[-1].split("_")[0]


def contrast(base, attr, hi_cond, lo_cond, keys, seed=0):
    """mean margin(hi) - margin(lo) over keys, video-clustered bootstrap."""
    a, b = L(f"{base}_{attr}_{hi_cond}"), L(f"{base}_{attr}_{lo_cond}")
    if a is None or b is None:
        return None
    ok = [k for k in keys if k in a and k in b]
    if len(ok) < 30:
        return None
    per = {k: margin(a[k]) - margin(b[k]) for k in ok}
    vids = sorted({video(k) for k in ok})
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(BOOTS):
        sel = set(rng.choice(vids, len(vids), replace=True))
        v = [per[k] for k in ok if video(k) in sel]
        if v:
            bs.append(np.mean(v))
    return (float(np.mean(list(per.values()))),
            float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(ok))


def por(base, attr, cond, keys):
    """share of baseline-correct items that flip to the claimed answer under cond."""
    c = L(f"{base}_{attr}_{cond}")
    if c is None:
        return None
    ok = [k for k in keys if k in c]
    return 100.0 * sum(not c[k]["correct"] for k in ok) / len(ok) if ok else None


def fmt(r):
    return "--".rjust(24) if r is None else f"{r[0]:7.3f} [{r[1]:6.2f},{r[2]:6.2f}]".rjust(24)


if __name__ == "__main__":
    print("Decomposing the signal-only -> confident-claim difference into one-step contrasts.")
    print("Truth-aligned margin, unaided-correct items, video-clustered 95% CI.\n")

    for lvl in ("high", "low"):
        print(f"=== declared confidence = {lvl} ===")
        print(f"{'model':18}{'block alone':>24}{'+ claim (Priority)':>24}"
              f"{'+ CONFIDENCE field':>24}{'n':>6}")
        for base, name, attr in MODELS:
            b = L(f"{base}_{attr}_none")
            if b is None:
                print(f"{name:18}{'--':>24}")
                continue
            keys = [k for k in b if b[k]["correct"]]
            d_struct = contrast(base, attr, "null_block", "none", keys)
            d_claim = contrast(base, attr, f"priority_{lvl}", "null_block", keys)
            d_conf = contrast(base, attr, f"anon_{lvl}", f"priority_{lvl}", keys)
            n = d_conf[3] if d_conf else (d_claim[3] if d_claim else 0)
            print(f"{name:18}{fmt(d_struct)}{fmt(d_claim)}{fmt(d_conf)}{n:>6}")
        print()

    print("=== is the field's effect about epistemic content, or about any filled field? ===")
    print("each column is (field at high) - (field at low), same claim, same block\n")
    print(f"{'model':18}{'Confidence':>24}{'Reliability':>24}{'Certainty':>24}{'Priority':>24}")
    for base, name, attr in MODELS:
        b = L(f"{base}_{attr}_none")
        if b is None:
            print(f"{name:18}{'--':>24}")
            continue
        keys = [k for k in b if b[k]["correct"]]
        cells = "".join(fmt(contrast(base, attr, f"{p}_high", f"{p}_low", keys))
                        for p in ("anon", "reliab", "certain", "priority"))
        print(f"{name:18}{cells}")

    print("\n=== the same three steps in accuracy terms (override rate, %) ===")
    print(f"{'model':18}{'none':>10}{'null_block':>12}{'priority_hi':>13}"
          f"{'anon_high':>11}{'anon_low':>10}")
    for base, name, attr in MODELS:
        b = L(f"{base}_{attr}_none")
        if b is None:
            print(f"{name:18}{'--':>10}")
            continue
        keys = [k for k in b if b[k]["correct"]]
        vals = [por(base, attr, c, keys)
                for c in ("none", "null_block", "priority_high", "anon_high", "anon_low")]
        print(f"{name:18}" + "".join(("--" if v is None else f"{v:.1f}").rjust(w)
                                     for v, w in zip(vals, (10, 12, 13, 11, 10))))
