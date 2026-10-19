#!/usr/bin/env python3
"""Does the Prior field move the model at all, or only not in the claim-aligned direction?

A reviewer raised the sharpest methodological objection of this round: Prior: high says the
CLASS is common in general, so it should push toward "yes" for true and false claims alike --
and a shift common to both branches cancels in beta by construction. If so, Prior's small beta
would show only that Prior is not claim-aligned, NOT that Prior barely moves the model, and the
paper's gloss ("the content has to be about the claim") would be over-read.

The test needs no new inference. Define the presence-aligned log-odds

    u = log p(Yes) - log p(No)

which is +logit(p_A) in the prespecified order and -logit(p_A) under the swap, since the swap
exchanges the option letters. Then compare the high-minus-low change in u under each field, on
the same false-claim block, over the same items.

  Confidence/Priority high->low should move the CLAIM-aligned response, which points at "yes"
  for some items and "no" for others, so it should largely cancel in u.
  Prior high->low, if the reviewer is right, should move u coherently toward presence.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, "scripts")

EPS = 1e-6
CELLS = [("Qwen2.5/sound", "s2", "sound"), ("Qwen2.5/colour", "s2", "colour"),
         ("Qwen3/sound", "s2q3", "sound"), ("Qwen3/colour", "s2q3", "colour"),
         ("Qwen2-Audio", "nm_qwen2audio", "sound"), ("AF3", "nm_af3", "sound"),
         ("Nemotron/snd", "nm_nemotron", "sound"), ("Nemotron/col", "nm_nemotron", "colour")]


def L(n):
    p = f"outputs/{n}_v1.jsonl"
    if not os.path.exists(p):
        return None
    d = {}
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            d[r["clip"]] = r
    return d if len(d) >= 300 else None


def u_of(r, swapped):
    """Presence-aligned log-odds: + for 'yes', regardless of which letter carries it."""
    pa = min(max(r["p_A"], EPS), 1 - EPS)
    v = np.log(pa) - np.log(1 - pa)
    return -v if swapped else v


def field_shift(pfx, attr, hi, lo):
    """high-minus-low change in presence-aligned u, order-averaged, on the false-claim block."""
    out = []
    for tag in ("", "_swap"):
        sw = tag == "_swap"
        H, Lo = L(f"{pfx}_{attr}_{hi}{tag}"), L(f"{pfx}_{attr}_{lo}{tag}")
        if H is None or Lo is None:
            return None
        ks = sorted(set(H) & set(Lo))
        if len(ks) < 100:
            return None
        out.append(np.mean([u_of(H[k], sw) - u_of(Lo[k], sw) for k in ks]))
    return float(np.mean(out)) if out else None


print(f"{'cell':16}{'du Confidence':>16}{'du Priority':>14}{'du Prior':>11}")
rows = []
for name, pfx, attr in CELLS:
    c = field_shift(pfx, attr, "anon_high", "anon_low")
    p = field_shift(pfx, attr, "priority_high", "priority_low")
    r = field_shift(pfx, attr, "prior_high", "prior_low")
    if None in (c, p, r):
        print(f"{name:16}데이터 부족")
        continue
    rows.append((name, c, p, r))
    print(f"{name:16}{c:16.3f}{p:14.3f}{r:11.3f}")

if rows:
    a = np.array([[r[1], r[2], r[3]] for r in rows])
    print(f"\n  |du| mean   Confidence {np.abs(a[:,0]).mean():.3f}"
          f"   Priority {np.abs(a[:,1]).mean():.3f}   Prior {np.abs(a[:,2]).mean():.3f}")
    print(f"  Prior du range {a[:,2].min():+.3f}..{a[:,2].max():+.3f};"
          f"  positive (toward presence) in {int((a[:,2] > 0).sum())}/{len(rows)} cells")
    print(f"  |Prior| > |Confidence| in {int((np.abs(a[:,2]) > np.abs(a[:,0])).sum())}/{len(rows)} cells")
