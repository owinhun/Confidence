#!/usr/bin/env python3
"""Does Table 3's interval reconcile with a per-item s.d.? Three reviews say no.

They are checking 1.96*s/sqrt(n) against the printed half-width and getting a mismatch.
The check fails for two reasons, and only one of them is the reviewers' error:

  1. (ours) The s.d. the caption prints is computed on the per-item Delta POOLED over both
     answer orders. The two orders have different means, so that s.d. carries the between-order
     mean gap as well as the within-order dispersion. The interval is on the average of the two
     per-order means, which the between-order term does not enter the same way. The printed
     s.d. therefore cannot reconcile with the printed interval by construction.
  2. (theirs) The estimate averages TWO orders, so the effective standard error is roughly
     half the sum of the per-order ones, not a single order's.

This prints, per cell, the within-order s.d.s, the naive two-order-averaged half-width, and the
bootstrap half-width, so the reconciliation can be checked line by line.
"""
import sys; sys.path.insert(0, "scripts")
import numpy as np, importlib
g = importlib.import_module("gen_swing_v1")

print(f"{'cell':22}{'n/order':>14}{'vids/order':>14}{'sd within':>14}"
      f"{'sd pooled':>11}{'naive HW':>10}{'boot HW':>9}{'ratio':>7}")
rows = []
for name, pfx, attr in g.CELLS:
    c  = g.per_item(pfx, attr, "true_high", "anon_high", "true_low", "anon_low")
    p_ = g.per_item(pfx, attr, "priority_true_high", "priority_high",
                    "priority_true_low", "priority_low")
    if not c or not p_ or len(c) != len(p_):
        continue
    ns, vs, sds, ses, pooled = [], [], [], [], []
    for a, b in zip(c, p_):
        ks = sorted(set(a) & set(b))
        d = np.array([a[k] - b[k] for k in ks])
        ns.append(len(ks)); vs.append(len({g.video(k) for k in ks}))
        sds.append(d.std(ddof=1)); ses.append(d.std(ddof=1) / len(ks) ** .5)
        pooled.append(d)
    # the estimate is the mean of the per-order means -> SE = sqrt(sum se^2)/k
    se_avg = (sum(s ** 2 for s in ses)) ** .5 / len(ses)
    naive_hw = 1.96 * se_avg
    sd_pool = np.concatenate(pooled).std(ddof=1)
    pt, lo, hi = g.boot(c, p_)
    boot_hw = (hi - lo) / 2
    rows.append((name, min(sds), max(sds), sd_pool, naive_hw, boot_hw))
    print(f"{name:22}{'/'.join(str(n) for n in ns):>14}{'/'.join(str(v) for v in vs):>14}"
          f"{f'{min(sds):.2f}-{max(sds):.2f}':>14}{sd_pool:11.2f}{naive_hw:10.4f}"
          f"{boot_hw:9.4f}{boot_hw/naive_hw:7.2f}")

wmin = min(r[1] for r in rows); wmax = max(r[2] for r in rows)
pmin = min(r[3] for r in rows); pmax = max(r[3] for r in rows)
rat  = [r[5] / r[4] for r in rows]
print(f"\n  within-order per-item s.d. across cells : {wmin:.2f}-{wmax:.2f}")
print(f"  pooled-over-orders s.d. (what the caption printed): {pmin:.2f}-{pmax:.2f}")
print(f"  bootstrap HW / naive two-order HW: {min(rat):.2f}-{max(rat):.2f}"
      f"  (>1 = clustering widened it, as it should)")
