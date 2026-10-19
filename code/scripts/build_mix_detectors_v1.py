#!/usr/bin/env python3
"""Detector-correctness sweep files for the crossover experiment.

Starting from the real cross-fitted CLAP decisions, correct a random subset of its errors
per fold until held-out correctness hits each target. Uncorrected errors remain a subset of
CLAP's own errors, so the error-difficulty correlation is partially preserved -- the honest
middle ground between an oracle mixture (independent errors, the assumption pi* already
makes) and a genuinely stronger detector (which does not exist here). Pre-registered in
outputs/log_v1.md before this script was written; seed and targets are fixed there.
"""
import json, pathlib, random

SRC = pathlib.Path("outputs/clapx_cascade_v1.jsonl")
TARGETS = [0.80, 0.85, 0.90, 0.95]

rows = [json.loads(l) for l in SRC.open() if l.strip()]
for t in TARGETS:
    out = []
    by_fold = {}
    for r in rows:
        by_fold.setdefault(r["fold"], []).append(dict(r))
    for f, rs in sorted(by_fold.items()):
        rs.sort(key=lambda r: r["clip"])
        wrong = [r for r in rs if r["det_pred"] != r["label"]]
        n = len(rs)
        need = round(t * n) - (n - len(wrong))
        rng = random.Random(0)
        rng.shuffle(wrong)
        for r in wrong[:max(0, need)]:
            r["det_pred"] = r["label"]
        out.extend(rs)
    acc = sum(r["det_pred"] == r["label"] for r in out) / len(out)
    p = pathlib.Path(f"outputs/mix{int(t*100)}_cascade_v1.jsonl")
    p.write_text("\n".join(json.dumps(r) for r in sorted(out, key=lambda r: r["clip"])) + "\n")
    print(f"  {p.name}: n={len(out)} held-out acc={acc:.3f} (target {t})")
