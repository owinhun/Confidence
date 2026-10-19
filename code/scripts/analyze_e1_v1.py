#!/usr/bin/env python3
"""E1 analysis: does the AV-LLM add value on a task the specialist cannot solve?

Conditions compared on the same clips (video-grouped 5-fold CV where a probe is
fitted), per the pre-registered decision rule in log_v1.md:
  (1) expert-only     : logistic on the full 21-dim offset posterior
  (2) frozen, no evidence
  (3) frozen + verdict evidence
  (4) frozen + probability evidence
  (4b) + evidence-use adapter
  (5) late fusion     : logistic on [21-dim posterior, frozen p_A]
Reports accuracy, McNemar vs (1) and vs (5).
"""

import json
from math import comb
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

O = Path("/workspace/svhalluc/claude_track2/outputs")


def load(f):
    p = O / f"{f}.jsonl"
    if not p.exists():
        return {}
    return {json.loads(l)["clip"]: json.loads(l) for l in p.open() if l.strip()}


def mcnemar(a, b):
    ids = sorted(set(a) & set(b))
    n01 = sum(1 for i in ids if a[i]["correct"] and not b[i]["correct"])
    n10 = sum(1 for i in ids if not a[i]["correct"] and b[i]["correct"])
    n = n01 + n10
    p = (min(1.0, sum(comb(n, k) for k in range(min(n01, n10) + 1)) * 2 / 2 ** n)
         if n else 1.0)
    return n01, n10, p


def cv_probe(X, y, groups, ids):
    pred = np.zeros(len(y), dtype=bool)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return {ids[i]: {"correct": bool(pred[i] == y[i])} for i in range(len(ids))}


def main():
    exp = load("synchformer_swap_v1")
    base = load("why_task_base_v1")
    verdict = load("why_task_verdict_v1")
    prob = load("why_task_prob_v1")
    adapter = load("why_task_prob_adapter_v1")
    if not exp or not base:
        print("missing inputs; expert/base not ready")
        return

    ids = sorted(set(exp) & set(base))
    y = np.array([exp[i]["label"] == "A" for i in ids])  # A = temporal
    groups = np.array([exp[i]["ytid"] for i in ids])
    post = np.array([exp[i]["probs"] for i in ids])
    print(f"n={len(ids)}  temporal={y.sum()}  semantic={(~y).sum()}  "
          f"groups={len(set(groups))}")

    expert_only = cv_probe(post, y, groups, ids)
    pA = np.array([base[i]["p_A"] for i in ids]).reshape(-1, 1)
    late = cv_probe(np.hstack([post, pA]), y, groups, ids)

    def acc(d):
        v = [d[i]["correct"] for i in ids if i in d]
        return sum(v) / len(v) if v else float("nan"), len(v)

    rows = [("expert-only (21-dim posterior probe)", expert_only),
            ("frozen, no evidence", base),
            ("frozen + verdict", verdict),
            ("frozen + probability", prob),
            ("+ evidence-use adapter", adapter),
            ("late fusion (posterior + frozen p_A)", late)]
    print(f"\n{'condition':<40} {'acc':>7} {'n':>5}")
    for name, d in rows:
        if not d:
            print(f"{name:<40} {'--':>7}")
            continue
        a, n = acc(d)
        print(f"{name:<40} {a*100:>6.1f}% {n:>5}")

    print("\nPre-registered gate: an injected condition must beat BOTH "
          "expert-only and late fusion (McNemar p<.05).")
    for name, d in rows[2:5]:
        if not d:
            continue
        for ref_name, ref in [("expert-only", expert_only), ("late-fusion", late)]:
            n01, n10, p = mcnemar(d, ref)
            flag = "PASS" if (p < 0.05 and n01 > n10) else "no"
            print(f"  {name:<26} vs {ref_name:<12} n01={n01:>3} n10={n10:>3} "
                  f"p={p:.4f}  {flag}")


if __name__ == "__main__":
    main()
