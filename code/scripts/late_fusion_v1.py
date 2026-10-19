#!/usr/bin/env python3
"""X5: archived late-fusion baseline with CORRECT source-video grouping.

Reviewer finding: the TA expert file's `video_id` is item-unique
(temporal/<n>/<ytid>/<hash>.mp4, 323 distinct), so grouping by it performs
UNgrouped CV. The true source id is path field 3 (184 distinct). This script
reports both, so the leakage cost is visible.

Late fusion = logistic regression on [expert score, frozen-model p(A)], 5-fold CV.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

O = Path("/workspace/svhalluc/claude_track2/outputs")


def fit_cv(X, y, groups=None):
    pred = np.zeros(len(y), dtype=bool)
    splitter = (GroupKFold(n_splits=5).split(X, y, groups) if groups is not None
                else KFold(n_splits=5, shuffle=True, random_state=0).split(X, y))
    for tr, te in splitter:
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def run(name, expert_file, base_file, key, sfield, group_fn):
    ex = {json.loads(l)[key]: json.loads(l) for l in (O / f"{expert_file}.jsonl").open()}
    ba = {json.loads(l)[key]: json.loads(l) for l in (O / f"{base_file}.jsonl").open()}
    ids = sorted(set(ex) & set(ba))
    y = np.array([ex[i].get("answer", ex[i].get("label")) == "A" for i in ids])
    s = np.array([ex[i][sfield] for i in ids])
    p = np.array([ba[i]["p_A"] for i in ids])
    g = np.array([group_fn(ex[i]) for i in ids])
    X = np.stack([s, p], axis=1)
    print(f"\n{name}: n={len(ids)}, true source groups={len(set(g))}")
    for gname, gg in [("source-video grouped", g), ("ungrouped (leaky)", None)]:
        acc_lf = np.mean(fit_cv(X, y, gg) == y)
        acc_ex = np.mean(fit_cv(X[:, :1], y, gg) == y)
        print(f"  {gname:22}: late fusion {acc_lf*100:.1f}%  |  expert-score-only "
              f"{acc_ex*100:.1f}%")


def main():
    run("SVHalluc TA", "synchformer_ta_probe_v1", "eval_ta_frozen_base_v1",
        "sample_id", "s_mean", lambda r: r["video_id"].split("/")[2])
    run("VGGSound-Sparse", "synchformer_vggsparse_v1", "eval_vggsparse_qwen_base_v1",
        "clip", "s_near0", lambda r: r["ytid"])


if __name__ == "__main__":
    main()
