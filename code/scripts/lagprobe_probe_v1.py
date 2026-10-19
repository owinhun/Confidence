#!/usr/bin/env python3
"""Layer-resolved lag-aware probes (pre-registered; log_v1.md).

Loads lagprobe_bins_v1.npz (per layer: A [N,32,D], V [N,32,D]) and, per layer,
runs video-grouped 5-fold CV (same folds all layers) with:
 (a) lag-profile logistic: cosine sim between mean-centered A/V bins at lags
     -8..+8 -> 17-dim profile -> logistic regression;
 (b) nonlinear MLP on the same profile;
 (c) sanity: logistic on PCA-64 of [A-mean; V-mean] concat.
Prints per-layer accuracies. CPU-only.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path("/workspace/svhalluc/claude_track2/outputs/lagprobe_bins_v1.npz")
LAYERS = [0, 4, 8, 14, 20, 27]
MAXLAG = 8


def lag_profile(A: np.ndarray, V: np.ndarray) -> np.ndarray:
    """A,V: [32, D] (float) -> [2*MAXLAG+1] mean cosine at each lag."""
    A = A - A.mean(0, keepdims=True)
    V = V - V.mean(0, keepdims=True)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-6)
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-6)
    prof = []
    T = A.shape[0]
    for lag in range(-MAXLAG, MAXLAG + 1):
        if lag >= 0:
            a, v = An[lag:], Vn[:T - lag]
        else:
            a, v = An[:T + lag], Vn[-lag:]
        prof.append(float((a * v).sum(1).mean()))
    return np.array(prof)


def main():
    z = np.load(OUT, allow_pickle=True)
    y = (z["labels"] == "A").astype(int)
    groups = z["ytids"]
    n = len(y)
    print(f"n={n} clips, pos={y.mean():.3f}, groups={len(set(groups))}")
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(np.zeros(n), y, groups))

    for L in LAYERS:
        A = z[f"A{L}"].astype(np.float32)
        V = z[f"V{L}"].astype(np.float32)
        prof = np.stack([lag_profile(A[i], V[i]) for i in range(n)])
        accs = {"lag-logreg": [], "lag-mlp": [], "concat-pca": []}
        for tr, te in folds:
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            m.fit(prof[tr], y[tr]); accs["lag-logreg"].append(m.score(prof[te], y[te]))
            m2 = make_pipeline(StandardScaler(), MLPClassifier(
                hidden_layer_sizes=(32,), max_iter=2000, random_state=0))
            m2.fit(prof[tr], y[tr]); accs["lag-mlp"].append(m2.score(prof[te], y[te]))
            X = np.concatenate([A.mean(1), V.mean(1)], axis=1)
            p = PCA(n_components=64, random_state=0).fit(X[tr])
            m3 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            m3.fit(p.transform(X[tr]), y[tr])
            accs["concat-pca"].append(m3.score(p.transform(X[te]), y[te]))
        msg = "  ".join(f"{k}={np.mean(v)*100:.1f}%" for k, v in accs.items())
        print(f"layer {L:>2}: {msg}")


if __name__ == "__main__":
    main()
