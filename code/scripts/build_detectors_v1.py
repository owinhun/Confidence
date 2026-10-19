#!/usr/bin/env python3
"""Real per-sample detectors for the three constructed acoustic attributes.

The utility analysis is only a simulation until the confidence values fed to the LLM come
from something that actually looks at the audio and is sometimes wrong. These are honest
detectors: a small logistic regression on classical spectral features, fit on a calibration
split and evaluated on a source-disjoint test split, with a tunable threshold so the whole
precision-coverage curve can be swept.

  noise    spectral flatness + high-band energy fraction (white noise is flat and broadband)
  lowpass  energy fraction above 1 kHz, and spectral centroid / rolloff
  reverb   decay-rate of the temporal envelope + low-frequency modulation depth

Deliberately weak-but-real. A detector that is never wrong would make the whole question
vacuous; what the interface study needs is a detector whose posterior is informative and
imperfect, and whose operating point can be moved.

Outputs, per attribute:
  data/detector_<attr>_v1.json   {clip: {"p": calibrated posterior, "y": true label,
                                         "split": "cal"|"test"}}
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path("/workspace/svhalluc")
DATA = ROOT / "claude_track2/data"
ATTRS = ["noise", "lowpass", "reverb"]
SR = 16000


def features(path: Path) -> np.ndarray | None:
    import librosa
    try:
        y, _ = librosa.load(str(path), sr=SR, mono=True, duration=10.0)
    except Exception:  # noqa: BLE001
        return None
    if y.size < SR // 2 or not np.isfinite(y).all():
        return None
    y = y / (np.abs(y).max() + 1e-9)          # remove any loudness confound
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)) + 1e-10
    freqs = librosa.fft_frequencies(sr=SR, n_fft=1024)
    p = S ** 2
    tot = p.sum(0) + 1e-10

    hi_1k = p[freqs > 1000].sum(0) / tot      # band limitation
    hi_4k = p[freqs > 4000].sum(0) / tot
    flat = np.exp(np.log(p + 1e-10).mean(0)) / (p.mean(0) + 1e-10)   # spectral flatness
    cent = librosa.feature.spectral_centroid(S=S, sr=SR)[0] / (SR / 2)
    roll = librosa.feature.spectral_rolloff(S=S, sr=SR, roll_percent=0.95)[0] / (SR / 2)

    # --- reverberation cues -------------------------------------------------
    # Reverberation fills the gaps between transients, so it suppresses fast envelope
    # modulation and lengthens envelope autocorrelation. These are the standard blind
    # features (modulation spectrum + envelope statistics); nothing here is tuned to the
    # particular echo delays we applied, which would make the detector unfairly matched.
    hop_hz = SR / 256.0                      # envelope frame rate
    env = np.sqrt(p.sum(0))
    env = env / (env.max() + 1e-9)
    e = env - env.mean()
    mod = np.abs(np.fft.rfft(e * np.hanning(e.size)))
    mfreq = np.fft.rfftfreq(e.size, d=1.0 / hop_hz)
    def band(lo, hi):
        m = (mfreq >= lo) & (mfreq < hi)
        return float(mod[m].sum() / (mod.sum() + 1e-10))
    m_slow, m_mid, m_fast = band(0.5, 3.0), band(3.0, 10.0), band(10.0, 40.0)

    ac = np.correlate(e, e, mode="full")
    ac = ac[ac.size // 2:] / (ac[ac.size // 2] + 1e-9)
    lag_half = float(np.argmax(ac < 0.5)) if (ac < 0.5).any() else float(ac.size)
    # long-window envelope autocorrelation in the 30-300 ms range, where a reverberant
    # tail keeps the signal correlated with itself
    lo_i, hi_i = int(0.03 * hop_hz), int(0.30 * hop_hz)
    tail = float(ac[lo_i:hi_i].mean()) if hi_i > lo_i + 1 else 0.0

    from scipy.stats import kurtosis, skew
    return np.array([
        hi_1k.mean(), hi_1k.std(), hi_4k.mean(),
        flat.mean(), flat.std(),
        cent.mean(), cent.std(), roll.mean(),
        env.std(), lag_half / 50.0,
        m_slow, m_mid, m_fast, m_fast / (m_slow + 1e-6),
        tail, float(kurtosis(env)), float(skew(env)),
        float(np.percentile(env, 10)),        # reverb raises the noise floor between events
    ], dtype=np.float64)


def main():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    for attr in ATTRS:
        man = DATA / f"held_attr_{attr}_v1.jsonl"
        rows = [json.loads(l) for l in man.open() if l.strip()]
        X, Y, C = [], [], []
        for r in rows:
            f = features(ROOT / r["clip"])
            if f is None or not np.isfinite(f).all():
                continue
            X.append(f)
            Y.append(1 if r["label"] == "A" else 0)
            C.append(r["clip"])
        X = np.array(X); Y = np.array(Y)
        # source-disjoint split on the YouTube id, so a clip and its modified twin
        # can never straddle calibration and test
        vid = np.array([Path(c).stem.split("_")[0] for c in C])
        uniq = sorted(set(vid))
        rng = np.random.default_rng(0)
        rng.shuffle(uniq)
        cal_ids = set(uniq[:len(uniq) // 2])
        is_cal = np.array([v in cal_ids for v in vid])

        base = make_pipeline(StandardScaler(),
                             LogisticRegression(C=0.5, max_iter=2000))
        clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
        clf.fit(X[is_cal], Y[is_cal])
        p = clf.predict_proba(X)[:, 1]
        auc_t = roc_auc_score(Y[~is_cal], p[~is_cal])
        acc_t = ((p[~is_cal] >= 0.5).astype(int) == Y[~is_cal]).mean()
        out = {c: {"p": float(pi), "y": int(yi), "split": "cal" if ic else "test"}
               for c, pi, yi, ic in zip(C, p, Y, is_cal)}
        (DATA / f"detector_{attr}_v1.json").write_text(json.dumps(out))
        print(f"{attr:8} n={len(C)}  test AUC={auc_t:.3f}  test acc={acc_t:.3f}  "
              f"cal/test videos={len(cal_ids)}/{len(uniq)-len(cal_ids)}", flush=True)


if __name__ == "__main__":
    main()
