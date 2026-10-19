#!/usr/bin/env python3
"""Level-5 probe: raw-signal correspondence. Audio RMS envelope vs visual
motion energy (frame-diff), per clip; features = xcorr profile (lags ±6s);
grouped CV. If separable -> raw-signal micro-module method viable. If not ->
narration-style synchrony lacks low-level correspondence entirely (level-5)."""
import json, subprocess
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT = Path("/workspace/svhalluc")
rows = [json.loads(l) for l in open(ROOT/"claude_track2/data/train_pairs_easy_v1.jsonl") if l.strip()]
FPS = 5; HOP = 0.2  # 5 Hz both streams

def feats(r):
    p = str(ROOT / r["clip"])
    try:
        # audio envelope at 5 Hz
        out = subprocess.run(["ffmpeg","-v","error","-i",p,"-vn","-ac","1","-ar","8000","-f","s16le","-"],capture_output=True)
        x = np.frombuffer(out.stdout, dtype="<i2").astype(np.float32)/32768.0
        n = len(x)//1600  # 0.2s hops
        env = np.sqrt((x[:n*1600].reshape(n,1600)**2).mean(1))
        # motion energy: grayscale 32x32 @5fps frame diffs
        out = subprocess.run(["ffmpeg","-v","error","-i",p,"-vf",f"fps={FPS},scale=32:32,format=gray","-f","rawvideo","-"],capture_output=True)
        v = np.frombuffer(out.stdout, dtype=np.uint8).astype(np.float32)
        m = len(v)//1024
        fr = v[:m*1024].reshape(m,32,32)
        mot = np.abs(np.diff(fr,axis=0)).mean((1,2))
        T = min(len(env)-1, len(mot))
        if T < 20: return None
        e, mo = env[1:T+1], mot[:T]
        e = (e-e.mean())/(e.std()+1e-8); mo = (mo-mo.mean())/(mo.std()+1e-8)
        # xcorr profile, lags -30..30 hops (±6s)
        prof = []
        for lag in range(-30,31):
            if lag>=0: a,b = e[lag:], mo[:T-lag]
            else: a,b = e[:T+lag], mo[-lag:]
            prof.append(float((a*b).mean()) if len(a)>5 else 0.0)
        prof = np.array(prof)
        extra = [prof[30], prof.max(), float(np.argmax(prof))-30, prof[28:33].mean()-np.r_[prof[:25],prof[36:]].mean()]
        return np.r_[prof, extra], 1 if r["label"]=="A" else 0, r["ytid"]
    except Exception:
        return None

with ProcessPoolExecutor(12) as ex:
    res = [r for r in ex.map(feats, rows) if r is not None]
X = np.array([r[0] for r in res]); y = np.array([r[1] for r in res]); g = np.array([r[2] for r in res])
print(f"n={len(y)}")
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
for name, clf in [("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))),
                  ("gbdt", GradientBoostingClassifier(random_state=0))]:
    sc = cross_val_score(clf, X, y, cv=GroupKFold(5), groups=g)
    print(f"{name}: {sc.mean()*100:.1f}% (+-{sc.std()*100:.1f})")
# zero-lag heuristic
h = (X[:,30] > np.median(X[:,30])).astype(int)
print(f"zero-lag-corr>median heuristic: {(h==y).mean()*100:.1f}%")
