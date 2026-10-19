#!/usr/bin/env python3
"""Tier-T: a genuinely stronger detector for the forwarding-gain direction.

Supervised probe on the concatenated audio embeddings of both CLAP checkpoints, plus the
positive/negative phrasing similarities each checkpoint already exposes, plus a query-class
one-hot. Cross-fitted on the same source-video folds as every cascade number: each fold's
decisions come from a probe that never saw that fold. Pre-registered in outputs/log_v1.md;
the >= 0.85 OOF gate and the per-cell sign predictions are fixed there.
"""
import json, os, sys
from pathlib import Path
os.environ.setdefault("HF_HOME", "/data2/svhalluc_models/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/data2/svhalluc_models/hf_cache/hub")
import numpy as np

ROOT = Path("/workspace/svhalluc")
rows = [json.loads(l) for l in open("outputs/clapx_cascade_v1.jsonl")]
CKPTS = ["laion/clap-htsat-unfused", "laion/larger_clap_general"]

import torch, librosa
from transformers import ClapModel, ClapProcessor
dev = "cuda"

feats = {r["clip"]: [] for r in rows}
sims  = {r["clip"]: [] for r in rows}
classes = sorted({r["query_class"] for r in rows})
ci = {c: i for i, c in enumerate(classes)}

for ck in CKPTS:
    print(f"[embed] {ck}", flush=True)
    model = ClapModel.from_pretrained(ck, use_safetensors=True).to(dev).eval()
    proc  = ClapProcessor.from_pretrained(ck)
    with torch.no_grad():
        tp = proc(text=[f"the sound of {c}" for c in classes], return_tensors="pt", padding=True)
        te_pos = model.get_text_features(**{k: v.to(dev) for k, v in tp.items()})
        tn = proc(text=[f"a clip without the sound of {c}" for c in classes], return_tensors="pt", padding=True)
        te_neg = model.get_text_features(**{k: v.to(dev) for k, v in tn.items()})
        te_pos = torch.nn.functional.normalize(te_pos, dim=-1)
        te_neg = torch.nn.functional.normalize(te_neg, dim=-1)
        for i, r in enumerate(rows):
            y, _ = librosa.load(str(ROOT / r["clip"]), sr=48000, mono=True, duration=10.0)
            a = proc(audios=[y], sampling_rate=48000, return_tensors="pt")
            ae = model.get_audio_features(**{k: v.to(dev) for k, v in a.items()})
            ae = torch.nn.functional.normalize(ae, dim=-1)[0]
            feats[r["clip"]].append(ae.cpu().numpy())
            k = ci[r["query_class"]]
            sims[r["clip"]].extend([float(ae @ te_pos[k]), float(ae @ te_neg[k])])
            if (i + 1) % 100 == 0: print(f"  {i+1}/{len(rows)}", flush=True)
    del model; torch.cuda.empty_cache()

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

X = np.array([np.concatenate(feats[r["clip"]] + [np.array(sims[r["clip"]])]
              + [np.eye(len(classes))[ci[r["query_class"]]]]) for r in rows])
y = np.array([1 if r["label"] == "A" else 0 for r in rows])
folds = np.array([r["fold"] for r in rows])

oof_p = np.zeros(len(rows))
for f in sorted(set(folds)):
    tr, te = folds != f, folds == f
    clf = LogisticRegression(C=0.5, max_iter=2000).fit(X[tr], y[tr])
    oof_p[te] = clf.predict_proba(X[te])[:, 1]
pred = (oof_p >= 0.5).astype(int)
acc = float((pred == y).mean())
print(f"[probe] OOF accuracy = {acc:.3f}  (게이트 0.85)", flush=True)

out = []
for i, r in enumerate(rows):
    q_raw = oof_p[i] if pred[i] == 1 else 1 - oof_p[i]
    rr = dict(r)
    rr["det_pred"] = "A" if pred[i] == 1 else "B"
    rr["p_present"] = round(float(oof_p[i]), 6)
    rr["q"] = round(float(q_raw), 6)
    out.append(rr)
# fold별 isotonic은 q를 쓰는 조건에서만 필요 -- plain 조건은 det_pred만 소비
accs = {f: float((pred[folds == f] == y[folds == f]).mean()) for f in sorted(set(folds))}
print("[probe] per-fold:", {k: round(v, 3) for k, v in accs.items()}, flush=True)
if acc >= 0.85:
    p = Path("outputs/mixstr_cascade_v1.jsonl")
    p.write_text("\n".join(json.dumps(r) for r in out) + "\n")
    print(f"[write] {p} (n={len(out)})", flush=True)
else:
    print("[gate] 0.85 미달 -- 파일 미작성, 음성 결과로 기록", flush=True)
