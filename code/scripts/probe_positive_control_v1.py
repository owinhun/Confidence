#!/usr/bin/env python3
"""Probe-family POSITIVE CONTROL (highest-priority review fix).

Runs the SAME lag-profile / nonlinear probes we applied to AVLLM representations
on Synchformer's OWN audio and visual feature sequences, for the SAME 246 easy
clips. If the probe family finds synchrony there (high accuracy) but not in
AVLLM features, the audit's null is a property of the representations, not of
our probes. If it fails here too, the audit is a probe artifact and must be
retracted.

Stage 1 (--stage extract): dump per-clip visual/audio feature sequences from
Synchformer (post extract_vfeats/extract_afeats + projections, pre transformer).
Stage 2 (--stage probe): identical probe suite, video-grouped 5-fold CV.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/workspace/svhalluc")
SF = ROOT / "claude_track2/external/Synchformer"
OUT = ROOT / "claude_track2/outputs/sf_features_v1.npz"
PART = ROOT / "claude_track2/outputs/sf_feat_part_v1"
MANIFEST = ROOT / "claude_track2/data/train_pairs_v2.jsonl"
EXP = "24-01-04T16-39-21"
VFPS, AFPS, SIDE = 25, 16000, 256
NBINS = 16


def extract():
    sys.path.insert(0, str(SF))
    os.chdir(SF)
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from dataset.dataset_utils import get_video_and_audio
    from scripts.train_utils import get_model, get_transforms, prepare_inputs

    cfg = OmegaConf.load(f"logs/sync_models/{EXP}/cfg-{EXP}.yaml")
    cfg.model.params.afeat_extractor.params.ckpt_path = None
    cfg.model.params.vfeat_extractor.params.ckpt_path = None
    cfg.model.params.transformer.target = cfg.model.params.transformer.target.replace(
        ".modules.feature_selector.", ".sync_model.")
    device = torch.device("cuda:0")
    _, model = get_model(cfg, device)
    ckpt = torch.load(f"logs/sync_models/{EXP}/{EXP}.pt", map_location="cpu",
                      weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tfm = get_transforms(cfg, ["test"])["test"]

    rows = [json.loads(l) for l in MANIFEST.open() if l.strip()]
    rows = [r for r in rows if float(r["delta"]) >= 8]
    print(f"easy clips: {len(rows)}")
    PART.mkdir(parents=True, exist_ok=True)
    TMP = Path(os.environ.get("CLAUDE_JOB_DIR", "/tmp")) / "tmp" / "sf_pc_reenc"
    TMP.mkdir(parents=True, exist_ok=True)

    def bin_mean(x, nb):
        T = x.shape[0]
        idx = np.linspace(0, T, nb + 1).astype(int)
        return np.stack([x[idx[i]:max(idx[i] + 1, idx[i + 1])].mean(0)
                         for i in range(nb)]).astype(np.float16)

    n_new = 0
    for r in rows:
        stem = Path(r["clip"]).stem
        dst = PART / f"{stem}.npz"
        if dst.exists():
            continue
        src = ROOT / r["clip"]
        re = TMP / f"{stem}_re.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(src), "-vf",
                        f"fps={VFPS},scale=iw*{SIDE}/'min(iw,ih)':ih*{SIDE}/'min(iw,ih)',"
                        "crop='trunc(iw/2)'*2:'trunc(ih/2)'*2",
                        "-ar", str(AFPS), str(re)], capture_output=True)
        try:
            rgb, audio, meta = get_video_and_audio(str(re), get_meta=True)
            item = dict(video=rgb, audio=audio, meta=meta, path=str(re),
                        split="test",
                        targets={"v_start_i_sec": 1.0, "offset_sec": 0.0})
            item = tfm(item)
            batch = torch.utils.data.default_collate([item])
            aud, vid, _ = prepare_inputs(batch, device)
            with torch.no_grad(), torch.autocast("cuda", enabled=cfg.training.use_half_precision):
                v = model.extract_vfeats(vid, for_loop=False)
                a = model.extract_afeats(aud, for_loop=False)
                v = model.vproj(v)
                a = model.aproj(a)
            # (B, S, tokens, D) -> flatten segment/token dims into time
            v = v.float().flatten(1, 2)[0].cpu().numpy()
            a = a.float().flatten(1, 2)[0].cpu().numpy()
            np.savez_compressed(dst, V=bin_mean(v, NBINS), A=bin_mean(a, NBINS),
                                label=r["label"], ytid=r["ytid"])
            n_new += 1
            if n_new % 25 == 0:
                print(f"{n_new} done", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {stem}: {type(e).__name__}: {e}", flush=True)
        finally:
            re.unlink(missing_ok=True)
    files = sorted(PART.glob("*.npz"))
    Vs, As, ys, gs = [], [], [], []
    for fp in files:
        z = np.load(fp, allow_pickle=True)
        Vs.append(z["V"]); As.append(z["A"])
        ys.append(str(z["label"])); gs.append(str(z["ytid"]))
    np.savez_compressed(OUT, V=np.stack(Vs), A=np.stack(As),
                        labels=np.array(ys), ytids=np.array(gs))
    print(f"saved {len(files)} clips -> {OUT}")


def probe():
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    z = np.load(OUT, allow_pickle=True)
    V, A = z["V"].astype(np.float32), z["A"].astype(np.float32)
    y = (z["labels"] == "A").astype(int)
    g = z["ytids"]
    n = len(y)
    print(f"n={n}, pos={y.mean():.3f}, groups={len(set(g))}")

    MAXLAG = 6

    def lag_profile(a, v):
        a = a - a.mean(0, keepdims=True); v = v - v.mean(0, keepdims=True)
        an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-6)
        vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-6)
        T = a.shape[0]; prof = []
        for lag in range(-MAXLAG, MAXLAG + 1):
            if lag >= 0:
                x, w = an[lag:], vn[:T - lag]
            else:
                x, w = an[:T + lag], vn[-lag:]
            prof.append(float((x * w).sum(1).mean()))
        return np.array(prof)

    prof = np.stack([lag_profile(A[i], V[i]) for i in range(n)])
    folds = list(GroupKFold(n_splits=5).split(prof, y, g))
    for name, mk in [("lag-logreg", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
                     ("lag-mlp", lambda: make_pipeline(StandardScaler(), MLPClassifier((32,), max_iter=2000, random_state=0)))]:
        accs = []
        for tr, te in folds:
            m = mk(); m.fit(prof[tr], y[tr]); accs.append(m.score(prof[te], y[te]))
        print(f"Synchformer features, {name}: {np.mean(accs)*100:.1f}%")
    # audio-only control (must stay at chance)
    Am = A.reshape(n, -1)
    accs = []
    for tr, te in folds:
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        m.fit(Am[tr], y[tr]); accs.append(m.score(Am[te], y[te]))
    print(f"Synchformer features, audio-only control: {np.mean(accs)*100:.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["extract", "probe"], required=True)
    a = ap.parse_args()
    extract() if a.stage == "extract" else probe()
