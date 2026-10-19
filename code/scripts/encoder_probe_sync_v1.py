#!/usr/bin/env python3
"""Encoder-level sync diagnostic: does AV temporal-alignment info survive at the
ENCODER outputs of Qwen2.5-Omni (audio tower / visual encoder, BEFORE LLM fusion)?
LLM-stream linear probe was at chance (50-53%); if encoder-level temporal-correlation
features also fail -> alignment info is lost at encoding itself. If they work ->
tiny external sync head on frozen encoders is viable.

Method: forward hooks capture (a) audio-tower output sequence, (b) visual encoder
output sequence during a normal forward (use_audio_in_video=True, fps 2.0,
max_frames 24). Both streams are temporally resampled to T=24 steps (0.5 s/step
over the 12 s clips), per-step centered (subtract each stream's temporal mean)
and L2-normalized; TxT cross-modal cosine matrix -> lag-profile features ->
logistic regression with ytid-grouped 5-fold CV + zero-training best-lag==0
heuristic.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, "/workspace/svhalluc/src/qwen")
import numpy as np, torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

ROOT = Path("/workspace/svhalluc")
T_COMMON = 24          # common temporal grid (0.5 s/step for 12 s clips)
LAGS = list(range(-6, 7))   # +-6 steps = +-3 s
Q = "When the speech is heard, is the narrated event happening at the same time in the video? (A) Yes (B) No"

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0, help="only first N clips (shape check)")
ap.add_argument("--out_npz", default=str(ROOT/"claude_track2/outputs/encoder_probe_feats_v1.npz"))
args = ap.parse_args()

rows = [json.loads(l) for l in open(ROOT/"claude_track2/data/train_pairs_easy_v1.jsonl") if l.strip()]
if args.limit: rows = rows[:args.limit]

model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(ROOT/"models/Qwen2.5-Omni-7B",
    dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="flash_attention_2", local_files_only=True)
model.eval()
proc = Qwen2_5OmniProcessor.from_pretrained(ROOT/"models/Qwen2.5-Omni-7B", local_files_only=True, use_fast=False)

# ---- module structure notes (top 2 levels) ----
print("== Thinker top-level modules ==", flush=True)
for name, mod in model.named_children():
    print(f"  {name}: {type(mod).__name__}")
    for n2, m2 in mod.named_children():
        print(f"    .{n2}: {type(m2).__name__}")

audio_tower = getattr(model, "audio_tower", None) or model.model.audio_tower
visual = getattr(model, "visual", None) or model.model.visual
print("audio_tower type:", type(audio_tower).__name__)
print("visual type:", type(visual).__name__, flush=True)

# ---- hooks: accumulate ALL calls per forward (chunking guard), concat in order ----
cap = {"audio": [], "visual": []}
def _grab(store):
    def hook(mod, inp, out):
        t = out
        if hasattr(t, "last_hidden_state"): t = t.last_hidden_state
        if isinstance(t, tuple): t = t[0]
        store.append(t.detach().float().cpu())
    return hook
h1 = audio_tower.register_forward_hook(_grab(cap["audio"]))
h2 = visual.register_forward_hook(_grab(cap["visual"]))

def cat2d(chunks):
    ts = [c.reshape(-1, c.shape[-1]) if c.dim() > 2 else c for c in chunks]  # drop batch dim if any
    return torch.cat(ts, dim=0).numpy()

def resample_T(x, T):
    """mean-pool/nearest-resample (n,d) -> (T,d) preserving temporal order."""
    n = x.shape[0]
    if n >= T:
        return np.stack([c.mean(0) for c in np.array_split(x, T, axis=0)])
    idx = np.minimum((np.arange(T) * n // T), n - 1)
    return x[idx]

def lag_features(A, V):
    """A,V: (T,d_a),(T,d_v) resampled streams -> feature vector + per-lag profile."""
    A = A - A.mean(0, keepdims=True); V = V - V.mean(0, keepdims=True)  # center over time
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    if A.shape[1] != V.shape[1]:  # different dims: cannot dot directly
        raise RuntimeError(f"dim mismatch audio {A.shape} vs visual {V.shape}")
    S = A @ V.T                                    # S[a_t, v_t]
    T = S.shape[0]
    ii, jj = np.meshgrid(np.arange(T), np.arange(T), indexing="ij")
    band = np.abs(ii - jj) <= 1
    band_mean = S[band].mean()
    off_mean = S[~band].mean()
    lag_prof = np.array([np.mean(np.diag(S, k=-l)) for l in LAGS])  # lag = a_t - v_t
    best_lag = LAGS[int(np.argmax(lag_prof))]
    feats = np.concatenate([[band_mean, lag_prof.max(), band_mean - off_mean, best_lag], lag_prof])
    return feats, best_lag

feats, labels, groups, best_lags = [], [], [], []
shape_note_done = False
for k, r in enumerate(rows):
    cap["audio"].clear(); cap["visual"].clear()
    try:
        msgs = [{"role":"user","content":[{"type":"video","video":str(ROOT/r["clip"]),"fps":2.0,"max_frames":24},{"type":"text","text":Q}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        a,i,v,vk = process_mm_info(msgs, use_audio_in_video=True, return_video_kwargs=True)
        if isinstance(vk.get("fps"), list) and len(vk["fps"]) == 1: vk["fps"] = vk["fps"][0]
        ins = proc(text=text,audio=a,images=i,videos=v,return_tensors="pt",padding=True,use_audio_in_video=True,**vk).to(model.device).to(model.dtype)
        grid = ins["video_grid_thw"][0].tolist()  # [t, h, w] in patch units
        with torch.inference_mode():
            model(**ins, return_dict=True, use_audio_in_video=True)
        A = cat2d(cap["audio"])   # (n_audio_tokens, d)
        Vt = cat2d(cap["visual"]) # (grid_t * h*w/4, d) after spatial merge
        gt = grid[0]
        spat = Vt.shape[0] // gt
        assert spat * gt == Vt.shape[0], f"visual tokens {Vt.shape[0]} not divisible by grid_t {gt}"
        Vframes = Vt.reshape(gt, spat, -1).mean(1)   # mean-pool spatial tokens per temporal step
        if not shape_note_done:
            print(f"[shapes] audio_calls={len(cap['audio'])} visual_calls={len(cap['visual'])} "
                  f"audio_out={A.shape} visual_out={Vt.shape} video_grid_thw={grid} "
                  f"-> visual temporal steps={gt}, spatial tokens/step={spat}, "
                  f"audio tokens/s={A.shape[0]/12.0:.2f}", flush=True)
            shape_note_done = True
        f, bl = lag_features(resample_T(A, T_COMMON), resample_T(Vframes, T_COMMON))
        feats.append(f); labels.append(1 if r["label"]=="A" else 0); groups.append(r["ytid"]); best_lags.append(bl)
        if k % 25 == 0: print(k, "ok", flush=True)
    except Exception as e:
        print(k, "ERR", repr(e), flush=True)
h1.remove(); h2.remove()

X, y, g, bl = np.array(feats), np.array(labels), np.array(groups), np.array(best_lags)
np.savez(args.out_npz, X=X, y=y, g=g, best_lag=bl)
print(f"n={len(y)} (A={int(y.sum())}, B={int((1-y).sum())}), feat_dim={X.shape[1]}", flush=True)

if len(y) < 20:
    print("too few samples for CV (shape-check mode); exiting.", flush=True)
    sys.exit(0)
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
sc = cross_val_score(clf, X, y, cv=GroupKFold(5), groups=g)
print(f"LOGREG (all feats): acc={sc.mean()*100:.1f}% (+-{sc.std()*100:.1f}) folds={[f'{s*100:.1f}' for s in sc]}")
# lag-profile-only ablation (13 dims)
sc2 = cross_val_score(clf, X[:, 4:], y, cv=GroupKFold(5), groups=g)
print(f"LOGREG (lag profile only): acc={sc2.mean()*100:.1f}% (+-{sc2.std()*100:.1f})")
# zero-training heuristic: aligned iff best-lag == 0
pred = (bl == 0).astype(int)
print(f"HEURISTIC best-lag==0 -> aligned: acc={(pred==y).mean()*100:.1f}%  "
      f"(best-lag==0 rate: aligned {np.mean(bl[y==1]==0)*100:.0f}%, shifted {np.mean(bl[y==0]==0)*100:.0f}%)")
print("best-lag histogram aligned:", dict(zip(*np.unique(bl[y==1], return_counts=True))))
print("best-lag histogram shifted:", dict(zip(*np.unique(bl[y==0], return_counts=True))))
