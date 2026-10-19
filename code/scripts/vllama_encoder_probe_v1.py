#!/usr/bin/env python3
"""Second-backbone encoder probe: does AV temporal-alignment info survive at the
ENCODER level of VideoLLaMA2.1-7B-AV (BEATs audio tower + SigLIP visual tower,
different families from Qwen2.5-Omni's Whisper-style/ViT stack)?

On Qwen2.5-Omni all levels were at chance (encoder cross-modal logreg 48%,
nonlinear head 47%). Question: universal info loss, or Qwen-specific?
Verdict rule (pre-registered): >=65% grouped-CV cross-modal accuracy with
unimodal controls at chance => REVIVAL (sync-head method viable on this
backbone); ~50% => universality strengthened.

Streams captured via forward hooks during the model's own AV forward
(generate(max_new_tokens=1) on the TA question, video+audio):
  - mm_projector_a  : input  = BEATs embedding      (T_a, 768)   [raw audio enc]
                      output = projected audio toks (T_a, 3584)  [as fed to LLM]
  - vision_tower    : output = SigLIP per-frame     (t, n, 1152) -> spatial mean
  - mm_projector    : output = STC-connector video  (1, t/2*s, 3584) -> per-step
                      spatial mean                  (t/2, 3584)  [as fed to LLM]
Audio preprocessing note: process_audio_from_video concatenates 8x2s uniformly
spaced clips (time-monotonic, overlapping for 12s videos => 16s) then zero-pads
to 30s; BEATs tokens are time-major (T x F=8 freq patches per 10ms*16 step).
We truncate to the real-audio fraction (16/30) before resampling to 64 steps.

Probes (protocol mirrors the Qwen scripts exactly):
  1. logreg on cross-modal cosine lag-profile features (T=24 grid, lags +-6),
     GroupKFold(5) by ytid  [encoder_probe_sync_v1.py]
  2. nonlinear cross-modal transformer head, leak-free fixed protocol
     [crossmodal_head_v1.py]: grouped 5-fold CV + inner grouped 15% early-stop
     split, per-fold-train standardization, + audio-only / visual-only /
     permuted-label controls.
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path("/workspace/svhalluc")
MANIFEST = ROOT / "claude_track2/data/train_pairs_easy_v1.jsonl"
CACHE = ROOT / "claude_track2/outputs/vllama_encoder_seqs_v1.npz"
T_AUD = 64
N_FRAMES = 16            # probe frame count (0.75 s/frame on 12 s clips); model default is 8
T_VIS_PROJ = N_FRAMES // 2   # STC connector downsamples time 2x
D_A_RAW, D_A_PROJ, D_V_RAW, D_V_PROJ = 768, 3584, 1152, 3584
REAL_AUDIO_FRAC = 16.0 / 30.0  # 8x2s concat clips of a >=2s video, zero-padded to 30s
T_COMMON = 24
LAGS = list(range(-6, 7))
Q = ("When the speech is heard, is the narrated event happening at the same time "
     "in the video? (A) Yes (B) No")

ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["all", "extract", "logreg", "head"], default="all")
ap.add_argument("--extract", action="store_true", help="force re-extraction")
ap.add_argument("--limit", type=int, default=0, help="extraction shape-check: first N clips, no probing")
ap.add_argument("--pair", choices=["proj", "raw"], default="proj",
                help="head feature pair: proj = post-projector both-3584 streams "
                "(as fed to LLM; analogous to Qwen probe); raw = BEATs 768 + SigLIP 1152")
ap.add_argument("--modality", choices=["both", "audio", "visual"], default="both")
ap.add_argument("--permute_labels", action="store_true")
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--patience", type=int, default=25)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()


def resample_T(x, T):
    """mean-pool/nearest-resample (n,d) -> (T,d) preserving temporal order."""
    n = x.shape[0]
    if n >= T:
        return np.stack([c.mean(0) for c in np.array_split(x, T, axis=0)])
    idx = np.minimum((np.arange(T) * n // T), n - 1)
    return x[idx]


# ---------------------------------------------------------------- phase 1: extract
def extract(rows, out_npz, limit=0):
    VLLAMA_ROOT = ROOT / "src/qwen/third_party/video_llama2_av"
    sys.path[:0] = [str(VLLAMA_ROOT / ".deps"), str(VLLAMA_ROOT)]
    import torch
    from videollama2 import model_init
    from videollama2.constants import DEFAULT_VIDEO_TOKEN
    from videollama2.mm_utils import tokenizer_multimodal_token

    if limit:
        rows = rows[:limit]
    model, processor, tokenizer = model_init("DAMO-NLP-SG/VideoLLaMA2.1-7B-AV",
                                             device_map={"": 0}, use_flash_attn=False)
    model.eval()

    print("== top-level modules ==", flush=True)
    for name, mod in model.named_children():
        print(f"  {name}: {type(mod).__name__}")
        for n2, m2 in mod.named_children():
            print(f"    .{n2}: {type(m2).__name__}")

    base = model.get_model() if hasattr(model, "get_model") else model.model
    vision_tower = base.vision_tower
    mm_projector = base.mm_projector
    mm_projector_a = base.mm_projector_a
    print("vision_tower:", type(vision_tower).__name__,
          "| mm_projector:", type(mm_projector).__name__,
          "| mm_projector_a:", type(mm_projector_a).__name__, flush=True)

    cap = {"a_raw": [], "a_proj": [], "v_frames": [], "v_proj": []}

    def hook_aproj(mod, inp, out):
        cap["a_raw"].append(inp[0].detach().float().cpu())
        cap["a_proj"].append(out.detach().float().cpu())

    def hook_vt(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        cap["v_frames"].append(t.detach().float().cpu())

    def hook_vproj(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        cap["v_proj"].append(t.detach().float().cpu())

    hooks = [mm_projector_a.register_forward_hook(hook_aproj),
             vision_tower.register_forward_hook(hook_vt),
             mm_projector.register_forward_hook(hook_vproj)]

    def cat2d(chunks):
        import torch as _t
        ts = [c.reshape(-1, c.shape[-1]) for c in chunks]
        return _t.cat(ts, dim=0).numpy()

    Ar_all, Ap_all, Vf_all, Vp_all, y_all, g_all, ok_idx, alen_all = [], [], [], [], [], [], [], []
    t0 = time.time()
    for k, r in enumerate(rows):
        for v in cap.values():
            v.clear()
        try:
            tensor = processor["video"](str(ROOT / r["clip"]), va=True, num_frames=N_FRAMES)
            vis = [({kk: vv.half().cuda() for kk, vv in tensor.items()}, "video")]
            message = [{"role": "user", "content": DEFAULT_VIDEO_TOKEN + "\n" + Q}]
            prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer_multimodal_token(prompt, tokenizer, DEFAULT_VIDEO_TOKEN,
                                                   return_tensors="pt").unsqueeze(0).long().cuda()
            attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()
            with torch.inference_mode():
                model.generate(input_ids, attention_mask=attention_masks, images=vis,
                               do_sample=False, max_new_tokens=1, use_cache=True,
                               pad_token_id=tokenizer.eos_token_id)

            Ar = cat2d(cap["a_raw"])          # (T_a, 768)  BEATs embedding (30 s incl. pad)
            Ap = cat2d(cap["a_proj"])         # (T_a, 3584) as fed to LLM
            assert Ar.shape[0] == Ap.shape[0]
            n_real = max(1, int(round(Ar.shape[0] * REAL_AUDIO_FRAC)))
            alen_all.append(n_real)
            Ar, Ap = Ar[:n_real], Ap[:n_real]  # drop zero-pad region (time-major tokens)

            Vf = cap["v_frames"][0]           # (t, n_patch, 1152)
            assert Vf.dim() == 3 and Vf.shape[0] == N_FRAMES, f"vision out {tuple(Vf.shape)}"
            Vf = Vf.mean(1).numpy()           # (t, 1152) spatial mean per frame

            Vp = cap["v_proj"][0]             # (1, new_t*spat, 3584)
            Vp = Vp.reshape(-1, Vp.shape[-1]).numpy()
            assert Vp.shape[0] % T_VIS_PROJ == 0, f"v_proj {Vp.shape} not divisible by {T_VIS_PROJ}"
            spat = Vp.shape[0] // T_VIS_PROJ
            Vp = Vp.reshape(T_VIS_PROJ, spat, -1).mean(1)   # (t/2, 3584)

            if k == 0:
                print(f"[shapes] a_raw_full={cat2d(cap['a_raw']).shape} -> real {n_real} "
                      f"| a_proj kept {Ap.shape} | v_frames {Vf.shape} "
                      f"| v_proj {Vp.shape} (spat={spat}) "
                      f"| audio calls={len(cap['a_raw'])} vt calls={len(cap['v_frames'])}", flush=True)

            Ar_all.append(resample_T(Ar, T_AUD).astype(np.float16))
            Ap_all.append(resample_T(Ap, T_AUD).astype(np.float16))
            Vf_all.append(Vf.astype(np.float16))
            Vp_all.append(Vp.astype(np.float16))
            y_all.append(1 if r["label"] == "A" else 0)
            g_all.append(r["ytid"]); ok_idx.append(k)
            if k % 25 == 0:
                print(f"extract {k}/{len(rows)} ok ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(k, "ERR", repr(e), flush=True)
    for h in hooks:
        h.remove()
    np.savez_compressed(out_npz, Ar=np.stack(Ar_all), Ap=np.stack(Ap_all),
                        Vf=np.stack(Vf_all), Vp=np.stack(Vp_all),
                        y=np.array(y_all), g=np.array(g_all),
                        idx=np.array(ok_idx), a_true_len=np.array(alen_all))
    print(f"cached {out_npz}: n={len(y_all)}/{len(rows)} "
          f"({Path(out_npz).stat().st_size/1e6:.0f} MB)", flush=True)
    del model


rows = [json.loads(l) for l in open(MANIFEST) if l.strip()]
if args.stage in ("all", "extract") and (args.extract or args.limit or not CACHE.exists()):
    out = CACHE if not args.limit else CACHE.with_name("vllama_encoder_seqs_shapecheck.npz")
    extract(rows, out, limit=args.limit)
    if args.limit:
        print("shape-check mode: no probing.", flush=True)
        sys.exit(0)
if args.stage == "extract":
    sys.exit(0)

d = np.load(CACHE, allow_pickle=False)
Ar, Ap = d["Ar"].astype(np.float32), d["Ap"].astype(np.float32)
Vf, Vp = d["Vf"].astype(np.float32), d["Vp"].astype(np.float32)
y, g = d["y"], d["g"]
print(f"loaded cache: Ar{Ar.shape} Ap{Ap.shape} Vf{Vf.shape} Vp{Vp.shape} n={len(y)} "
      f"(pos={int(y.sum())}, neg={int((1-y).sum())}, ytids={len(set(g))})", flush=True)

# ------------------------------------------------------- probe 1: lag-profile logreg
if args.stage in ("all", "logreg"):
    def lag_features(A, V):
        A = A - A.mean(0, keepdims=True); V = V - V.mean(0, keepdims=True)
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
        S = A @ V.T
        T = S.shape[0]
        ii, jj = np.meshgrid(np.arange(T), np.arange(T), indexing="ij")
        band = np.abs(ii - jj) <= 1
        band_mean = S[band].mean(); off_mean = S[~band].mean()
        lag_prof = np.array([np.mean(np.diag(S, k=-l)) for l in LAGS])
        best_lag = LAGS[int(np.argmax(lag_prof))]
        return np.concatenate([[band_mean, lag_prof.max(), band_mean - off_mean, best_lag],
                               lag_prof]), best_lag

    feats, bls = [], []
    for i in range(len(y)):
        f, bl = lag_features(resample_T(Ap[i], T_COMMON), resample_T(Vp[i], T_COMMON))
        feats.append(f); bls.append(bl)
    X, bl = np.array(feats), np.array(bls)

    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    sc = cross_val_score(clf, X, y, cv=GroupKFold(5), groups=g)
    print(f"\n===== PROBE 1: cross-modal cosine lag-profile logreg (proj streams) =====")
    print(f"LOGREG (all feats): acc={sc.mean()*100:.1f}% (+-{sc.std()*100:.1f}) "
          f"folds={[f'{s*100:.1f}' for s in sc]}")
    sc2 = cross_val_score(clf, X[:, 4:], y, cv=GroupKFold(5), groups=g)
    print(f"LOGREG (lag profile only): acc={sc2.mean()*100:.1f}% (+-{sc2.std()*100:.1f})")
    pred = (bl == 0).astype(int)
    print(f"HEURISTIC best-lag==0 -> aligned: acc={(pred==y).mean()*100:.1f}%  "
          f"(rate: aligned {np.mean(bl[y==1]==0)*100:.0f}%, shifted {np.mean(bl[y==0]==0)*100:.0f}%)")
    print("best-lag histogram aligned:", dict(zip(*[a.tolist() for a in np.unique(bl[y==1], return_counts=True)])))
    print("best-lag histogram shifted:", dict(zip(*[a.tolist() for a in np.unique(bl[y==0], return_counts=True)])), flush=True)

if args.stage == "logreg":
    sys.exit(0)

# ------------------------------------------------------- probe 2: nonlinear head
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

if args.pair == "proj":
    A, V, d_a, d_v = Ap, Vp, D_A_PROJ, D_V_PROJ
else:
    A, V, d_a, d_v = Ar, Vf, D_A_RAW, D_V_RAW
T_A, T_V = A.shape[1], V.shape[1]

if args.permute_labels:
    y = y[np.random.RandomState(777).permutation(len(y))]
    print("permute_labels: labels randomly permuted (pipeline integrity control)", flush=True)

dev = torch.device("cuda:0")
torch.manual_seed(args.seed); np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True

MODALITY = args.modality
SEQ_LEN = 1 + (T_A if MODALITY in ("both", "audio") else 0) + \
          (T_V if MODALITY in ("both", "visual") else 0)


class CrossModalHead(nn.Module):
    def __init__(self, d_model=256, n_layers=2, n_heads=4, d_ff=512, drop=0.1):
        super().__init__()
        self.proj_a = nn.Linear(d_a, d_model)
        self.proj_v = nn.Linear(d_v, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        self.mod = nn.Parameter(torch.zeros(2, d_model))
        nn.init.normal_(self.cls, std=0.02); nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.mod, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout=drop,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, a, v):
        toks = [self.cls.expand(a.shape[0], -1, -1)]
        if MODALITY in ("both", "audio"):
            toks.append(self.proj_a(a) + self.mod[0])
        if MODALITY in ("both", "visual"):
            toks.append(self.proj_v(v) + self.mod[1])
        x = torch.cat(toks, dim=1) + self.pos
        x = self.enc(x)
        return self.head(x[:, 0]).squeeze(-1)


def run_epoch(model, Xa, Xv, yy, opt=None, batch=16):
    train = opt is not None
    model.train(train)
    idx = np.random.permutation(len(yy)) if train else np.arange(len(yy))
    tot_loss, tot_correct = 0.0, 0
    lossf = nn.BCEWithLogitsLoss(reduction="sum")
    for s in range(0, len(idx), batch):
        b = idx[s:s + batch]
        a = torch.from_numpy(Xa[b]).to(dev); v = torch.from_numpy(Xv[b]).to(dev)
        t = torch.from_numpy(yy[b].astype(np.float32)).to(dev)
        with torch.set_grad_enabled(train):
            logit = model(a, v)
            loss = lossf(logit, t)
        if train:
            opt.zero_grad(); (loss / len(b)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        tot_loss += loss.item()
        tot_correct += ((logit > 0).float() == t).sum().item()
    return tot_loss / len(idx), tot_correct / len(idx)


gkf = GroupKFold(n_splits=5)
fold_res = []
for fi, (tr_all, te) in enumerate(gkf.split(A, y, groups=g)):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    tr_i, va_i = next(gss.split(tr_all, y[tr_all], groups=g[tr_all]))
    tr, va = tr_all[tr_i], tr_all[va_i]
    assert not (set(g[tr]) & set(g[va])) and not (set(g[tr_all]) & set(g[te]))

    mu_a, sd_a = A[tr].reshape(-1, d_a).mean(0), A[tr].reshape(-1, d_a).std(0) + 1e-6
    mu_v, sd_v = V[tr].reshape(-1, d_v).mean(0), V[tr].reshape(-1, d_v).std(0) + 1e-6
    Za, Zv = (A - mu_a) / sd_a, (V - mu_v) / sd_v

    torch.manual_seed(args.seed + fi)
    model = CrossModalHead().to(dev)
    if fi == 0:
        print(f"model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M "
              f"(pair={args.pair}, T_A={T_A}, T_V={T_V})", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_va, best_state, best_ep, bad = math.inf, None, -1, 0
    max_tr_acc, tr_acc_at_best = 0.0, 0.0
    for ep in range(args.epochs):
        # train on the fold's training subset ONLY (leak-free protocol)
        tr_loss, tr_acc = run_epoch(model, Za[tr], Zv[tr], y[tr], opt, batch=args.batch)
        with torch.no_grad():
            va_loss, va_acc = run_epoch(model, Za[va], Zv[va], y[va])
        max_tr_acc = max(max_tr_acc, tr_acc)
        if va_loss < best_va - 1e-4:
            best_va, best_ep, bad = va_loss, ep, 0
            tr_acc_at_best = tr_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % 20 == 0 or bad == args.patience:
            print(f"  f{fi} ep{ep:3d} tr_loss {tr_loss:.4f} tr_acc {tr_acc:.3f} "
                  f"va_loss {va_loss:.4f} va_acc {va_acc:.3f} (best ep{best_ep})", flush=True)
        if bad >= args.patience:
            break

    def _ep(model_, Xa_, Xv_, yy_):
        with torch.no_grad():
            return run_epoch(model_, Xa_, Xv_, yy_)

    model.load_state_dict(best_state)
    _, te_acc = _ep(model, Za[te], Zv[te], y[te])
    _, tr_full_acc = _ep(model, Za[tr], Zv[tr], y[tr])
    fold_res.append(dict(fold=fi, test_acc=te_acc, n_test=len(te),
                         train_acc_at_best=tr_full_acc, train_acc_running_at_best=tr_acc_at_best,
                         max_train_acc=max_tr_acc, best_epoch=best_ep, epochs_run=ep + 1))
    print(f"FOLD {fi}: test {te_acc*100:.1f}% (n={len(te)}) | train@best {tr_full_acc*100:.1f}% "
          f"| max train acc {max_tr_acc*100:.1f}% | best ep {best_ep}/{ep+1}", flush=True)

te = np.array([r["test_acc"] for r in fold_res])
tr = np.array([r["train_acc_at_best"] for r in fold_res])
mx = np.array([r["max_train_acc"] for r in fold_res])
print(f"\n===== PROBE 2 RESULTS (pair={args.pair}, modality={MODALITY}, "
      f"permuted={args.permute_labels}) =====", flush=True)
print(f"test-fold acc:  {te.mean()*100:.1f}% +- {te.std()*100:.1f}  "
      f"folds={[f'{x*100:.1f}' for x in te]}")
print(f"train acc @best-val checkpoint: {tr.mean()*100:.1f}%")
print(f"max train acc over epochs:      {mx.mean()*100:.1f}%")
print(json.dumps(fold_res, indent=1))
