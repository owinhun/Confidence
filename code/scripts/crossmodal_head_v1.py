#!/usr/bin/env python3
"""Final in-budget variant: small NONLINEAR trainable cross-modal head on RAW frozen
encoder sequences of Qwen2.5-Omni. Linear probes were at chance on (a) LLM hidden
states (51%) and (b) encoder cross-modal cosine-correlation features (48%). Question:
can a 2-layer transformer over the raw [audio; visual] token sequences find what
linear probes missed? Pre-registered protocol:
  - 5-fold GroupKFold by ytid over all 246 easy clips
  - within each fold's training portion, ~15% grouped holdout for early stopping
  - AdamW lr 1e-4, batch 16, up to 200 epochs, early stop on grouped-val loss
  - report mean+-std test-fold accuracy AND train accuracy (overfit diagnostic)
Verdict thresholds: >=65% strong / 55-65% weak-positive / ~50% negative.

Phase 1 (--extract or missing cache): hook audio_tower + visual encoder outputs
(same approach as encoder_probe_sync_v1.py), pool audio to 64 steps x 3584 and
visual to its 12 temporal steps x 3584 (spatial mean-pool), cache one npz.
Phase 2: train the head on GPU (plain torch).

Architecture choice (noted per task): CONCATENATED-sequence transformer —
[CLS; audio(64); visual(12)] tokens, each modality linearly projected 3584->256,
learned positional + modality embeddings, 2-layer TransformerEncoder (pre-norm,
4 heads, ff 512), CLS pooling -> binary logit. Chosen over audio->visual
cross-attention because full self-attention lets the model discover ANY
cross-modal temporal pairing (incl. within-modality context), strictly more
expressive at this tiny scale.
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path("/workspace/svhalluc")
MANIFEST = ROOT / "claude_track2/data/train_pairs_easy_v1.jsonl"
CACHE = ROOT / "claude_track2/outputs/encoder_raw_seqs_v1.npz"
T_AUD, T_VIS, D_ENC = 64, 12, 3584

ap = argparse.ArgumentParser()
ap.add_argument("--extract", action="store_true", help="force re-extraction of raw sequences")
ap.add_argument("--limit", type=int, default=0, help="extraction shape-check: first N clips only, no training")
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--patience", type=int, default=25)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--modality", choices=["both", "audio", "visual"], default="both",
                help="ablation control: audio-only / visual-only token sequences. "
                "Audio-only ~chance is REQUIRED for a genuine cross-modal claim "
                "(audio alone from an unseen ytid cannot legitimately reveal alignment); "
                "visual-only must be ~50% by construction (identical videos within pairs).")
ap.add_argument("--permute_labels", action="store_true",
                help="pipeline integrity control: randomly permute labels across all "
                "clips before CV; a sound pipeline must yield ~50% test accuracy.")
ap.add_argument("--shuffle_time", action="store_true",
                help="interpretive control: randomly permute token order WITHIN each "
                "clip's audio and visual sequences (fixed per clip). If accuracy holds, "
                "the cue is time-invariant content correspondence; if it drops, "
                "temporal structure carries the signal.")
args = ap.parse_args()


# ---------------------------------------------------------------- phase 1: extract
def resample_T(x, T):
    """mean-pool/nearest-resample (n,d) -> (T,d) preserving temporal order."""
    n = x.shape[0]
    if n >= T:
        return np.stack([c.mean(0) for c in np.array_split(x, T, axis=0)])
    idx = np.minimum((np.arange(T) * n // T), n - 1)
    return x[idx]


def extract(rows, out_npz, limit=0):
    import torch
    sys.path.insert(0, str(ROOT / "src/qwen"))
    from qwen_omni_utils import process_mm_info
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    if limit:
        rows = rows[:limit]
    Q = ("When the speech is heard, is the narrated event happening at the same time "
         "in the video? (A) Yes (B) No")
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        ROOT / "models/Qwen2.5-Omni-7B", dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="flash_attention_2", local_files_only=True)
    model.eval()
    proc = Qwen2_5OmniProcessor.from_pretrained(ROOT / "models/Qwen2.5-Omni-7B",
                                                local_files_only=True, use_fast=False)
    audio_tower = getattr(model, "audio_tower", None) or model.model.audio_tower
    visual = getattr(model, "visual", None) or model.model.visual
    cap = {"audio": [], "visual": []}

    def _grab(store):
        def hook(mod, inp, out):
            t = out
            if hasattr(t, "last_hidden_state"):
                t = t.last_hidden_state
            if isinstance(t, tuple):
                t = t[0]
            store.append(t.detach().float().cpu())
        return hook

    h1 = audio_tower.register_forward_hook(_grab(cap["audio"]))
    h2 = visual.register_forward_hook(_grab(cap["visual"]))

    def cat2d(chunks):
        ts = [c.reshape(-1, c.shape[-1]) if c.dim() > 2 else c for c in chunks]
        return torch.cat(ts, dim=0).numpy()

    A_all, V_all, y_all, g_all, ok_idx = [], [], [], [], []
    t0 = time.time()
    for k, r in enumerate(rows):
        cap["audio"].clear(); cap["visual"].clear()
        try:
            msgs = [{"role": "user", "content": [
                {"type": "video", "video": str(ROOT / r["clip"]), "fps": 2.0, "max_frames": 24},
                {"type": "text", "text": Q}]}]
            text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            a, i, v, vk = process_mm_info(msgs, use_audio_in_video=True, return_video_kwargs=True)
            if isinstance(vk.get("fps"), list) and len(vk["fps"]) == 1:
                vk["fps"] = vk["fps"][0]
            ins = proc(text=text, audio=a, images=i, videos=v, return_tensors="pt", padding=True,
                       use_audio_in_video=True, **vk).to(model.device).to(model.dtype)
            grid = ins["video_grid_thw"][0].tolist()
            with torch.inference_mode():
                model(**ins, return_dict=True, use_audio_in_video=True)
            A = cat2d(cap["audio"])            # (n_audio_tokens, 3584)
            Vt = cat2d(cap["visual"])          # (grid_t * spat, 3584)
            gt = grid[0]
            spat = Vt.shape[0] // gt
            assert spat * gt == Vt.shape[0], f"visual {Vt.shape[0]} not divisible by grid_t {gt}"
            Vframes = Vt.reshape(gt, spat, -1).mean(1)      # (gt, 3584) spatial mean-pool
            A_all.append(resample_T(A, T_AUD).astype(np.float16))
            V_all.append(resample_T(Vframes, T_VIS).astype(np.float16))
            y_all.append(1 if r["label"] == "A" else 0)
            g_all.append(r["ytid"]); ok_idx.append(k)
            if k == 0:
                print(f"[shapes] audio_raw={A.shape} visual_raw={Vt.shape} grid={grid} "
                      f"-> A pooled {A_all[0].shape}, V pooled {V_all[0].shape}", flush=True)
            if k % 25 == 0:
                print(f"extract {k}/{len(rows)} ok ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(k, "ERR", repr(e), flush=True)
    h1.remove(); h2.remove()
    A_arr = np.stack(A_all); V_arr = np.stack(V_all)
    np.savez_compressed(out_npz, A=A_arr, V=V_arr, y=np.array(y_all),
                        g=np.array(g_all), idx=np.array(ok_idx))
    print(f"cached {out_npz}: A{A_arr.shape} V{V_arr.shape} n={len(y_all)}/{len(rows)} "
          f"({Path(out_npz).stat().st_size/1e6:.0f} MB)", flush=True)
    del model
    return out_npz


rows = [json.loads(l) for l in open(MANIFEST) if l.strip()]
if args.extract or args.limit or not CACHE.exists():
    out = CACHE if not args.limit else CACHE.with_name("encoder_raw_seqs_shapecheck.npz")
    extract(rows, out, limit=args.limit)
    if args.limit:
        print("shape-check mode: no training.", flush=True)
        sys.exit(0)

# ---------------------------------------------------------------- phase 2: train
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

d = np.load(CACHE, allow_pickle=False)
A, V, y, g = d["A"].astype(np.float32), d["V"].astype(np.float32), d["y"], d["g"]
print(f"loaded cache: A{A.shape} V{V.shape} n={len(y)} "
      f"(pos={int(y.sum())}, neg={int((1-y).sum())}, ytids={len(set(g))})", flush=True)
if args.permute_labels:
    y = y[np.random.RandomState(777).permutation(len(y))]
    print("permute_labels: labels randomly permuted (pipeline integrity control)", flush=True)
if args.shuffle_time:
    rs = np.random.RandomState(1234)
    for i in range(len(y)):
        A[i] = A[i][rs.permutation(A.shape[1])]
        V[i] = V[i][rs.permutation(V.shape[1])]
    print("shuffle_time: per-clip random temporal permutation applied to both streams", flush=True)

dev = torch.device("cuda:0")
torch.manual_seed(args.seed); np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True


MODALITY = args.modality
SEQ_LEN = 1 + (T_AUD if MODALITY in ("both", "audio") else 0) + \
          (T_VIS if MODALITY in ("both", "visual") else 0)


class CrossModalHead(nn.Module):
    def __init__(self, d_enc=D_ENC, d_model=256, n_layers=2, n_heads=4, d_ff=512, drop=0.1):
        super().__init__()
        self.proj_a = nn.Linear(d_enc, d_model)
        self.proj_v = nn.Linear(d_enc, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        self.mod = nn.Parameter(torch.zeros(2, d_model))      # modality embeddings
        nn.init.normal_(self.cls, std=0.02); nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.mod, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout=drop,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, a, v):                                   # a:(B,64,De) v:(B,12,De)
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
    # grouped ~15% early-stop holdout inside the fold's training portion
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    tr_i, va_i = next(gss.split(tr_all, y[tr_all], groups=g[tr_all]))
    tr, va = tr_all[tr_i], tr_all[va_i]
    assert not (set(g[tr]) & set(g[va])) and not (set(g[tr_all]) & set(g[te]))

    # per-dim standardization from THIS fold's training portion only
    mu_a, sd_a = A[tr].reshape(-1, D_ENC).mean(0), A[tr].reshape(-1, D_ENC).std(0) + 1e-6
    mu_v, sd_v = V[tr].reshape(-1, D_ENC).mean(0), V[tr].reshape(-1, D_ENC).std(0) + 1e-6
    Za, Zv = (A - mu_a) / sd_a, (V - mu_v) / sd_v

    torch.manual_seed(args.seed + fi)
    model = CrossModalHead().to(dev)
    if fi == 0:
        n_par = sum(p.numel() for p in model.parameters())
        print(f"model params: {n_par/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_va, best_state, best_ep, bad = math.inf, None, -1, 0
    max_tr_acc, tr_acc_at_best = 0.0, 0.0
    for ep in range(args.epochs):
        # CRITICAL: train on the fold's training subset ONLY (a previous version
        # passed the full arrays here -> memorization of test/val = invalid results)
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
print(f"\n================ RESULTS (modality={MODALITY}) ================", flush=True)
print(f"test-fold acc:  {te.mean()*100:.1f}% +- {te.std()*100:.1f}  "
      f"folds={[f'{x*100:.1f}' for x in te]}")
print(f"train acc @best-val checkpoint: {tr.mean()*100:.1f}% "
      f"folds={[f'{x*100:.1f}' for x in tr]}")
print(f"max train acc over epochs:      {mx.mean()*100:.1f}% "
      f"folds={[f'{x*100:.1f}' for x in mx]}")
print(json.dumps(fold_res, indent=1))
