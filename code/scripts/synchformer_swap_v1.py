#!/usr/bin/env python3
"""VGGSound-Sparse pair scoring (amended protocol in log_v1.md): can Synchformer (ICASSP'24, AudioSet ckpt) detect
audio-visual misalignment in OUR narration-domain pairs (delta=2 s, in-grid)?

Pre-registered protocol: see outputs/log_v1.md entry [2026-08-05] Teacher-viability probe.
Runs from anywhere; imports Synchformer from claude_track2/external/Synchformer.
Resumes by clip path. GPU 0 only.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

ROOT = Path("/workspace/svhalluc")
SF = ROOT / "claude_track2/external/Synchformer"
sys.path.insert(0, str(SF))
os.chdir(SF)  # ckpt paths in cfg are relative

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from dataset.dataset_utils import get_video_and_audio  # noqa: E402
from dataset.transforms import make_class_grid  # noqa: E402
from scripts.train_utils import get_model, get_transforms, prepare_inputs  # noqa: E402

EXP = "24-01-04T16-39-21"
MANIFEST = ROOT / "claude_track2/data/swap_pairs_v1.jsonl"
OUT = ROOT / "claude_track2/outputs/synchformer_swap_v1.jsonl"
TMP = Path(os.environ.get("CLAUDE_JOB_DIR", "/tmp")) / "tmp" / "sf_swap_reenc"
TMP.mkdir(parents=True, exist_ok=True)
V_STARTS = [1.0]  # 6 s clips admit one 5 s crop
VFPS, AFPS, SIDE = 25, 16000, 256


def patch_config(cfg):
    cfg.model.params.afeat_extractor.params.ckpt_path = None
    cfg.model.params.vfeat_extractor.params.ckpt_path = None
    cfg.model.params.transformer.target = cfg.model.params.transformer.target.replace(
        ".modules.feature_selector.", ".sync_model.")
    return cfg


def reencode(src: Path, dst: Path) -> bool:
    if dst.exists() and dst.stat().st_size > 0:
        return True
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-vf", f"fps={VFPS},scale=iw*{SIDE}/'min(iw,ih)':ih*{SIDE}/'min(iw,ih)',"
                "crop='trunc(iw/2)'*2:'trunc(ih/2)'*2",
         "-ar", str(AFPS), str(dst)], capture_output=True)
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def main():
    rows = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    print(f"delta=2 clips: {len(rows)}")

    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["clip"])

    cfg = OmegaConf.load(f"logs/sync_models/{EXP}/cfg-{EXP}.yaml")
    cfg = patch_config(cfg)
    device = torch.device("cuda:0")
    _, model = get_model(cfg, device)
    ckpt = torch.load(f"logs/sync_models/{EXP}/{EXP}.pt",
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    grid = make_class_grid(-cfg.data.max_off_sec, cfg.data.max_off_sec,
                           cfg.model.params.transformer.params.off_head_cfg.params.out_features)
    near0 = [i for i, g in enumerate(grid) if abs(float(g)) <= 0.3]
    tfm = get_transforms(cfg, ["test"])["test"]

    n_new = 0
    with OUT.open("a", encoding="utf-8") as f:
        for r in rows:
            if r["clip"] in done:
                continue
            src = ROOT / r["clip"]
            dst = TMP / (Path(r["clip"]).stem + "_re.mp4")
            if not reencode(src, dst):
                print(f"REENC FAIL {r['clip']}")
                continue
            try:
                rgb, audio, meta = get_video_and_audio(str(dst), get_meta=True)
                probs_acc = None
                for vs in V_STARTS:
                    item = dict(video=rgb, audio=audio, meta=meta, path=str(dst),
                                split="test",
                                targets={"v_start_i_sec": vs, "offset_sec": 0.0})
                    item = tfm(item)
                    batch = torch.utils.data.default_collate([item])
                    aud, vid, _ = prepare_inputs(batch, device)
                    with torch.no_grad(), torch.autocast("cuda", enabled=cfg.training.use_half_precision):
                        _, logits = model(vid, aud)
                    p = torch.softmax(logits.float(), dim=-1)[0].cpu()
                    probs_acc = p if probs_acc is None else probs_acc + p
                probs = (probs_acc / len(V_STARTS)).tolist()
                s_near0 = float(sum(probs[i] for i in near0))
                amax = int(max(range(len(probs)), key=lambda i: probs[i]))
                f.write(json.dumps({
                    "clip": r["clip"], "label": "A" if r["kind"]=="temporal" else "B",
                    "ytid": r["ytid"], "direction": r["kind"], "s_near0": s_near0,
                    "argmax_off": float(grid[amax]), "probs": probs}) + "\n")
                f.flush()
                n_new += 1
                if n_new % 20 == 0:
                    print(f"{n_new} done")
            except Exception as e:
                print(f"FAIL {r['clip']}: {type(e).__name__}: {e}")
            finally:
                dst.unlink(missing_ok=True)
    print(f"DONE: {n_new} new results -> {OUT}")


if __name__ == "__main__":
    main()
