#!/usr/bin/env python3
"""Synchformer -> SVHalluc TA transfer probe (pre-registered; see log_v1.md).

For each TA eval clip: re-encode to 25fps/16kHz/256, take up to 4 evenly spaced
5 s crops, score s_near0 = P(|offset| <= 0.3 s) per crop. Writes mean/max per
clip. Resumes by sample_id. GPU 0 only.
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
os.chdir(SF)

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from dataset.dataset_utils import get_video_and_audio  # noqa: E402
from dataset.transforms import make_class_grid  # noqa: E402
from scripts.train_utils import get_model, get_transforms, prepare_inputs  # noqa: E402

EXP = "24-01-04T16-39-21"
MANIFEST = ROOT / "data/ter_large_eval_v1.jsonl"
OUT = ROOT / "claude_track2/outputs/synchformer_ta_probe_v1.jsonl"
TMP = Path(os.environ.get("CLAUDE_JOB_DIR", "/tmp")) / "tmp" / "sf_ta_reenc"
TMP.mkdir(parents=True, exist_ok=True)
VFPS, AFPS, SIDE = 25, 16000, 256
MAX_CROPS = 4


def patch_config(cfg):
    cfg.model.params.afeat_extractor.params.ckpt_path = None
    cfg.model.params.vfeat_extractor.params.ckpt_path = None
    cfg.model.params.transformer.target = cfg.model.params.transformer.target.replace(
        ".modules.feature_selector.", ".sync_model.")
    return cfg


def reencode(src: Path, dst: Path, ss: float | None = None, t: float | None = None) -> bool:
    if dst.exists() and dst.stat().st_size > 0:
        return True
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if ss is not None:
        cmd += ["-ss", str(ss)]
    if t is not None:
        cmd += ["-t", str(t)]
    cmd += ["-i", str(src),
            "-vf", f"fps={VFPS},scale=iw*{SIDE}/'min(iw,ih)':ih*{SIDE}/'min(iw,ih)',"
                   "crop='trunc(iw/2)'*2:'trunc(ih/2)'*2",
            "-ar", str(AFPS), str(dst)]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def dur_of(p: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", p], capture_output=True, text=True)
    return float(out.stdout.strip())


def main():
    rows = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["task"] == "TA"]
    print(f"TA clips: {len(rows)}")

    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["sample_id"])

    cfg = OmegaConf.load(f"logs/sync_models/{EXP}/cfg-{EXP}.yaml")
    cfg = patch_config(cfg)
    device = torch.device("cuda:0")
    _, model = get_model(cfg, device)
    ckpt = torch.load(f"logs/sync_models/{EXP}/{EXP}.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    grid = make_class_grid(-cfg.data.max_off_sec, cfg.data.max_off_sec,
                           cfg.model.params.transformer.params.off_head_cfg.params.out_features)
    near0 = [i for i, g in enumerate(grid) if abs(float(g)) <= 0.3]
    tfm = get_transforms(cfg, ["test"])["test"]

    n_new = 0
    with OUT.open("a", encoding="utf-8") as f:
        for r in rows:
            if r["sample_id"] in done:
                continue
            src = Path(r["video_path"])
            try:
                d = dur_of(str(src))
                # crop starts on the ORIGINAL timeline; each crop becomes its own
                # 7 s sub-clip (transform caps audio at 10 s), model sees [1, 6] of it
                hi = max(0.0, d - 7.0)
                k = min(MAX_CROPS, max(1, int(hi // 4) + 1))
                starts = [i * hi / max(1, k - 1) for i in range(k)] if k > 1 else [0.0]
                scores, argmaxes = [], []
                for ci, ss in enumerate(starts):
                    dst = TMP / f"{r['sample_id']}_c{ci}_re.mp4"
                    if not reencode(src, dst, ss=ss, t=7.0):
                        raise RuntimeError(f"reencode crop {ci}")
                    try:
                        rgb, audio, meta = get_video_and_audio(str(dst), get_meta=True)
                        item = dict(video=rgb, audio=audio, meta=meta, path=str(dst),
                                    split="test",
                                    targets={"v_start_i_sec": 1.0, "offset_sec": 0.0})
                        item = tfm(item)
                        batch = torch.utils.data.default_collate([item])
                        aud, vid, _ = prepare_inputs(batch, device)
                        with torch.no_grad(), torch.autocast("cuda", enabled=cfg.training.use_half_precision):
                            _, logits = model(vid, aud)
                        p = torch.softmax(logits.float(), dim=-1)[0].cpu()
                        scores.append(float(sum(p[i] for i in near0)))
                        argmaxes.append(float(grid[int(p.argmax())]))
                    finally:
                        dst.unlink(missing_ok=True)
                f.write(json.dumps({
                    "sample_id": r["sample_id"], "answer": r["answer"],
                    "video_id": r["video_id"], "dur": d, "starts": starts,
                    "s_list": scores, "argmax_list": argmaxes,
                    "s_mean": sum(scores) / len(scores), "s_max": max(scores)}) + "\n")
                f.flush()
                n_new += 1
                if n_new % 25 == 0:
                    print(f"{n_new} done")
            except Exception as e:
                print(f"FAIL {r['sample_id']}: {type(e).__name__}: {e}")
    print(f"DONE: {n_new} new results -> {OUT}")


if __name__ == "__main__":
    main()
