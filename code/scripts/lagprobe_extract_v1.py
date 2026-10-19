#!/usr/bin/env python3
"""Layer-resolved time-binned representation extraction (pre-registered; log_v1.md).

For each easy-subset clip: one forward through Qwen2.5-Omni Thinker with
output_hidden_states=True; locate audio/video placeholder-token spans; split
each span into 32 temporal bins; save per-bin mean vectors at selected layers.

Output: outputs/lagprobe_bins_v1.npz  (per layer: A [N,32,D], V [N,32,D]),
plus labels/ytids. Resumes via a .part dir of per-clip npy files.
Run: CUDA_VISIBLE_DEVICES=1 python3 scripts/lagprobe_extract_v1.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.path.insert(0, "/workspace/svhalluc/src/qwen")

import numpy as np
import torch
from qwen_omni_utils import process_mm_info
from transformers import (Qwen2_5OmniProcessor,
                          Qwen2_5OmniThinkerForConditionalGeneration)

ROOT = Path("/workspace/svhalluc")
MODEL_PATH = ROOT / "models/Qwen2.5-Omni-7B"
MANIFEST = ROOT / "claude_track2/data/train_pairs_v2.jsonl"
PART = ROOT / "claude_track2/outputs/lagprobe_part_v1"
OUT = ROOT / "claude_track2/outputs/lagprobe_bins_v1.npz"
LAYERS = [0, 4, 8, 14, 20, 27]
NBINS = 32
FPS, MAX_FRAMES = 2.0, 64
QUESTION = ("When the speech is heard, is the narrated event happening at the "
            "same time in the video? (A) Yes (B) No")


def build_inputs(processor, video_path: str):
    messages = [{"role": "user", "content": [
        {"type": "video", "video": video_path, "fps": FPS, "max_frames": MAX_FRAMES},
        {"type": "text", "text": QUESTION},
    ]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True,
                                         tokenize=False)
    audios, images, videos, video_kwargs = process_mm_info(
        messages, use_audio_in_video=True, return_video_kwargs=True)
    if isinstance(video_kwargs.get("fps"), list) and len(video_kwargs["fps"]) == 1:
        video_kwargs["fps"] = video_kwargs["fps"][0]
    return processor(text=text, audio=audios, images=images, videos=videos,
                     return_tensors="pt", padding=True,
                     use_audio_in_video=True, **video_kwargs)


def bin_mean(x: torch.Tensor, nbins: int) -> np.ndarray:
    """x: [T, D] -> [nbins, D] by equal temporal binning."""
    T = x.shape[0]
    idx = np.linspace(0, T, nbins + 1).astype(int)
    out = np.stack([x[idx[i]:max(idx[i] + 1, idx[i + 1])].mean(0).float().cpu().numpy()
                    for i in range(nbins)])
    return out.astype(np.float16)


def main():
    rows = [json.loads(l) for l in MANIFEST.open() if l.strip()]
    rows = [r for r in rows if r["delta"] >= 8]
    print(f"easy subset clips: {len(rows)}")
    PART.mkdir(parents=True, exist_ok=True)

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="flash_attention_2", local_files_only=True)
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(
        MODEL_PATH, local_files_only=True, use_fast=False)

    cfg = model.config
    audio_id = getattr(cfg, "audio_token_index", None)
    video_id = getattr(cfg, "video_token_index", None)
    assert audio_id is not None and video_id is not None, \
        f"token indices missing: audio={audio_id} video={video_id}"
    print(f"audio_token_index={audio_id} video_token_index={video_id}")

    n_new = 0
    for r in rows:
        stem = Path(r["clip"]).stem
        dst = PART / f"{stem}.npz"
        if dst.exists():
            continue
        try:
            inputs = build_inputs(processor, str(ROOT / r["clip"]))
            ids = inputs["input_ids"][0]
            a_pos = (ids == audio_id).nonzero().squeeze(-1)
            v_pos = (ids == video_id).nonzero().squeeze(-1)
            assert len(a_pos) >= NBINS and len(v_pos) >= NBINS, \
                f"too few tokens: a={len(a_pos)} v={len(v_pos)}"
            inputs = inputs.to(model.device).to(torch.bfloat16)
            with torch.inference_mode():
                out = model(**inputs, use_audio_in_video=True,
                            output_hidden_states=True, return_dict=True)
            payload = {}
            for L in LAYERS:
                h = out.hidden_states[L][0]  # [T, D]
                payload[f"A{L}"] = bin_mean(h[a_pos.to(h.device)], NBINS)
                payload[f"V{L}"] = bin_mean(h[v_pos.to(h.device)], NBINS)
            np.savez_compressed(dst, label=r["label"], ytid=r["ytid"], **payload)
            n_new += 1
            if n_new % 20 == 0:
                print(f"{n_new} done", flush=True)
            del out
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {stem}: {type(e).__name__}: {e}", flush=True)
    print(f"extraction DONE, new={n_new}")

    # consolidate
    files = sorted(PART.glob("*.npz"))
    labels, ytids = [], []
    stacks = {f"{m}{L}": [] for L in LAYERS for m in "AV"}
    for fp in files:
        z = np.load(fp, allow_pickle=True)
        labels.append(str(z["label"])); ytids.append(str(z["ytid"]))
        for k in stacks:
            stacks[k].append(z[k])
    np.savez_compressed(OUT, labels=np.array(labels), ytids=np.array(ytids),
                        **{k: np.stack(v) for k, v in stacks.items()})
    print(f"consolidated {len(files)} clips -> {OUT}")


if __name__ == "__main__":
    main()
