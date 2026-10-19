#!/usr/bin/env python3
"""Pairwise interface control (pre-registered).

The scalar Likert judge is flat against ground-truth offset. Two explanations:
(i) the model cannot perceive the offset; (ii) it perceives it but the absolute
score interface does not express it. This test discriminates them: show the
ALIGNED and the SHIFTED version of the SAME source clip together and ask which
is better synchronized, with the presentation order counterbalanced.

Run: CUDA_VISIBLE_DEVICES=k python3 eval_pairwise_v1.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/workspace/svhalluc/src/qwen")

import torch
import torch.nn.functional as F
from qwen_omni_utils import process_mm_info
from transformers import (Qwen2_5OmniProcessor,
                          Qwen2_5OmniThinkerForConditionalGeneration)

ROOT = Path("/workspace/svhalluc")
MODEL_PATH = ROOT / "models/Qwen2.5-Omni-7B"
MANIFEST = ROOT / "claude_track2/data/vggsparse_pairs_v2.jsonl"
OUT = ROOT / "claude_track2/outputs/judge_pairwise_v1.jsonl"
FPS, MAX_FRAMES = 2.0, 24
QUESTION = ("You have watched two clips. In one of them the audio is correctly "
            "aligned with the video; in the other the audio is shifted in time. "
            "Which clip is better synchronized? (A) The first clip (B) The "
            "second clip")
LID = {"A": 32, "B": 33}


def build_inputs(processor, paths, question):
    content = [{"type": "video", "video": p, "fps": FPS, "max_frames": MAX_FRAMES}
               for p in paths]
    content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True,
                                         tokenize=False)
    audios, images, videos, vk = process_mm_info(
        messages, use_audio_in_video=True, return_video_kwargs=True)
    if isinstance(vk.get("fps"), list) and len(set(vk["fps"])) == 1:
        vk["fps"] = vk["fps"][0]
    return processor(text=text, audio=audios, images=images, videos=videos,
                     return_tensors="pt", padding=True, use_audio_in_video=True, **vk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in MANIFEST.open() if l.strip()]
    pairs = {}
    for r in rows:
        k = f"{r['ytid']}_{r['start_sec']:06d}"
        pairs.setdefault(k, {})[r["label"]] = r["clip"]
    keys = [k for k, v in sorted(pairs.items()) if "A" in v and "B" in v]
    if args.limit:
        keys = keys[:args.limit]
    print(f"[data] {len(keys)} pairs (each scored in both orders)", flush=True)

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="flash_attention_2", local_files_only=True)
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(
        MODEL_PATH, local_files_only=True, use_fast=False)

    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                done.add((d["pair"], d["order"]))

    n_ok = n_corr = 0
    t0 = time.time()
    with OUT.open("a", encoding="utf-8") as f:
        for i, k in enumerate(keys):
            for order in ("aligned_first", "aligned_second"):
                if (k, order) in done:
                    continue
                a, b = pairs[k]["A"], pairs[k]["B"]
                paths = ([str(ROOT / a), str(ROOT / b)] if order == "aligned_first"
                         else [str(ROOT / b), str(ROOT / a)])
                gt = "A" if order == "aligned_first" else "B"
                try:
                    inputs = build_inputs(processor, paths, QUESTION)
                    inputs = inputs.to(model.device).to(torch.bfloat16)
                    with torch.inference_mode():
                        out = model(**inputs, use_audio_in_video=True, return_dict=True)
                    lg = out.logits[0, -1, :].float()
                    ab = torch.tensor([lg[LID["A"]].item(), lg[LID["B"]].item()])
                    p = F.softmax(ab, dim=0)
                    pred = "A" if p[0] >= 0.5 else "B"
                    corr = pred == gt
                    n_ok += 1
                    n_corr += int(corr)
                    f.write(json.dumps({"pair": k, "order": order, "gt": gt,
                                        "pred": pred, "p_first": float(p[0]),
                                        "correct": corr}) + "\n")
                    f.flush()
                    if n_ok % 40 == 0:
                        print(f"[{i+1}/{len(keys)}] acc {n_corr/n_ok:.3f} "
                              f"({time.time()-t0:.0f}s)", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"SKIP {k}/{order}: {type(e).__name__}: {e}", flush=True)
    print(f"DONE n={n_ok} acc={n_corr/max(1,n_ok):.4f}", flush=True)


if __name__ == "__main__":
    main()
