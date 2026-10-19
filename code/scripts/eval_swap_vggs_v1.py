#!/usr/bin/env python3
"""Frozen Qwen2.5-Omni on VGGSound-Sparse aligned/shifted pairs — baseline and
expert-injection (T2) conditions, per the amended pre-registered protocol.

Usage:
  CUDA_VISIBLE_DEVICES=1 python3 eval_vggsparse_qwen_v1.py --mode base
  CUDA_VISIBLE_DEVICES=1 python3 eval_vggsparse_qwen_v1.py --mode inject
`inject` requires outputs/synchformer_vggsparse_v1.jsonl (expert scores).
Constrained A/B letter scoring; resumes by clip.
"""

from __future__ import annotations

import argparse
import json
import os
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
EXPERT = ROOT / "claude_track2/outputs/synchformer_vggsparse_v1.jsonl"
LETTER_IDS = {"A": 32, "B": 33}
FPS, MAX_FRAMES = 2.0, 64
QUESTION = ("Is the audio track temporally aligned with the visual content of "
            "this video? (A) Yes (B) No")


def expert_report(s: float) -> str:
    return (
        "An audio-visual synchronization expert model analyzed this clip. "
        f"Estimated probability that the audio and the visual content are "
        f"time-aligned: {s:.2f}. Probabilities near 0 indicate the audio is "
        "out of sync with the video; higher values indicate alignment. Use "
        "this evidence together with the video and audio to answer.\n\n")


def build_inputs(processor, video_path: str, question: str, no_av: bool = False):
    content = [] if no_av else [
        {"type": "video", "video": video_path, "fps": FPS, "max_frames": MAX_FRAMES}]
    content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True,
                                         tokenize=False)
    audios, images, videos, video_kwargs = process_mm_info(
        messages, use_audio_in_video=True, return_video_kwargs=True)
    if isinstance(video_kwargs.get("fps"), list) and len(video_kwargs["fps"]) == 1:
        video_kwargs["fps"] = video_kwargs["fps"][0]
    return processor(text=text, audio=audios, images=images, videos=videos,
                     return_tensors="pt", padding=True,
                     use_audio_in_video=True, **video_kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["base", "inject"], required=True)
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--suffix", type=str, default="")
    ap.add_argument("--no_av", action="store_true", help="text-only: drop video/audio input")
    ap.add_argument("--permute", action="store_true", help="shuffle expert scores across clips (seed 0)")
    ap.add_argument("--binary", action="store_true", help="ablation: binary verdict wording (thr 0.175)")
    ap.add_argument("--swap", action="store_true", help="swap option layout")
    args = ap.parse_args()

    rows = [json.loads(l) for l in MANIFEST.open() if l.strip()]
    exp = {}
    if args.mode == "inject":
        exp = {json.loads(l)["clip"]: json.loads(l)
               for l in EXPERT.open() if l.strip()}
        rows = [r for r in rows if r["clip"] in exp]
        if args.permute:
            import numpy as np
            keys = [r["clip"] for r in rows]
            vals = [exp[k]["s_near0"] for k in keys]
            perm = np.random.RandomState(0).permutation(len(vals))
            exp = {k: {"s_near0": vals[perm[i]]} for i, k in enumerate(keys)}
    out_path = ROOT / f"claude_track2/outputs/eval_vggsparse_qwen_{args.mode}{args.suffix}_v1.jsonl"
    print(f"[data] {len(rows)} clips, mode={args.mode}")

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="flash_attention_2", local_files_only=True)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
        print(f"[model] adapter applied: {args.adapter}")
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(
        MODEL_PATH, local_files_only=True, use_fast=False)
    tok = processor.tokenizer
    for letter, expect in LETTER_IDS.items():
        assert tok.encode(letter) == [expect]

    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["clip"])
        print(f"[resume] {len(done)} done")

    n_correct = n_ok = 0
    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            if row["clip"] in done:
                continue
            try:
                q = QUESTION
                label = row["label"]
                if args.swap:
                    q = ("Is the audio track temporally aligned with the visual "
                         "content of this video? (A) No (B) Yes")
                    label = {"A": "B", "B": "A"}[label]
                if args.mode == "inject":
                    if args.binary:
                        verdict = ("time-aligned" if exp[row["clip"]]["s_near0"] >= 0.175
                                   else "OUT OF SYNC")
                        q = ("An audio-visual synchronization expert model analyzed "
                             f"this clip and judged that the audio and the visual "
                             f"content are {verdict}. Use this evidence together "
                             "with the video and audio to answer.\n\n") + q
                    else:
                        q = expert_report(exp[row["clip"]]["s_near0"]) + q
                inputs = build_inputs(processor, str(ROOT / row["clip"]), q, no_av=args.no_av)
                inputs = inputs.to(model.device).to(torch.bfloat16)
                with torch.inference_mode():
                    out = model(**inputs, use_audio_in_video=True, return_dict=True)
                logits = out.logits[0, -1, :].float()
                ab = torch.tensor([logits[LETTER_IDS["A"]].item(),
                                   logits[LETTER_IDS["B"]].item()])
                probs = F.softmax(ab, dim=0)
                pred = "A" if probs[0] >= 0.5 else "B"
                correct = pred == label
                n_ok += 1
                n_correct += int(correct)
                f.write(json.dumps({
                    "clip": row["clip"], "label": label, "pred": pred,
                    "ytid": row["ytid"], "direction": row["direction"],
                    "p_A": round(probs[0].item(), 6), "correct": correct}) + "\n")
                f.flush()
                if n_ok % 25 == 0:
                    print(f"[{i+1}/{len(rows)}] acc {n_correct/n_ok:.4f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"SKIP {row['clip']}: {type(e).__name__}: {e}", flush=True)
    print(f"DONE mode={args.mode} new={n_ok} acc(new)={n_correct/max(1,n_ok):.4f}")


if __name__ == "__main__":
    main()
