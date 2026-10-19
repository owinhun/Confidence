#!/usr/bin/env python3
"""E1: "why is this clip inconsistent?" — temporal vs semantic (pre-registered).

Every clip is inconsistent; the model must classify the CAUSE:
  (A) correct sound at the wrong time   [temporal: our shifted clips]
  (B) a different sound entirely        [semantic: audio swapped across classes]
A synchrony specialist outputs only an offset posterior and cannot separate these;
visual semantics are required.

Modes: base (no evidence) | verdict | prob. Constrained A/B letter scoring.
Run: CUDA_VISIBLE_DEVICES=k python3 eval_why_task_v1.py --mode base
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
MANIFEST = ROOT / "claude_track2/data/swap_pairs_v1.jsonl"
EXPERT = ROOT / "claude_track2/outputs/synchformer_swap_v1.jsonl"
LETTER_IDS = {"A": 32, "B": 33}
FPS, MAX_FRAMES = 2.0, 64
QUESTION = ("The audio in this clip does not match the video. What is the "
            "problem? (A) The sound is a different sound that does not belong "
            "to this video. (B) The sound is the correct sound for this video "
            "but occurs at the wrong time.")


def build_inputs(processor, video_path: str, question: str):
    messages = [{"role": "user", "content": [
        {"type": "video", "video": video_path, "fps": FPS, "max_frames": MAX_FRAMES},
        {"type": "text", "text": question},
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["base", "verdict", "prob"], required=True)
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    rows = [json.loads(l) for l in MANIFEST.open() if l.strip()]
    exp = {}
    if args.mode != "base":
        exp = {json.loads(l)["clip"]: json.loads(l)
               for l in EXPERT.open() if l.strip()}
        rows = [r for r in rows if r["clip"] in exp]
    out_path = ROOT / f"claude_track2/outputs/why_task_swap_{args.mode}{args.tag}_v1.jsonl"
    print(f"[data] {len(rows)} clips, mode={args.mode}", flush=True)

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="flash_attention_2", local_files_only=True)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
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
        print(f"[resume] {len(done)} done", flush=True)

    n_ok = n_correct = 0
    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            if row["clip"] in done:
                continue
            gt = "B" if row["kind"] == "temporal" else "A"  # swapped layout
            try:
                q = QUESTION
                if args.mode == "verdict":
                    s = exp[row["clip"]]["s_near0"]
                    v = "time-aligned" if s >= 0.175 else "OUT OF SYNC"
                    q = ("An audio-visual synchronization expert model analyzed "
                         f"this clip and judged that the audio and the visual "
                         f"content are {v}. Use this evidence together with the "
                         "video and audio to answer.\n\n") + q
                elif args.mode == "prob":
                    s = exp[row["clip"]]["s_near0"]
                    off = exp[row["clip"]]["argmax_off"]
                    q = ("An audio-visual synchronization expert model analyzed "
                         f"this clip. Probability that audio and video are "
                         f"time-aligned: {s:.2f}; most likely offset: {off:+.1f} "
                         "seconds. Use this evidence together with the video and "
                         "audio to answer.\n\n") + q
                inputs = build_inputs(processor, str(ROOT / row["clip"]), q)
                inputs = inputs.to(model.device).to(torch.bfloat16)
                with torch.inference_mode():
                    out = model(**inputs, use_audio_in_video=True, return_dict=True)
                logits = out.logits[0, -1, :].float()
                ab = torch.tensor([logits[LETTER_IDS["A"]].item(),
                                   logits[LETTER_IDS["B"]].item()])
                probs = F.softmax(ab, dim=0)
                pred = "A" if probs[0] >= 0.5 else "B"
                n_ok += 1
                n_correct += int(pred == gt)
                f.write(json.dumps({
                    "clip": row["clip"], "kind": row["kind"], "answer": gt,
                    "pred": pred, "ytid": row["ytid"],
                    "p_A": round(probs[0].item(), 6),
                    "correct": pred == gt}) + "\n")
                f.flush()
                if n_ok % 50 == 0:
                    print(f"[{i+1}/{len(rows)}] acc {n_correct/n_ok:.4f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"SKIP {row['clip']}: {type(e).__name__}: {e}", flush=True)
    print(f"DONE mode={args.mode} new={n_ok} acc={n_correct/max(1,n_ok):.4f}",
          flush=True)


if __name__ == "__main__":
    main()
