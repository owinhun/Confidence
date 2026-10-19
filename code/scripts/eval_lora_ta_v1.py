#!/usr/bin/env python3
"""Constrained-scoring eval on SVHalluc TA with an optional LoRA adapter.

Loads Qwen2.5-Omni-7B Thinker (bf16), optionally applies a saved LoRA
adapter, then for every TA row of the manifest scores the letter logits
(A=32 / B=33) at the first generated position — single forward pass, no
generation — and writes jsonl results + prints accuracy.

Run (GPU1):
  CUDA_VISIBLE_DEVICES=1 python claude_track2/scripts/eval_lora_ta_v1.py \
      --adapter claude_track2/outputs/lora_sync_v1/epoch2 --limit 3
Omit --adapter for the frozen base model. --limit N processes only the
first N samples (smoke test).
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
MANIFEST = ROOT / "data/ter_large_eval_v1.jsonl"
LETTER_IDS = {"A": 32, "B": 33}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", type=str, default=None,
                   help="path to a saved peft adapter dir; omit for base model")
    p.add_argument("--out", type=str, default=None,
                   help="output jsonl (default derived from adapter name)")
    p.add_argument("--manifest", type=str, default=str(MANIFEST))
    p.add_argument("--limit", type=int, default=0, help="if >0, only N samples")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--max_frames", type=int, default=64)
    return p.parse_args()


def build_inputs(processor, video_path: str, question: str, fps: float,
                 max_frames: int):
    messages = [{"role": "user", "content": [
        {"type": "video", "video": video_path, "fps": fps, "max_frames": max_frames},
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
    args = parse_args()
    rows = [json.loads(l) for l in Path(args.manifest).open() if l.strip()]
    ta = [r for r in rows if r.get("task") == "TA"]
    print(f"[data] {len(ta)} TA rows in manifest")
    if args.limit > 0:
        ta = ta[:args.limit]
        print(f"[data] --limit {args.limit} -> processing {len(ta)}")

    tag = Path(args.adapter).parent.name + "_" + Path(args.adapter).name \
        if args.adapter else "base"
    out_path = Path(args.out) if args.out else \
        ROOT / f"claude_track2/outputs/eval_ta_{tag}_v1.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[model] loading...", flush=True)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="flash_attention_2", local_files_only=True)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter,
                                          is_trainable=False)
        print(f"[model] adapter applied: {args.adapter}")
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(
        MODEL_PATH, local_files_only=True, use_fast=False)
    tok = processor.tokenizer
    for letter, expect in LETTER_IDS.items():
        assert tok.encode(letter) == [expect], f"letter id mismatch for {letter}"

    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["sample_id"])
        print(f"[resume] {len(done)} already done in {out_path}")

    n_correct = n_ok = 0
    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as f:
        for i, row in enumerate(ta):
            sid = row["sample_id"]
            if sid in done:
                continue
            try:
                inputs = build_inputs(processor, row["video_path"],
                                      row["question"], args.fps, args.max_frames)
                inputs = inputs.to(model.device).to(torch.bfloat16)
                with torch.inference_mode():
                    out = model(**inputs, use_audio_in_video=True,
                                return_dict=True)
                logits = out.logits[0, -1, :].float()
                ab = torch.tensor([logits[LETTER_IDS["A"]].item(),
                                   logits[LETTER_IDS["B"]].item()])
                probs = F.softmax(ab, dim=0)
                pred = "A" if probs[0] >= 0.5 else "B"
                correct = pred == row["answer"]
                n_ok += 1
                n_correct += int(correct)
                f.write(json.dumps({
                    "sample_id": sid, "answer": row["answer"], "pred": pred,
                    "p_A": round(probs[0].item(), 6),
                    "p_B": round(probs[1].item(), 6),
                    "correct": correct, "adapter": args.adapter or "base",
                }) + "\n")
                f.flush()
                print(f"[{i+1}/{len(ta)}] {sid} pred={pred} ans={row['answer']} "
                      f"p_A={probs[0]:.3f} ({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[eval] SKIP {sid}: {type(e).__name__}: {e}", flush=True)

    # accuracy over the whole results file (including resumed rows)
    all_rows = [json.loads(l) for l in out_path.read_text().splitlines()
                if l.strip()]
    n = len(all_rows)
    c = sum(r["correct"] for r in all_rows)
    print(f"\n[result] adapter={args.adapter or 'base'} "
          f"acc={c}/{n}={c/n*100:.1f}% -> {out_path}")


if __name__ == "__main__":
    main()
