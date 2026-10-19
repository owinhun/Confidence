#!/usr/bin/env python3
"""N1 (pre-registered): does the two-way MCQ read-out cap the model?

X3 showed the frozen model's logit carries information its argmax discards.
Here we vary ONLY the read-out format on the same VGGSound-Sparse clips, with no
expert evidence, and ask whether any format escapes chance:
  mcq2   : the paper's binary A/B (baseline)
  mcq4   : four options (aligned / audio early / audio late / different sound)
  gen    : free generation, short answer parsed for aligned-vs-not
  reg    : open numeric answer ("offset in seconds"), scored by |pred| threshold
Run: CUDA_VISIBLE_DEVICES=k python3 eval_readout_n1_v1.py --fmt mcq4
"""

from __future__ import annotations

import argparse
import json
import re
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
FPS, MAX_FRAMES = 2.0, 64
LETTERS = {"A": 32, "B": 33, "C": 34, "D": 35}

PROMPTS = {
    "mcq2": ("Is the audio track temporally aligned with the visual content of "
             "this video? (A) Yes (B) No"),
    "mcq4": ("How does the audio relate in time to the visual content? "
             "(A) Aligned (B) Audio occurs earlier than the visuals "
             "(C) Audio occurs later than the visuals (D) Cannot tell"),
    "gen": ("Listen to the audio and watch the video. Are they synchronised in "
            "time, or is the audio shifted? Answer in one short sentence."),
    "reg": ("Estimate how far the audio is shifted relative to the video, in "
            "seconds. Answer with a single number (0 means perfectly aligned)."),
}


def build_inputs(processor, video_path: str, question: str):
    messages = [{"role": "user", "content": [
        {"type": "video", "video": video_path, "fps": FPS, "max_frames": MAX_FRAMES},
        {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True,
                                         tokenize=False)
    audios, images, videos, vk = process_mm_info(
        messages, use_audio_in_video=True, return_video_kwargs=True)
    if isinstance(vk.get("fps"), list) and len(vk["fps"]) == 1:
        vk["fps"] = vk["fps"][0]
    return processor(text=text, audio=audios, images=images, videos=videos,
                     return_tensors="pt", padding=True, use_audio_in_video=True, **vk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmt", choices=list(PROMPTS), required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in MANIFEST.open() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    out_path = ROOT / f"claude_track2/outputs/readout_{args.fmt}_v1.jsonl"
    print(f"[data] {len(rows)} clips, fmt={args.fmt}", flush=True)

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="flash_attention_2", local_files_only=True)
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(
        MODEL_PATH, local_files_only=True, use_fast=False)

    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["clip"])
        print(f"[resume] {len(done)}", flush=True)

    n_ok = 0
    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            if row["clip"] in done:
                continue
            try:
                inputs = build_inputs(processor, str(ROOT / row["clip"]),
                                      PROMPTS[args.fmt])
                inputs = inputs.to(model.device).to(torch.bfloat16)
                rec = {"clip": row["clip"], "label": row["label"],
                       "ytid": row["ytid"], "fmt": args.fmt}
                if args.fmt.startswith("mcq"):
                    with torch.inference_mode():
                        out = model(**inputs, use_audio_in_video=True, return_dict=True)
                    lg = out.logits[0, -1, :].float()
                    keys = ["A", "B"] if args.fmt == "mcq2" else ["A", "B", "C", "D"]
                    v = torch.tensor([lg[LETTERS[k]].item() for k in keys])
                    p = F.softmax(v, dim=0)
                    rec["probs"] = {k: round(float(x), 6) for k, x in zip(keys, p)}
                    rec["pred_letter"] = keys[int(p.argmax())]
                    # aligned iff first option
                    rec["pred_aligned"] = rec["pred_letter"] == "A"
                    rec["p_aligned"] = float(p[0])
                else:
                    with torch.inference_mode():
                        gen = model.generate(**inputs, use_audio_in_video=True,
                                             max_new_tokens=48, do_sample=False)
                    txt = processor.batch_decode(
                        gen[:, inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True)[0].strip()
                    rec["text"] = txt
                    low = txt.lower()
                    if args.fmt == "gen":
                        neg = any(w in low for w in
                                  ["not synchron", "shifted", "out of sync",
                                   "delay", "mismatch", "ahead", "behind", "lag"])
                        pos = any(w in low for w in
                                  ["synchronis", "synchroniz", "aligned", "in sync",
                                   "match"])
                        rec["pred_aligned"] = bool(pos and not neg)
                    else:
                        m = re.search(r"-?\d+\.?\d*", low)
                        val = float(m.group()) if m else 0.0
                        rec["pred_offset"] = val
                        rec["pred_aligned"] = abs(val) < 0.5
                rec["correct"] = rec["pred_aligned"] == (row["label"] == "A")
                f.write(json.dumps(rec) + "\n")
                f.flush()
                n_ok += 1
                if n_ok % 50 == 0:
                    print(f"[{i+1}/{len(rows)}] {time.time()-t0:.0f}s", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"SKIP {row['clip']}: {type(e).__name__}: {e}", flush=True)
    print(f"DONE fmt={args.fmt} new={n_ok}", flush=True)


if __name__ == "__main__":
    main()
