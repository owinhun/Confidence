#!/usr/bin/env python3
"""T2 (pre-registered): training-free expert injection into frozen Qwen2.5-Omni.

Prepends a plain-text expert sync report (per-position P(aligned) from
synchformer_ta_probe_v1.jsonl — same values as T1, no new expert calls) to the
TA question; constrained A/B letter scoring, video+audio inputs unchanged.

Run: CUDA_VISIBLE_DEVICES=1 python3 scripts/expert_inject_ta_t2_v1.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.path.insert(0, "/workspace/svhalluc/src/qwen")

import torch
import torch.nn.functional as F
from qwen_omni_utils import process_mm_info
from transformers import (Qwen2_5OmniProcessor,
                          Qwen2_5OmniThinkerForConditionalGeneration)

ROOT = Path("/workspace/svhalluc")
MODEL_PATH = ROOT / "models/Qwen2.5-Omni-7B"
MANIFEST = ROOT / "data/ter_large_eval_v1.jsonl"
EXPERT = ROOT / "claude_track2/outputs/synchformer_ta_probe_v1.jsonl"
OUT = ROOT / "claude_track2/outputs/expert_inject_ta_t2_v1.jsonl"
LETTER_IDS = {"A": 32, "B": 33}
FPS, MAX_FRAMES = 2.0, 64


def expert_report(rec: dict, k1: bool = False, binary: bool = False) -> str:
    s_list = rec["s_list"][:1] if k1 else rec["s_list"]
    if binary:
        mean_s = sum(s_list) / len(s_list)
        verdict = "time-aligned" if mean_s >= 0.175 else "OUT OF SYNC"
        return ("An audio-visual synchronization expert model analyzed this clip "
                f"and judged that the speech and the visual content are {verdict}. "
                "Use this evidence together with the video and audio to answer.\n\n")
    probs = ", ".join(f"{s:.2f}" for s in s_list)
    k = len(s_list)
    return (
        "An audio-visual synchronization expert model analyzed "
        f"{k} position(s) of this clip. Estimated probability that the speech "
        f"and the visual content are time-aligned at each position: {probs}. "
        "Probabilities near 0 indicate the audio is out of sync with the video; "
        "higher values indicate alignment. Use this evidence together with the "
        "video and audio to answer.\n\n"
    )


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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--no_av", action="store_true")
    ap.add_argument("--permute", action="store_true")
    ap.add_argument("--k1", action="store_true", help="ablation: use only the first position score")
    ap.add_argument("--binary", action="store_true", help="ablation: verbalize binary verdict instead of probabilities")
    ap.add_argument("--swap", action="store_true", help="swap option layout: (A) No (B) Yes")
    ap.add_argument("--nobase", action="store_true", help="frozen baseline: no evidence text")
    args = ap.parse_args()
    global OUT
    if args.out:
        OUT = Path(args.out)
    rows = [json.loads(l) for l in MANIFEST.open() if l.strip()]
    ta = [r for r in rows if r.get("task") == "TA"]
    exp = {json.loads(l)["sample_id"]: json.loads(l)
           for l in EXPERT.open() if l.strip()}
    ta = [r for r in ta if r["sample_id"] in exp]
    if args.permute:
        import numpy as np
        keys = [r["sample_id"] for r in ta]
        vals = [exp[k] for k in keys]
        perm = np.random.RandomState(0).permutation(len(vals))
        exp = {k: vals[perm[i]] for i, k in enumerate(keys)}
    print(f"[data] {len(ta)} TA rows with expert scores")

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
        assert tok.encode(letter) == [expect], f"letter id mismatch for {letter}"

    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["sample_id"])
        print(f"[resume] {len(done)} already done")

    n_correct = n_ok = 0
    t0 = time.time()
    with OUT.open("a", encoding="utf-8") as f:
        for i, row in enumerate(ta):
            sid = row["sample_id"]
            if sid in done:
                continue
            try:
                question = row["question"]
                answer = row["answer"]
                if args.swap:
                    suffix = "(A) Yes (B) No"
                    assert suffix in question, question
                    question = question.replace(suffix, "(A) No (B) Yes")
                    answer = {"A": "B", "B": "A"}[answer]
                q = question if args.nobase else (
                    expert_report(exp[sid], k1=args.k1, binary=args.binary) + question)
                inputs = build_inputs(processor, row["video_path"], q, no_av=args.no_av)
                inputs = inputs.to(model.device).to(torch.bfloat16)
                with torch.inference_mode():
                    out = model(**inputs, use_audio_in_video=True, return_dict=True)
                logits = out.logits[0, -1, :].float()
                ab = torch.tensor([logits[LETTER_IDS["A"]].item(),
                                   logits[LETTER_IDS["B"]].item()])
                probs = F.softmax(ab, dim=0)
                pred = "A" if probs[0] >= 0.5 else "B"
                correct = pred == answer
                n_ok += 1
                n_correct += int(correct)
                f.write(json.dumps({
                    "sample_id": sid, "answer": answer, "pred": pred,
                    "p_A": round(probs[0].item(), 6),
                    "p_B": round(probs[1].item(), 6),
                    "correct": correct}) + "\n")
                f.flush()
                if n_ok % 20 == 0:
                    print(f"[{i+1}/{len(ta)}] acc so far {n_correct/n_ok:.4f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"SKIP {sid}: {type(e).__name__}: {e}", flush=True)
    print(f"DONE new={n_ok} acc(new)={n_correct/max(1,n_ok):.4f} -> {OUT}")


if __name__ == "__main__":
    main()
