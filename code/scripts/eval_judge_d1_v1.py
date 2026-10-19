#!/usr/bin/env python3
"""D1 (pre-registered): are MLLM-judged synchronization sub-scores valid?

2026 audio-video generation benchmarks (Omni-Judge, AVE-Compass, T2AV-Compass,
VABench, AVGen-Bench) all report a synchronization sub-score produced by an
omni-LLM judge, and none validates it against ground-truth offsets. We replicate
the published read-outs verbatim and test them on clips whose desynchronization
we control.

Read-outs:
  likert   : "Rate audio-visual synchronization from 1 (poor) to 5 (excellent)"
             scored as constrained logits over the digits 1..5 (T2AV/VABench style)
  checklist: atomic Yes/No item (AVE-Compass style)
Data: --delta selects one of our ladder manifests; aligned clips are shared.
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
FPS, MAX_FRAMES = 2.0, 64

PROMPTS = {
    "likert": ("Rate the audio-visual synchronization quality of this video on a "
               "scale from 1 to 5, where 1 means the audio and video are badly "
               "out of sync and 5 means they are perfectly synchronized. "
               "Answer with a single digit."),
    "checklist": ("Checklist item: the audio and visual signals in this video "
                  "remain temporally consistent with each other. Is this "
                  "satisfied? (A) Yes (B) No"),
    # wording robustness variants (same construct, different phrasing)
    "likert_v2": ("On a scale of 1 to 5, how well do the sounds line up in time "
                  "with what happens on screen? 1 = the timing is clearly wrong, "
                  "5 = the timing is perfect. Reply with only the number."),
    "likert_v3": ("Evaluate the temporal alignment between the audio track and "
                  "the visual events in this clip. Score from 1 (severe lag or "
                  "lead between sound and picture) to 5 (no perceptible offset). "
                  "Output a single digit."),
}
DIGITS = ["1", "2", "3", "4", "5"]


def build_inputs(processor, video_path, question: str):
    vids = video_path if isinstance(video_path, (list, tuple)) else [video_path]
    content = [{"type": "video", "video": v, "fps": FPS, "max_frames": MAX_FRAMES}
               for v in vids]
    content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": content}]
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
    ap.add_argument("--manifest", required=True,
                    help="e.g. data/vggsparse_delta0.4_v1.jsonl or "
                         "data/swap_pairs_v1.jsonl")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in (ROOT / "claude_track2" / args.manifest).open()
            if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    out_path = ROOT / f"claude_track2/outputs/judge_{args.fmt}_{args.tag}_v1.jsonl"
    print(f"[data] {len(rows)} clips, fmt={args.fmt}, tag={args.tag}", flush=True)

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="flash_attention_2", local_files_only=True)
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(
        MODEL_PATH, local_files_only=True, use_fast=False)
    tok = processor.tokenizer
    if args.fmt.startswith("likert"):
        ids = [tok.encode(d, add_special_tokens=False) for d in DIGITS]
        assert all(len(v) == 1 for v in ids), ids
        LID = [v[0] for v in ids]
    else:
        LID = [tok.encode(c, add_special_tokens=False)[0] for c in ("A", "B")]

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
                with torch.inference_mode():
                    out = model(**inputs, use_audio_in_video=True, return_dict=True)
                lg = out.logits[0, -1, :].float()
                v = torch.tensor([lg[t].item() for t in LID])
                p = F.softmax(v, dim=0)
                rec = {"clip": row["clip"], "label": row.get("label"),
                       "delta": row.get("delta"), "kind": row.get("kind"),
                       "ytid": row.get("ytid"), "fmt": args.fmt}
                if args.fmt.startswith("likert"):
                    rec["probs"] = [round(float(x), 6) for x in p]
                    rec["score"] = float(sum((k + 1) * float(p[k]) for k in range(5)))
                    rec["argmax"] = int(p.argmax()) + 1
                else:
                    rec["p_yes"] = float(p[0])
                    rec["pred_consistent"] = bool(p[0] >= 0.5)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                n_ok += 1
                if n_ok % 50 == 0:
                    print(f"[{i+1}/{len(rows)}] {time.time()-t0:.0f}s", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"SKIP {row['clip']}: {type(e).__name__}: {e}", flush=True)
    print(f"DONE fmt={args.fmt} tag={args.tag} new={n_ok}", flush=True)


if __name__ == "__main__":
    main()
