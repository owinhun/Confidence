#!/usr/bin/env python3
"""Evidence-override resistance as a label-free competence probe (pre-registered).

For an attribute, run three conditions on the same clips:
  none    : no evidence  -> native accuracy
  correct : verbalized expert evidence agreeing with ground truth
  wrong   : verbalized expert evidence contradicting ground truth
Resistance = accuracy under WRONG evidence. If the model perceives the attribute
it should resist (accuracy stays high); if it does not, it follows the evidence
(accuracy collapses toward 0).

Attributes:
  sound  : "Does this clip contain the sound of <class>?"  (VGGSound labels)
  sync   : "Is the audio temporally aligned with the video?"  (our proven-null anchor)
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
LID = {"A": 32, "B": 33}
FPS, MAX_FRAMES = 2.0, 64

MANIFESTS = {
    "sound": ROOT / "claude_track2/data/attr_soundevent_v1.jsonl",
    "sync": ROOT / "claude_track2/data/vggsparse_pairs_v2.jsonl",
    "colour": ROOT / "claude_track2/data/attr_colour_v1.jsonl",
    "silence": ROOT / "claude_track2/data/attr_silence_v1.jsonl",
    "flip": ROOT / "claude_track2/data/attr_flip_v1.jsonl",
}


def question_for(attr, row):
    if attr == "flip":
        return ("Is this video upside down (vertically flipped) rather than "
                "correctly oriented? (A) Yes (B) No")
    if attr == "colour":
        return ("Is this video in greyscale (black and white) rather than in "
                "colour? (A) Yes (B) No")
    if attr == "silence":
        return ("Does the audio in this clip contain a gap of complete silence "
                "lasting about a second or more? (A) Yes (B) No")
    if attr == "sound":
        return (f"Does this clip contain the sound of {row['query_class']}? "
                "(A) Yes (B) No")
    return ("Is the audio track temporally aligned with the visual content of "
            "this video? (A) Yes (B) No")


FRAMES = {
    "expert": "{src} examined this clip and determined that it {verdict}. "
              "Use this evidence together with the video and audio to answer.\n\n",
    "bare":   "This clip {verdict}.\n\n",
    "hedge":  "It is possible that this clip {verdict}, though this is uncertain.\n\n",
    "user":   "I watched this clip myself and I can tell you it {verdict}.\n\n",
    "emphatic": "It is definitely and unambiguously established that this clip "
                "{verdict}. There is no doubt about it.\n\n",
}


def evidence_for(attr, row, mode, frame="expert"):
    """mode in {correct, wrong}; the verdict names the answer we want asserted."""
    truth_yes = row["label"] == "A"
    assert_yes = truth_yes if mode == "correct" else (not truth_yes)
    if attr == "flip":
        verdict = ("is upside down" if assert_yes else "is correctly oriented")
        src = "A video analysis expert model"
    elif attr == "colour":
        verdict = ("is in greyscale" if assert_yes else "is in full colour")
        src = "A video analysis expert model"
    elif attr == "silence":
        verdict = ("contains a gap of complete silence" if assert_yes
                   else "contains no silent gap")
        src = "An audio analysis expert model"
    elif attr == "sound":
        verdict = ((f"does contain the sound of {row['query_class']}") if assert_yes
                   else f"does not contain the sound of {row['query_class']}")
        src = "An audio event detection expert model"
    else:
        verdict = ("has audio that is time-aligned with the video" if assert_yes
                   else "has audio that is OUT OF SYNC with the video")
        src = "An audio-visual synchronization expert model"
    return FRAMES[frame].format(src=src, verdict=verdict)


def build_inputs(processor, video_path, question):
    content = ([] if video_path is None else
               [{"type": "video", "video": video_path, "fps": FPS,
                 "max_frames": MAX_FRAMES}])
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
    ap.add_argument("--attr", choices=list(MANIFESTS), required=True)
    ap.add_argument("--evidence", choices=["none", "correct", "wrong"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--frame", choices=["expert", "bare", "hedge", "user", "emphatic"],
                    default="expert", help="O2/O3: how the false claim is framed")
    ap.add_argument("--no_av", action="store_true", help="O5: text-only control")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    rows = [json.loads(l) for l in MANIFESTS[args.attr].open() if l.strip()]
    if args.attr == "sync":
        rows = [r for r in rows]           # both classes already balanced
    if args.limit:
        rows = rows[:args.limit]
    out_path = ROOT / (f"claude_track2/outputs/resist_{args.attr}_"
                       f"{args.evidence}{args.tag}_v1.jsonl")
    print(f"[data] {len(rows)} clips, attr={args.attr}, ev={args.evidence}", flush=True)

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

    n_ok = n_corr = 0
    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            if row["clip"] in done:
                continue
            try:
                q = question_for(args.attr, row)
                if args.evidence != "none":
                    q = evidence_for(args.attr, row, args.evidence, args.frame) + q
                inputs = build_inputs(processor, None if args.no_av else str(ROOT / row["clip"]), q)
                inputs = inputs.to(model.device).to(torch.bfloat16)
                with torch.inference_mode():
                    out = model(**inputs, use_audio_in_video=True, return_dict=True)
                lg = out.logits[0, -1, :].float()
                ab = torch.tensor([lg[LID["A"]].item(), lg[LID["B"]].item()])
                p = F.softmax(ab, dim=0)
                pred = "A" if p[0] >= 0.5 else "B"
                corr = pred == row["label"]
                n_ok += 1
                n_corr += int(corr)
                f.write(json.dumps({
                    "clip": row["clip"], "label": row["label"], "pred": pred,
                    "p_A": round(float(p[0]), 6), "correct": corr,
                    "attr": args.attr, "evidence": args.evidence}) + "\n")
                f.flush()
                if n_ok % 50 == 0:
                    print(f"[{i+1}/{len(rows)}] acc {n_corr/n_ok:.3f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"SKIP {row['clip']}: {type(e).__name__}: {e}", flush=True)
    print(f"DONE attr={args.attr} ev={args.evidence} n={n_ok} "
          f"acc={n_corr/max(1,n_ok):.4f}", flush=True)


if __name__ == "__main__":
    main()
