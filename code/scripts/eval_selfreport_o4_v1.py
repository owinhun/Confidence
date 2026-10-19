#!/usr/bin/env python3
"""O4 (pre-registered): does the percept survive when the ANSWER flips?

Under a false statement the model's binary answer flips. Two possibilities:
 (a) the false text corrupts the representation -> the model can no longer
     report what it actually perceived;
 (b) the percept survives and only the answer is captured -> the failure is at
     the answer level, which is the strongest form of the claim and implies a
     cheap mitigation (ask for the percept, not the verdict).

Test: same clip, same false statement, but ask the model to DESCRIBE what it
observes rather than to answer yes/no, then score the description.
  colour  : "Describe the colour appearance of this video in one short phrase."
            -> scored by whether the description says greyscale/black-and-white
  silence : "Describe the audio in one short phrase: is it continuous or does it
             stop at some point?"
Both are run with the same false evidence sentence used in the yes/no runs.
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
from qwen_omni_utils import process_mm_info
from transformers import (Qwen2_5OmniProcessor,
                          Qwen2_5OmniThinkerForConditionalGeneration)

ROOT = Path("/workspace/svhalluc")
MODEL_PATH = ROOT / "models/Qwen2.5-Omni-7B"
FPS, MAX_FRAMES = 2.0, 64
MANIFESTS = {
    "colour": ROOT / "claude_track2/data/attr_colour_v1.jsonl",
    "silence": ROOT / "claude_track2/data/attr_silence_v1.jsonl",
}
DESCRIBE = {
    "colour": ("Describe the colour appearance of this video in one short "
               "phrase (for example: 'full colour' or 'black and white')."),
    "silence": ("Describe the audio of this clip in one short phrase: is the "
                "sound continuous throughout, or does it stop completely at "
                "some point?"),
}
FALSE_EV = {
    "colour": {True: "This clip is in full colour.\n\n",
               False: "This clip is in greyscale.\n\n"},
    "silence": {True: "This clip contains no silent gap.\n\n",
                False: "This clip contains a gap of complete silence.\n\n"},
}


def score(attr, text, truth_yes):
    t = text.lower()
    if attr == "colour":
        said_grey = bool(re.search(r"black[- ]and[- ]white|greyscale|grayscale|"
                                   r"monochrom|b&w", t))
        said_colour = bool(re.search(r"full colou?r|colou?rful|in colou?r", t))
        if said_grey and not said_colour:
            return True == truth_yes
        if said_colour and not said_grey:
            return False == truth_yes
        return None
    stops = bool(re.search(r"stops|silence|silent|pause|gap|cuts out|goes quiet", t))
    cont = bool(re.search(r"continuous|throughout|uninterrupted|constant", t))
    if stops and not cont:
        return True == truth_yes
    if cont and not stops:
        return False == truth_yes
    return None


def build(processor, path, q):
    messages = [{"role": "user", "content": [
        {"type": "video", "video": path, "fps": FPS, "max_frames": MAX_FRAMES},
        {"type": "text", "text": q}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True,
                                         tokenize=False)
    a, i, v, vk = process_mm_info(messages, use_audio_in_video=True,
                                  return_video_kwargs=True)
    if isinstance(vk.get("fps"), list) and len(vk["fps"]) == 1:
        vk["fps"] = vk["fps"][0]
    return processor(text=text, audio=a, images=i, videos=v, return_tensors="pt",
                     padding=True, use_audio_in_video=True, **vk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", choices=list(MANIFESTS), required=True)
    ap.add_argument("--evidence", choices=["none", "wrong"], required=True)
    ap.add_argument("--limit", type=int, default=250)
    args = ap.parse_args()

    rows = [json.loads(l) for l in MANIFESTS[args.attr].open() if l.strip()]
    rows = rows[:args.limit]
    out = ROOT / f"claude_track2/outputs/selfreport_{args.attr}_{args.evidence}_v1.jsonl"
    print(f"[data] {len(rows)} clips", flush=True)

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="flash_attention_2", local_files_only=True).eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(
        MODEL_PATH, local_files_only=True, use_fast=False)

    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip():
                done.add(json.loads(l)["clip"])

    n = ok = amb = 0
    t0 = time.time()
    with out.open("a", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            if r["clip"] in done:
                continue
            truth_yes = r["label"] == "A"
            try:
                q = DESCRIBE[args.attr]
                if args.evidence == "wrong":
                    q = FALSE_EV[args.attr][truth_yes] + q
                inp = build(processor, str(ROOT / r["clip"]), q)
                inp = inp.to(model.device).to(torch.bfloat16)
                with torch.inference_mode():
                    g = model.generate(**inp, use_audio_in_video=True,
                                       max_new_tokens=24, do_sample=False)
                txt = processor.batch_decode(
                    g[:, inp["input_ids"].shape[1]:],
                    skip_special_tokens=True)[0].split("\n")[0].strip()
                s = score(args.attr, txt, truth_yes)
                n += 1
                if s is None:
                    amb += 1
                else:
                    ok += int(s)
                f.write(json.dumps({"clip": r["clip"], "label": r["label"],
                                    "text": txt, "correct": s,
                                    "evidence": args.evidence}) + "\n")
                f.flush()
                if n % 40 == 0:
                    d = n - amb
                    print(f"[{i+1}/{len(rows)}] scored {ok}/{d} amb={amb} "
                          f"({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"SKIP {r['clip']}: {type(e).__name__}: {e}", flush=True)
    d = max(1, n - amb)
    print(f"DONE n={n} scorable={n-amb} acc={ok/d:.4f} ambiguous={amb}", flush=True)


if __name__ == "__main__":
    main()
