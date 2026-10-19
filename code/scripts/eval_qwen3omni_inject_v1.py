#!/usr/bin/env python3
"""R1 (pre-registered): does the NEWER omni generation solve synchrony?

Existential check for the paper's premise. Qwen3-Omni-30B-A3B (MoE, absolute-time
anchored temporal IDs) on a balanced subset of the VGGSound-Sparse pairs, frozen,
constrained option-letter scoring at the first answer position -- the same
instrument used for Qwen2.5-Omni.

Decision rule (fixed before running): accuracy >= 60% on n=200 means the newer
generation partially solves the task and the paper must be scoped to the
Qwen2.5 era; ~50% adds a third-backbone generality row.
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
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

ROOT = Path("/workspace/svhalluc")
MODEL_PATH = ROOT / "models/Qwen3-Omni-30B-A3B-Instruct"
MANIFEST = ROOT / "claude_track2/data/vggsparse_pairs_v2.jsonl"
QUESTION = ("Is the audio track temporally aligned with the visual content of "
            "this video? (A) Yes (B) No")
EXPERT = ROOT / "claude_track2/outputs/synchformer_vggsparse_v1.jsonl"
FPS, MAX_FRAMES = 2.0, 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--inject", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in MANIFEST.open() if l.strip()]
    # balanced subset: first N/2 of each class, deterministic
    a = [r for r in rows if r["label"] == "A"][: args.limit // 2]
    b = [r for r in rows if r["label"] == "B"][: args.limit // 2]
    rows = a + b
    exp = {json.loads(l)["clip"]: json.loads(l) for l in EXPERT.open() if l.strip()} if args.inject else {}
    if args.inject:
        rows = [r for r in rows if r["clip"] in exp]
    out_path = ROOT / f"claude_track2/outputs/qwen3omni_r1_{'verdict' if args.inject else 'base'}_v1.jsonl"
    print(f"[data] {len(rows)} clips (balanced)", flush=True)

    processor = Qwen3OmniMoeProcessor.from_pretrained(str(MODEL_PATH))
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        str(MODEL_PATH), dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa")
    model.eval()
    tok = processor.tokenizer
    ids = {L: tok.encode(L, add_special_tokens=False) for L in ("A", "B")}
    assert all(len(v) == 1 for v in ids.values()), ids
    LID = {L: v[0] for L, v in ids.items()}
    print(f"[model] loaded; letter ids {LID}", flush=True)

    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["clip"])

    n_ok = n_corr = 0
    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            if row["clip"] in done:
                continue
            try:
                messages = [{"role": "user", "content": [
                    {"type": "video", "video": str(ROOT / row["clip"])},
                    {"type": "text", "text": (("An audio-visual synchronization expert model analyzed this clip and judged that the audio and the visual content are " + ("time-aligned" if exp[row["clip"]]["s_near0"] >= 0.175 else "OUT OF SYNC") + ". Use this evidence together with the video and audio to answer.\n\n") if args.inject else "") + QUESTION}]}]
                text = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False)
                from qwen_omni_utils import process_mm_info
                audios, images, videos = process_mm_info(
                    messages, use_audio_in_video=True)[:3]
                inputs = processor(text=text, audio=audios, images=images,
                                   videos=videos, return_tensors="pt",
                                   padding=True, use_audio_in_video=True)
                inputs = inputs.to(model.device)
                for k, v in inputs.items():
                    if torch.is_tensor(v) and v.dtype == torch.float32:
                        inputs[k] = v.to(torch.bfloat16)
                with torch.inference_mode():
                    out = model.thinker(**inputs, use_audio_in_video=True,
                                        return_dict=True)
                lg = out.logits[0, -1, :].float()
                ab = torch.tensor([lg[LID["A"]].item(), lg[LID["B"]].item()])
                p = F.softmax(ab, dim=0)
                pred = "A" if p[0] >= 0.5 else "B"
                corr = pred == row["label"]
                n_ok += 1
                n_corr += int(corr)
                f.write(json.dumps({"clip": row["clip"], "label": row["label"],
                                    "pred": pred, "p_A": round(float(p[0]), 6),
                                    "correct": corr}) + "\n")
                f.flush()
                if n_ok % 25 == 0:
                    print(f"[{i+1}/{len(rows)}] acc {n_corr/n_ok:.3f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"SKIP {row['clip']}: {type(e).__name__}: {e}", flush=True)
    print(f"DONE n={n_ok} acc={n_corr/max(1,n_ok):.4f}", flush=True)


if __name__ == "__main__":
    main()
