#!/usr/bin/env python3
"""B1 backbone-generality (pre-registered): frozen VideoLLaMA2.1-7B-AV on
SVHalluc TA, base vs expert-injection, constrained A/B letter scoring.

Audio ON (va=True) — the AV variant fuses audio+video tokens.
Expert scores: outputs/synchformer_ta_probe_v1.jsonl (same as Qwen T2).

Run: CUDA_VISIBLE_DEVICES=1 python3 vllama_inject_ta_b1_v1.py --mode base|inject
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/workspace/svhalluc")
VLLAMA_ROOT = ROOT / "src/qwen/third_party/video_llama2_av"
DEPS = VLLAMA_ROOT / ".deps"
sys.path[:0] = [str(DEPS), str(VLLAMA_ROOT)]

import torch
import torch.nn.functional as F
from videollama2 import model_init
from videollama2.constants import DEFAULT_VIDEO_TOKEN
from videollama2.mm_utils import tokenizer_multimodal_token

MODEL_ID = "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"
import os
MANIFEST = ROOT / os.environ.get("VJ_MANIFEST", "claude_track2/data/vggsparse_pairs_v2.jsonl")
EXPERT = ROOT / "claude_track2/outputs/synchformer_vggsparse_v1.jsonl"


LIKERT = ("Rate the audio-visual synchronization quality of this video on a scale "
          "from 1 to 5, where 1 means the audio and video are badly out of sync and "
          "5 means they are perfectly synchronized. Answer with a single digit.")
CHECK = ("Checklist item: the audio and visual signals in this video remain temporally "
         "consistent with each other. Is this satisfied? (A) Yes (B) No")


def expert_report(rec: dict) -> str:
    probs = ", ".join(f"{s:.2f}" for s in rec["s_list"])
    k = len(rec["s_list"])
    return (
        "An audio-visual synchronization expert model analyzed "
        f"{k} position(s) of this clip. Estimated probability that the speech "
        f"and the visual content are time-aligned at each position: {probs}. "
        "Probabilities near 0 indicate the audio is out of sync with the video; "
        "higher values indicate alignment. Use this evidence together with the "
        "video and audio to answer.\n\n"
    )


def first_token_logits(model, tokenizer, tensor, instruct: str) -> torch.Tensor:
    vis = [(tensor.half().cuda() if not isinstance(tensor, dict)
            else {k: v.half().cuda() for k, v in tensor.items()}, "video")]
    message = [{"role": "user", "content": DEFAULT_VIDEO_TOKEN + "\n" + instruct}]
    prompt = tokenizer.apply_chat_template(message, tokenize=False,
                                           add_generation_prompt=True)
    input_ids = tokenizer_multimodal_token(prompt, tokenizer, DEFAULT_VIDEO_TOKEN,
                                           return_tensors="pt").unsqueeze(0).long().cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()
    with torch.inference_mode():
        out = model.generate(
            input_ids, attention_mask=attention_masks, images=vis,
            do_sample=False, max_new_tokens=1, use_cache=True,
            output_scores=True, return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return out.scores[0][0].float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["likert", "checklist"], required=True)
    ap.add_argument("--binary", action="store_true", help="verdict wording instead of probabilities")
    args = ap.parse_args()

    ta = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]

    print(f"[data] {len(ta)} TA rows, mode={args.mode}", flush=True)

    model, processor, tokenizer = model_init(MODEL_ID, device_map={"": 0},
                                             use_flash_attn=False)
    model.eval()
    keys = [str(d) for d in range(1, 6)] if args.mode == "likert" else ["A", "B"]
    ids = {L: tokenizer.encode(L, add_special_tokens=False) for L in keys}
    assert all(len(v) == 1 for v in ids.values()), ids
    letter_ids = {L: v[0] for L, v in ids.items()}

    out_path = ROOT / f"claude_track2/outputs/vllama_judge_{args.mode}{os.environ.get('VJ_TAG','')}_v1.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["sample_id"])
        print(f"[resume] {len(done)} done", flush=True)

    n_ok = n_correct = 0
    with out_path.open("a", encoding="utf-8") as f:
        for row in ta:
            sid = row["clip"]
            if sid in done:
                continue
            try:
                tensor = processor["video"](str(ROOT / row["clip"]), va=True)
                q = LIKERT if args.mode == "likert" else CHECK
                logits = first_token_logits(model, tokenizer, tensor, q)
                v = torch.tensor([logits[letter_ids[k]].item() for k in keys])
                probs = F.softmax(v, dim=0)
                rec = {"clip": sid, "label": row.get("label"), "kind": row.get("kind"),
                       "ytid": row["ytid"],
                       "delta": row.get("delta")}
                if args.mode == "likert":
                    rec["score"] = float(sum((j + 1) * float(probs[j]) for j in range(5)))
                else:
                    rec["p_yes"] = float(probs[0])
                n_ok += 1
                f.write(json.dumps(rec) + "\n")
                f.flush()
                if n_ok % 25 == 0:
                    print(f"{n_ok} done", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"SKIP {sid}: {type(e).__name__}: {e}", flush=True)
    print(f"DONE mode={args.mode} new={n_ok}", flush=True)


if __name__ == "__main__":
    main()
