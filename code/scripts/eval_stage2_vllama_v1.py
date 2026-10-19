#!/usr/bin/env python3
"""Stage 2 confirmatory factorial on a THIRD model: VideoLLaMA2.1-7B-AV.

Identical design and identical held-out manifests to eval_stage2_factorial_v1.py
(Qwen2.5) and eval_stage2_q3_v1.py (Qwen3): the claim proposition is byte-identical
across cells and only the Source and Confidence fields move.

This exists to answer the one limitation left after Stage 2 closed on two models:
both are Qwen-family Omni models, so a shared instruction-tuning recipe could in
principle produce the confidence effect. VideoLLaMA2.1-AV is a different lineage
(Mistral/Qwen2 LLM + separate audio branch, DAMO), so a replication here rules that out.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# All model weights live on /data4; /workspace is at 99%. Pin the hub cache here too so a
# load-by-name never re-downloads into /workspace.
os.environ.setdefault("HF_HOME", "/data4/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/data4/hf_cache/hub")
os.environ.setdefault("HF_MODULES_CACHE", "/data4/hf_cache/modules")

VL = Path("/workspace/svhalluc/src/qwen/third_party/video_llama2_av")
sys.path[:0] = [str(VL / ".deps"), str(VL)]

import torch
import torch.nn.functional as F
from videollama2 import model_init
from videollama2.constants import DEFAULT_VIDEO_TOKEN
from videollama2.mm_utils import tokenizer_multimodal_token

ROOT = Path("/workspace/svhalluc")
CACHE = ROOT / "claude_track2/data/vl_pre_cache_v1"


def preprocess(processor, clip: str):
    """Decode+preprocess once per clip and reuse across all conditions.

    Under heavy external CPU load the video decode, not the GPU, is the bottleneck,
    and each of the nine conditions was re-decoding the same 400 clips. Written via a
    temp file + atomic rename so concurrent chains cannot read a half-written tensor.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / (Path(clip).stem + ".pt")
    if f.exists():
        try:
            return torch.load(f, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001
            f.unlink(missing_ok=True)
    t = processor["video"](str(ROOT / clip), va=True)
    tmp = f.with_suffix(f".pt.{os.getpid()}.tmp")
    torch.save(t, tmp)
    os.replace(tmp, f)
    return t

MANIFESTS = {
    "colour": ROOT / "claude_track2/data/held_attr_colour_v1.jsonl",
    "silence": ROOT / "claude_track2/data/held_attr_silence_v1.jsonl",
    "sound": ROOT / "claude_track2/data/held_attr_soundevent_v1.jsonl",
    "noise": ROOT / "claude_track2/data/held_attr_noise_v1.jsonl",
    # cross-proposition cells: the QUESTION is the sound event, the CLAIM is about audio
    # quality on the same clip. Same percept, different proposition.
    "grad_noise_g1": ROOT / "claude_track2/data/held_grad_noise_g1_v1.jsonl",
    "grad_noise_g2": ROOT / "claude_track2/data/held_grad_noise_g2_v1.jsonl",
    "grad_noise_g3": ROOT / "claude_track2/data/held_grad_noise_g3_v1.jsonl",
    "grad_noise_g4": ROOT / "claude_track2/data/held_grad_noise_g4_v1.jsonl",
    "grad_lowpass_g1": ROOT / "claude_track2/data/held_grad_lowpass_g1_v1.jsonl",
    "grad_lowpass_g2": ROOT / "claude_track2/data/held_grad_lowpass_g2_v1.jsonl",
    "grad_lowpass_g3": ROOT / "claude_track2/data/held_grad_lowpass_g3_v1.jsonl",
    "grad_lowpass_g4": ROOT / "claude_track2/data/held_grad_lowpass_g4_v1.jsonl",
    "grad_reverb_g1": ROOT / "claude_track2/data/held_grad_reverb_g1_v1.jsonl",
    "grad_reverb_g2": ROOT / "claude_track2/data/held_grad_reverb_g2_v1.jsonl",
    "grad_reverb_g3": ROOT / "claude_track2/data/held_grad_reverb_g3_v1.jsonl",
    "grad_reverb_g4": ROOT / "claude_track2/data/held_grad_reverb_g4_v1.jsonl",
    "aux_noise": ROOT / "claude_track2/data/held_aux_noise_v1.jsonl",
    "aux_lowpass": ROOT / "claude_track2/data/held_aux_lowpass_v1.jsonl",
    "lowpass": ROOT / "claude_track2/data/held_attr_lowpass_v1.jsonl",
    "reverb": ROOT / "claude_track2/data/held_attr_reverb_v1.jsonl",
}
QUESTIONS = {
    "colour": ("Is this video in greyscale (black and white) rather than in "
               "colour? (A) Yes (B) No"),
    "silence": ("Does the audio in this clip contain a gap of complete silence "
                "lasting about a second or more? (A) Yes (B) No"),
    "sound": None,
    "noise": ("Is the audio in this clip dominated by a loud continuous background "
              "hiss? (A) Yes (B) No"),
    "grad_noise_g1": None,
    "grad_noise_g2": None,
    "grad_noise_g3": None,
    "grad_noise_g4": None,
    "grad_lowpass_g1": None,
    "grad_lowpass_g2": None,
    "grad_lowpass_g3": None,
    "grad_lowpass_g4": None,
    "grad_reverb_g1": None,
    "grad_reverb_g2": None,
    "grad_reverb_g3": None,
    "grad_reverb_g4": None,
    "aux_noise": None, "aux_lowpass": None,
    "lowpass": ("Does the audio in this clip sound muffled, as if all the high "
                "frequencies had been removed? (A) Yes (B) No"),
    "reverb": ("Does the audio in this clip sound like it was recorded in a large "
               "echoing space? (A) Yes (B) No"),
}
PROP = {
    "colour": lambda yes, row: ("The clip is in greyscale." if yes
                                else "The clip is in full colour."),
    "silence": lambda yes, row: ("The clip contains a gap of complete silence."
                                 if yes else "The clip contains no silent gap."),
    "sound": lambda yes, row: (f"The clip contains the sound of "
                               f"{row['query_class']}." if yes else
                               f"The clip does not contain the sound of "
                               f"{row['query_class']}."),
    "noise": lambda yes, row: ("The clip's audio is dominated by a loud continuous hiss."
                               if yes else "The clip's audio is clean, with no added hiss."),
    "grad_noise_g1": lambda yes, row: ("The clip's audio has {p}." if yes else "The clip's audio has no {p}.").format(p=row["property"]),
    "grad_noise_g2": lambda yes, row: ("The clip's audio has {p}." if yes else "The clip's audio has no {p}.").format(p=row["property"]),
    "grad_noise_g3": lambda yes, row: ("The clip's audio has {p}." if yes else "The clip's audio has no {p}.").format(p=row["property"]),
    "grad_noise_g4": lambda yes, row: ("The clip's audio has {p}." if yes else "The clip's audio has no {p}.").format(p=row["property"]),
    "grad_lowpass_g1": lambda yes, row: ("The clip's audio has {p}." if yes else "The clip's audio has no {p}.").format(p=row["property"]),
    "grad_lowpass_g2": lambda yes, row: ("The clip's audio has {p}." if yes else "The clip's audio has no {p}.").format(p=row["property"]),
    "grad_lowpass_g3": lambda yes, row: ("The clip's audio has {p}." if yes else "The clip's audio has no {p}.").format(p=row["property"]),
    "grad_lowpass_g4": lambda yes, row: ("The clip's audio has {p}." if yes else "The clip's audio has no {p}.").format(p=row["property"]),
    "grad_reverb_g1": lambda yes, row: ("The clip's audio has {p}." if yes else "The clip's audio has no {p}.").format(p=row["property"]),
    "grad_reverb_g2": lambda yes, row: ("The clip's audio has {p}." if yes else "The clip's audio has no {p}.").format(p=row["property"]),
    "grad_reverb_g3": lambda yes, row: ("The clip's audio has {p}." if yes else "The clip's audio has no {p}.").format(p=row["property"]),
    "grad_reverb_g4": lambda yes, row: ("The clip's audio has {p}." if yes else "The clip's audio has no {p}.").format(p=row["property"]),
    "aux_noise": lambda yes, row: ("The audio in this clip is degraded by loud continuous "
                                   "background noise." if yes else
                                   "The audio in this clip is clean, with no added noise."),
    "aux_lowpass": lambda yes, row: ("The audio in this clip is severely band-limited." if yes
                                     else "The audio in this clip has its full frequency range."),
    "lowpass": lambda yes, row: ("The clip's audio is muffled and band-limited." if yes
                                 else "The clip's audio has its full frequency range."),
    "reverb": lambda yes, row: ("The clip's audio has the reverberation of a large space."
                                if yes else "The clip's audio is dry, with no reverberation."),
}
IRRELEVANT = "The clip was uploaded on a weekday."
# reworded questions that share no content words with the claim
PARAPHRASE = {
    "sound": None,
    "colour": ("Does this video lack colour, appearing only in shades of grey? "
               "(A) Yes (B) No"),
    "noise": ("Does a steady rushing sound cover the recording in this clip? "
              "(A) Yes (B) No"),
}
CONDS = ["none", "anon_high", "anon_low", "expert_high", "expert_low",
         "true_high", "true_low", "irrelevant_high", "irrelevant_low", "bare_false",
         "hedge_plain", "prior_high", "prior_low", "reliab_high", "reliab_low", "certain_high", "certain_low",
         "otherclip_high", "otherclip_low", "null_block",
         "priority_high", "priority_low", "priority_true_high", "priority_true_low", "srconly_high", "srconly_low", "urgency_high", "urgency_low", "opaque_high", "opaque_low",
         "n_055", "n_065", "n_075"]


def block(source: str, confidence: str, claim: str) -> str:
    return (f"Source: {source}\nClaim: \"{claim}\"\nConfidence: {confidence}\n\n")


def evidence(cond: str, attr: str, row) -> str:
    truth_yes = row["cond_label" if attr.startswith("aux_") else "label"] == "A"
    false_claim = PROP[attr](not truth_yes, row)
    true_claim = PROP[attr](truth_yes, row)
    return {
        "none": "",
        "anon_high": block("anonymous system", "high", false_claim),
        "anon_low": block("anonymous system", "low", false_claim),
        "expert_high": block("expert model", "high", false_claim),
        "expert_low": block("expert model", "low", false_claim),
        "true_high": block("anonymous system", "high", true_claim),
        "true_low": block("anonymous system", "low", true_claim),
        "irrelevant_high": block("anonymous system", "high", IRRELEVANT),
        "irrelevant_low": block("anonymous system", "low", IRRELEVANT),
        "bare_false": false_claim + "\n\n",
        # the block with nothing in it: whatever a structured field costs by merely being
        # present is measured here, with no claim to attribute it to
        "null_block": 'Source: ---\nClaim: ""\nConfidence: ---\n\n',
        # epistemic synonyms -- these should behave like Confidence if the effect is about
        # declared confidence rather than about the particular word "Confidence"
        # epistemic vocabulary, but scoped to the class base rate rather than to this claim --
        # the missing cell between the epistemic fields and the non-epistemic controls
        "prior_high": f'Source: anonymous system\nClaim: "{false_claim}"\nPrior: high\n\n',
        "prior_low": f'Source: anonymous system\nClaim: "{false_claim}"\nPrior: low\n\n',
        "reliab_high": f'Source: anonymous system\nClaim: "{false_claim}"\nReliability: high\n\n',
        "reliab_low": f'Source: anonymous system\nClaim: "{false_claim}"\nReliability: low\n\n',
        "certain_high": f'Source: anonymous system\nClaim: "{false_claim}"\nCertainty: high\n\n',
        "certain_low": f'Source: anonymous system\nClaim: "{false_claim}"\nCertainty: low\n\n',
        # the non-epistemic field: same block, same claim, same filled third slot, but the
        # value carries no statement about how much to believe the claim
        "priority_high": f'Source: anonymous system\nClaim: "{false_claim}"\nPriority: high\n\n',
        "priority_low": f'Source: anonymous system\nClaim: "{false_claim}"\nPriority: low\n\n',
        "priority_true_high": f'Source: anonymous system\nClaim: "{true_claim}"\nPriority: high\n\n',
        "priority_true_low": f'Source: anonymous system\nClaim: "{true_claim}"\nPriority: low\n\n',
        # everything priority_c has except the claim, so the difference isolates claim insertion
        "urgency_high": f'Source: anonymous system\nClaim: "{false_claim}"\nUrgency: high\n\n',
        "urgency_low": f'Source: anonymous system\nClaim: "{false_claim}"\nUrgency: low\n\n',
        "opaque_high": f'Source: anonymous system\nClaim: "{false_claim}"\nMetric-X: high\n\n',
        "opaque_low": f'Source: anonymous system\nClaim: "{false_claim}"\nMetric-X: low\n\n',
        "srconly_high": 'Source: anonymous system\nClaim: ""\nPriority: high\n\n',
        "srconly_low": 'Source: anonymous system\nClaim: ""\nPriority: low\n\n',
    }[cond]


def first_logits(model, tokenizer, tensor, instruct, mask_sensory=False):
    if mask_sensory:
        message = [{"role": "user", "content": instruct}]
        vis = None  # VideoLLaMA's generate() cannot take an empty list; None skips
                    # the multimodal path entirely, which is what "clip withheld" means
    else:
        vis = [(tensor.half().cuda() if not isinstance(tensor, dict)
                else {k: v.half().cuda() for k, v in tensor.items()}, "video")]
        message = [{"role": "user", "content": DEFAULT_VIDEO_TOKEN + "\n" + instruct}]
    prompt = tokenizer.apply_chat_template(message, tokenize=False,
                                           add_generation_prompt=True)
    input_ids = tokenizer_multimodal_token(prompt, tokenizer, DEFAULT_VIDEO_TOKEN,
                                           return_tensors="pt").unsqueeze(0).long().cuda()
    am = input_ids.ne(tokenizer.pad_token_id).long().cuda()
    with torch.inference_mode():
        out = model.generate(input_ids, attention_mask=am, images=vis,
                             do_sample=False, max_new_tokens=1, use_cache=True,
                             output_scores=True, return_dict_in_generate=True,
                             pad_token_id=tokenizer.eos_token_id)
    return out.scores[0][0].float()



def apply_swap(q: str) -> str:
    """(A) Yes (B) No  ->  (A) No (B) Yes. The label mapping is inverted at read-out."""
    return q.replace("(A) Yes (B) No", "(A) No (B) Yes")

SWAPPED = "(A) No (B) Yes"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", choices=list(MANIFESTS), required=True)
    ap.add_argument("--cond", choices=CONDS, required=True)
    ap.add_argument("--mask_sensory", action="store_true")
    ap.add_argument("--swap_options", action="store_true",
                    help="render (A) No (B) Yes and invert the read-out")
    ap.add_argument("--paraphrase", action="store_true",
                    help="reword the question so it shares no content words with the claim")
    ap.add_argument("--post", action="store_true",
                    help="place the metadata block after the question")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()
    flagtag = (("_swap" if args.swap_options else "") + ("_post" if args.post else "")
               + ("_para" if args.paraphrase else ""))

    rows = [json.loads(l) for l in MANIFESTS[args.attr].open() if l.strip()]
    out = ROOT / (f"claude_track2/outputs/s2vl_{args.attr}_"
                  f"{args.cond}{flagtag}{args.tag}_v1.jsonl")
    print(f"[data] {len(rows)} clips, attr={args.attr}, cond={args.cond}", flush=True)

    model, processor, tokenizer = model_init("DAMO-NLP-SG/VideoLLaMA2.1-7B-AV",
                                             device_map={"": 0}, use_flash_attn=False)
    model.eval()
    LIDS = {L: tokenizer.encode(L, add_special_tokens=False)[0] for L in ("A", "B")}

    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip():
                done.add(json.loads(l)["clip"])
        print(f"[resume] {len(done)}", flush=True)

    n = ok = 0
    t0 = time.time()
    with out.open("a", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            if r["clip"] in done:
                continue
            try:
                if args.attr.startswith("grad_"):
                    q = (f"Does the audio in this clip have {r['property']}? "
                         "(A) Yes (B) No")
                elif True:
                    q = (PARAPHRASE.get(args.attr) if args.paraphrase
                     else QUESTIONS[args.attr])
                if q is None:
                    q = ((f"Can the sound of {r['query_class']} be heard anywhere "
                          "in this recording? (A) Yes (B) No") if args.paraphrase
                         else (f"Does this clip contain the sound of "
                               f"{r['query_class']}? (A) Yes (B) No"))
                q = apply_swap(q) if args.swap_options else q
                ev = evidence(args.cond, args.attr, r)
                q = (q + "\n\n" + ev) if args.post else (ev + q)
                tensor = (None if args.mask_sensory else
                          preprocess(processor, r["clip"]))
                lg = first_logits(model, tokenizer, tensor, q, args.mask_sensory)
                ab = torch.tensor([lg[LIDS["A"]].item(), lg[LIDS["B"]].item()])
                p = F.softmax(ab, dim=0)
                pred = "A" if p[0] >= 0.5 else "B"
                if args.swap_options:      # "A" now means "No", i.e. label B
                    pred = "B" if pred == "A" else "A"
                n += 1
                ok += int(pred == r["label"])
                f.write(json.dumps({"clip": r["clip"], "label": r["label"],
                                    "pred": pred, "p_A": round(float(p[0]), 6),
                                    "correct": pred == r["label"],
                                    "attr": args.attr, "cond": args.cond}) + "\n")
                f.flush()
                if n % 50 == 0:
                    print(f"[{i+1}/{len(rows)}] acc {ok/n:.3f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"SKIP {r['clip']}: {type(e).__name__}: {e}", flush=True)
    print(f"DONE attr={args.attr} cond={args.cond} n={n} acc={ok/max(1,n):.4f}",
          flush=True)


if __name__ == "__main__":
    main()
