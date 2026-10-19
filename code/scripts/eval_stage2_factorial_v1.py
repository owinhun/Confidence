#!/usr/bin/env python3
"""Stage 2 confirmatory factorial (pre-registered; run only after Stage 1 closes).

Design fix demanded by the review: keep the CLAIM PROPOSITION byte-identical and
vary only two structured fields, so source authority and epistemic commitment are
manipulated independently of verb choice, sentence length and syntax.

  Source: anonymous system | expert model
  Confidence: high | low
  Claim: "<identical sentence>"

Main 2x2 (all false claims) plus separate controls with source fixed:
  none | true-high | false-high | irrelevant-high | bare-false-declarative

Run on a HELD-OUT video set that Stage 1 never touched.
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
    "sound": None,  # built per row from query_class
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
PROP = {  # the claim sentence, parameterised only by which value is asserted
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


def block(source: str, confidence: str, claim: str) -> str:
    return (f"Source: {source}\nClaim: \"{claim}\"\nConfidence: {confidence}\n\n")


def evidence(cond: str, attr: str, row) -> str:
    truth_yes = row["cond_label" if attr.startswith("aux_") else "label"] == "A"
    false_claim = PROP[attr](not truth_yes, row)
    true_claim = PROP[attr](truth_yes, row)
    if cond == "none":
        return ""
    if cond == "anon_high":
        return block("anonymous system", "high", false_claim)
    if cond == "anon_low":
        return block("anonymous system", "low", false_claim)
    if cond == "expert_high":
        return block("expert model", "high", false_claim)
    if cond == "expert_low":
        return block("expert model", "low", false_claim)
    if cond == "true_low":
        # the true-claim branch at LOW declared confidence. Without it the claim-aligned
        # contrast exists at one rung only, and the high-minus-low swing -- the quantity that
        # separates the confidence field from a filled field -- cannot be order-averaged.
        return block("anonymous system", "low", true_claim)
    if cond == "true_high":
        return block("anonymous system", "high", true_claim)
    if cond == "irrelevant_high":
        return block("anonymous system", "high", IRRELEVANT)
    if cond == "irrelevant_low":
        # decisive control: does LOW confidence damage accuracy even when the claim bears
        # on nothing? If yes, low confidence is disruptive per se and the harm seen on
        # true claims is not about the claim.
        return block("anonymous system", "low", IRRELEVANT)
    if cond.startswith("n_"):
        # numeric declared confidence, all values above 0.5 so that "low confidence asserts
        # the complement" cannot account for the effect
        return block("anonymous system",
                     f"{int(cond.split('_')[1]) / 100:.2f}", false_claim)
    if cond == "null_block":
        # regression-to-the-mean control: the block is present and empty, so any drop that
        # comes from perturbing the prompt at all is measured with no claim in it
        return 'Source: ---\nClaim: ""\nConfidence: ---\n\n'
    if cond.startswith("urgency_") or cond.startswith("opaque_"):
        # Two further non-epistemic fields, because Priority is not perfectly neutral: it can be
        # read as an instruction to prioritise the claim. Urgency is action-relevant like
        # Priority; Metric-X is graded but bears on nothing at all. If the confidence effect
        # were really about having any filled graded field, these should reproduce it.
        field = "Urgency" if cond.startswith("urgency_") else "Metric-X"
        lvl = "high" if cond.endswith("high") else "low"
        return f'Source: anonymous system\nClaim: "{false_claim}"\n{field}: {lvl}\n\n'
    if cond.startswith("detclaim_"):
        # The controlled branches matched to the claim-only cascade policy. pi* was previously
        # built from true_high/anon_high, which carry a Confidence field, and then applied to
        # cascade_bare, which does not -- and this paper's own result is that the field changes
        # the branch gains, so the threshold and the policy were not the same object. These two
        # differ from cascade_bare in exactly one thing: whether the claim is true.
        # BUG FIX 2026-08-20: this read PROP[attr](cond.endswith("true"), row), which asserts
        # the A-side proposition regardless of the clip's label. On the balanced manifest that
        # made detclaim_true a claim that is true on only half the clips -- an "always says
        # present" constant detector, not the true-claim branch the comment below describes.
        # detclaim_false was its complement, also 50% true. Every break-even built from this
        # pair was therefore built from two 50%-accurate branches. bare_true/bare_false were
        # always truth-relative, which is why the bare and block halves disagreed.
        claim = true_claim if cond.endswith("true") else false_claim
        return f'Source: sound-event detector\nClaim: "{claim}"\n\n'
    if cond in ("cascadex_plain", "cascade2x_plain", "cascade3x_plain", "cascademix80_plain", "cascademix85_plain", "cascademix90_plain", "cascademix95_plain", "cascademix80s1_plain", "cascademix85s1_plain", "cascademix90s1_plain", "cascademix95s1_plain", "cascademix80s2_plain", "cascademix85s2_plain", "cascademix90s2_plain", "cascademix95s2_plain", "cascademixstr_plain"):
        # The detector's own decision, sent as a plain sentence instead of inside the
        # Source/Claim block. This is the policy the packaging argument is about: the controlled
        # branches show a bare true claim helps and a bare false one hurts, but only running the
        # detector's actual mixture in bare form says whether bare forwarding pays at the
        # accuracy this tagger has.
        import json as _json, os as _os
        # cascade3x is Qwen2-Audio acting as the detector -- a second *family*, not a
        # second checkpoint of the same one
        _w = (cond[7:cond.rfind("_")] if cond.startswith("cascademix")
              else "clap2x" if cond.startswith("cascade2x")
              else "q2ax" if cond.startswith("cascade3x") else "clapx")
        global _XP
        try:
            _XP
        except NameError:
            _XP = {}
        if _w not in _XP:
            _d = {}
            _pp = f"/workspace/svhalluc/claude_track2/outputs/{_w}_cascade_v1.jsonl"
            if _os.path.exists(_pp):
                for _l in open(_pp):
                    if _l.strip():
                        _r = _json.loads(_l)
                        _d[_r["clip"]] = _r["det_pred"]
            _XP[_w] = _d
        dp = _XP[_w].get(row["clip"])
        if dp is None:
            raise ValueError(f"no {_w} prediction for {row['clip']}")
        return PROP[attr](dp == "A", row) + "\n\n"
    if cond.startswith("cascadex_") or cond.startswith("cascade2x_"):
        # Cross-fitted cascade: the threshold, calibration and band precisions that produced this
        # clip's message were fitted on the other four folds of source videos, so all 400 clips
        # are usable without the leak that restricted the earlier analysis to half of them.
        import json as _json, os as _os
        _which = "clap2x" if cond.startswith("cascade2x_") else "clapx"
        global _XC
        try:
            _XC
        except NameError:
            _XC = {}
        if _which not in _XC:
            _d = {}
            _pp = f"/workspace/svhalluc/claude_track2/outputs/{_which}_cascade_v1.jsonl"
            if _os.path.exists(_pp):
                for _l in open(_pp):
                    if _l.strip():
                        _r = _json.loads(_l)
                        _d[_r["clip"]] = (_r["det_pred"], _r["det_level"], _r["cal_pct"])
            _XC[_which] = _d
        entx = _XC[_which].get(row["clip"])
        if entx is None:
            raise ValueError(f"no {_which} output for {row['clip']}")
        dpx, dlx, cpx = entx
        claim = PROP[attr](dpx == "A", row)
        head = f'Source: sound-event detector\nClaim: "{claim}"'
        if cond.endswith("_bare"):
            return head + "\n\n"
        val = dlx if cond.endswith("_conf") else f"{cpx}%"
        return head + f"\nConfidence: {val}\n\n"
    if cond.startswith("cascade2_"):
        # Same cascade, second detector. A finding that survives a change of upstream tagger is
        # about the interface rather than about CLAP.
        import json as _json, os as _os
        global _CASCADE2
        try:
            _CASCADE2
        except NameError:
            _CASCADE2 = {}
            _p2 = "/workspace/svhalluc/claude_track2/outputs/clap2_cascade_v1.jsonl"
            if _os.path.exists(_p2):
                for _l in open(_p2):
                    if _l.strip():
                        _r = _json.loads(_l)
                        _CASCADE2[_r["clip"]] = (_r["det_pred"], _r["det_level"], _r["cal_pct"])
        ent2 = _CASCADE2.get(row["clip"])
        if ent2 is None:
            raise ValueError(f"no second-detector output for {row['clip']}")
        dp, dl, cp = ent2
        claim = PROP[attr](dp == "A", row)
        head = f'Source: sound-event detector\nClaim: "{claim}"'
        if cond == "cascade2_bare":
            return head + "\n\n"
        val = dl if cond == "cascade2_conf" else f"{cp}%"
        return head + f"\nConfidence: {val}\n\n"
    if cond.startswith("cascade_"):
        # The real cascade. The claim is whatever CLAP predicted for this clip -- right on about
        # seven clips in ten and wrong on the rest, with the errors falling where the detector
        # actually struggles rather than where we chose to put them. cascade_bare forwards the
        # claim alone; cascade_conf adds the confidence band the detector's own score falls in.
        import json as _json, os as _os
        global _CASCADE
        try:
            _CASCADE
        except NameError:
            _CASCADE = {}
            _p = "/workspace/svhalluc/claude_track2/outputs/clap_cascade_v1.jsonl"
            if _os.path.exists(_p):
                for _l in open(_p):
                    if _l.strip():
                        _r = _json.loads(_l)
                        _CASCADE[_r["clip"]] = (_r["det_pred"], _r["det_level"],
                                                _r.get("mono_level"), _r.get("cal_pct"))
        ent = _CASCADE.get(row["clip"])
        if ent is None:
            raise ValueError(f"no detector output for {row['clip']}")
        det_pred, det_level, mono_level, cal_pct = ent
        # the detector says "present" (A) or "absent" (B); phrase its decision as the claim
        claim = PROP[attr](det_pred == "A", row)
        head = f'Source: sound-event detector\nClaim: "{claim}"'
        if cond == "cascade_bare":
            return head + "\n\n"
        # cascade_mono re-labels the bands so the word ordering matches measured precision;
        # cascade_pct sends that precision itself. Both are calibrated on the half of the source
        # videos the threshold was chosen on, so the scored half stays untouched.
        val = {"cascade_conf": det_level, "cascade_mono": mono_level,
               "cascade_pct": f"{cal_pct}%"}[cond]
        return head + f"\nConfidence: {val}\n\n"
    if cond.startswith("srconly_"):
        # the missing rung of the ladder. priority_c already has Source, a claim and a filled
        # non-epistemic field; this has everything EXCEPT the claim. The difference between the
        # two is therefore claim insertion alone, with source, block and field held fixed --
        # which is what the earlier "+ claim" column was wrongly assumed to be.
        lvl = "high" if cond.endswith("high") else "low"
        return f'Source: anonymous system\nClaim: ""\nPriority: {lvl}\n\n'
    if cond.startswith("priority_true_"):
        # the non-epistemic field with a TRUE claim. Pairing this with priority_<lvl> gives
        # the claim-aligned contrast for a field that declares no confidence, which is the
        # baseline the declared-confidence contrast has to beat to be about confidence.
        lvl = "high" if cond.endswith("high") else "low"
        return f'Source: anonymous system\nClaim: "{true_claim}"\nPriority: {lvl}\n\n'
    if cond.startswith("priority_"):
        # a field with no epistemic content: if the effect is about the WORD rather than
        # about expressed confidence, this should move the answer too
        lvl = "high" if cond.endswith("high") else "low"
        return f'Source: anonymous system\nClaim: "{false_claim}"\nPriority: {lvl}\n\n'
    if cond.startswith("otherclip_"):
        lvl = "high" if cond.endswith("high") else "low"
        return block("anonymous system", lvl,
                     "A different clip in this collection " +
                     PROP[attr](True, row)[len("The clip "):].replace("is ", "is ", 1))
    if cond == "hedge_plain":
        # directionless: no claim is asserted, only that the matter is unsettled
        prop = PROP[attr](True, row)
        subject = prop.rstrip(".")
        return f'Source: anonymous system\nClaim: "It is unclear whether {subject[0].lower()}{subject[1:]}."\n\n'
    if cond.startswith("prior_"):
        # Epistemic register, but scoped to the class base rate rather than to this clip's
        # claim. Confidence/Reliability/Certainty all say how much to believe *this* claim;
        # Priority/Urgency/Metric-X say nothing epistemic at all. Neither separates "epistemic
        # wording" from "epistemic wording aimed at the claim", which is the reading a reviewer
        # can still offer for the whole 2.1-3.9x result. This field is the missing cell of that
        # 2x2: same block, same graded value, epistemic vocabulary, different referent.
        lvl = "high" if cond.endswith("high") else "low"
        return (f'Source: anonymous system\nClaim: "{false_claim}"\nPrior: {lvl}\n\n')
    if cond.startswith("reliab_") or cond.startswith("certain_"):
        field = "Reliability" if cond.startswith("reliab_") else "Certainty"
        lvl = "high" if cond.endswith("high") else "low"
        return (f'Source: anonymous system\nClaim: "{false_claim}"\n{field}: {lvl}\n\n')
    if cond == "bare_true":
        # The true claim as a bare sentence: no Source, no field, no key:value framing. Paired
        # with bare_false it gives the claim-aligned contrast for an unstructured message, and
        # paired with detclaim_* it separates what the claim asserts from how it is packaged.
        return true_claim + "\n\n"
    if cond == "null_priority":
        # The empty block with the same third field the source-only rung uses; without it the
        # step from empty block to source-only moves source, field name and field value at once.
        return 'Source: ---\nClaim: ""\nPriority: ---\n\n'
    if cond == "bare_false":
        return false_claim + "\n\n"
    raise ValueError(cond)


CONDS = ["none", "anon_high", "anon_low", "expert_high", "expert_low",
         "true_high", "true_low", "irrelevant_high", "irrelevant_low", "bare_false", "bare_true", "null_priority",
         "hedge_plain", "prior_high", "prior_low", "reliab_high", "reliab_low", "certain_high", "certain_low",
         "otherclip_high", "otherclip_low", "null_block",
         "priority_high", "priority_low", "priority_true_high", "priority_true_low", "srconly_high", "srconly_low", "cascadex_plain", "cascade2x_plain", "cascade3x_plain", "cascademix80_plain", "cascademix85_plain", "cascademix90_plain", "cascademix95_plain", "cascademix80s1_plain", "cascademix85s1_plain", "cascademix90s1_plain", "cascademix95s1_plain", "cascademix80s2_plain", "cascademix85s2_plain", "cascademix90s2_plain", "cascademix95s2_plain", "cascademixstr_plain", "cascadex_bare", "cascadex_conf", "cascadex_pct", "cascade2x_bare", "cascade2x_conf", "cascade2x_pct", "cascade_bare", "cascade_conf", "cascade2_bare", "cascade2_conf", "cascade2_pct", "detclaim_true", "detclaim_false", "cascade_mono", "cascade_pct", "urgency_high", "urgency_low", "opaque_high", "opaque_low",
         "n_055", "n_065", "n_075"]



def apply_swap(q: str) -> str:
    """(A) Yes (B) No  ->  (A) No (B) Yes. The label mapping is inverted at read-out."""
    return q.replace("(A) Yes (B) No", "(A) No (B) Yes")

SWAPPED = "(A) No (B) Yes"

def build_inputs(processor, path, question):
    content = ([] if path is None else
               [{"type": "video", "video": path, "fps": FPS, "max_frames": MAX_FRAMES}])
    content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": content}]
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
    ap.add_argument("--cond", choices=CONDS, required=True)
    ap.add_argument("--mask_sensory", action="store_true",
                    help="engagement control: withhold the clip, so we can see whether the "
                         "structured block is READ at all when there is no percept to defend")
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
    out = ROOT / f"claude_track2/outputs/s2_{args.attr}_{args.cond}{flagtag}{args.tag}_v1.jsonl"
    print(f"[data] {len(rows)} clips, attr={args.attr}, cond={args.cond}", flush=True)

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
                inp = build_inputs(processor,
                                   None if args.mask_sensory else str(ROOT / r["clip"]), q)
                inp = inp.to(model.device).to(torch.bfloat16)
                with torch.inference_mode():
                    o = model(**inp, use_audio_in_video=True, return_dict=True)
                lg = o.logits[0, -1, :].float()
                ab = torch.tensor([lg[LID["A"]].item(), lg[LID["B"]].item()])
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
