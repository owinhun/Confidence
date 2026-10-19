#!/usr/bin/env python3
"""The same protocol on three additional model families, to widen the evidence base.

The confirmatory study so far covers Qwen2.5-Omni, Qwen3-Omni and VideoLLaMA2.1-AV -- two of
which share a lineage. This script adds Qwen2-Audio (audio-only), Phi-4-multimodal (Microsoft)
and MiniCPM-o 2.6 (OpenBMB) so the claim rests on genuinely distinct families and, in the
audio-only case, does not depend on the presence of a video stream at all.

Conditions are identical to the main study: the claim proposition is byte-identical across
cells and only Source/Confidence move. Scoring is a single forward pass over the option-letter
logits, as before.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data4/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/data4/hf_cache/hub")
os.environ.setdefault("HF_MODULES_CACHE", "/data4/hf_cache/modules")

ROOT = Path("/workspace/svhalluc")
MODELS = {
    "qwen2audio": "/data4/svhalluc_models/Qwen2-Audio-7B-Instruct",
    "phi4mm": "/data4/svhalluc_models/Phi-4-multimodal-instruct",
    "minicpmo": "/data4/svhalluc_models/MiniCPM-o-2_6",
}
AUDIO_ONLY = {"qwen2audio"}

MANIFESTS = {
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
    "colour": ROOT / "claude_track2/data/held_attr_colour_v1.jsonl",
}
QUESTIONS = {
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
    "colour": ("Is this video in greyscale (black and white) rather than in "
               "colour? (A) Yes (B) No"),
}
PROP = {
    "sound": lambda yes, r: (f"The clip contains the sound of {r['query_class']}." if yes
                             else f"The clip does not contain the sound of {r['query_class']}."),
    "noise": lambda yes, r: ("The clip's audio is dominated by a loud continuous hiss." if yes
                             else "The clip's audio is clean, with no added hiss."),
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
    "aux_noise": lambda yes, r: ("The audio in this clip is degraded by loud continuous "
                                 "background noise." if yes else
                                 "The audio in this clip is clean, with no added noise."),
    "aux_lowpass": lambda yes, r: ("The audio in this clip is severely band-limited." if yes
                                   else "The audio in this clip has its full frequency range."),
    "colour": lambda yes, r: ("The clip is in greyscale." if yes
                              else "The clip is in full colour."),
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
LEVELS = {"n_055": "0.55", "n_065": "0.65", "n_075": "0.75",
          "v_vlow": "very low", "v_low": "low", "v_mod": "moderate",
          "v_high": "high", "v_vhigh": "very high",
          "n_015": "0.15", "n_035": "0.35", "n_060": "0.60",
          "n_085": "0.85", "n_095": "0.95"}
CONDS = (["none", "anon_high", "anon_low", "expert_high", "expert_low", "true_high",
          "irrelevant_high", "irrelevant_low", "bare_false", "bare_true", "null_priority",
          "hedge_plain", "prior_high", "prior_low", "reliab_high", "reliab_low", "certain_high", "certain_low",
         "otherclip_high", "otherclip_low", "null_block",
         "priority_high", "priority_low", "priority_true_high", "priority_true_low", "srconly_high", "srconly_low", "cascadex_plain", "cascade2x_plain", "cascade3x_plain", "cascademix80_plain", "cascademix85_plain", "cascademix90_plain", "cascademix95_plain", "cascademix80s1_plain", "cascademix85s1_plain", "cascademix90s1_plain", "cascademix95s1_plain", "cascademix80s2_plain", "cascademix85s2_plain", "cascademix90s2_plain", "cascademix95s2_plain", "cascademixstr_plain", "cascadex_bare", "cascadex_conf", "cascadex_pct", "cascade2x_bare", "cascade2x_conf", "cascade2x_pct", "cascade_bare", "cascade_conf", "cascade2_bare", "cascade2_conf", "cascade2_pct", "detclaim_true", "detclaim_false", "cascade_mono", "cascade_pct", "urgency_high", "urgency_low", "opaque_high", "opaque_low",
         "true_low",
         "n_055", "n_065", "n_075"] + list(LEVELS))


def block(source, confidence, claim):
    return f'Source: {source}\nClaim: "{claim}"\nConfidence: {confidence}\n\n'


def evidence(cond, attr, row, claim_true=False):
    truth_yes = row["cond_label" if attr.startswith("aux_") else "label"] == "A"
    false_claim = PROP[attr](not truth_yes, row)
    true_claim = PROP[attr](truth_yes, row)
    if cond == "none":
        return ""
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
        return PROP[attr](row["label"] == "A", row) + "\n\n"
    if cond == "null_priority":
        return 'Source: ---\nClaim: ""\nPriority: ---\n\n'
    if cond == "bare_false":
        return false_claim + "\n\n"
    if cond in LEVELS:
        return block("anonymous system", LEVELS[cond],
                     true_claim if claim_true else false_claim)
    src = "expert model" if cond.startswith("expert") else "anonymous system"
    conf = "high" if cond.endswith("high") else "low"
    if cond.startswith("true"):
        return block(src, conf, true_claim)
    if cond.startswith("irrelevant"):
        return block(src, conf, IRRELEVANT)
    return block(src, conf, false_claim)


def question_for(attr, r, paraphrase=False, readback=False):
    """The question put to the model.

    `readback` is the manipulation check Audio Flamingo 3 needs. Its override rate is 49.3 vs
    49.2 -- a flat null -- and a null has two readings: the model never attends the metadata
    block, or it attends and does not integrate. Asking about the *message* instead of the clip
    separates them, because answering needs only that the block was read. Scoring is against the
    claim's polarity, not the ground truth, so a false-claim condition is answered correctly by
    contradicting the manifest; the read-out inverts accordingly at analysis time.
    """
    if readback:
        # "the sound described in it" is ambiguous when the claim is a negation -- it is unclear
        # whether the negation is part of what is being referred to. Prefixing the ordinary
        # question instead keeps the proposition byte-identical to the one the claim asserts,
        # so the only thing that changes is whose answer is being asked for.
        q = question_for(attr, r)
        return "According to the message above, " + q[0].lower() + q[1:]
    if paraphrase:
        q = PARAPHRASE.get(attr)
        return q if q is not None else (
            f"Can the sound of {r['query_class']} be heard anywhere in this "
            "recording? (A) Yes (B) No")
    q = QUESTIONS[attr]
    if q is None:
        q = f"Does this clip contain the sound of {r['query_class']}? (A) Yes (B) No"
    return q


def read_audio(path, sr=16000):
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=True, duration=30.0)
    return y



def apply_swap(q: str) -> str:
    """(A) Yes (B) No  ->  (A) No (B) Yes. The label mapping is inverted at read-out."""
    return q.replace("(A) Yes (B) No", "(A) No (B) Yes")

SWAPPED = "(A) No (B) Yes"

# --------------------------------------------------------------------- backends
def backend_qwen2audio():
    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    mp = MODELS["qwen2audio"]
    proc = AutoProcessor.from_pretrained(mp)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        mp, dtype=torch.bfloat16, device_map="cuda:0").eval()
    ids = {L: proc.tokenizer.encode(L, add_special_tokens=False)[0] for L in ("A", "B")}

    def score(clip, q, mask=False):
        content = ([] if mask else [{"type": "audio", "audio_url": "x"}])
        content.append({"type": "text", "text": q})
        conv = [{"role": "user", "content": content}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        if mask:
            inp = proc(text=text, return_tensors="pt", padding=True)
        else:
            au = read_audio(ROOT / clip, proc.feature_extractor.sampling_rate)
            inp = proc(text=text, audio=[au], return_tensors="pt", padding=True,
                       sampling_rate=proc.feature_extractor.sampling_rate)
        inp = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inp.items()}
        with torch.inference_mode():
            lg = model(**inp).logits[0, -1, :].float()
        return lg[ids["A"]].item(), lg[ids["B"]].item()
    return score


def backend_phi4mm():
    # Phi-4-multimodal's remote code targets an older peft API; the installed 0.19 raises
    # on prepare_inputs_for_generation. A pinned 0.13.2 is kept in its own directory and
    # prepended here rather than downgrading the shared environment for every other user.
    sys.path.insert(0, "/data4/pyenv_phi4")
    for m in [k for k in list(sys.modules) if k == "peft" or k.startswith("peft.")]:
        del sys.modules[m]
    import peft  # noqa: F401
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    mp = MODELS["phi4mm"]
    # The remote code also predates the transformers release installed here, which moved
    # prepare_inputs_for_generation off the model class; peft then fails looking for it.
    # Patch the dynamically loaded class rather than pinning transformers, which Qwen3-Omni
    # needs at a newer version.
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.generation import GenerationMixin
    for name in ("modeling_phi4mm.Phi4MMModel", "modeling_phi4mm.Phi4MMForCausalLM"):
        try:
            cls = get_class_from_dynamic_module(name, mp)
        except Exception:  # noqa: BLE001
            continue
        if not hasattr(cls, "prepare_inputs_for_generation"):
            cls.prepare_inputs_for_generation = GenerationMixin.prepare_inputs_for_generation
    proc = AutoProcessor.from_pretrained(mp, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        mp, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True,
        _attn_implementation="eager").eval()
    ids = {L: proc.tokenizer.encode(L, add_special_tokens=False)[0] for L in ("A", "B")}

    def score(clip, q, mask=False):
        if mask:
            inp = proc(text=f"<|user|>{q}<|end|><|assistant|>", return_tensors="pt")
        else:
            prompt = f"<|user|><|audio_1|>{q}<|end|><|assistant|>"
            au = read_audio(ROOT / clip)
            inp = proc(text=prompt, audios=[(au, 16000)], return_tensors="pt")
        inp = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inp.items()}
        with torch.inference_mode():
            # use_cache=False: the remote code calls a Cache method that newer transformers
            # removed, and a single scoring pass has no use for the cache anyway.
            lg = model(**inp, use_cache=False).logits[0, -1, :].float()
        return lg[ids["A"]].item(), lg[ids["B"]].item()
    return score


def backend_minicpmo():
    import torch
    from transformers import AutoModel, AutoTokenizer
    mp = MODELS["minicpmo"]
    tok = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        mp, trust_remote_code=True, attn_implementation="sdpa", dtype=torch.bfloat16,
        init_vision=False, init_audio=True, init_tts=False).eval().cuda()
    ids = {L: tok.encode(L, add_special_tokens=False)[0] for L in ("A", "B")}

    def score(clip, q):
        au = read_audio(ROOT / clip)
        msgs = [{"role": "user", "content": [au, q]}]
        out = model.chat(msgs=msgs, tokenizer=tok, sampling=False, max_new_tokens=1,
                         output_logits=True, return_dict_in_generate=True)
        lg = out.logits[0][0].float()
        return lg[ids["A"]].item(), lg[ids["B"]].item()
    return score


BACKENDS = {"qwen2audio": backend_qwen2audio, "phi4mm": backend_phi4mm,
            "minicpmo": backend_minicpmo}


def main():
    import math
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--attr", choices=list(MANIFESTS), required=True)
    ap.add_argument("--cond", choices=CONDS, required=True)
    ap.add_argument("--claim", choices=["false", "true"], default="false")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mask_sensory", action="store_true",
                    help="engagement control: withhold the signal so only the claim remains")
    ap.add_argument("--paraphrase", action="store_true",
                    help="reword the question so it shares no content words with the claim")
    ap.add_argument("--swap_options", action="store_true")
    ap.add_argument("--readback", action="store_true")
    ap.add_argument("--post", action="store_true")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()
    flagtag = (("_swap" if args.swap_options else "") + ("_post" if args.post else "")
               + ("_para" if args.paraphrase else "")
               + ("_masked" if args.mask_sensory else "")
               + ("_readback" if args.readback else ""))

    if args.model in AUDIO_ONLY and args.attr == "colour":
        print("SKIP: audio-only model cannot judge a visual attribute", flush=True)
        return

    rows = [json.loads(l) for l in MANIFESTS[args.attr].open() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    tag = "" if args.claim == "false" else "_T"
    out = ROOT / (f"claude_track2/outputs/nm_{args.model}_{args.attr}_"
                  f"{args.cond}{tag}{flagtag}_v1.jsonl")
    print(f"[data] {len(rows)} clips  {args.model}/{args.attr}/{args.cond}", flush=True)

    score = BACKENDS[args.model]()
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
                qq = question_for(args.attr, r, args.paraphrase, args.readback)
                qq = apply_swap(qq) if args.swap_options else qq
                ev = evidence(args.cond, args.attr, r, args.claim == "true")
                q = (qq + "\n\n" + ev) if args.post else (ev + qq)
                la, lb = score(r["clip"], q, args.mask_sensory)
                pa = 1.0 / (1.0 + math.exp(-(la - lb)))
                pred = "A" if pa >= 0.5 else "B"
                if args.swap_options:      # "A" now means "No", i.e. label B
                    pred = "B" if pred == "A" else "A"
                n += 1
                ok += int(pred == r["label"])
                f.write(json.dumps({"clip": r["clip"], "label": r["label"], "pred": pred,
                                    "p_A": round(pa, 6), "correct": pred == r["label"],
                                    "attr": args.attr, "cond": args.cond,
                                    "claim": args.claim, "model": args.model}) + "\n")
                f.flush()
                if n % 50 == 0:
                    print(f"[{i+1}/{len(rows)}] acc {ok/n:.3f} ({time.time()-t0:.0f}s)",
                          flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"SKIP {r['clip']}: {type(e).__name__}: {e}", flush=True)
    print(f"DONE {args.model} {args.attr} {args.cond} n={n} acc={ok/max(1,n):.4f}", flush=True)


if __name__ == "__main__":
    main()
