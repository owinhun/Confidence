#!/usr/bin/env python3
"""Confidence as a CONTROL KNOB for cascaded audio-visual systems (pre-registered).

Why this experiment exists, in ICASSP terms.

Deployed audio-visual systems are cascades: an upstream detector (sound-event tagger,
sync estimator, VAD) emits a decision, that decision is serialised into text, and a
multimodal LLM consumes it alongside the raw signal. Stage 2 showed the failure mode of
that cascade -- when the upstream module is wrong the LLM abandons its own percept -- and
showed that a single declared-confidence field, not the claim content, governs how far it
abandons it. Two levels ("high" / "low") establish that the field is causal. They do not
tell a system designer what to put in it.

This experiment turns the field into a calibrated interface. Two scales over the SAME
byte-identical false claim:

  verbal  : very low -> low -> moderate -> high -> very high
  numeric : 0.15 -> 0.35 -> 0.60 -> 0.85 -> 0.95     (what a detector actually outputs)

Pre-registered predictions:
  P1 override rate increases monotonically with declared confidence on both scales
  P2 the numeric scale works at least as well as the verbal one -- if it does, the design
     rule is "forward the detector's calibrated posterior verbatim; do not verbalise it"
  P3 the mapping is model-specific, so the knob must be characterised per model

Run on the SAME held-out manifest and the SAME claim strings as Stage 2, so this is an
extension of the confirmatory design, not a new one.
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

ROOT = Path("/workspace/svhalluc")

MANIFESTS = {
    "colour": ROOT / "claude_track2/data/held_attr_colour_v1.jsonl",
    "sound": ROOT / "claude_track2/data/held_attr_soundevent_v1.jsonl",
    "noise": ROOT / "claude_track2/data/held_attr_noise_v1.jsonl",
    "lowpass": ROOT / "claude_track2/data/held_attr_lowpass_v1.jsonl",
    "reverb": ROOT / "claude_track2/data/held_attr_reverb_v1.jsonl",
    # auxiliary-evidence task: the QUESTION is the sound event, the CLAIM is signal quality.
    # The detector can produce the claim (it hears the audio) but cannot answer the question,
    # so metadata is auxiliary evidence rather than the answer -- the detector-only baseline
    # is undefined here by construction.
    "aux_noise": ROOT / "claude_track2/data/held_aux_noise_v1.jsonl",
    "aux_lowpass": ROOT / "claude_track2/data/held_aux_lowpass_v1.jsonl",
    "aux_reverb": ROOT / "claude_track2/data/held_aux_reverb_v1.jsonl",
}
QUESTIONS = {
    "colour": ("Is this video in greyscale (black and white) rather than in "
               "colour? (A) Yes (B) No"),
    "sound": None,
    "noise": ("Is the audio in this clip dominated by a loud continuous background "
              "hiss? (A) Yes (B) No"),
    "lowpass": ("Does the audio in this clip sound muffled, as if all the high "
                "frequencies had been removed? (A) Yes (B) No"),
    "reverb": ("Does the audio in this clip sound like it was recorded in a large "
               "echoing space? (A) Yes (B) No"),
    "aux_noise": None, "aux_lowpass": None, "aux_reverb": None,
}
PROP = {
    "colour": lambda yes, row: ("The clip is in greyscale." if yes
                                else "The clip is in full colour."),
    "sound": lambda yes, row: (f"The clip contains the sound of "
                               f"{row['query_class']}." if yes else
                               f"The clip does not contain the sound of "
                               f"{row['query_class']}."),
    "noise": lambda yes, row: ("The clip's audio is dominated by a loud continuous hiss."
                               if yes else "The clip's audio is clean, with no added hiss."),
    "lowpass": lambda yes, row: ("The clip's audio is muffled and band-limited." if yes
                                 else "The clip's audio has its full frequency range."),
    "reverb": lambda yes, row: ("The clip's audio has the reverberation of a large space."
                                if yes else "The clip's audio is dry, with no reverberation."),
    # quality claims for the auxiliary task ("yes" = the degradation is present)
    "aux_noise": lambda yes, row: ("The audio in this clip is degraded by loud continuous "
                                   "background noise." if yes else
                                   "The audio in this clip is clean, with no added noise."),
    "aux_lowpass": lambda yes, row: ("The audio in this clip is severely band-limited (high "
                                     "frequencies removed)." if yes else
                                     "The audio in this clip has its full frequency range."),
    "aux_reverb": lambda yes, row: ("The audio in this clip is heavily reverberant." if yes
                                    else "The audio in this clip is dry, with no added "
                                         "reverberation."),
}
# the two scales; keys are the condition names on the command line
LEVELS = {
    "v_vlow": "very low", "v_low": "low", "v_mod": "moderate",
    "v_high": "high", "v_vhigh": "very high",
    "n_015": "0.15", "n_035": "0.35", "n_060": "0.60",
    "n_085": "0.85", "n_095": "0.95",
}
CONDS = list(LEVELS) + ["none"]


def block(confidence: str, claim: str) -> str:
    # source is held at "anonymous system" throughout: Stage 2 showed source is
    # equivalent to null in 3 of 4 cells, so it is a nuisance factor here.
    return (f"Source: anonymous system\nClaim: \"{claim}\"\n"
            f"Confidence: {confidence}\n\n")


def evidence(cond: str, attr: str, row, claim_true: bool = False) -> str:
    """The claim asserts the FALSE value by default; --claim true asserts the true one.

    Both branches are needed to turn the diagnosis into a design procedure. The false
    branch measures the harm a wrong upstream detection does at each confidence level;
    the true branch measures the benefit a correct one buys at the same level. A detector
    of known precision pi has expected accuracy pi*Acc_true(c) + (1-pi)*Acc_false(c), so
    only with both curves can one choose the encoding c that maximises it.
    """
    key = "cond_label" if attr.startswith("aux_") else "label"
    truth_yes = row[key] == "A"
    assert_yes = truth_yes if claim_true else (not truth_yes)
    return block(LEVELS[cond], PROP[attr](assert_yes, row))


def question_for(attr, r):
    q = QUESTIONS[attr]
    if q is None:
        q = (f"Does this clip contain the sound of {r['query_class']}? "
             "(A) Yes (B) No")
    return q


# ---------------------------------------------------------------- model backends
def backend_qwen25():
    sys.path.insert(0, "/workspace/svhalluc/src/qwen")
    import torch
    from qwen_omni_utils import process_mm_info
    from transformers import (Qwen2_5OmniProcessor,
                              Qwen2_5OmniThinkerForConditionalGeneration)
    mp = ROOT / "models/Qwen2.5-Omni-7B"
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        mp, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="flash_attention_2", local_files_only=True).eval()
    proc = Qwen2_5OmniProcessor.from_pretrained(mp, local_files_only=True,
                                                use_fast=False)
    LID = {"A": 32, "B": 33}

    def score(clip, q):
        messages = [{"role": "user", "content": [
            {"type": "video", "video": str(ROOT / clip), "fps": 2.0,
             "max_frames": 64},
            {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(messages, add_generation_prompt=True,
                                        tokenize=False)
        a, i, v, vk = process_mm_info(messages, use_audio_in_video=True,
                                      return_video_kwargs=True)
        if isinstance(vk.get("fps"), list) and len(vk["fps"]) == 1:
            vk["fps"] = vk["fps"][0]
        inp = proc(text=text, audio=a, images=i, videos=v, return_tensors="pt",
                   padding=True, use_audio_in_video=True, **vk)
        inp = inp.to(model.device).to(torch.bfloat16)
        with torch.inference_mode():
            o = model(**inp, use_audio_in_video=True, return_dict=True)
        lg = o.logits[0, -1, :].float()
        return lg[LID["A"]].item(), lg[LID["B"]].item()
    return score


def backend_qwen3():
    sys.path.insert(0, "/workspace/svhalluc/src/qwen")
    import torch
    from qwen_omni_utils import process_mm_info
    from transformers import (Qwen3OmniMoeProcessor,
                              Qwen3OmniMoeForConditionalGeneration)
    mp = ROOT / "models/Qwen3-Omni-30B-A3B-Instruct"
    proc = Qwen3OmniMoeProcessor.from_pretrained(str(mp))
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        str(mp), dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa").eval()
    LID = {"A": 32, "B": 33}

    def score(clip, q):
        messages = [{"role": "user", "content": [
            {"type": "video", "video": str(ROOT / clip), "fps": 2.0,
             "max_frames": 64},
            {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(messages, add_generation_prompt=True,
                                        tokenize=False)
        a, i, v, vk = process_mm_info(messages, use_audio_in_video=True,
                                      return_video_kwargs=True)
        if isinstance(vk.get("fps"), list) and len(vk["fps"]) == 1:
            vk["fps"] = vk["fps"][0]
        inp = proc(text=text, audio=a, images=i, videos=v, return_tensors="pt",
                   padding=True, use_audio_in_video=True)
        inp = inp.to(model.device)
        for k, val in inp.items():
            if torch.is_tensor(val) and val.dtype == torch.float32:
                inp[k] = val.to(torch.bfloat16)
        with torch.inference_mode():
            o = model.thinker(**inp, use_audio_in_video=True, return_dict=True)
        lg = o.logits[0, -1, :].float()
        return lg[LID["A"]].item(), lg[LID["B"]].item()
    return score


def backend_vllama():
    VL = Path("/workspace/svhalluc/src/qwen/third_party/video_llama2_av")
    sys.path[:0] = [str(VL / ".deps"), str(VL)]
    import torch
    from videollama2 import model_init
    from videollama2.constants import DEFAULT_VIDEO_TOKEN
    from videollama2.mm_utils import tokenizer_multimodal_token
    CACHE = ROOT / "claude_track2/data/vl_pre_cache_v1"
    model, processor, tokenizer = model_init(
        "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV", device_map={"": 0},
        use_flash_attn=False)
    model.eval()
    LID = {L: tokenizer.encode(L, add_special_tokens=False)[0] for L in ("A", "B")}

    def pre(clip):
        f = CACHE / (Path(clip).stem + ".pt")
        if f.exists():
            try:
                return torch.load(f, map_location="cpu", weights_only=False)
            except Exception:  # noqa: BLE001
                f.unlink(missing_ok=True)
        t = processor["video"](str(ROOT / clip), va=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(f".pt.{os.getpid()}.tmp")
        torch.save(t, tmp)
        os.replace(tmp, f)
        return t

    def score(clip, q):
        t = pre(clip)
        vis = [(t.half().cuda() if not isinstance(t, dict)
                else {k: v.half().cuda() for k, v in t.items()}, "video")]
        message = [{"role": "user", "content": DEFAULT_VIDEO_TOKEN + "\n" + q}]
        prompt = tokenizer.apply_chat_template(message, tokenize=False,
                                               add_generation_prompt=True)
        ids = tokenizer_multimodal_token(prompt, tokenizer, DEFAULT_VIDEO_TOKEN,
                                         return_tensors="pt").unsqueeze(0).long().cuda()
        am = ids.ne(tokenizer.pad_token_id).long().cuda()
        with torch.inference_mode():
            out = model.generate(ids, attention_mask=am, images=vis,
                                 do_sample=False, max_new_tokens=1, use_cache=True,
                                 output_scores=True, return_dict_in_generate=True,
                                 pad_token_id=tokenizer.eos_token_id)
        lg = out.scores[0][0].float()
        return lg[LID["A"]].item(), lg[LID["B"]].item()
    return score


BACKENDS = {"qwen25": backend_qwen25, "qwen3": backend_qwen3,
            "vllama": backend_vllama}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(BACKENDS), required=True)
    ap.add_argument("--attr", choices=list(MANIFESTS), required=True)
    ap.add_argument("--cond", choices=CONDS, required=True)
    ap.add_argument("--claim", choices=["false", "true"], default="false",
                    help="whether the injected claim agrees with ground truth")
    # the read-out inversion and the filename tag below both already consulted this flag; the
    # declaration was missing, so every swapped ladder job died at argument parsing and the
    # runner's `|| true` turned the failure into a silent zero-row output
    ap.add_argument("--swap_options", action="store_true",
                    help="present the two answer choices in the opposite order")
    args = ap.parse_args()

    import math
    rows = [json.loads(l) for l in MANIFESTS[args.attr].open() if l.strip()]
    tag = ("" if args.claim == "false" else "_T") \
          + ("_swap" if args.swap_options else "")
    out = ROOT / (f"claude_track2/outputs/cs_{args.model}_{args.attr}_"
                  f"{args.cond}{tag}_v1.jsonl")
    print(f"[data] {len(rows)} clips  model={args.model} attr={args.attr} "
          f"cond={args.cond} ({LEVELS.get(args.cond, 'none')})", flush=True)

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
                ev = ("" if args.cond == "none" else
                      evidence(args.cond, args.attr, r, args.claim == "true"))
                qq = question_for(args.attr, r)
                if args.swap_options:
                    qq = qq.replace("(A) Yes (B) No", "(A) No (B) Yes")
                q = ev + qq
                la, lb = score(r["clip"], q)
                pa = 1.0 / (1.0 + math.exp(-(la - lb)))
                pred = "A" if pa >= 0.5 else "B"
                if args.swap_options:      # "A" now means "No", i.e. label B
                    pred = "B" if pred == "A" else "A"
                n += 1
                ok += int(pred == r["label"])
                f.write(json.dumps({"clip": r["clip"], "label": r["label"],
                                    "pred": pred, "p_A": round(pa, 6),
                                    "correct": pred == r["label"],
                                    "attr": args.attr, "cond": args.cond,
                                    "level": LEVELS.get(args.cond, "none"),
                                    "claim": args.claim,
                                    "model": args.model}) + "\n")
                f.flush()
                if n % 50 == 0:
                    print(f"[{i+1}/{len(rows)}] acc {ok/n:.3f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"SKIP {r['clip']}: {type(e).__name__}: {e}", flush=True)
    print(f"DONE model={args.model} attr={args.attr} cond={args.cond} n={n} "
          f"acc={ok/max(1,n):.4f}", flush=True)


if __name__ == "__main__":
    main()
