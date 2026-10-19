#!/usr/bin/env python3
"""One harness, several model families, so adding a backbone is an adapter and not a rewrite.

Every model already in the paper carries a Qwen text backbone, which makes "in every model" a
statement about one lineage. The fix is more lineages, and the cost of more lineages is almost
entirely plumbing: each release wants its own prompt string, its own way of being handed frames
or a waveform, and its own class. The scoring is identical everywhere -- one forward pass, read
the two option-letter logits at the last position -- so that part is written once here and each
family supplies only:

    load()   -> anything the adapter needs
    score(state, clip, question, mask_sensory) -> (logit_A, logit_B)

Adapters are registered in MODELS with the backbone they carry, because the backbone is the
whole point of adding them:

    vllama2_mistral   Mistral-7B    same DAMO codebase as VideoLLaMA2.1-AV with the LLM swapped,
                                    so it is a controlled backbone substitution on the visual cell
    video_llava       Vicuna/LLaMA  the standard video-LLM baseline in the sycophancy literature
    llava_next_video  Vicuna
    internvl2         InternLM2.5
    ultravox          Llama-3.1     audio only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# weights fetched during this round live on /data2; the older ones stay on /data4
os.environ.setdefault("HF_HOME", "/data2/svhalluc_models/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/data2/svhalluc_models/hf_cache/hub")
os.environ.setdefault("HF_MODULES_CACHE", "/data2/svhalluc_models/hf_cache/modules")

ROOT = Path("/workspace/svhalluc")
NUM_FRAMES = 8
AUDIO_SR = 16000

sys.path.insert(0, str(ROOT / "claude_track2/scripts"))
from eval_newmodels_v1 import (CONDS, MANIFESTS, apply_swap, evidence,  # noqa: E402
                               question_for, read_audio)

# name -> (repo id, backbone, modality). "video" means no audio branch, so audio attributes
# are skipped rather than answered blind.
MODELS = {
    "vllama2_mistral":  ("DAMO-NLP-SG/VideoLLaMA2-7B", "Mistral-7B", "video"),
    "video_llava":      ("LanguageBind/Video-LLaVA-7B-hf", "Vicuna/LLaMA", "video"),
    # lmms-lab/ ships the training-repo layout with no image-processor config;
    # llava-hf/ is the converted checkpoint the transformers class can load
    "llava_next_video": ("llava-hf/LLaVA-NeXT-Video-7B-hf", "Vicuna", "video"),
    "internvl2":        ("OpenGVLab/InternVL2-8B", "InternLM2.5", "video"),
    "ultravox":         ("fixie-ai/ultravox-v0_5-llama-3_1-8b", "Llama-3.1", "audio"),
    # SEA-LION is a Gemma-2 derivative, so this is a Gemma-lineage audio model that needs no
    # gated download -- the independence gemma-3n would have given us, without the licence gate.
    "meralion":         ("MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION", "Gemma-2/SEA-LION",
                         "audio"),
}
VISUAL_ATTRS = {"colour"}


def read_frames(path, n=NUM_FRAMES):
    import numpy as np
    from decord import VideoReader
    vr = VideoReader(str(path))
    idx = np.linspace(0, max(0, len(vr) - 1), num=min(n, max(1, len(vr)))).astype(int)
    return vr.get_batch(idx).asnumpy()


# --------------------------------------------------------------------------- adapters
def load_video_llava():
    import torch
    from transformers import AutoProcessor, VideoLlavaForConditionalGeneration
    rid = MODELS["video_llava"][0]
    proc = AutoProcessor.from_pretrained(rid)
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        rid, dtype=torch.float16, device_map="cuda:0").eval()
    return {"proc": proc, "model": model, "tk": proc.tokenizer}


def score_video_llava(st, clip, question, mask):
    import torch
    prompt = (f"USER: {question} ASSISTANT:" if mask
              else f"USER: <video>\n{question} ASSISTANT:")
    kw = {} if mask else {"videos": [read_frames(clip)]}
    inp = st["proc"](text=prompt, return_tensors="pt", **kw).to(st["model"].device)
    with torch.inference_mode():
        lg = st["model"](**inp).logits[0, -1, :].float()
    return lg, st["tk"]


def load_llava_next_video():
    import torch
    from transformers import AutoProcessor, LlavaNextVideoForConditionalGeneration
    rid = MODELS["llava_next_video"][0]
    proc = AutoProcessor.from_pretrained(rid)
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        rid, dtype=torch.float16, device_map="cuda:0").eval()
    return {"proc": proc, "model": model, "tk": proc.tokenizer}


def score_llava_next_video(st, clip, question, mask):
    import torch
    content = [{"type": "text", "text": question}]
    if not mask:
        content.insert(0, {"type": "video"})
    prompt = st["proc"].apply_chat_template([{"role": "user", "content": content}],
                                            add_generation_prompt=True)
    kw = {} if mask else {"videos": [read_frames(clip)]}
    inp = st["proc"](text=prompt, return_tensors="pt", **kw).to(st["model"].device)
    with torch.inference_mode():
        lg = st["model"](**inp).logits[0, -1, :].float()
    return lg, st["tk"]


def load_vllama2_mistral():
    """Same inference stack as the VideoLLaMA2.1-AV run, different checkpoint."""
    VL = ROOT / "src/qwen/third_party/video_llama2_av"
    sys.path[:0] = [str(VL / ".deps"), str(VL)]
    from videollama2 import model_init
    model, processor, tokenizer = model_init(MODELS["vllama2_mistral"][0],
                                             device_map={"": "cuda:0"})
    model.eval()
    return {"model": model, "processor": processor, "tk": tokenizer}


def score_vllama2_mistral(st, clip, question, mask):
    import torch
    from videollama2.constants import DEFAULT_VIDEO_TOKEN
    from videollama2.mm_utils import tokenizer_multimodal_token
    tk = st["tk"]
    if mask:
        msg, vis = [{"role": "user", "content": question}], None
    else:
        t = st["processor"]["video"](str(clip))
        vis = [(t.half().cuda(), "video")]
        msg = [{"role": "user", "content": DEFAULT_VIDEO_TOKEN + "\n" + question}]
    prompt = tk.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    ids = tokenizer_multimodal_token(prompt, tk, DEFAULT_VIDEO_TOKEN,
                                     return_tensors="pt").unsqueeze(0).long().cuda()
    am = ids.ne(tk.pad_token_id).long().cuda()
    with torch.inference_mode():
        out = st["model"].generate(ids, attention_mask=am, images=vis, do_sample=False,
                                   max_new_tokens=1, use_cache=True, output_scores=True,
                                   return_dict_in_generate=True, pad_token_id=tk.eos_token_id)
    return out.scores[0][0].float(), tk


def load_ultravox():
    import torch
    from transformers import AutoModel, AutoProcessor
    rid = MODELS["ultravox"][0]
    proc = AutoProcessor.from_pretrained(rid, trust_remote_code=True)
    model = AutoModel.from_pretrained(rid, dtype=torch.bfloat16, device_map="cuda:0",
                                      trust_remote_code=True).eval()
    return {"proc": proc, "model": model, "tk": proc.tokenizer}


def score_ultravox(st, clip, question, mask):
    import torch
    turns = [{"role": "user", "content": ("" if mask else "<|audio|>\n") + question}]
    text = st["tk"].apply_chat_template(turns, tokenize=False, add_generation_prompt=True)
    kw = {} if mask else {"audio": read_audio(clip, sr=AUDIO_SR), "sampling_rate": AUDIO_SR}
    inp = st["proc"](text=text, return_tensors="pt", **kw)
    inp = {k: (v.to(st["model"].device) if hasattr(v, "to") else v) for k, v in inp.items()}
    with torch.inference_mode():
        lg = st["model"](**inp).logits[0, -1, :].float()
    return lg, st["tk"]


def load_meralion():
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    rid = MODELS["meralion"][0]
    proc = AutoProcessor.from_pretrained(rid, trust_remote_code=True)
    # the remote code predates transformers' attention-implementation dispatch and never sets
    # _supports_sdpa, so the automatic sdpa probe raises on load; pinning eager skips the probe
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        rid, use_safetensors=True, trust_remote_code=True, attn_implementation="eager",
        dtype=torch.bfloat16, device_map="cuda:0").eval()
    return {"proc": proc, "model": model, "tk": proc.tokenizer}


def score_meralion(st, clip, question, mask):
    """The card's own template: the audio is spliced in at <SpeechHere>."""
    import torch
    body = (question if mask
            else f"Given the following audio context: <SpeechHere>\n\nText instruction: {question}")
    chat = st["tk"].apply_chat_template(conversation=[[{"role": "user", "content": body}]],
                                        tokenize=False, add_generation_prompt=True)
    kw = {} if mask else {"audios": [read_audio(clip, sr=AUDIO_SR)]}
    inp = st["proc"](text=chat, **kw)
    inp = {k: (v.to(st["model"].device) if hasattr(v, "to") else v) for k, v in inp.items()}
    inp = {k: (v.to(torch.bfloat16) if torch.is_tensor(v) and v.dtype == torch.float32 else v)
           for k, v in inp.items()}
    with torch.inference_mode():
        lg = st["model"](**inp).logits[0, -1, :].float()
    return lg, st["tk"]


ADAPTERS = {
    "meralion": (load_meralion, score_meralion),
    "video_llava": (load_video_llava, score_video_llava),
    "llava_next_video": (load_llava_next_video, score_llava_next_video),
    "vllama2_mistral": (load_vllama2_mistral, score_vllama2_mistral),
    "ultravox": (load_ultravox, score_ultravox),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(ADAPTERS), required=True)
    ap.add_argument("--attr", choices=list(MANIFESTS), required=True)
    ap.add_argument("--cond", choices=CONDS, required=True)
    ap.add_argument("--claim", choices=["false", "true"], default="false")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--swap_options", action="store_true")
    ap.add_argument("--post", action="store_true")
    ap.add_argument("--paraphrase", action="store_true")
    ap.add_argument("--mask_sensory", action="store_true")
    args = ap.parse_args()

    modality = MODELS[args.model][2]
    if modality == "video" and args.attr not in VISUAL_ATTRS:
        print(f"SKIP: {args.model} has no audio branch; {args.attr} is not visual", flush=True)
        return
    if modality == "audio" and args.attr in VISUAL_ATTRS:
        print(f"SKIP: {args.model} is audio-only; {args.attr} is visual", flush=True)
        return

    flagtag = (("_swap" if args.swap_options else "") + ("_post" if args.post else "")
               + ("_para" if args.paraphrase else "") + ("_masked" if args.mask_sensory else ""))
    rows = [json.loads(l) for l in MANIFESTS[args.attr].open() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    tag = "" if args.claim == "false" else "_T"
    out = ROOT / (f"claude_track2/outputs/gm_{args.model}_{args.attr}_"
                  f"{args.cond}{tag}{flagtag}_v1.jsonl")
    print(f"[data] {len(rows)} clips  {args.model}"
          f" ({MODELS[args.model][1]}) /{args.attr}/{args.cond}{tag}", flush=True)

    load, score = ADAPTERS[args.model]
    st = load()
    ids = {L: st["tk"].encode(L, add_special_tokens=False)[-1] for L in ("A", "B")}
    print(f"[option ids] A={ids['A']} B={ids['B']}", flush=True)

    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["clip"])
        print(f"[resume] {len(done)}", flush=True)

    n = ok = 0
    t0 = time.time()
    with out.open("a", encoding="utf-8") as f:
        for r in rows:
            if r["clip"] in done:
                continue
            try:
                qq = question_for(args.attr, r, args.paraphrase)
                if args.swap_options:
                    qq = apply_swap(qq)
                ev = evidence(args.cond, args.attr, r, args.claim == "true")
                q = (qq + "\n\n" + ev) if args.post else (ev + qq)
                lg, _ = score(st, ROOT / r["clip"], q, args.mask_sensory)
                la, lb = lg[ids["A"]].item(), lg[ids["B"]].item()
                pa = 1.0 / (1.0 + math.exp(-(la - lb)))
                pred = "A" if pa >= 0.5 else "B"
                if args.swap_options:
                    pred = "B" if pred == "A" else "A"
                n += 1
                ok += int(pred == r["label"])
                f.write(json.dumps({"clip": r["clip"], "label": r["label"], "pred": pred,
                                    "p_A": round(pa, 6), "correct": pred == r["label"],
                                    "attr": args.attr, "cond": args.cond,
                                    "claim": args.claim}) + "\n")
                f.flush()
            except Exception as e:  # noqa: BLE001
                print(f"[skip] {r['clip']}: {type(e).__name__}: {e}", flush=True)
            if n and n % 25 == 0:
                print(f"[{n}] acc={ok / n:.3f} {(time.time() - t0) / n:.2f}s/clip", flush=True)
    print(f"[done] n={n} acc={ok / max(1, n):.4f}", flush=True)


if __name__ == "__main__":
    main()
