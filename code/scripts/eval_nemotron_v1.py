#!/usr/bin/env python3
"""The same factorial on a model that shares nothing with the other five.

Reading the configs on disk showed that every model evaluated so far carries a Qwen text
backbone -- Qwen2.5-Omni and Qwen3-Omni obviously, Qwen2-Audio (`qwen2`), VideoLLaMA2.1-AV
(`videollama2_qwen2`), and Audio Flamingo 3, whose `text_config.model_type` is also `qwen2`
despite being an NVIDIA release. MiniCPM-o 2.6 is the same story: 3584 hidden, 28 layers,
vocab 151700 is Qwen2.5-7B's geometry. So "the effect appears in every model we tested" was at
risk of meaning "in every Qwen".

Nemotron 3 Nano Omni is the exception available to us, and it is different in all three places
at once:

    LLM              Nemotron-H, a Mamba2-Transformer hybrid MoE (31B total, ~3B active)
    speech encoder   Parakeet
    vision encoder   C-RADIO v4-H

No shared backbone, no shared audio front end, no shared vision tower, no shared tokenizer
(vocab 131072 against Qwen's 151k). If the confidence effect reproduces here, the finding is
about instruction-tuned multimodal models rather than about one lineage.

Two honest caveats to carry into the paper. The model card states it was improved using data
from Qwen3-VL, Qwen3.5 and Qwen2.5-VL among others, so independence is architectural, not
a guarantee of independent training signal. And it is a reasoning model whose thinking mode is
on by default; we disable it, because a single scored forward pass over the two option letters
is only meaningful if the next token is the answer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data4/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/data4/hf_cache/hub")
os.environ.setdefault("HF_MODULES_CACHE", "/data4/hf_cache/modules")

ROOT = Path("/workspace/svhalluc")
MODEL = "/data4/svhalluc_models/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
NUM_FRAMES = 8          # matches the frame budget the other video models are given
AUDIO_SR = 16000

sys.path.insert(0, str(ROOT / "claude_track2/scripts"))
from eval_newmodels_v1 import (CONDS, MANIFESTS, apply_swap, evidence,  # noqa: E402
                               question_for, read_audio)


def read_frames(path, n=NUM_FRAMES):
    """Uniformly sample n frames as PIL Images.

    The C-RADIO image processor indexes `.width` on whatever it is given, so a raw
    (n, H, W, 3) array raises AttributeError on the first clip -- returning PIL is what the
    processor's video path expects.
    """
    import numpy as np
    from decord import VideoReader
    from PIL import Image
    vr = VideoReader(str(path))
    total = len(vr)
    idx = np.linspace(0, max(0, total - 1), num=min(n, max(1, total))).astype(int)
    return [Image.fromarray(f) for f in vr.get_batch(idx).asnumpy()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", choices=list(MANIFESTS), required=True)
    ap.add_argument("--cond", choices=CONDS, required=True)
    ap.add_argument("--claim", choices=["false", "true"], default="false")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--swap_options", action="store_true")
    ap.add_argument("--post", action="store_true")
    ap.add_argument("--paraphrase", action="store_true")
    ap.add_argument("--mask_sensory", action="store_true")
    args = ap.parse_args()
    flagtag = (("_swap" if args.swap_options else "") + ("_post" if args.post else "")
               + ("_para" if args.paraphrase else "")
               + ("_masked" if args.mask_sensory else ""))

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    # The checkpoint's remote code was written against an older masking_utils: it passes
    # input_embeds= (no "s") and cache_position=, neither of which the installed
    # create_causal_mask accepts. Patch the symbol before from_pretrained imports the remote
    # module, so the shim is what the model binds to.
    import transformers.masking_utils as _mu
    _orig_ccm = _mu.create_causal_mask

    def _ccm_shim(*a, **kw):
        if "input_embeds" in kw:
            kw["inputs_embeds"] = kw.pop("input_embeds")
        kw.pop("cache_position", None)
        return _orig_ccm(*a, **kw)

    _mu.create_causal_mask = _ccm_shim

    rows = [json.loads(l) for l in MANIFESTS[args.attr].open() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    tag = "" if args.claim == "false" else "_T"
    out = ROOT / (f"claude_track2/outputs/nm_nemotron_{args.attr}_{args.cond}{tag}"
                  f"{flagtag}_v1.jsonl")
    print(f"[data] {len(rows)} clips  nemotron/{args.attr}/{args.cond}{tag}", flush=True)

    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        # 62 GB in BF16: fits one A100-80GB when a card is free, otherwise shards over the
        # pair given in CUDA_VISIBLE_DEVICES, as Qwen3-Omni does
        MODEL, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True).eval()
    # the custom generate() forwards use_cache itself, so it cannot be passed as a kwarg;
    # setting it on the generation config is the only way in, and it has to be off because the
    # checkpoint's cache class predates the API the installed transformers drives
    model.generation_config.use_cache = False

    # The custom generate() builds a NemotronHHybridDynamicCache regardless, and that class was
    # written before masking_utils started asking caches for their query offset and mask sizes.
    # One token is generated from an empty cache, so the offset is 0 and the mask spans the
    # whole prefill; supplying those two answers is enough to let the mask code run.
    import sys as _sys
    for _m in list(_sys.modules.values()):
        _C = getattr(_m, "NemotronHHybridDynamicCache", None)
        if _C is None:
            continue
        if not hasattr(_C, "get_query_offset"):
            _C.get_query_offset = lambda self, *a, **k: 0
        if not hasattr(_C, "get_mask_sizes"):
            def _mask_sizes(self, cache_position, layer_idx=None):
                n = int(cache_position.shape[-1])
                return n, n
            _C.get_mask_sizes = _mask_sizes
        print(f"[cache shim] patched {_C.__name__}", flush=True)
        break

    tk = proc.tokenizer
    ids = {L: tk.encode(L, add_special_tokens=False)[0] for L in ("A", "B")}
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
        for i, r in enumerate(rows):
            if r["clip"] in done:
                continue
            try:
                qq = question_for(args.attr, r, args.paraphrase)
                if args.swap_options:
                    qq = apply_swap(qq)
                ev = evidence(args.cond, args.attr, r, args.claim == "true")
                q = (qq + "\n\n" + ev) if args.post else (ev + qq)

                clip = ROOT / r["clip"]
                videos = audio = None
                prefix = ""
                if not args.mask_sensory:
                    videos = [read_frames(clip)]
                    audio = [read_audio(clip, sr=AUDIO_SR)]
                    # the processor expands "<video>" into per-frame <img> chunks and
                    # "<so_embedding>" into the audio placeholder run; "<audio>" is only
                    # the name used in its docstring and matches nothing
                    prefix = "<video>\n<so_embedding>\n"
                conv = [{"role": "user", "content": prefix + q}]
                text = tk.apply_chat_template(conv, tokenize=False,
                                              add_generation_prompt=True,
                                              enable_thinking=False)
                inp = proc(text=[text], videos=videos, audio=audio, return_tensors="pt")
                dev = next(model.parameters()).device
                inp = {k: (v.to(dev) if hasattr(v, "to") else v)
                       for k, v in inp.items()}
                inp = {k: (v.to(torch.bfloat16)
                           if torch.is_tensor(v) and v.dtype == torch.float32 else v)
                       for k, v in inp.items()}
                # forward() takes pixel_values as a required positional and knows nothing
                # about video or sound; the model's own generate() is the entry point that
                # splices the three encoders in. One step with scores gives the same
                # next-token logits a forward pass would, which is all the read-out needs.
                gen = {k: v for k, v in inp.items()
                       if k in ("input_ids", "attention_mask", "pixel_values",
                                "pixel_values_videos", "sound_clips", "sound_length")}
                with torch.inference_mode():
                    # the checkpoint's NemotronHHybridDynamicCache predates the cache API the
                    # installed transformers drives (no get_query_offset). Only one token is
                    # ever generated here, so the cache buys nothing and can be switched off.
                    out = model.generate(**gen, max_new_tokens=1, do_sample=False,
                                         output_scores=True, return_dict_in_generate=True)
                lg = out.scores[0][0].float()
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
                print(f"[{n}] acc={ok / n:.3f}  {(time.time() - t0) / n:.2f}s/clip",
                      flush=True)
    print(f"[done] n={n} acc={ok / max(1, n):.4f}", flush=True)


if __name__ == "__main__":
    main()
