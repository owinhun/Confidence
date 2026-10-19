#!/usr/bin/env python3
"""Audio Flamingo 3 (NVIDIA), run through the same protocol as every other model.

AF3's transformers class only exists in transformers 5.x, which requires torch >= 2.5 while
this machine's shared environment is on 2.4.1. Upgrading globally would break the five models
already producing results (Qwen3-Omni in particular), so AF3 runs against a pinned interpreter
path assembled under /data4/pyenv_af3. Launch with scripts/run_af3_v1.sh, which sets the
PYTHONPATH and LD_LIBRARY_PATH; running this file bare will pick up the shared environment and
fail at import.

Conditions, manifests and scoring are identical to the other models: the claim proposition is
byte-identical across cells, only Source/Confidence move, and the answer is read from the
option-letter logits of a single forward pass.
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

ROOT = Path("/workspace/svhalluc")
MODEL = "/data4/svhalluc_models/audio-flamingo-3-hf"

sys.path.insert(0, str(ROOT / "claude_track2/scripts"))
from eval_newmodels_v1 import (CONDS, MANIFESTS, evidence,  # noqa: E402
                               question_for, read_audio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", choices=list(MANIFESTS), required=True)
    ap.add_argument("--cond", choices=CONDS, required=True)
    ap.add_argument("--claim", choices=["false", "true"], default="false")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--swap_options", action="store_true")
    ap.add_argument("--readback", action="store_true")
    ap.add_argument("--post", action="store_true")
    ap.add_argument("--paraphrase", action="store_true")
    ap.add_argument("--mask_sensory", action="store_true")
    args = ap.parse_args()
    flagtag = (("_swap" if args.swap_options else "") + ("_post" if args.post else "")
               + ("_para" if args.paraphrase else "")
               + ("_masked" if args.mask_sensory else "")
               + ("_readback" if args.readback else ""))
    if args.attr == "colour":
        print("SKIP: audio-only model, visual attribute not applicable", flush=True)
        return

    import torch
    from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

    rows = [json.loads(l) for l in MANIFESTS[args.attr].open() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    tag = "" if args.claim == "false" else "_T"
    out = ROOT / (f"claude_track2/outputs/nm_af3_{args.attr}_{args.cond}{tag}"
                  f"{flagtag}_v1.jsonl")
    print(f"[data] {len(rows)} clips  af3/{args.attr}/{args.cond}{tag}", flush=True)

    proc = AutoProcessor.from_pretrained(MODEL)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0").eval()
    tk = proc.tokenizer
    ids = {L: tk.encode(L, add_special_tokens=False)[0] for L in ("A", "B")}

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
                if args.swap_options:
                    qq = qq.replace("(A) Yes (B) No", "(A) No (B) Yes")
                ev = evidence(args.cond, args.attr, r, args.claim == "true")
                q = (qq + "\n\n" + ev) if args.post else (ev + qq)
                content = [{"type": "text", "text": q}]
                if not args.mask_sensory:
                    content.append({"type": "audio", "audio": read_audio(ROOT / r["clip"])})
                conv = [{"role": "user", "content": content}]
                inp = proc.apply_chat_template(conv, tokenize=True, return_dict=True,
                                              add_generation_prompt=True,
                                              return_tensors="pt")
                # the feature extractor emits float32 features; cast them to the model
                # dtype, as with Qwen3-Omni
                inp = {k: (v.to(model.device) if hasattr(v, "to") else v)
                       for k, v in inp.items()}
                inp = {k: (v.to(torch.bfloat16)
                           if torch.is_tensor(v) and v.dtype == torch.float32 else v)
                       for k, v in inp.items()}
                with torch.inference_mode():
                    lg = model(**inp).logits[0, -1, :].float()
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
                                    "claim": args.claim, "model": "af3"}) + "\n")
                f.flush()
                if n % 50 == 0:
                    print(f"[{i+1}/{len(rows)}] acc {ok/n:.3f} ({time.time()-t0:.0f}s)",
                          flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"SKIP {r['clip']}: {type(e).__name__}: {e}", flush=True)
    print(f"DONE af3 {args.attr} {args.cond}{tag} n={n} acc={ok/max(1,n):.4f}", flush=True)


if __name__ == "__main__":
    main()
