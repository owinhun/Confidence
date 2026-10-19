#!/usr/bin/env python3
"""Does Phi-4's position answering survive a free-generation read-out?

Five non-Qwen releases were screened out because they answer by option position: Phi-4 is right
99.5% of the time when the answer is (A) and 1.5% when it is (B). We read every model by
comparing the option-letter logits at the first generated position, so that pattern has two
readings. Either the model really is answering by position, or our read-out is the artefact --
and if it is the read-out, every number in the paper inherits the problem, not just the
screening.

This settles it on the model where the pattern is starkest. Same clips, same prompt, but the
answer is taken from what the model actually writes, mapped to yes/no by the words rather than
by the letters. If the direction gap closes, the read-out is at fault; if it persists, the
screening was measuring the model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data2/svhalluc_models/hf_cache")
ROOT = Path("/workspace/svhalluc")
sys.path.insert(0, str(ROOT / "claude_track2/scripts"))
from eval_newmodels_v1 import MANIFESTS, MODELS, evidence, question_for, read_audio  # noqa: E402

YES = ("yes", "(a)", "a.", "a)", "contains", "does contain")
NO = ("no", "(b)", "b.", "b)", "does not", "doesn't", "not contain")


def decode(text):
    """Map free text to the manifest's letters; None when it commits to neither."""
    t = text.strip().lower()
    hit_y = any(t.startswith(k) or f" {k}" in f" {t}" for k in YES)
    hit_n = any(t.startswith(k) or f" {k}" in f" {t}" for k in NO)
    if hit_n and not hit_y:
        return "B"
    if hit_y and not hit_n:
        return "A"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", default="sound", choices=list(MANIFESTS))
    ap.add_argument("--cond", default="none")
    ap.add_argument("--swap_options", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    mp = MODELS["phi4mm"]
    proc = AutoProcessor.from_pretrained(mp, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        mp, trust_remote_code=True, torch_dtype=torch.bfloat16,
        _attn_implementation="eager").eval().cuda()

    rows = [json.loads(l) for l in MANIFESTS[a.attr].open() if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    tag = "_swap" if a.swap_options else ""
    out = ROOT / f"claude_track2/outputs/phi4_freegen_{a.attr}_{a.cond}{tag}_v1.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["clip"] for l in out.open() if l.strip()}
        print(f"[resume] {len(done)}", flush=True)

    n = ok = undec = 0
    with out.open("a", encoding="utf-8") as f:
        for r in rows:
            if r["clip"] in done:
                continue
            q = question_for(a.attr, r)
            if a.swap_options:
                q = q.replace("(A) Yes (B) No", "(A) No (B) Yes")
            ev = "" if a.cond == "none" else evidence(a.cond, a.attr, r)
            prompt = f"<|user|><|audio_1|>{ev}{q}<|end|><|assistant|>"
            try:
                au = read_audio(ROOT / r["clip"])
                inp = proc(text=prompt, audios=[(au, 16000)], return_tensors="pt")
                inp = {k: (v.to(model.device) if hasattr(v, "to") else v)
                       for k, v in inp.items()}
                with torch.inference_mode():
                    gen = model.generate(**inp, max_new_tokens=12, do_sample=False,
                                         use_cache=False)
                txt = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:],
                                        skip_special_tokens=True)[0]
                pred = decode(txt)
                if a.swap_options and pred is not None:
                    pred = "B" if pred == "A" else "A"
                n += 1
                if pred is None:
                    undec += 1
                else:
                    ok += int(pred == r["label"])
                f.write(json.dumps({"clip": r["clip"], "label": r["label"], "pred": pred,
                                    "text": txt.strip()[:80],
                                    "correct": pred == r["label"]}) + "\n")
                f.flush()
            except Exception as e:  # noqa: BLE001
                print(f"[skip] {r['clip']}: {type(e).__name__}: {e}", flush=True)
            if n and n % 100 == 0:
                print(f"[{n}] acc={ok/max(1,n-undec):.3f} undecided={undec}", flush=True)
    print(f"[done] n={n} acc={ok/max(1,n-undec):.4f} undecided={undec}", flush=True)


if __name__ == "__main__":
    main()
