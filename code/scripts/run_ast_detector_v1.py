#!/usr/bin/env python3
"""Stage 1 of the real cascade: run an actual sound-event tagger over the held-out clips.

Every claim in the paper so far was written by us, with its truth set by the manifest and its
confidence chosen from a ladder. That is what makes the forwarding analysis a counterfactual:
it mixes a true branch and a false branch at a precision supplied from outside, and it has to
assume the detector's errors are independent of how hard the item is for the downstream model.

A real tagger removes both assumptions at once. AST fine-tuned on AudioSet emits a posterior
over 527 classes; taking its top class as the claim and its calibrated probability as the
declared confidence produces exactly the message a deployed pipeline would serialise -- wrong
on the clips it finds hard, confident in the way it is actually confident, and with its errors
correlated with item difficulty in whatever way they truly are.

This script only produces the detector side: per clip, the top class, its raw probability, the
probability assigned to the manifest's ground-truth class, and whether the top class matches.
Calibration and prompt construction happen downstream, so the split used to fit a temperature
can be chosen after seeing how well-separated the scores are.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data2/svhalluc_models/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/data2/svhalluc_models/hf_cache/hub")

ROOT = Path("/workspace/svhalluc")
MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SR = 16000


def read_audio(path, sr=SR):
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=True, duration=10.0)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="claude_track2/data/held_attr_soundevent_v1.jsonl")
    ap.add_argument("--out", default="claude_track2/outputs/ast_detector_v1.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import ASTForAudioClassification, AutoFeatureExtractor

    rows = [json.loads(l) for l in (ROOT / args.manifest).open() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    out = ROOT / args.out
    print(f"[data] {len(rows)} clips -> {out}", flush=True)

    fe = AutoFeatureExtractor.from_pretrained(MODEL)
    model = ASTForAudioClassification.from_pretrained(MODEL).eval().cuda()
    id2label = model.config.id2label
    labels_lower = {i: v.lower() for i, v in id2label.items()}

    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["clip"])
        print(f"[resume] {len(done)}", flush=True)

    n = 0
    with out.open("a", encoding="utf-8") as f:
        for r in rows:
            if r["clip"] in done:
                continue
            try:
                y = read_audio(ROOT / r["clip"])
                inp = fe(y, sampling_rate=SR, return_tensors="pt")
                inp = {k: v.cuda() for k, v in inp.items()}
                with torch.inference_mode():
                    probs = torch.sigmoid(model(**inp).logits)[0].float().cpu()
                top = int(probs.argmax())
                # the manifest's class名 is a VGGSound string; match it loosely against the
                # AudioSet label set so the detector's score for the *asked-about* class is
                # recoverable even when the two taxonomies word things differently
                # VGGSound and AudioSet name the same event differently ("dog barking" vs
                # "Bark", "people eating crisps" vs "Chewing, mastication"), so a one-way
                # substring test finds almost nothing. Match on shared word stems in either
                # direction, after dropping the generic words VGGSound uses as connectives.
                STOP = {"people", "playing", "sound", "with", "from", "male", "female"}
                q = r["query_class"].lower()
                toks = [t for t in q.replace(",", " ").replace("-", " ").split()
                        if len(t) > 2 and t not in STOP]
                stems = {t[:max(4, len(t) - 3)] for t in toks}
                cand = [i for i, v in labels_lower.items()
                        if any(st in v for st in stems)
                        or any(w[:max(4, len(w) - 3)] in q
                               for w in v.replace(",", " ").split() if len(w) > 2)]
                gt_i = max(cand, key=lambda i: float(probs[i])) if cand else None
                f.write(json.dumps({
                    "clip": r["clip"], "label": r["label"], "query_class": r["query_class"],
                    "top_class": id2label[top], "top_prob": round(float(probs[top]), 6),
                    "gt_class": id2label[gt_i] if gt_i is not None else None,
                    "gt_prob": round(float(probs[gt_i]), 6) if gt_i is not None else None,
                    "n_matched": len(cand)}) + "\n")
                f.flush()
                n += 1
            except Exception as e:  # noqa: BLE001
                print(f"[skip] {r['clip']}: {type(e).__name__}: {e}", flush=True)
            if n and n % 100 == 0:
                print(f"[{n}]", flush=True)
    print(f"[done] n={n}", flush=True)


if __name__ == "__main__":
    main()
