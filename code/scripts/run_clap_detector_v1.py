#!/usr/bin/env python3
"""Stage 1 of the real cascade, using CLAP rather than a fixed-taxonomy tagger.

The first attempt used AST fine-tuned on AudioSet. It runs, but the manifest names classes the
way VGGSound does and AST answers in AudioSet's vocabulary, so recovering the detector's score
for the asked-about class needs a mapping between the two taxonomies. Loose stem matching gave
nine matches in ten and most were wrong -- "people eating apple" matched "Applause", "playing
badminton" matched "Children playing". A detector score attached to the wrong class would
corrupt everything downstream, and a hand-built VGGSound-to-AudioSet table is a project of its
own.

CLAP removes the problem instead of patching it. It scores audio against arbitrary text, so the
query class can be used verbatim: the detector is asked exactly the question the downstream
model is asked. Its decision is the sign of the contrast between a positive and a negative
phrasing, and its confidence is the softmax over that pair -- a real posterior from a real
model, with errors that fall where the model actually finds the clip hard.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data2/svhalluc_models/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/data2/svhalluc_models/hf_cache/hub")

ROOT = Path("/workspace/svhalluc")
# a second checkpoint answers whether the non-monotone precision bands are a property of this
# detector or of the band-to-word mapping in general. A different CLAP is the apples-to-apples
# choice: same zero-shot text-scoring interface, so no taxonomy mapping enters the comparison,
# but different training data and capacity.
MODEL = "laion/clap-htsat-unfused"
SR = 48000


def read_audio(path, sr=SR):
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=True, duration=10.0)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="claude_track2/data/held_attr_soundevent_v1.jsonl")
    ap.add_argument("--out", default="claude_track2/outputs/clap_detector_v1.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    import torch
    from transformers import ClapModel, ClapProcessor

    rows = [json.loads(l) for l in (ROOT / args.manifest).open() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    out = ROOT / args.out
    print(f"[data] {len(rows)} clips -> {out}", flush=True)

    proc = ClapProcessor.from_pretrained(args.model)
    # the repo ships both a .bin and a safetensors revision; transformers refuses torch.load
    # under torch < 2.6, so the safetensors one has to be requested explicitly
    model = ClapModel.from_pretrained(args.model, use_safetensors=True).eval().cuda()

    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["clip"])
        print(f"[resume] {len(done)}", flush=True)

    n = ok = 0
    with out.open("a", encoding="utf-8") as f:
        for r in rows:
            if r["clip"] in done:
                continue
            try:
                y = read_audio(ROOT / r["clip"])
                cls = r["query_class"]
                texts = [f"the sound of {cls}", f"a recording with no {cls}"]
                inp = proc(text=texts, audios=y, sampling_rate=SR,
                           return_tensors="pt", padding=True)
                inp = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in inp.items()}
                with torch.inference_mode():
                    logits = model(**inp).logits_per_audio[0].float().cpu()
                p_present = float(torch.softmax(logits, dim=-1)[0])
                # the manifest's label A means the event is present
                pred = "A" if p_present >= 0.5 else "B"
                correct = pred == r["label"]
                n += 1
                ok += int(correct)
                f.write(json.dumps({
                    "clip": r["clip"], "label": r["label"], "query_class": cls,
                    "pred": pred, "p_present": round(p_present, 6),
                    "conf": round(max(p_present, 1 - p_present), 6),
                    "correct": correct}) + "\n")
                f.flush()
            except Exception as e:  # noqa: BLE001
                print(f"[skip] {r['clip']}: {type(e).__name__}: {e}", flush=True)
            if n and n % 100 == 0:
                print(f"[{n}] acc={ok / n:.3f}", flush=True)
    print(f"[done] n={n} acc={ok / max(1, n):.4f}", flush=True)


if __name__ == "__main__":
    main()
