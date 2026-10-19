#!/usr/bin/env python3
"""Turn Qwen2-Audio's unaided answers into a second-family detector for the cascade.

Both cascade detectors so far are CLAP checkpoints. That answers whether the downstream findings
depend on one checkpoint, not whether they depend on CLAP: the reviews ask for a detector from a
different family. AST and PANNs cannot supply one -- they answer in AudioSet's vocabulary while
the manifest names classes the way VGGSound does, and loose stem matching put "people eating
apple" under "Applause".

An audio LLM sidesteps the taxonomy entirely: it is asked the query class verbatim, exactly as
CLAP is. Qwen2-Audio's no-metadata pass is already that measurement -- a yes/no decision plus the
option-letter probability behind it -- so no new inference is needed, only a reshape into the
schema `build_crossfit_cascade_v1.py` expects. Its unaided accuracy is 77.0%, close enough to
CLAP's out-of-fold 74.8% that the two detectors sit at a comparable operating point.

Qwen2-Audio is also a downstream model. Its own cell is therefore excluded downstream: a model
reading back a claim it produced itself is a different experiment from a cascade.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path("/workspace/svhalluc/claude_track2")
SRC = ROOT / "outputs/nm_qwen2audio_sound_none_v1.jsonl"
OUT = ROOT / "outputs/q2a_cascade_v1.jsonl"
MAN = ROOT / "data/held_attr_soundevent_v1.jsonl"

if __name__ == "__main__":
    man = {json.loads(l)["clip"]: json.loads(l) for l in MAN.open() if l.strip()}
    rows = []
    for l in SRC.open():
        if not l.strip():
            continue
        r = json.loads(l)
        # p_A is P(option A) and option A is "Yes", so it is P(event present) directly
        p = float(r["p_A"])
        rows.append({"clip": r["clip"], "label": r["label"],
                     "query_class": man[r["clip"]]["query_class"],
                     "pred": r["pred"], "p_present": round(p, 6),
                     "conf": round(max(p, 1 - p), 6), "correct": bool(r["correct"])})
    acc = sum(x["correct"] for x in rows) / len(rows)
    with OUT.open("w") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
    print(f"[q2a] {len(rows)} clips, raw accuracy {acc*100:.1f}%  -> {OUT.name}")
