#!/usr/bin/env python3
"""The auxiliary-evidence cascade: metadata the detector CAN produce, on a question it CANNOT answer.

Every utility measurement so far shares a defect that would sink the method claim: the
downstream question is exactly what the upstream detector detects. Ask "is the audio dominated
by hiss?" while an audio detector answers that at AUC 0.966 and the omni model is redundant --
detector-only wins and the cascade is pointless.

This builds the version where the two roles come apart.

  detector  : hears audio only, estimates the ACOUSTIC CONDITION (hiss / band-limited /
              reverberant) -- exactly what our spectral detectors already do, AUC 0.92-1.00
  question  : the SOUND EVENT ("does this clip contain the sound of <class>?") on the same
              clip, which the acoustic-condition detector cannot answer at all
  role of metadata: signal-quality context. Knowing the audio is badly degraded is a reason to
              weight the video more heavily; knowing it is clean is a reason to trust what was
              heard. The claim is auxiliary evidence, never the answer.

The clips are the sound-event held-out clips (half degraded by us, half untouched), so the
sound-event ground truth and the query class carry over exactly, and the acoustic-condition
ground truth is exact by construction. Detector-only is undefined on this question, which is
the point; the baseline becomes audio-only vs audio+video.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/workspace/svhalluc")
DATA = ROOT / "claude_track2/data"
ATTRS = ["noise", "lowpass", "reverb"]


def main():
    # sound-event ground truth, keyed by the ORIGINAL clip stem
    se = {}
    for line in (DATA / "held_attr_soundevent_v1.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        se[Path(r["clip"]).stem] = r

    for attr in ATTRS:
        src = DATA / f"held_attr_{attr}_v1.jsonl"
        out = DATA / f"held_aux_{attr}_v1.jsonl"
        n = miss = 0
        with out.open("w", encoding="utf-8") as f:
            for line in src.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                stem = Path(r["clip"]).stem
                base_stem = stem[: -len(f"_{attr}")] if stem.endswith(f"_{attr}") else stem
                s = se.get(base_stem)
                if s is None:
                    miss += 1
                    continue
                f.write(json.dumps({
                    "clip": r["clip"],
                    "label": s["label"],              # the SOUND EVENT answer -- the question
                    "query_class": s["query_class"],
                    "cond_label": r["label"],         # whether the degradation is present
                    "attr": attr,
                    "degraded": r["label"] == "A",
                }) + "\n")
                n += 1
        print(f"{attr:8} wrote {n} rows (unmatched {miss}) -> {out.name}", flush=True)

    # sanity: the two ground truths must be independent, otherwise the condition claim
    # leaks the sound-event answer and the whole design is circular
    import collections
    for attr in ATTRS:
        rows = [json.loads(l) for l in (DATA / f"held_aux_{attr}_v1.jsonl").read_text().splitlines() if l.strip()]
        t = collections.Counter((r["degraded"], r["label"]) for r in rows)
        n = len(rows)
        p_deg = sum(v for k, v in t.items() if k[0]) / n
        p_a = sum(v for k, v in t.items() if k[1] == "A") / n
        joint = t[(True, "A")] / n
        print(f"{attr:8} P(degraded)={p_deg:.3f} P(event=A)={p_a:.3f} "
              f"P(both)={joint:.3f} vs independent {p_deg*p_a:.3f}  "
              f"-> {'INDEPENDENT' if abs(joint - p_deg*p_a) < 0.05 else 'LEAKAGE'}")


if __name__ == "__main__":
    main()
