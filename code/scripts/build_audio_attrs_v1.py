#!/usr/bin/env python3
"""Three more AUDITORY attributes with ground truth true by construction.

Why this is the most important remaining experiment. The confirmatory design currently
rests on exactly one auditory attribute (sound-event presence). For an audio conference
that is a thin base, and it leaves open the reading that the effect belongs to semantic
event labels rather than to audition. These three are classic signal-processing
manipulations, they are perceptible without any semantic knowledge, and the label is
exact because we apply the operation ourselves:

  noise    additive white noise at a fixed SNR      "is there a loud continuous hiss?"
  lowpass  band-limited to 1 kHz                    "is the audio muffled / band-limited?"
  reverb   strong artificial reverberation          "does it sound like a large echoey room?"

The video track is copied untouched in all three, so the input stays intact-AV exactly as
in the rest of the study, and half of each set is the unmodified original.
"""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

ROOT = Path("/workspace/svhalluc")
SRC = ROOT / "claude_track2/data/held_attr_soundevent_v1.jsonl"
OUT_DIR = ROOT / "claude_track2/data/held_audio_clips_v1"

FILTERS = {
    # 0 dB SNR white noise mixed in; the hiss is unmistakable but the content survives
    "noise": ("[0:a]volume=1.0[a0];anoisesrc=c=white:a=0.06:d=12[n];"
              "[a0][n]amix=inputs=2:duration=first:dropout_transition=0[aout]"),
    # hard band limitation -- everything above 1 kHz gone
    "lowpass": ("[0:a]lowpass=f=1000:poles=2,lowpass=f=1000:poles=2,"
                "lowpass=f=1000:poles=2[aout]"),   # poles is capped at 2, so cascade
    # long, obvious tail
    "reverb": "[0:a]aecho=0.8:0.9:60|110|180:0.5|0.4|0.3[aout]",
}
QUESTION_PROP = {
    "noise": "loud continuous background hiss",
    "lowpass": "muffled, band-limited sound",
    "reverb": "large echoing room",
}


def run(cmd):
    return subprocess.run(cmd, capture_output=True).returncode == 0


def make(attr, src: Path, dst: Path) -> bool:
    return run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                "-filter_complex", FILTERS[attr], "-map", "0:v", "-map", "[aout]",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", str(dst)])


def main():
    rows = [json.loads(l) for l in SRC.open() if l.strip()]
    # take the ORIGINAL clip of every row (the sound-event manifest points at originals)
    clips = sorted({r["clip"] for r in rows})
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for attr in FILTERS:
        rng = random.Random(hash(attr) % 10_000)
        man = ROOT / f"claude_track2/data/held_attr_{attr}_v1.jsonl"
        done = set()
        if man.exists():
            for line in man.read_text().splitlines():
                if line.strip():
                    done.add(json.loads(line)["clip"])
        order = list(clips)
        rng.shuffle(order)
        half = len(order) // 2
        n_w = n_f = 0
        with man.open("a", encoding="utf-8") as f:
            for k, clip in enumerate(order):
                positive = k < half           # exactly balanced by construction
                if positive:
                    dst = OUT_DIR / f"{Path(clip).stem}_{attr}.mp4"
                    rel = str(dst.relative_to(ROOT))
                    if rel in done:
                        continue
                    if not dst.exists() and not make(attr, ROOT / clip, dst):
                        n_f += 1
                        continue
                    out_clip, label = rel, "A"
                else:
                    out_clip, label = clip, "B"
                    if out_clip in done:
                        continue
                f.write(json.dumps({"clip": out_clip, "label": label, "attr": attr,
                                    "property": QUESTION_PROP[attr]}) + "\n")
                n_w += 1
        print(f"{attr}: wrote {n_w} rows (ffmpeg failures {n_f}) -> {man.name}", flush=True)


if __name__ == "__main__":
    main()
