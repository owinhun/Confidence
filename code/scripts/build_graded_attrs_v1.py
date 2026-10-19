#!/usr/bin/env python3
"""Graded acoustic degradations, so the operating point can be chosen rather than guessed.

Eight of nine low-level acoustic cells failed the perceptual-eligibility gate, and the reason
was not that the degradations were too subtle -- they were too strong and too obvious in the
wrong way. At 0 dB SNR and a 1 kHz cutoff the models answered one direction almost constantly
(recall 100/2), which is a response collapse, not a judgement. What we never did was vary the
intensity and look for a level at which the property is actually discriminable.

This builds each degradation at four strengths. Two things are then screened for, and they are
the two open directions at once:

  (a) a level where the model can judge the property itself -- unaided accuracy in [0.60, 0.95]
      with both-direction recall >= 0.60. That is what makes the confidence contrast askable
      on a low-level acoustic attribute.
  (b) a level where the degradation actually impairs SOUND-EVENT recognition. The earlier
      auxiliary-evidence task died because 0 dB noise left event accuracy at 94 %: nothing for
      a quality claim to repair. A level that costs 10-20 points gives the real-detector
      cascade something to do.

The two need not be the same level, and finding both is the point.
"""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

ROOT = Path("/workspace/svhalluc")
DATA = ROOT / "claude_track2/data"
SRC = DATA / "held_attr_soundevent_v1.jsonl"
OUT_DIR = DATA / "graded_clips_v1"

# name -> ffmpeg audio filter chain, ordered mild to severe
GRADES = {
    "noise": {  # anoisesrc amplitude; larger = louder hiss relative to the clip
        "g1": 0.005, "g2": 0.015, "g3": 0.04, "g4": 0.10,
    },
    "lowpass": {  # cutoff in Hz, one biquad stage
        "g1": 6000, "g2": 3000, "g3": 1500, "g4": 800,
    },
    "reverb": {  # aecho decay of the three taps
        "g1": 0.15, "g2": 0.30, "g3": 0.50, "g4": 0.70,
    },
}


def run(cmd):
    return subprocess.run(cmd, capture_output=True).returncode == 0


def chain(attr, level):
    v = GRADES[attr][level]
    if attr == "noise":
        return (f"[0:a]volume=1.0[a0];anoisesrc=c=white:a={v}:d=12[n];"
                f"[a0][n]amix=inputs=2:duration=first:dropout_transition=0[aout]")
    if attr == "lowpass":
        return f"[0:a]lowpass=f={v}:poles=2,lowpass=f={v}:poles=2[aout]"
    return f"[0:a]aecho=0.8:0.9:60|110|180:{v}|{v*0.8:.2f}|{v*0.6:.2f}[aout]"


def make(attr, level, src: Path, dst: Path) -> bool:
    return run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                "-filter_complex", chain(attr, level), "-map", "0:v", "-map", "[aout]",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", str(dst)])


QPROP = {"noise": "loud continuous background hiss",
         "lowpass": "muffled, band-limited sound",
         "reverb": "large echoing room"}


def main():
    se = {}
    for line in SRC.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            se[Path(r["clip"]).stem] = r
    clips = sorted(se)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for attr in GRADES:
        for level in GRADES[attr]:
            rng = random.Random(hash(attr + level) % 10_000)
            order = list(clips)
            rng.shuffle(order)
            half = len(order) // 2
            man = DATA / f"held_grad_{attr}_{level}_v1.jsonl"
            done = set()
            if man.exists():
                for line in man.read_text().splitlines():
                    if line.strip():
                        done.add(json.loads(line)["clip"])
            n = fail = 0
            with man.open("a", encoding="utf-8") as f:
                for k, stem in enumerate(order):
                    src = ROOT / se[stem]["clip"]
                    if k < half:
                        dst = OUT_DIR / f"{stem}_{attr}_{level}.mp4"
                        rel = str(dst.relative_to(ROOT))
                        if rel in done:
                            continue
                        if not dst.exists() and not make(attr, level, src, dst):
                            fail += 1
                            continue
                        clip, lab = rel, "A"
                    else:
                        clip, lab = se[stem]["clip"], "B"
                        if clip in done:
                            continue
                    f.write(json.dumps({
                        "clip": clip, "label": lab, "attr": f"{attr}_{level}",
                        "cond_label": lab, "level": level,
                        "query_class": se[stem]["query_class"],
                        "property": QPROP[attr]}) + "\n")
                    n += 1
            print(f"{attr}/{level}: {n} rows (ffmpeg fail {fail})", flush=True)


if __name__ == "__main__":
    main()
