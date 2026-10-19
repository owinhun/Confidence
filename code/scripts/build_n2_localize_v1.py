#!/usr/bin/env python3
"""N2 (pre-registered): temporal localization of a desync — a task needing
visual semantics, not just a global offset posterior.

Each clip is 8 s of video from ONE source. The audio matches the video for one
half and is replaced by audio from a DIFFERENT VGGSound class for the other
half (contiguous crops, no splices inside a half). The model must say WHICH
half is wrong. A global sync score cannot answer this; per-half expert scores
can, so we will compute them as the expert-only baseline (giving the specialist
every advantage, per the review's matched-protocol requirement).

Output: data/n2_localize_v1.jsonl + clips under data/n2_localize_v1/
Source read-only.
"""

import csv
import json
import random
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/svhalluc")
SRC = Path("/data4/jh/vggsound/data/video")
SF = ROOT / "claude_track2/external/Synchformer/data"
OUT_DIR = ROOT / "claude_track2/data/n2_localize_v1"
MANIFEST = ROOT / "claude_track2/data/n2_localize_v1.jsonl"
V0, HALF = 1.0, 4.0          # video window [1, 9]; halves [1,5] and [5,9]
RMS_MIN = 0.005
N_TARGET = 400


def rms(p: str, start: float, length: float) -> float:
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(length),
                          "-i", p, "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
                         capture_output=True)
    if not out.stdout:
        return 0.0
    x = np.frombuffer(out.stdout, dtype="<i2").astype("float32") / 32768.0
    return float(np.sqrt((x ** 2).mean())) if len(x) else 0.0


def build(vsrc: str, asrc: str, dst: Path, bad_half: str) -> bool:
    """Video [V0, V0+2*HALF]; audio = own for the good half, donor for the bad half."""
    if bad_half == "second":
        fc = (f"[0:v]trim={V0}:{V0+2*HALF},setpts=PTS-STARTPTS[v];"
              f"[0:a]atrim={V0}:{V0+HALF},asetpts=PTS-STARTPTS[a0];"
              f"[1:a]atrim={V0+HALF}:{V0+2*HALF},asetpts=PTS-STARTPTS[a1];"
              f"[a0][a1]concat=n=2:v=0:a=1[a]")
    else:
        fc = (f"[0:v]trim={V0}:{V0+2*HALF},setpts=PTS-STARTPTS[v];"
              f"[1:a]atrim={V0}:{V0+HALF},asetpts=PTS-STARTPTS[a0];"
              f"[0:a]atrim={V0+HALF}:{V0+2*HALF},asetpts=PTS-STARTPTS[a1];"
              f"[a0][a1]concat=n=2:v=0:a=1[a]")
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", vsrc, "-i", asrc,
                        "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                        "-c:a", "aac", "-b:a", "96k", str(dst)], capture_output=True)
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def main():
    rng = random.Random(0)
    sparse = set(r[0].strip() for r in csv.reader((SF / "picked_sparse_classes.csv").open(),
                                                  delimiter="\t")
                 if len(r) == 2 and r[1].strip() == "y")
    by_class = {}
    for ytid, start, klass, split in csv.reader((SF / "vggsound.csv").open()):
        if klass in sparse and split == "test":
            p = SRC / f"{ytid}_{int(start):06d}.mp4"
            if p.exists():
                by_class.setdefault(klass, []).append((str(p), ytid, int(start)))
    classes = sorted(by_class)
    print(f"classes {len(classes)}, clips {sum(len(v) for v in by_class.values())}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["clip"])

    pool = [(c, e) for c in classes for e in by_class[c]]
    rng.shuffle(pool)
    n_w = n_fail = 0
    with MANIFEST.open("a", encoding="utf-8") as f:
        for cls, (vpath, ytid, start) in pool:
            if n_w >= N_TARGET:
                break
            donor_cls = rng.choice([c for c in classes if c != cls])
            dpath, dytid, dstart = rng.choice(by_class[donor_cls])
            bad = rng.choice(["first", "second"])
            base = f"{ytid}_{start:06d}_bad-{bad}"
            dst = OUT_DIR / f"{base}.mp4"
            rel = str(dst.relative_to(ROOT))
            if rel in done:
                continue
            if rms(vpath, V0, 2 * HALF) < RMS_MIN or rms(dpath, V0, 2 * HALF) < RMS_MIN:
                continue
            if not build(vpath, dpath, dst, bad):
                n_fail += 1
                continue
            f.write(json.dumps({
                "clip": rel, "label": "A" if bad == "first" else "B",
                "bad_half": bad, "ytid": ytid, "start_sec": start,
                "class": cls, "donor_class": donor_cls, "donor": dytid}) + "\n")
            n_w += 1
            if n_w % 50 == 0:
                print(f"{n_w} written", flush=True)
    print(f"DONE wrote {n_w} clips (fail={n_fail})")


if __name__ == "__main__":
    main()
