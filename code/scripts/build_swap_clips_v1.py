#!/usr/bin/env python3
"""E1: build audio-SWAP clips (semantic mismatch) for the "why inconsistent?" task.

For each VGGSound-Sparse entry used in vggsparse_pairs_v2, emit a clip whose
video is the usual [2,8] s window and whose audio comes from a DIFFERENT source
video of a DIFFERENT sparse class (same window length). Contiguous crop, no
splices; RMS filter on the donor audio.

Manifest: data/swap_pairs_v1.jsonl with kind in {temporal, semantic}; the
temporal rows point at the existing delta=2 shifted clips.
"""

import csv
import json
import random
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/svhalluc")
SRC_DIR = Path("/data4/jh/vggsound/data/video")
SF_DATA = ROOT / "claude_track2/external/Synchformer/data"
BASE = ROOT / "claude_track2/data/vggsparse_pairs_v2.jsonl"
OUT_DIR = ROOT / "claude_track2/data/swap_clips_v1"
MANIFEST = ROOT / "claude_track2/data/swap_pairs_v1.jsonl"
L, V0 = 6.0, 2.0
RMS_MIN = 0.005


def window_rms(p: str, start: float, length: float) -> float:
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(length),
                          "-i", p, "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
                         capture_output=True)
    if not out.stdout:
        return 0.0
    x = np.frombuffer(out.stdout, dtype="<i2").astype("float32") / 32768.0
    return float(np.sqrt((x ** 2).mean())) if len(x) else 0.0


def cut_swap(vsrc: str, asrc: str, dst: Path) -> bool:
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", vsrc, "-i", asrc,
                        "-filter_complex",
                        f"[0:v]trim={V0}:{V0+L},setpts=PTS-STARTPTS[v];"
                        f"[1:a]atrim={V0}:{V0+L},asetpts=PTS-STARTPTS[a]",
                        "-map", "[v]", "-map", "[a]", "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "23", "-c:a", "aac",
                        "-b:a", "96k", str(dst)], capture_output=True)
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def main():
    rng = random.Random(0)
    # class per (ytid, start)
    cls = {}
    for ytid, start, klass, split in csv.reader((SF_DATA / "vggsound.csv").open()):
        cls[(ytid, int(start))] = klass

    base = [json.loads(l) for l in BASE.open() if l.strip()]
    shifted = [r for r in base if r["label"] == "B"]
    keys = [(r["ytid"], r["start_sec"]) for r in shifted]
    by_class = {}
    for k in keys:
        by_class.setdefault(cls.get(k, "?"), []).append(k)
    print(f"entries {len(keys)}, classes {len(by_class)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["clip"])

    n_sem = n_tmp = n_fail = 0
    with MANIFEST.open("a", encoding="utf-8") as f:
        for r in shifted:
            k = (r["ytid"], r["start_sec"])
            # temporal row: reuse the existing shifted clip
            if r["clip"] not in done:
                f.write(json.dumps({"clip": r["clip"], "kind": "temporal",
                                    "ytid": r["ytid"], "start_sec": r["start_sec"],
                                    "class": cls.get(k, "?")}) + "\n")
                n_tmp += 1
            # semantic row: donor from a different class
            my_class = cls.get(k, "?")
            others = [c for c in by_class if c != my_class]
            if not others:
                continue
            donor_class = rng.choice(others)
            dk = rng.choice(by_class[donor_class])
            vsrc = SRC_DIR / f"{k[0]}_{k[1]:06d}.mp4"
            asrc = SRC_DIR / f"{dk[0]}_{dk[1]:06d}.mp4"
            dst = OUT_DIR / f"{k[0]}_{k[1]:06d}_swapfrom_{dk[0]}.mp4"
            rel = str(dst.relative_to(ROOT))
            if rel in done or not (vsrc.exists() and asrc.exists()):
                continue
            if window_rms(str(asrc), V0, L) < RMS_MIN:
                continue
            if not cut_swap(str(vsrc), str(asrc), dst):
                n_fail += 1
                continue
            f.write(json.dumps({"clip": rel, "kind": "semantic",
                                "ytid": k[0], "start_sec": k[1],
                                "class": my_class, "donor_class": donor_class,
                                "donor": f"{dk[0]}_{dk[1]:06d}"}) + "\n")
            n_sem += 1
    print(f"DONE temporal rows {n_tmp}, semantic clips {n_sem}, fail {n_fail}")


if __name__ == "__main__":
    main()
