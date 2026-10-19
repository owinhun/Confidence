#!/usr/bin/env python3
"""Build T3 training pairs from the VGGSound TRAIN split (sparse classes, on-server).

Per pre-registered protocol (log_v1.md): sample 1,250 train-split sparse-class
clips (seed 0), one aligned/shifted pair each (6 s, +-2 s contiguous, RMS filter).
Source read-only; outputs under claude_track2/data/. 10% of ytids -> split=val.
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
OUT_DIR = ROOT / "claude_track2/data/vggtrain_pairs_v1"
MANIFEST = ROOT / "claude_track2/data/vggtrain_pairs_v1.jsonl"
N_CLIPS = 1250
L, DELTA, V0 = 6.0, 2.0, 2.0
RMS_MIN = 0.005
VAL_FRAC = 0.10


def dur(p: str) -> float | None:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", p], capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return None


def window_rms(p: str, start: float, length: float) -> float:
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(length),
                          "-i", p, "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
                         capture_output=True)
    if not out.stdout:
        return 0.0
    x = np.frombuffer(out.stdout, dtype="<i2").astype("float32") / 32768.0
    return float(np.sqrt((x ** 2).mean())) if len(x) else 0.0


def cut(p: str, dst: Path, v0: float, a0: float) -> bool:
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p, "-filter_complex",
                        f"[0:v]trim={v0}:{v0+L},setpts=PTS-STARTPTS[v];"
                        f"[0:a]atrim={a0}:{a0+L},asetpts=PTS-STARTPTS[a]",
                        "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "23", "-c:a", "aac", "-b:a", "96k", str(dst)],
                       capture_output=True, text=True)
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def main():
    rng = random.Random(0)
    sparse_classes = set(r[0].strip() for r in csv.reader(
        (SF_DATA / "picked_sparse_classes.csv").open(), delimiter="\t")
        if len(r) == 2 and r[1].strip() == "y")
    cands = []
    for ytid, start, klass, split in csv.reader((SF_DATA / "vggsound.csv").open()):
        if klass in sparse_classes and split == "train":
            p = SRC_DIR / f"{ytid}_{int(start):06d}.mp4"
            if p.exists():
                cands.append((ytid, int(start), klass))
    rng.shuffle(cands)
    picked = cands[:N_CLIPS]
    ytids = sorted(set(c[0] for c in picked))
    rng2 = random.Random(1)
    rng2.shuffle(ytids)
    val_ids = set(ytids[:max(1, int(len(ytids) * VAL_FRAC))])
    print(f"candidates {len(cands)}, picked {len(picked)}, "
          f"ytids {len(ytids)} (val {len(val_ids)})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["clip"])

    n_short = n_silent = n_fail = n_written = 0
    with MANIFEST.open("a", encoding="utf-8") as f:
        for ytid, ss, klass in picked:
            src = SRC_DIR / f"{ytid}_{ss:06d}.mp4"
            T = dur(str(src))
            if T is None or T < V0 + L + DELTA - 0.2:
                n_short += 1
                continue
            direction = rng.choice(["lead", "lag"])
            a0 = V0 + DELTA if direction == "lead" else V0 - DELTA
            if window_rms(str(src), V0, L) < RMS_MIN:
                n_silent += 1
                continue
            base = f"{ytid}_{ss:06d}"
            rows_out = []
            for kind, astart in (("aligned", V0), ("shifted", a0)):
                clip = OUT_DIR / f"{base}_{kind}.mp4"
                rel = str(clip.relative_to(ROOT))
                if rel in done:
                    continue
                if not cut(str(src), clip, V0, astart):
                    rows_out = []
                    n_fail += 1
                    break
                rows_out.append({"clip": rel, "label": "A" if kind == "aligned" else "B",
                                 "ytid": ytid, "start_sec": ss, "class": klass,
                                 "delta": DELTA, "direction": direction,
                                 "split": "val" if ytid in val_ids else "train"})
            for r in rows_out:
                f.write(json.dumps(r) + "\n")
                n_written += 1
            if n_written and n_written % 200 == 0:
                print(f"{n_written} clips written")
    print(f"DONE wrote {n_written} clips (short={n_short} silent={n_silent} fail={n_fail})")


if __name__ == "__main__":
    main()
