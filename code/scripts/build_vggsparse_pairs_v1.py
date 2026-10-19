#!/usr/bin/env python3
"""Build aligned/shifted eval pairs from downloaded VGGSound-Sparse segments.

Per pre-registered protocol (log_v1.md): each segment (~16 s, [start-2, start+14]
of the source video) yields video crop [2, 14] (12 s, matching the training-pair
clip length so the expert probe config and calibrated threshold transfer as-is);
ALIGNED audio = same window; SHIFTED audio = contiguous ±2 s crop (direction
random, seed 0). RMS silence filter. Eval-only data — never used for training.
"""

import glob
import json
import random
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/svhalluc")
SRC_DIR = ROOT / "claude_track2/data/vggsound_sparse_v1"
OUT_DIR = ROOT / "claude_track2/data/vggsparse_pairs_v1"
MANIFEST = ROOT / "claude_track2/data/vggsparse_pairs_v1.jsonl"
L, DELTA = 12.0, 2.0
V0 = 2.0
RMS_MIN = 0.005


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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["clip"])

    srcs = sorted(glob.glob(str(SRC_DIR / "*.mp4")))
    print(f"segments: {len(srcs)}")
    n_short = n_silent = n_fail = n_written = 0
    with MANIFEST.open("a", encoding="utf-8") as f:
        for p in srcs:
            stem = Path(p).stem  # {vid}_{s0}
            ytid = stem.rsplit("_", 1)[0]
            T = dur(p)
            if T is None or T < V0 + L + DELTA - 0.2:
                n_short += 1
                continue
            direction = rng.choice(["lead", "lag"])
            a0 = V0 + DELTA if direction == "lead" else V0 - DELTA
            if a0 + L > T + 0.2:
                direction = "lag"
                a0 = V0 - DELTA
            if window_rms(p, V0, L) < RMS_MIN:
                n_silent += 1
                continue
            rows = []
            for kind, astart in (("aligned", V0), ("shifted", a0)):
                clip = OUT_DIR / f"{stem}_{kind}.mp4"
                rel = str(clip.relative_to(ROOT))
                if rel in done:
                    continue
                if not cut(p, clip, V0, astart):
                    rows = []
                    n_fail += 1
                    break
                rows.append({"clip": rel, "label": "A" if kind == "aligned" else "B",
                             "ytid": ytid, "delta": DELTA, "direction": direction})
            for r in rows:
                f.write(json.dumps(r) + "\n")
                n_written += 1
    print(f"DONE wrote {n_written} clips (short={n_short} silent={n_silent} fail={n_fail})")


if __name__ == "__main__":
    main()
