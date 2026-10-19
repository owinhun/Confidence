#!/usr/bin/env python3
"""Build aligned/shifted eval pairs from ON-SERVER VGGSound clips (read-only source).

Amended protocol (log_v1.md): source = /data4/jh/vggsound/data/video/{vid}_{start:06d}.mp4
for VGGSound-Sparse is_sync=1 entries. L=6 s, video [2,8]; aligned audio [2,8];
shifted audio [4,10] (lead) or [0,6] (lag), balanced random seed 0. RMS filter.
Eval-only data; zero training contact.
"""

import csv
import json
import random
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/svhalluc")
SRC_DIR = Path("/data4/jh/vggsound/data/video")
SPARSE = ROOT / "claude_track2/external/Synchformer/data/vggsound_sparse_clean.csv"
OUT_DIR = ROOT / "claude_track2/data/vggsparse_pairs_v2"
MANIFEST = ROOT / "claude_track2/data/vggsparse_pairs_v2.jsonl"
L, DELTA, V0 = 6.0, 2.0, 2.0
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
    rows = list(csv.DictReader(SPARSE.open()))
    entries = sorted(set((r["video_id"], int(r["start_sec"])) for r in rows
                         if r["vid_is_missing"] == "0" and r["is_sync"] == "1"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["clip"])

    n_missing = n_short = n_silent = n_fail = n_written = 0
    with MANIFEST.open("a", encoding="utf-8") as f:
        for vid, ss in entries:
            src = SRC_DIR / f"{vid}_{ss:06d}.mp4"
            if not src.exists():
                n_missing += 1
                continue
            T = dur(str(src))
            if T is None or T < V0 + L + DELTA - 0.2:
                n_short += 1
                continue
            direction = rng.choice(["lead", "lag"])
            a0 = V0 + DELTA if direction == "lead" else V0 - DELTA
            if window_rms(str(src), V0, L) < RMS_MIN:
                n_silent += 1
                continue
            base = f"{vid}_{ss:06d}"
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
                                 "ytid": vid, "start_sec": ss, "delta": DELTA,
                                 "direction": direction})
            for r in rows_out:
                f.write(json.dumps(r) + "\n")
                n_written += 1
    print(f"DONE wrote {n_written} clips "
          f"(missing={n_missing} short={n_short} silent={n_silent} fail={n_fail})")


if __name__ == "__main__":
    main()
