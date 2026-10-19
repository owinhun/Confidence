#!/usr/bin/env python3
"""Third-party public benchmark: official VGGSound-Sync test protocol.

Uses Synchformer's shipped fixed-offset file
(fixed_offsets_vggsound/test_size21_crop5_min-2.00_max2.00.csv), which defines,
per clip, the official visual start (vstart_sec) and the audio offset
(offset_sec) — the same protocol published sync numbers are computed on.
We build 5 s clips exactly as specified and binarize for the MCQ format:
aligned iff |offset| <= 0.2 s (Synchformer's own near-zero tolerance),
misaligned iff |offset| >= 1.0 s. Balanced sample, seed 0.

Source read-only. Output: data/vggsoundsync_v1.jsonl + clips.
"""

import csv
import json
import random
import subprocess
from pathlib import Path

ROOT = Path("/workspace/svhalluc")
SRC_DIR = Path("/data4/jh/vggsound/data/video")
FIXED = (ROOT / "claude_track2/external/Synchformer/data/fixed_offsets_vggsound"
         / "test_size21_crop5_min-2.00_max2.00.csv")
OUT_DIR = ROOT / "claude_track2/data/vggsoundsync_v2"
MANIFEST = ROOT / "claude_track2/data/vggsoundsync_v2.jsonl"
L = 6.0            # official 5 s crop + 1 s margin required by the scorer's transform
N_PER_CLASS = 300


def cut(src: str, dst: Path, v0: float, off: float) -> bool:
    a0 = v0 + off   # positive offset = audio starts earlier in the file => audio ahead
    if a0 < 0:
        return False
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-filter_complex",
                        f"[0:v]trim={v0}:{v0+L},setpts=PTS-STARTPTS[v];"
                        f"[0:a]atrim={a0}:{a0+L},asetpts=PTS-STARTPTS[a]",
                        "-map", "[v]", "-map", "[a]", "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "23", "-c:a", "aac",
                        "-b:a", "96k", str(dst)], capture_output=True)
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def main():
    rng = random.Random(0)
    rows = list(csv.DictReader(FIXED.open()))
    pool = {"aligned": [], "misaligned": []}
    for r in rows:
        ytid, start_ms, _ = r["path"].rsplit("_", 2)
        src = SRC_DIR / f"{ytid}_{int(start_ms)//1000:06d}.mp4"
        if not src.exists():
            continue
        off = float(r["offset_sec"])
        v0 = float(r["vstart_sec"])
        if v0 + L + abs(off) > 10.0:
            continue
        if abs(off) <= 0.2:
            pool["aligned"].append((src, r["path"], v0, off))
        elif abs(off) >= 1.0:
            pool["misaligned"].append((src, r["path"], v0, off))
    for k in pool:
        rng.shuffle(pool[k])
    print(f"pool aligned={len(pool['aligned'])} misaligned={len(pool['misaligned'])}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["clip"])

    n_written = n_fail = 0
    with MANIFEST.open("a", encoding="utf-8") as f:
        for kind, label in (("aligned", "A"), ("misaligned", "B")):
            taken = 0
            for src, name, v0, off in pool[kind]:
                if taken >= N_PER_CLASS:
                    break
                dst = OUT_DIR / f"{name}_off{off:+.1f}.mp4"
                rel = str(dst.relative_to(ROOT))
                if rel in done:
                    taken += 1
                    continue
                if not cut(str(src), dst, v0, off):
                    n_fail += 1
                    continue
                f.write(json.dumps({
                    "clip": rel, "label": label, "ytid": name.rsplit("_", 2)[0],
                    "offset_sec": off, "vstart_sec": v0,
                    "official_path": name}) + "\n")
                n_written += 1
                taken += 1
    print(f"DONE wrote {n_written} clips (fail={n_fail})")


if __name__ == "__main__":
    main()
