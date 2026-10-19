#!/usr/bin/env python3
"""W1: build VGGSound-Sparse pairs at multiple offset magnitudes (pre-registered).

Same 403 source entries / 6 s window as vggsparse_pairs_v2, but with
delta in {0.2, 0.4, 0.8, 1.2} s (2.0 s already built). Aligned clips are
identical across deltas, so only ONE aligned clip per source is emitted
(reused from vggsparse_pairs_v2 via its path) and a shifted clip per delta.

Output: data/vggsparse_delta{D}_v1.jsonl (+ clips under data/vggsparse_deltas_v1/).
Source read-only. Run: python3 scripts/build_vggsparse_deltas_v1.py
"""

import json
import random
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/svhalluc")
SRC_DIR = Path("/data4/jh/vggsound/data/video")
BASE_MANIFEST = ROOT / "claude_track2/data/vggsparse_pairs_v2.jsonl"
OUT_DIR = ROOT / "claude_track2/data/vggsparse_deltas_v1"
DELTAS = [0.2, 0.4, 0.8, 1.2]
L, V0 = 6.0, 2.0
RMS_MIN = 0.005


def cut(p: str, dst: Path, v0: float, a0: float) -> bool:
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p, "-filter_complex",
                        f"[0:v]trim={v0}:{v0+L},setpts=PTS-STARTPTS[v];"
                        f"[0:a]atrim={a0}:{a0+L},asetpts=PTS-STARTPTS[a]",
                        "-map", "[v]", "-map", "[a]", "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "23", "-c:a", "aac",
                        "-b:a", "96k", str(dst)], capture_output=True)
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def main():
    base = [json.loads(l) for l in BASE_MANIFEST.open() if l.strip()]
    aligned = {r["ytid"] + "_" + f"{r['start_sec']:06d}": r
               for r in base if r["label"] == "A"}
    shifted = {r["ytid"] + "_" + f"{r['start_sec']:06d}": r
               for r in base if r["label"] == "B"}
    print(f"base pairs: {len(aligned)} aligned / {len(shifted)} shifted")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for delta in DELTAS:
        man = ROOT / f"claude_track2/data/vggsparse_delta{delta}_v1.jsonl"
        done = set()
        if man.exists():
            for line in man.read_text().splitlines():
                if line.strip():
                    done.add(json.loads(line)["clip"])
        n_written = n_fail = 0
        with man.open("a", encoding="utf-8") as f:
            for key, arow in aligned.items():
                srow = shifted.get(key)
                if srow is None:
                    continue
                direction = srow["direction"]  # keep the same direction as delta=2
                src = SRC_DIR / f"{key}.mp4"
                if not src.exists():
                    continue
                # aligned row reuses the existing clip (identical for all deltas)
                arel = arow["clip"]
                if arel not in done:
                    f.write(json.dumps({**arow, "delta": delta}) + "\n")
                    n_written += 1
                a0 = V0 + delta if direction == "lead" else V0 - delta
                dst = OUT_DIR / f"{key}_shift{delta}.mp4"
                rel = str(dst.relative_to(ROOT))
                if rel in done:
                    continue
                if not cut(str(src), dst, V0, a0):
                    n_fail += 1
                    continue
                f.write(json.dumps({
                    "clip": rel, "label": "B", "ytid": srow["ytid"],
                    "start_sec": srow["start_sec"], "delta": delta,
                    "direction": direction}) + "\n")
                n_written += 1
        print(f"delta={delta}: wrote {n_written} rows (fail={n_fail}) -> {man.name}")


if __name__ == "__main__":
    main()
