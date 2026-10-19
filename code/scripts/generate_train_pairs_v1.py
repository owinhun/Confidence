#!/usr/bin/env python3
"""Self-supervised aligned/shifted pair generation for synchrony tuning (Route B).

Sources: CrossTask-family external videos with audio (113 unique YouTube IDs,
verified ZERO overlap with SVHalluc TA eval sources). For each source video,
sample non-overlapping windows of length L; each window yields one ALIGNED
clip (label Yes) and one SHIFTED clip (label No; audio taken from a contiguous
segment offset by delta in a random direction — contiguous crop, no splices,
so no editing-artifact shortcuts). Near-silent windows are skipped via RMS.

Outputs: clips under claude_track2/data/train_pairs_v1/ + manifest jsonl with
{clip, label, ytid, window_start, delta, direction, split}. 15 held-out ytids
form the synthetic validation split; SVHalluc TA remains the untouched
real-benchmark test set.
"""

from __future__ import annotations

import glob
import json
import math
import os
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/workspace/svhalluc")
OUT_DIR = ROOT / "claude_track2/data/train_pairs_v1"
MANIFEST = ROOT / "claude_track2/data/train_pairs_v1.jsonl"
L = 12.0            # clip length (sec)
DELTAS = [2.0, 4.0, 8.0, 16.0]
STRIDE = 20.0       # window stride within a long video
RMS_MIN = 0.005     # skip near-silent windows
VAL_YTIDS = 15
SEED = 0

SRC_PATTERNS = [
    "data/external/timelogic_crosstask_external80_clips_v1/*.mp4",
    "data/external/timelogic_crosstask_local_pilot_v1/*.mp4",
    "data/external/timelogic_crosstask_media_pilot_v1/*.mp4",
    "data/external/crosstask_speech_turn_pointer_v1/*.mp4",
]


def ytid(p: str) -> str:
    return os.path.basename(p).rsplit(".", 1)[0].split("_")[-1]


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
    import numpy as np
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


def main() -> None:
    rng = random.Random(SEED)
    # dedupe sources by ytid (keep first)
    seen, sources = set(), []
    for pat in SRC_PATTERNS:
        for p in sorted(glob.glob(str(ROOT / pat))):
            y = ytid(p)
            if y not in seen:
                seen.add(y)
                sources.append((y, p))
    ids = sorted(seen)
    rng.shuffle(ids)
    val_ids = set(ids[:VAL_YTIDS])
    print(f"sources: {len(sources)} unique ytids; val ytids: {len(val_ids)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["clip"])

    jobs = []
    for y, p in sources:
        T = dur(p)
        if T is None or T < L + min(DELTAS) + 1:
            continue
        n_windows = max(1, int((T - L - 16.0) // STRIDE) + 1) if T >= L + 16.0 else 1
        for w in range(n_windows):
            # choose delta/direction feasible for this window
            feas = []
            for d in DELTAS:
                w0_lead = w * STRIDE          # audio from later: needs w0+d+L <= T
                if w0_lead + L + d <= T:
                    feas.append((d, "lead", w0_lead))
                w0_lag = w * STRIDE + d       # audio from earlier: needs w0 >= d
                if w0_lag + L <= T:
                    feas.append((d, "lag", w0_lag))
            if not feas:
                continue
            d, direction, v0 = rng.choice(feas)
            jobs.append((y, p, w, d, direction, v0))

    print(f"windows planned: {len(jobs)}")

    def process(job):
        y, p, w, d, direction, v0 = job
        base = f"{y}_w{w}"
        a0 = v0 + d if direction == "lead" else v0 - d
        rows = []
        for kind, astart in (("aligned", v0), ("shifted", a0)):
            clip = OUT_DIR / f"{base}_{kind}.mp4"
            rel = str(clip.relative_to(ROOT))
            if rel in done:
                rows.append(None)
                continue
            if kind == "aligned" and window_rms(p, v0, L) < RMS_MIN:
                return []  # silent window — drop the whole pair
            if not cut(p, clip, v0, astart):
                return []
            rows.append({"clip": rel, "label": "A" if kind == "aligned" else "B",
                         "ytid": y, "window_start": v0, "delta": d, "direction": direction,
                         "split": "val" if y in val_ids else "train"})
        return [r for r in rows if r]

    n_written = 0
    with MANIFEST.open("a", encoding="utf-8") as f:
        with ThreadPoolExecutor(8) as ex:
            for rows in ex.map(process, jobs):
                for r in rows:
                    f.write(json.dumps(r) + "\n")
                    n_written += 1
        f.flush()
    print(f"DONE: wrote {n_written} new clip rows -> {MANIFEST}")


if __name__ == "__main__":
    main()
