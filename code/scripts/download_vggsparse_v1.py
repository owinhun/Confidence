#!/usr/bin/env python3
"""Download VGGSound-Sparse (is_sync=1) source segments from YouTube.

For each (video_id, start_sec) with vid_is_missing=0 and is_sync=1, grabs the
[start-2, start+14] section at <=360p into claude_track2/data/vggsound_sparse_v1/.
Resumes by existing file. Sequential, 120 s timeout per video.
"""

import csv
import subprocess
from pathlib import Path

SRC = Path("/workspace/svhalluc/claude_track2/external/Synchformer/data/vggsound_sparse_clean.csv")
OUT = Path("/workspace/svhalluc/claude_track2/data/vggsound_sparse_v1")
OUT.mkdir(parents=True, exist_ok=True)

rows = list(csv.DictReader(SRC.open()))
jobs, seen = [], set()
for r in rows:
    if r["vid_is_missing"] == "0" and r["is_sync"] == "1" and r["video_id"] not in seen:
        seen.add(r["video_id"])
        jobs.append((r["video_id"], int(r["start_sec"])))
print(f"jobs: {len(jobs)}")

ok = fail = skip = 0
for vid, ss in jobs:
    s0 = max(0, ss - 2)
    dst = OUT / f"{vid}_{s0}.mp4"
    if dst.exists() and dst.stat().st_size > 0:
        skip += 1
        continue
    try:
        r = subprocess.run(
            ["python3", "-m", "yt_dlp", "-q", "--no-warnings",
             "-f", "bv*[height<=360]+ba/b[height<=360]/b",
             "--download-sections", f"*{s0}-{ss+14}",
             "-o", str(OUT / f"{vid}_%(section_start)s.%(ext)s"),
             "--merge-output-format", "mp4",
             f"https://www.youtube.com/watch?v={vid}"],
            capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            ok += 1
        else:
            fail += 1
    except subprocess.TimeoutExpired:
        fail += 1
    if (ok + fail) % 25 == 0 and (ok + fail) > 0:
        print(f"progress: ok={ok} fail={fail} skip={skip}")
print(f"DONE ok={ok} fail={fail} skip={skip}")
