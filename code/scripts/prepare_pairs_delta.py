#!/usr/bin/env python3
"""Evidence-Contrast Scoring, step 1: build matched aligned/shifted AV pairs.

For each ground-truth-Yes TA sample with duration >= MIN_DUR, produce two clips
with IDENTICAL visual content (video[0 : T-DELTA]) and different audio:
  - aligned: audio[0 : T-DELTA]          (original AV relationship preserved)
  - shifted: audio[DELTA : T]            (audio leads video by DELTA seconds)

The within-pair comparison p(Yes|aligned) vs p(Yes|shifted) cancels option-slot
priors, option-content priors, and decoding effects by construction: question,
options, letters, and visual stream are identical across the pair; only the
audio-visual temporal relationship differs. Under the null (no synchrony
sensitivity) the sign of the difference is symmetric -> chance = 50%.

Re-encodes with libx264/aac for exact cuts (stream-copy cuts snap to keyframes
and would corrupt the intended offset).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import os
ROOT = Path("/workspace/svhalluc")
DELTA_ENV = float(os.environ.get("DELTA", "8"))
NMAX = int(os.environ.get("NMAX", "80"))
MANIFEST = ROOT / "data/ter_large_eval_v1.jsonl"
OUT_DIR = ROOT / f"claude_track2/outputs/ta_contrast_clips_d{int(DELTA_ENV)}"
PAIR_MANIFEST = ROOT / f"claude_track2/outputs/ta_contrast_pairs_d{int(DELTA_ENV)}.jsonl"
DELTA = DELTA_ENV
MIN_DUR = DELTA_ENV * 2


def probe_duration(path: str) -> float | None:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return None


def make_clip(src: str, dst: Path, v_start: float, v_end: float, a_start: float, a_end: float) -> bool:
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", src,
        "-filter_complex",
        f"[0:v]trim={v_start}:{v_end},setpts=PTS-STARTPTS[v];"
        f"[0:a]atrim={a_start}:{a_end},asetpts=PTS-STARTPTS[a]",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def main() -> None:
    rows = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    ta_yes = [r for r in rows if r["task"] == "TA" and r["answer"] == "A"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    done = set()
    if PAIR_MANIFEST.exists():
        for line in PAIR_MANIFEST.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["sample_id"])

    n_ok = n_skip = n_fail = 0
    with PAIR_MANIFEST.open("a", encoding="utf-8") as f:
        for r in ta_yes[:NMAX]:
            sid = r["sample_id"]
            if sid in done:
                continue
            dur = probe_duration(r["video_path"])
            if dur is None or dur < MIN_DUR:
                n_skip += 1
                continue
            t_end = dur - DELTA
            aligned = OUT_DIR / f"{sid}_aligned.mp4"
            shifted = OUT_DIR / f"{sid}_shifted.mp4"
            ok1 = make_clip(r["video_path"], aligned, 0, t_end, 0, t_end)
            ok2 = make_clip(r["video_path"], shifted, 0, t_end, DELTA, dur)
            if not (ok1 and ok2):
                n_fail += 1
                print(sid, "FFMPEG FAIL", flush=True)
                continue
            f.write(json.dumps({
                "sample_id": sid, "question": r["question"],
                "orig_video": r["video_path"], "duration": dur, "delta": DELTA,
                "aligned_path": str(aligned), "shifted_path": str(shifted),
            }) + "\n")
            f.flush()
            n_ok += 1
            print(sid, f"ok (dur={dur:.1f})", flush=True)

    print(f"\nDONE: pairs={n_ok} skipped_short={n_skip} failed={n_fail}")


if __name__ == "__main__":
    main()
