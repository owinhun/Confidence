#!/usr/bin/env python3
"""Build two more attributes with ground truth true BY CONSTRUCTION, so that
competence can be measured without label noise and the resistance scale gets
more than two anchor points.

  colour  : half the clips are converted to greyscale (trivially perceptible)
  silence : half the clips get a 1.5 s digital-silence gap inserted mid-clip
            (audio-side, perceptible but less salient)

Both keep the input otherwise intact — no channel is removed, so the
intact-input property of our anchor is preserved across the whole scale.
"""

import json
import random
import subprocess
from pathlib import Path

ROOT = Path("/workspace/svhalluc")
SRC_MANIFEST = ROOT / "claude_track2/data/vggsparse_pairs_v2.jsonl"
OUT_DIR = ROOT / "claude_track2/data/attr_clips_v1"
N = 400


def run(cmd):
    return subprocess.run(cmd, capture_output=True).returncode == 0


def make_grey(src: Path, dst: Path) -> bool:
    return run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf", "hue=s=0",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "copy", str(dst)])


def make_silence(src: Path, dst: Path) -> bool:
    # mute audio between 2.0 s and 3.5 s, video untouched
    return run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                "-af", "volume=enable='between(t,2.0,3.5)':volume=0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "96k", str(dst)])


def main():
    rng = random.Random(0)
    rows = [json.loads(l) for l in SRC_MANIFEST.open() if l.strip()]
    aligned = [r for r in rows if r["label"] == "A"]
    rng.shuffle(aligned)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for attr, maker, q_yes in (("colour", make_grey, "greyscale"),
                               ("silence", make_silence, "silent gap")):
        man = ROOT / f"claude_track2/data/attr_{attr}_v1.jsonl"
        done = set()
        if man.exists():
            for line in man.read_text().splitlines():
                if line.strip():
                    done.add(json.loads(line)["clip"])
        n_w = n_f = 0
        with man.open("a", encoding="utf-8") as f:
            for r in aligned[:N]:
                src = ROOT / r["clip"]
                positive = rng.random() < 0.5
                stem = Path(r["clip"]).stem
                if positive:
                    dst = OUT_DIR / f"{stem}_{attr}.mp4"
                    rel = str(dst.relative_to(ROOT))
                    if rel in done:
                        continue
                    if not maker(src, dst):
                        n_f += 1
                        continue
                    clip, label = rel, "A"          # A = yes, the property is present
                else:
                    clip, label = r["clip"], "B"    # untouched original
                    if clip in done:
                        continue
                f.write(json.dumps({"clip": clip, "label": label, "attr": attr,
                                    "ytid": r["ytid"], "property": q_yes}) + "\n")
                n_w += 1
        print(f"{attr}: wrote {n_w} rows (fail {n_f}) -> {man.name}")


if __name__ == "__main__":
    main()
