#!/bin/bash
# The upstream detector for the real cascade. AST fine-tuned on AudioSet gives a 527-class
# posterior over an audio clip, which is exactly what a deployed sound-event tagger emits: a
# predicted class and a confidence that can be calibrated and then serialised into the prompt.
# Until now every claim in this paper was constructed by us; this is what makes the cascade real.
set -u
export HF_HOME=/data2/svhalluc_models/hf_cache HF_HUB_CACHE=/data2/svhalluc_models/hf_cache/hub
S=/workspace/svhalluc/claude_track2/outputs/fetch_det_status.txt
K(){ python3 -c "import datetime;print((datetime.datetime.utcnow()+datetime.timedelta(hours=9)).strftime('%H:%M'))"; }
for R in MIT/ast-finetuned-audioset-10-10-0.4593 laion/clap-htsat-unfused; do
  echo "[KST $(K)] start $R" >> "$S"
  python3 - "$R" << 'PY' >> "$S" 2>&1
import sys
from huggingface_hub import snapshot_download
try: print("  ->", snapshot_download(sys.argv[1], max_workers=4, ignore_patterns=["*.msgpack","*.h5","*.pth"]))
except Exception as e: print("  FAILED:", type(e).__name__, str(e)[:150])
PY
  echo "[KST $(K)] done  $R  (/data2 여유 $(df -h /data2 | awk 'NR==2{print $4}'))" >> "$S"
done
