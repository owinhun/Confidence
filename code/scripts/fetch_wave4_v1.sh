#!/bin/bash
# LLaVA-NeXT-Video failed to load from lmms-lab/, which ships the original training-repo layout
# with no image-processor config. llava-hf/ is the converted checkpoint that the transformers
# LlavaNextVideoForConditionalGeneration class expects.
set -u
export HF_HOME=/data2/svhalluc_models/hf_cache HF_HUB_CACHE=/data2/svhalluc_models/hf_cache/hub
S=/workspace/svhalluc/claude_track2/outputs/fetch4_status.txt
K(){ python3 -c "import datetime;print((datetime.datetime.utcnow()+datetime.timedelta(hours=9)).strftime('%H:%M'))"; }
for R in llava-hf/LLaVA-NeXT-Video-7B-hf; do
  echo "[KST $(K)] start $R" >> "$S"
  python3 - "$R" << 'PY' >> "$S" 2>&1
import sys
from huggingface_hub import snapshot_download
try: print("  ->", snapshot_download(sys.argv[1], max_workers=4, ignore_patterns=["*.msgpack","*.h5","*.pth"]))
except Exception as e: print("  FAILED:", type(e).__name__, str(e)[:150])
PY
  echo "[KST $(K)] done $R" >> "$S"
done
