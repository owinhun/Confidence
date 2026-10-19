#!/bin/bash
# The models the neighbouring literature actually evaluates, chosen so that no two share a
# text backbone with each other or with the five already in the paper.
#
#   VideoLLaMA2-7B      Mistral-7B    same DAMO codebase as our AV model, LLM swapped.
#                                     A controlled backbone swap on the visual cell.
#   Video-LLaVA-7B-hf   Vicuna/LLaMA  the standard video-LLM baseline in the sycophancy work;
#                                     native transformers class, so the cheapest port.
#   InternVL2-8B        InternLM2.5   another independent lineage, widely benchmarked.
#   LLaVA-NeXT-Video-7B Vicuna        standard video baseline.
set -u
export HF_HOME=/data2/svhalluc_models/hf_cache HF_HUB_CACHE=/data2/svhalluc_models/hf_cache/hub
S=/workspace/svhalluc/claude_track2/outputs/fetch3_status.txt
K(){ python3 -c "import datetime;print((datetime.datetime.utcnow()+datetime.timedelta(hours=9)).strftime('%H:%M'))"; }
log(){ echo "[KST $(K)] $*" >> "$S"; }
for R in DAMO-NLP-SG/VideoLLaMA2-7B LanguageBind/Video-LLaVA-7B-hf OpenGVLab/InternVL2-8B lmms-lab/LLaVA-NeXT-Video-7B; do
  log "start $R"
  python3 - "$R" << 'PY' >> "$S" 2>&1
import sys
from huggingface_hub import snapshot_download
try: print("  ->", snapshot_download(sys.argv[1], max_workers=4,
        ignore_patterns=["*.msgpack","*.h5","*consolidated*","*.pth"]))
except Exception as e: print("  FAILED:", type(e).__name__, str(e)[:150])
PY
  log "done  $R  (/data2 여유 $(df -h /data2 | awk 'NR==2{print $4}'))"
done
log "WAVE 3 DONE"
