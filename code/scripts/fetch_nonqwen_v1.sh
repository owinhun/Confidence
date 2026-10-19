#!/bin/bash
# Pull the non-Qwen candidates. Ordered by porting cost, cheapest first: models with native
# transformers classes need only a chat-template + option-logit read-out, while repo-specific
# ones (SALMONN, Baichuan-Omni) need their own inference stack.
#
#   gemma-3n-E4B      Google      Gemma 3n LLM + USM audio + MobileNet-v5 vision   audio+vision
#   ultravox v0.5     Fixie       Llama-3.1-8B + Whisper encoder                   audio
#   MERaLiON          A*STAR      SEA-LION LLM + Whisper                           audio
#
# None of the three shares a backbone with Qwen, or with each other.
set -u
export HF_HOME=/data4/hf_cache HF_HUB_CACHE=/data4/hf_cache/hub
S=/workspace/svhalluc/claude_track2/outputs/fetch_status.txt
K(){ python3 -c "import datetime;print((datetime.datetime.utcnow()+datetime.timedelta(hours=9)).strftime('%H:%M'))"; }
log(){ echo "[KST $(K)] $*" >> "$S"; }
for R in google/gemma-3n-E4B-it fixie-ai/ultravox-v0_5-llama-3_1-8b MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION; do
  log "fetch start $R"
  python3 - "$R" << 'PY' >> "$S" 2>&1
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1], max_workers=4,
                      ignore_patterns=["*.pth","*.msgpack","*.h5","*consolidated*"])
print("  ->", p)
PY
  log "fetch done  $R  (남은 공간 $(df -h /data4 | awk 'NR==2{print $4}'))"
done
log "ALL FETCHES DONE"
