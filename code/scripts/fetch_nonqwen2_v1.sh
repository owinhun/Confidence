#!/bin/bash
# Second fetch wave, landing on /data2 (690 GB free; /data4 is at 97%).
# gemma-3n is omitted: it is a gated repo and needs the account holder to accept its licence.
set -u
export HF_HOME=/data2/svhalluc_models/hf_cache HF_HUB_CACHE=/data2/svhalluc_models/hf_cache/hub
S=/workspace/svhalluc/claude_track2/outputs/fetch2_status.txt
K(){ python3 -c "import datetime;print((datetime.datetime.utcnow()+datetime.timedelta(hours=9)).strftime('%H:%M'))"; }
log(){ echo "[KST $(K)] $*" >> "$S"; }
for R in tsinghua-ee/SALMONN baichuan-inc/Baichuan-Omni-1d5; do
  log "start $R"
  python3 - "$R" << 'PY' >> "$S" 2>&1
import sys
from huggingface_hub import snapshot_download
try:
    print("  ->", snapshot_download(sys.argv[1], max_workers=4,
          ignore_patterns=["*.msgpack","*.h5","*consolidated*"]))
except Exception as e:
    print("  FAILED:", type(e).__name__, str(e)[:160])
PY
  log "done  $R  (/data2 남은 공간 $(df -h /data2 | awk 'NR==2{print $4}'))"
done
log "WAVE 2 DONE"
