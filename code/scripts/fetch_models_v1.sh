#!/bin/bash
# Additional AV/audio LLMs, downloaded to /data4 (workspace is at 99%).
set -u
D=/data4/svhalluc_models
# keep every byte off /workspace, which is at 98%: the hub cache defaults to
# /workspace/.cache/huggingface, so redirect it as well as the download target.
export HF_HOME=/data4/hf_cache
export HF_HUB_CACHE=/data4/hf_cache/hub
export HF_HUB_ENABLE_HF_TRANSFER=0
log(){ echo "[$(TZ=Asia/Seoul date +%H:%M)] $*" >> /workspace/svhalluc/claude_track2/outputs/fetch_status.txt; }
for M in Qwen/Qwen2-Audio-7B-Instruct microsoft/Phi-4-multimodal-instruct openbmb/MiniCPM-o-2_6; do
  N=$(basename "$M")
  log "start $N"
  python3 - "$M" "$D/$N" << 'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2],
                  ignore_patterns=["*.pth","*.onnx","*.msgpack","*.h5"],
                  max_workers=8)
PY
  log "done $N -> $(du -sh $D/$N 2>/dev/null | cut -f1)"
done
log "FETCH ALL DONE"
