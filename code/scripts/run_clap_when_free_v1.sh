#!/bin/bash
# CLAP over the full held-out set, started the moment a card has room. It needs ~6 GB, so it
# fits beside a 7B job and only has to wait out a Nemotron one; polling for space rather than
# for wave3 to finish keeps it off the critical path.
set -u
cd /workspace/svhalluc/claude_track2
S=outputs/clap_status.txt
K(){ python3 -c "import datetime;print((datetime.datetime.utcnow()+datetime.timedelta(hours=9)).strftime('%m-%d %H:%M'))"; }
log(){ echo "[KST $(K)] $*" >> "$S"; }
log "waiting for a card with 4.5 GB free"
while :; do
  for g in 0 1 2 3; do
    f=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits)
    # CLAP-HTSAT is a 600 MB model; it fits beside the 7B and 30B jobs rather than waiting for a
    # slot it would never win against workers polling on a 30 GB threshold.
    if [ "$f" -gt 4500 ]; then
      log "GPU $g free (${f} MiB); starting CLAP over 400 clips"
      CUDA_VISIBLE_DEVICES=$g timeout 7200 python3 scripts/run_clap_detector_v1.py \
        > outputs/clap_detector_v1.log 2>&1
      n=$(wc -l < outputs/clap_detector_v1.jsonl 2>/dev/null || echo 0)
      log "CLAP done -> $n rows"
      exit 0
    fi
  done
  sleep 20
done
