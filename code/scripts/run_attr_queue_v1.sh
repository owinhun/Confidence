#!/bin/bash
set -u
CT2=/workspace/svhalluc/claude_track2; cd "$CT2"
S=/root/.claude/jobs/65f6dcfb/tmp/attr_status.txt; : > "$S"
log(){ echo "[$(date -u +%H:%M)] $*" | tee -a "$S"; }
for pair in "colour 0" "silence 2"; do
  set -- $pair; a=$1; g=$2
  ( for ev in none correct wrong; do
      CUDA_VISIBLE_DEVICES=$g python3 scripts/eval_resistance_v1.py --attr $a --evidence $ev \
        > outputs/resist_${a}_${ev}.log 2>&1
      log "$a/$ev done: $(wc -l < outputs/resist_${a}_${ev}_v1.jsonl 2>/dev/null || echo 0)"
    done ) &
done
wait
log "ATTR QUEUE TERMINATED"
