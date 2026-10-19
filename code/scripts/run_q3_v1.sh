#!/bin/bash
set -u
CT2=/workspace/svhalluc/claude_track2; cd "$CT2"
S=/root/.claude/jobs/65f6dcfb/tmp/q3_status.txt; : > "$S"
log(){ echo "[$(date -u +%H:%M)] $*" | tee -a "$S"; }
for a in colour silence; do
  for e in none wrong; do
    CUDA_VISIBLE_DEVICES=2 python3 scripts/eval_resistance_q3_v1.py --attr $a --evidence $e --limit 250 \
      > outputs/q3resist_${a}_${e}.log 2>&1
    log "q3 $a/$e -> $(wc -l < outputs/q3resist_${a}_${e}_v1.jsonl 2>/dev/null || echo 0)"
  done
done
log "Q3 TERMINATED"
