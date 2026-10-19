#!/bin/bash
set -u
CT2=/workspace/svhalluc/claude_track2; cd "$CT2"
S=/root/.claude/jobs/65f6dcfb/tmp/o4_status.txt; : > "$S"
log(){ echo "[$(date -u +%H:%M)] $*" | tee -a "$S"; }
for a in colour silence; do
  for e in none wrong; do
    CUDA_VISIBLE_DEVICES=3 python3 scripts/eval_selfreport_o4_v1.py --attr $a --evidence $e \
      > outputs/selfreport_${a}_${e}.log 2>&1
    log "$a/$e -> $(wc -l < outputs/selfreport_${a}_${e}_v1.jsonl 2>/dev/null || echo 0) rows"
  done
done
log "O4 TERMINATED"
