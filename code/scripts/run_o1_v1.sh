#!/bin/bash
set -u
CT2=/workspace/svhalluc/claude_track2; cd "$CT2"
S=/root/.claude/jobs/65f6dcfb/tmp/o1_status.txt; : > "$S"
log(){ echo "[$(date -u +%H:%M)] $*" | tee -a "$S"; }
( for a in colour silence sound; do
    for e in none wrong; do
      CUDA_VISIBLE_DEVICES=1 python3 scripts/eval_resistance_vllama_v1.py --attr $a --evidence $e \
        > outputs/vlresist_${a}_${e}.log 2>&1
      log "vllama $a/$e -> $(wc -l < outputs/vlresist_${a}_${e}_v1.jsonl 2>/dev/null || echo 0)"
    done
  done; log "O1-VLLAMA DONE" ) &
wait
log "O1 TERMINATED"
