#!/bin/bash
set -u
CT2=/workspace/svhalluc/claude_track2; cd "$CT2"
S=/root/.claude/jobs/65f6dcfb/tmp/s2q3_status.txt; : > "$S"
log(){ echo "[$(date -u +%H:%M)] $*" | tee -a "$S"; }
# core 4 factorial cells + the two decisive controls, on one visual and one audio attribute
for a in colour silence; do
  for c in none anon_high anon_low expert_high expert_low true_high irrelevant_high; do
    CUDA_VISIBLE_DEVICES=3 python3 scripts/eval_stage2_q3_v1.py --attr $a --cond $c \
      > outputs/s2q3_${a}_${c}.log 2>&1
    log "q3 $a/$c -> $(wc -l < outputs/s2q3_${a}_${c}_v1.jsonl 2>/dev/null || echo 0)"
  done
done
log "STAGE2-Q3 TERMINATED"
