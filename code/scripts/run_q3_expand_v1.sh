#!/bin/bash
set -u
CT2=/workspace/svhalluc/claude_track2; cd "$CT2"
S=/root/.claude/jobs/65f6dcfb/tmp/q3exp_status.txt; : > "$S"
log(){ echo "[$(date -u +%H:%M)] $*" | tee -a "$S"; }
# wait for GPU2's current batch to finish, then expand the weakest cell to full n
until ! ps aux | awk '/eval_resistance_q3_v1/ && !/awk/' | grep -q .; do sleep 180; done
for e in none wrong; do
  CUDA_VISIBLE_DEVICES=2 python3 scripts/eval_resistance_q3_v1.py --attr colour --evidence $e --limit 0 \
    > outputs/q3resist_colour_${e}_full.log 2>&1
  log "q3 colour/$e expanded -> $(wc -l < outputs/q3resist_colour_${e}_v1.jsonl 2>/dev/null || echo 0)"
done
log "Q3EXPAND TERMINATED"
