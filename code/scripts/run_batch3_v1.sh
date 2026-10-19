#!/bin/bash
set -u
CT2=/workspace/svhalluc/claude_track2; cd "$CT2"
S=/root/.claude/jobs/65f6dcfb/tmp/batch3_status.txt; : > "$S"
log(){ echo "[$(date -u +%H:%M)] $*" | tee -a "$S"; }
# GPU2: Qwen3 — flip attribute (does the visual defence generalise?) + hedge + text-only
( for e in none wrong; do
    CUDA_VISIBLE_DEVICES=2 python3 scripts/eval_resistance_q3_v1.py --attr flip --evidence $e --limit 250 \
      > outputs/q3resist_flip_${e}.log 2>&1
    log "q3 flip/$e -> $(wc -l < outputs/q3resist_flip_${e}_v1.jsonl 2>/dev/null || echo 0)"
  done
  CUDA_VISIBLE_DEVICES=2 python3 scripts/eval_resistance_q3_v1.py --attr colour --evidence wrong \
    --frame hedge --tag _hedge --limit 250 > outputs/q3resist_colour_hedge.log 2>&1
  log "q3 colour/hedge -> $(wc -l < outputs/q3resist_colour_wrong_hedge_v1.jsonl 2>/dev/null || echo 0)"
  CUDA_VISIBLE_DEVICES=2 python3 scripts/eval_resistance_q3_v1.py --attr silence --evidence wrong \
    --frame hedge --tag _hedge --limit 250 > outputs/q3resist_silence_hedge.log 2>&1
  log "q3 silence/hedge -> $(wc -l < outputs/q3resist_silence_wrong_hedge_v1.jsonl 2>/dev/null || echo 0)"
  log "GPU2 BATCH DONE" ) &
# GPU3: Qwen2.5 — text-only for the remaining attribute, sound self-report, evidence-position control
( CUDA_VISIBLE_DEVICES=3 python3 scripts/eval_resistance_v1.py --attr flip --evidence wrong \
    --tag _noav --no_av > outputs/resist_flip_noav.log 2>&1
  log "q2.5 flip/noav -> $(wc -l < outputs/resist_flip_wrong_noav_v1.jsonl 2>/dev/null || echo 0)"
  for a in colour silence sound flip; do
    CUDA_VISIBLE_DEVICES=3 python3 scripts/eval_resistance_v1.py --attr $a --evidence correct \
      --tag _chk > outputs/resist_${a}_correct_chk.log 2>&1
    log "q2.5 $a/correct -> $(wc -l < outputs/resist_${a}_correct_chk_v1.jsonl 2>/dev/null || echo 0)"
  done
  log "GPU3 BATCH DONE" ) &
wait
log "BATCH3 TERMINATED"
