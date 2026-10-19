#!/bin/bash
set -u
CT2=/workspace/svhalluc/claude_track2; cd "$CT2"
S=/root/.claude/jobs/65f6dcfb/tmp/batch2_status.txt; : > "$S"
log(){ echo "[$(date -u +%H:%M)] $*" | tee -a "$S"; }
# GPU0: second visual attribute on Qwen2.5 + hedge generalisation
( for e in none wrong hedge; do
    fr=expert; ev=$e; tg=""
    [ "$e" = hedge ] && { ev=wrong; fr=hedge; tg=_hedge; }
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_resistance_v1.py --attr flip --evidence $ev \
      --frame $fr --tag "$tg" > outputs/resist_flip_${e}.log 2>&1
    log "q2.5 flip/$e -> $(wc -l < outputs/resist_flip_${ev}${tg}_v1.jsonl 2>/dev/null || echo 0)"
  done
  for a in silence sound; do
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_resistance_v1.py --attr $a --evidence wrong \
      --frame hedge --tag _hedge > outputs/resist_${a}_hedge.log 2>&1
    log "q2.5 $a/hedge -> $(wc -l < outputs/resist_${a}_wrong_hedge_v1.jsonl 2>/dev/null || echo 0)"
  done
  log "GPU0 BATCH DONE" ) &
# GPU1: VideoLLaMA — hedge + text-only + correct-evidence completeness
( for a in colour silence sound; do
    CUDA_VISIBLE_DEVICES=1 python3 scripts/eval_resistance_vllama_v1.py --attr $a --evidence correct \
      > outputs/vlresist_${a}_correct.log 2>&1
    log "vl $a/correct -> $(wc -l < outputs/vlresist_${a}_correct_v1.jsonl 2>/dev/null || echo 0)"
  done
  CUDA_VISIBLE_DEVICES=1 python3 scripts/eval_resistance_vllama_v1.py --attr colour --evidence wrong \
    --frame hedge --tag _hedge > outputs/vlresist_colour_hedge.log 2>&1
  log "vl colour/hedge -> $(wc -l < outputs/vlresist_colour_wrong_hedge_v1.jsonl 2>/dev/null || echo 0)"
  log "GPU1 BATCH DONE" ) &
wait
log "BATCH2 TERMINATED"
