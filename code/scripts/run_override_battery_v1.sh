#!/bin/bash
set -u
CT2=/workspace/svhalluc/claude_track2; cd "$CT2"
S=/root/.claude/jobs/65f6dcfb/tmp/ovr_status.txt; : > "$S"
log(){ echo "[$(date -u +%H:%M)] $*" | tee -a "$S"; }
R(){ # gpu attr evidence frame tag  -> resistance run
  local g=$1 a=$2 e=$3 fr=$4 tg=$5
  CUDA_VISIBLE_DEVICES=$g python3 scripts/eval_resistance_v1.py --attr $a --evidence $e \
    --frame $fr --tag "$tg" ${6:-} > outputs/resist_${a}_${e}${tg}.log 2>&1
  log "$a/$e$tg (frame=$fr) -> $(wc -l < outputs/resist_${a}_${e}${tg}_v1.jsonl 2>/dev/null || echo 0)"
}
# GPU0: O2 framing ablation on the two extreme attributes
( for fr in bare user hedge; do
    R 0 colour wrong $fr _$fr
    R 0 silence wrong $fr _$fr
  done
  log "O2 DONE" ) &
# GPU2: O5 text-only control + O3 emphatic
( R 2 colour wrong expert _noav --no_av
  R 2 silence wrong expert _noav --no_av
  R 2 sound wrong expert _noav --no_av
  R 2 colour wrong emphatic _emphatic
  R 2 silence wrong emphatic _emphatic
  log "O5+O3 DONE" ) &
wait
log "OVERRIDE BATTERY TERMINATED"
