#!/bin/bash
# Cross-proposition test. Three levels of relatedness between the claim and the question,
# all at the same declared confidence:
#   same proposition   claim and question are both about the sound event   (already measured)
#   same clip, other   claim about audio quality, question about the event (this run)
#   irrelevant         "uploaded on a weekday"                             (already measured)
# If destabilisation is proposition-specific the middle level costs nothing; if it is a
# general loss of confidence in the percept, it costs something between the other two.
set -u
cd /workspace/svhalluc/claude_track2
S=outputs/crossprop_status.txt
log(){ echo "[$(TZ=Asia/Seoul date +%H:%M)] $*" >> "$S"; }
wait_free(){ local need=$1; shift; local g tot
  while :; do tot=0
    for g in "$@"; do tot=$((tot+$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits))); done
    [ "$tot" -gt "$need" ] && return 0; sleep 60
  done; }
s2(){ local gpus="$1" scr=$2 pfx=$3 c=$4 need=$5 out n
  out="outputs/${pfx}_aux_noise_${c}_v1.jsonl"
  for t in 1 2 3; do
    [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && { log "$pfx/$c done"; return; }
    wait_free "$need" $gpus
    CUDA_VISIBLE_DEVICES=$(echo $gpus|tr ' ' ,) python3 "$scr" --attr aux_noise --cond "$c" \
      > "outputs/${pfx}_aux_noise_${c}.log" 2>&1
    n=$(wc -l < "$out" 2>/dev/null||echo 0); log "$pfx/$c try$t -> $n"; [ "$n" -ge 400 ] && return; sleep 20
  done; log "$pfx/$c FAILED"; }
nm(){ local gpus="$1" m=$2 c=$3 need=$4 out n
  out="outputs/nm_${m}_aux_noise_${c}_v1.jsonl"
  for t in 1 2 3; do
    [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 400 ] && { log "$m/$c done"; return; }
    wait_free "$need" $gpus
    CUDA_VISIBLE_DEVICES=$(echo $gpus|tr ' ' ,) python3 scripts/eval_newmodels_v1.py \
      --model "$m" --attr aux_noise --cond "$c" > "outputs/nm_${m}_aux_noise_${c}.log" 2>&1
    n=$(wc -l < "$out" 2>/dev/null||echo 0); log "$m/$c try$t -> $n"; [ "$n" -ge 400 ] && return; sleep 20
  done; log "$m/$c FAILED"; }

( for c in none anon_high anon_low; do s2 0 scripts/eval_stage2_factorial_v1.py s2 $c 27000; done; log "L1 DONE" ) &
( for c in none anon_high anon_low; do s2 1 scripts/eval_stage2_vllama_v1.py s2vl $c 27000; done; log "L2 DONE" ) &
( for c in none anon_high anon_low; do nm 3 qwen2audio $c 27000; done; log "L3 DONE" ) &
( for c in none anon_high anon_low; do s2 "2 3" scripts/eval_stage2_q3_v1.py s2q3 $c 78000; done; log "L4 DONE" ) &
wait
log "CROSSPROP ALL DONE"
